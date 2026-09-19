import os
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from typing import List
import torch
from ultralytics import YOLO
import uvicorn
import logging
import json
import asyncio


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GPU_Pod_Server")

app = FastAPI(title="GPU Video Inference Pod API")

# Thread lock to protect TensorRT execution context — not thread-safe for concurrent calls.
# FastAPI runs 'def' endpoints in a thread pool, so the lock serializes GPU access
# without blocking the event loop (unlike async def which caused 5-6s latency).
import threading
gpu_lock = threading.Lock()

# Global model loading with TensorRT acceleration
PT_MODEL = "yolo11l.pt"
TRT_ENGINE = "yolo11l.engine"

if os.path.exists(TRT_ENGINE):
    logger.info(f"Loading pre-compiled TensorRT engine: {TRT_ENGINE}")
    model = YOLO(TRT_ENGINE)
else:
    logger.info(f"TensorRT engine not found. Exporting {PT_MODEL} → TensorRT FP16...")
    logger.info("This takes 2-5 minutes on first boot only.")
    base_model = YOLO(PT_MODEL)
    base_model.export(format='engine', half=True, imgsz=640, device=0, batch=8)
    logger.info("TensorRT export complete. Loading engine...")
    model = YOLO(TRT_ENGINE)

if torch.cuda.is_available():
    logger.info("Model loaded on CUDA GPU via TensorRT (FP16 fused)")
else:
    logger.info("WARNING: CUDA not available")

@app.post("/predict")
async def predict(request: Request):
    body = await request.body()
    if not body:
        return []
    
    try:
        frame_counts_str = request.headers.get("X-Frame-Counts", "[]")
        counts = json.loads(frame_counts_str)
        
        imgs = []
        valid_indices = []
        
        offset = 0
        idx = 0
        # Parse exact byte payload! 0% JSON overhead
        while offset < len(body):
            size = int.from_bytes(body[offset:offset+4], 'little')
            offset += 4
            frame_bytes = body[offset:offset+size]
            offset += size
            
            nparr = np.frombuffer(frame_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                imgs.append(img)
                valid_indices.append(idx)
            else:
                logger.error(f"Failed to decode frame {idx}")
            idx += 1

        total_frames = sum(counts) if counts else len(imgs)
        results = [[] for _ in range(total_frames)]

        if imgs:
            def _infer():
                all_raw_results = []
                with gpu_lock:
                    try:
                        # 1. Parallel Host->Device Upload & Tensor Formatting (GPU Offloaded)
                        tensors = []
                        for img in imgs:
                            # img is BGR from cv2 (H, W, 3). Convert to NCHW RGB Float16.
                            gpu_frame = torch.from_numpy(img).cuda(device=0, non_blocking=True)
                            gpu_frame = gpu_frame.flip(-1).permute(2, 0, 1).half() / 255.0
                            # YOLO11l requires 640x640
                            gpu_frame = torch.nn.functional.interpolate(
                                gpu_frame.unsqueeze(0), size=(640, 640), mode='bilinear', align_corners=False
                            )
                            tensors.append(gpu_frame)
                            
                        # 2. Massive Tensor Fusion
                        super_tensor = torch.cat(tensors, dim=0)
                        
                        # 3. Internal Slicing Engine (Bypassing TensorRT Compiler Limit natively)
                        SAFE_BATCH = 8
                        for i in range(0, super_tensor.shape[0], SAFE_BATCH):
                            chunk = super_tensor[i : i + SAFE_BATCH]
                            
                            # Pad chunk if it's the very last one and < 8
                            if chunk.shape[0] < SAFE_BATCH:
                                pad = torch.zeros((SAFE_BATCH - chunk.shape[0], 3, 640, 640), dtype=torch.float16, device=0)
                                chunk = torch.cat([chunk, pad], dim=0)
                            
                            # Bare-metal __call__ prediction
                            chunk_results = model(chunk, verbose=False)
                            all_raw_results.extend(chunk_results[: super_tensor.shape[0] - i])
                            
                    except Exception as e:
                        logger.error(f"Tensor Fusion Error: {e}")
                return all_raw_results

            # Execute without blocking asyncio loop
            preds = await asyncio.to_thread(_infer)
            
            # Fast De-Aggregation
            for i, res in enumerate(preds):
                original_idx = valid_indices[i]
                detections = []
                
                if len(res.boxes) > 0:
                    data = res.boxes.data.cpu().numpy()
                    for row in data:
                        detections.append({
                            "bbox": row[:4].tolist(),
                            "confidence": float(row[4]),
                            "class_id": int(row[5])
                        })
                results[original_idx] = detections

        return results
        
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {
        "status": "Healthy",
        "gpu": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
