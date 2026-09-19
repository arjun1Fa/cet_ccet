import os
import cv2
import base64
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
from ultralytics import YOLO
import uvicorn
import logging
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GPU_Pod_Server")

app = FastAPI(title="GPU Video Inference Pod API")

# Global Thread Lock to protect PyTorch from concurrent CUDA racing 
# when 8 asynchronous loops hit the server simultaneously.
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

class InferenceRequest(BaseModel):
    frames: List[str]  # List of base64 encoded JPEG strings

@app.post("/predict")
async def predict(request: InferenceRequest):
    if not request.frames:
        return []
    
    try:
        imgs = []
        valid_indices = []

        for i, frame_b64 in enumerate(request.frames):
            # Decode base64 to numpy array
            img_data = base64.b64decode(frame_b64)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is not None:
                imgs.append(img)
                valid_indices.append(i)
            else:
                logger.error(f"Failed to decode frame {i}")

        # Initialize results array with empty lists
        results = [[] for _ in range(len(request.frames))]

        if imgs:
            # Sub-Batching: Chunk the network payload into 8-frame GPU compute chunks
            # to stay safely within T4 VRAM limits.
            MAX_GPU_BATCH = 8
            
            for chunk_start in range(0, len(imgs), MAX_GPU_BATCH):
                chunk_imgs = imgs[chunk_start : chunk_start + MAX_GPU_BATCH]
                actual_count = len(chunk_imgs)
                
                # TensorRT engine is compiled for EXACTLY 8 frames.
                # Pad short batches by duplicating the last frame.
                while len(chunk_imgs) < MAX_GPU_BATCH:
                    chunk_imgs.append(chunk_imgs[-1])
                
                with gpu_lock:
                    preds = model.predict(chunk_imgs, batch=MAX_GPU_BATCH, verbose=False)
                
                # Only use the first 'actual_count' results (ignore padding)
                
                # Reassemble the results into the master array
                for i, res in enumerate(preds[:actual_count]):
                    original_list_idx = chunk_start + i
                    original_idx = valid_indices[original_list_idx]
                    
                    detections = []
                    for box in res.boxes:
                        bbox = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        detections.append({
                            "bbox": bbox,
                            "confidence": conf,
                            "class_id": cls
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
