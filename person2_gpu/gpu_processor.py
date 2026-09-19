import torch
from ultralytics import YOLO
import threading
import os
import cv2
import time
import logging
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('NativeGPUProcessor')

class GPUProcessor:
    """
    Cloud-Native GPU Processor — runs YOLO inference directly on the A40.
    No HTTP, no network. Pure local GPU execution.
    """
    def __init__(self, device_id=0, use_tensorrt=False, use_fp16=False):
        self.executor = ThreadPoolExecutor(max_workers=4)

        TRT_ENGINE = "/workspace/yolo11l.engine"
        PT_MODEL  = "/workspace/yolo11l.pt"

        self.device_id = device_id
        self.gpu_lock = threading.Lock()

        if os.path.exists(TRT_ENGINE):
            logger.info(f"Loading TensorRT engine: {TRT_ENGINE}")
            self.model = YOLO(TRT_ENGINE, task='detect')
        else:
            logger.info(f"Loading PyTorch model: {PT_MODEL}")
            self.model = YOLO(PT_MODEL, task='detect')

    def process_batch_async(self, batch, callback=None):
        if not batch: return
        future = self.executor.submit(self._process_batch_internal, batch)
        if callback:
            def _done_callback(fut):
                try:
                    res = fut.result()
                    if res: callback(batch, res)
                except Exception as e:
                    logger.error(f"Async callback error: {e}")
            future.add_done_callback(_done_callback)

    def _process_batch_internal(self, batch):
        if not batch: return []

        all_results = []
        try:
            all_imgs = []
            frame_counts = []

            for payload in batch:
                frames_cv = payload.get('frames')
                if frames_cv is None or len(frames_cv) == 0:
                    frame_counts.append(0)
                    continue

                frame_counts.append(len(frames_cv))
                for img in frames_cv:
                    # Resize to 640x640 to guarantee uniform batch dimensions
                    # (videos have mixed resolutions: 720p, 546p, etc.)
                    resized = cv2.resize(img, (640, 640))
                    all_imgs.append(resized)

            if not all_imgs:
                return []

            MAX_GPU_BATCH = 8
            all_raw_results = []

            with self.gpu_lock:
                for chunk_start in range(0, len(all_imgs), MAX_GPU_BATCH):
                    chunk_imgs = all_imgs[chunk_start : chunk_start + MAX_GPU_BATCH]
                    actual_count = len(chunk_imgs)

                    # Pad to exact batch size for TRT engine
                    while len(chunk_imgs) < MAX_GPU_BATCH:
                        chunk_imgs.append(chunk_imgs[-1])

                    preds = self.model.predict(chunk_imgs, batch=MAX_GPU_BATCH, verbose=False)
                    all_raw_results.extend(preds[:actual_count])

            # Parse detections
            current_frame_idx = 0
            for i, payload in enumerate(batch):
                stream_id = payload.get('stream_id', -1)
                timestamps = payload.get('timestamps', [])
                num_frames = frame_counts[i]

                payload_dets = all_raw_results[current_frame_idx : current_frame_idx + num_frames]
                current_frame_idx += num_frames

                parsed = []
                for res in payload_dets:
                    boxes, scores, classes = [], [], []
                    if hasattr(res, 'boxes') and res.boxes is not None and len(res.boxes) > 0:
                        for box in res.boxes:
                            boxes.append(box.xyxy[0].tolist())
                            scores.append(float(box.conf[0]))
                            classes.append(int(box.cls[0]))
                    parsed.append({"boxes": boxes, "scores": scores, "classes": classes})

                all_results.append({
                    'stream_id': stream_id,
                    'detections': parsed,
                    'timestamps': timestamps,
                    'num_frames': num_frames
                })

            return all_results

        except Exception as e:
            logger.error(f"Inference error: {e}")
            import traceback
            traceback.print_exc()
            return []
