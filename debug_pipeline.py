#!/usr/bin/env python3
"""Debug script: test the exact inference path on the pod"""
import cv2
import numpy as np
import sys

# 1. Check video files
print("=== CHECKING VIDEO FILES ===")
paths = [
    'data/ForBiggerBlazes.mp4',
    'data/BigBuckBunny.mp4',
    'data/ElephantsDream.mp4',
    'data/Sintel.mp4',
]
for p in paths:
    cap = cv2.VideoCapture(p)
    ret, frame = cap.read()
    if ret:
        print(f"  {p}: OK shape={frame.shape} dtype={frame.dtype}")
    else:
        print(f"  {p}: FAILED TO READ")
    cap.release()

# 2. Check what video_capture actually produces
print("\n=== CHECKING VIDEO_CAPTURE OUTPUT ===")
from person3_video.queue_manager import StreamQueueManager
from person3_video.video_capture import MultiStreamCapture
import time

qm = StreamQueueManager(num_streams=4, maxsize=10)
sources = {i: p for i, p in enumerate(paths)}
mc = MultiStreamCapture(queue_manager=qm, video_sources=sources, target_fps=30, use_gpu=False)
mc.start()
time.sleep(3)

queues = qm.get_queues()
for i, q in enumerate(queues):
    if not q.empty():
        payload = q.get_nowait()
        frames = payload.get('frames')
        print(f"  Stream {i}: type={type(frames)}, len={len(frames) if hasattr(frames, '__len__') else 'N/A'}")
        if hasattr(frames, '__len__') and len(frames) > 0:
            f = frames[0] if isinstance(frames, list) else frames[0]
            print(f"    First frame: type={type(f)}, shape={f.shape}, dtype={f.dtype}")
            # Check if it's a valid BGR image (H, W, 3)
            if len(f.shape) == 3 and f.shape[2] == 3:
                print(f"    Format: VALID BGR (H={f.shape[0]}, W={f.shape[1]}, C={f.shape[2]})")
            elif len(f.shape) == 3 and f.shape[0] == 3:
                print(f"    Format: INVALID - Channel-first (C={f.shape[0]}, H={f.shape[1]}, W={f.shape[2]})")
            else:
                print(f"    Format: UNKNOWN shape {f.shape}")
    else:
        print(f"  Stream {i}: EMPTY QUEUE")

mc.stop()

# 3. Test model.predict with a known good frame
print("\n=== TESTING MODEL.PREDICT ===")
from ultralytics import YOLO
cap = cv2.VideoCapture('data/BigBuckBunny.mp4')
frames = []
for _ in range(8):
    ret, frame = cap.read()
    if ret:
        frames.append(frame)
cap.release()

print(f"  Frames collected: {len(frames)}, shape={frames[0].shape}")

model = YOLO('/workspace/yolo11l.engine', task='detect')
try:
    results = model.predict(frames, batch=8, verbose=False)
    print(f"  model.predict SUCCESS: {len(results)} results")
except Exception as e:
    print(f"  model.predict FAILED: {e}")

print("\n=== DONE ===")
