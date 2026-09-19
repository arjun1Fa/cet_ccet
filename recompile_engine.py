#!/usr/bin/env python3
"""
Re-exports yolo11l.pt to a TensorRT engine with batch=32.
Run ONCE on the RunPod: python recompile_engine.py
This takes ~3-5 minutes. Only needed when changing batch size.
"""
import os
from ultralytics import YOLO

PT_MODEL = '/workspace/yolo11l.pt'
TRT_ENGINE = '/workspace/yolo11l_b32.engine'

print(f"Exporting {PT_MODEL} -> TensorRT FP16, batch=32...")
print("This will take 3-5 minutes. Go get a coffee.")

model = YOLO(PT_MODEL)
model.export(
    format='engine',
    half=True,
    imgsz=640,
    device=0,
    batch=32,
    workspace=4,   # GB of GPU workspace for TRT optimizer
)

# Ultralytics saves it next to the .pt file
generated = '/workspace/yolo11l.engine'
if os.path.exists(generated):
    os.rename(generated, TRT_ENGINE)
    print(f"\nDone! Engine saved to: {TRT_ENGINE}")
else:
    print("\nExport complete. Check /workspace/ for the .engine file.")
