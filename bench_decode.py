import cv2, time, os
from ultralytics import YOLO

print("=== DECODE BENCHMARK ===")
cap = cv2.VideoCapture('data/BigBuckBunny.mp4')
t0 = time.time()
frames = []
for i in range(30):
    ret, f = cap.read()
    if ret:
        frames.append(f)
t1 = time.time()
print(f"30 frames decoded in {t1-t0:.3f}s = {len(frames)/(t1-t0):.1f} FPS, shape={frames[0].shape}")
cap.release()

print("\n=== TRT INFERENCE BENCHMARK ===")
ENGINE = '/workspace/yolo11l.engine'
model = YOLO(ENGINE, task='detect')

# Warmup
batch = frames[:8]
model.predict(batch, batch=8, verbose=False)

# Benchmark 5 batches of 8
times = []
for _ in range(5):
    t0 = time.time()
    model.predict(batch, batch=8, verbose=False)
    times.append(time.time() - t0)

avg = sum(times) / len(times)
fps = 8 / avg
print(f"Batch of 8 frames: avg {avg*1000:.1f}ms -> {fps:.1f} FPS per batch")
print(f"With 4 streams x 8 frames theoretical max: {fps * 1:.1f} FPS total if sequential")
