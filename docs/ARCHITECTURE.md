# Architecture

## Data flow

`StreamWorker` reads frames and emits fixed-size frame payloads into a bounded queue. `FairScheduler` selects payloads according to per-stream weights. `GPUProcessor` flattens selected payloads, resizes frames, performs YOLO inference in bounded GPU chunks, and reconstructs results per stream. `MetricsTracker` records processed frame counts and per-frame latency.

## Scheduling model

Each stream accumulates credits according to its configured weight. A payload consumes one credit. When a queue is empty, its unused credit remains available for a later cycle, providing deficit-style compensation.

## Back-pressure

Capture workers avoid unlimited queue growth. When a queue is near capacity, incoming frames are skipped; when the queue is full, the oldest payload may be discarded before enqueueing the newest payload. This prioritizes recent video over stale backlog.

## GPU execution

`GPUProcessor` uses a thread pool for asynchronous work and protects model execution with a lock. Frames are resized to 640×640 and processed in bounded chunks. TensorRT engines are loaded when present; otherwise the configured `.pt` model is used.

## Validation

Fairness is evaluated with coefficient of variation for equal weights and weight-normalized throughput for weighted schedules. Starvation and proportionality checks live in `person4_validation/fairness.py`.

## Current limitations

The repository currently contains legacy RunPod HTTP client tests and two GPU-server implementations alongside the local `GPUProcessor`. Consolidating these paths into one interface is a recommended next step.
