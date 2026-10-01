# CET CCET — High-Throughput Fair-Share Video Inference

A multi-stream YOLO11 video inference pipeline focused on **fair scheduling, GPU batching, and measurable performance**.

## What this project does

The system ingests multiple video streams, groups frames into payloads, schedules payloads using weighted deficit accounting, runs batched YOLO11 inference, and records per-stream throughput and latency.

```text
Video Streams
     │
     ▼
MultiStreamCapture ──► bounded queues ──► FairScheduler
                                            │
                                            ▼
                                      GPUProcessor
                                            │
                                            ▼
                                  YOLO11 / TensorRT
                                            │
                                            ▼
                                     MetricsTracker
```

## Repository structure

| Path | Responsibility |
|---|---|
| `main.py` | End-to-end controller |
| `shared/fair_scheduler.py` | Weighted fair/deficit scheduler |
| `shared/config.py` | Runtime configuration |
| `person2_gpu/gpu_processor.py` | Batched GPU inference |
| `person3_video/video_capture.py` | Multi-stream capture and back-pressure |
| `person3_video/queue_manager.py` | Thread-safe bounded queues |
| `person4_validation/metrics.py` | Throughput and latency measurements |
| `person4_validation/fairness.py` | Fairness validation |
| `person4_validation/dashboard.py` | Console dashboard adapter |
| `gpu_server.py` | FastAPI GPU inference service |
| `gpu_server_new.py` | Experimental binary-payload GPU server |
| `DEPLOY.md` | RunPod deployment notes |

## Configuration

Edit `shared/config.py` for stream count, batch size, queue capacity, weights, model, and capture targets. Keep secrets such as RunPod credentials in environment variables or `.env`; do not commit them.

## Running

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the pipeline:

```bash
python main.py
```

Run the standalone endpoint test:

```bash
python test_endpoint.py
```

## Validation

The validation modules measure:

- total and per-stream throughput;
- p50/p95 latency;
- coefficient of variation for equal-weight fairness;
- proportionality for weighted streams;
- starvation detection.

A performance claim should always be accompanied by the hardware, model, input resolution, batch size, and test duration used to obtain it.

## Engineering notes

The current repository contains both local-GPU and RunPod/server versions because the project evolved during development. `main.py` currently uses `GPUProcessor`, while the FastAPI server files are kept as deployment alternatives.

Known cleanup opportunities include consolidating the duplicated GPU server implementations, aligning the test suite with the current `GPUProcessor` API, replacing development-only scripts with explicit examples, and adding automated CI.

## Contributing

1. Create a branch.
2. Make a focused change.
3. Run relevant tests.
4. Open a pull request describing the problem, implementation, and validation.

## License

No license file is currently present. Add an explicit license before distributing the project publicly.

## Status

Experimental research/hackathon code. Interfaces may change while the pipeline is consolidated.
