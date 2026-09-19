# 🚀 High-Throughput Fair-Share Video Inference

![Throughput](https://img.shields.io/badge/Throughput-100%2B_FPS-success)
![Fairness](https://img.shields.io/badge/Fairness_CV-<0.10-blue)
![GPU](https://img.shields.io/badge/GPU_Target-TensorRT_FP16-orange)

[cite_start]A highly optimized, multi-threaded video processing pipeline capable of running YOLOv11 object detection on 10 simultaneous video streams. [cite_start]Built for a 36-hour hackathon, this project solves the problem of "stream starvation" by combining aggressive GPU optimization with a mathematically fair scheduling algorithm[cite: 10, 42].

## 📖 Plain-English Overview
Imagine a busy restaurant kitchen (the GPU) trying to cook orders for 10 different tables (the video streams) at the same time. [cite_start]If the chef only cooks for the loudest tables, the quiet ones will starve. 

This project acts as the perfect restaurant manager. [cite_start]It uses a **Fair Scheduler** to ensure every table gets their food fairly, while the **GPU Optimizer** upgrades the kitchen equipment so the chef can cook massive batches of food in a fraction of the time. [cite_start]The result is blazing-fast object detection (100+ FPS) without ignoring any video feeds[cite: 33, 35].

## ✨ Key Features
* [cite_start]**Massive Throughput:** Achieves >100 FPS across 10 streams using TensorRT and CUDA stream pipelining[cite: 43, 67].
* [cite_start]**Provable Fairness:** Implements a Weighted Round-Robin scheduler ensuring proportional frame processing (CV < 0.10)[cite: 31, 44].
* [cite_start]**Deficit Compensation:** If a stream briefly drops out, it saves its "credits" to catch up later without penalizing others[cite: 17].
* [cite_start]**Real-Time Dashboard:** Live visualization of GPU utilization, latency (p95 < 200ms), and fairness metrics[cite: 127, 130].

---

## 🏗️ Architecture & Modules

The system is cleanly decoupled into four specialized roles:

### 1. The Video Pipeline (Data Capture)
* [cite_start]**What it does:** Feeds 10 video streams into the system fast and reliably[cite: 87].
* [cite_start]**How it works:** Spawns 10 independent OpenCV daemon threads[cite: 95, 117]. [cite_start]Each thread grabs frames at 30 FPS and places them into thread-safe holding queues (maximum 50 frames per queue)[cite: 91, 100]. [cite_start]It includes back-pressure mechanisms to slow down if the queues get too full[cite: 117].

### 2. The Fair Scheduler (The Brain)
* [cite_start]**What it does:** Decides exactly which frames go to the GPU and when[cite: 9].
* **How it works:** It uses a "credit" system. [cite_start]Every round, each stream gets a certain number of credits based on its priority weight[cite: 15]. [cite_start]The scheduler pulls frames from streams with the most credits[cite: 16]. [cite_start]If a stream is empty, it keeps its credits for the next round (Deficit Round-Robin)[cite: 17]. [cite_start]This adds less than 2ms of overhead per batch[cite: 34].

### 3. GPU Optimization (The Engine)
* [cite_start]**What it does:** Squeezes maximum performance out of the graphics card[cite: 57].
* [cite_start]**How it works:** It climbs a 3-tier optimization ladder[cite: 58]:
  1. [cite_start]*PyTorch Batching:* Stacks 8 frames together to process at once[cite: 63].
  2. [cite_start]*TensorRT FP16:* Converts the YOLOv11 model into a highly compressed, fast format, dropping inference time to ~8ms per batch[cite: 65, 66].
  3. [cite_start]*CUDA Streams:* Creates a pipeline so the system can upload the next batch of video to the GPU at the exact same time it is processing the current batch[cite: 67].

### 4. Validation & Demo (The Judge)
* [cite_start]**What it does:** Proves the system works through data visualization and automated testing.
* [cite_start]**How it works:** Tracks every frame's latency and outputs a live dashboard showing FPS and GPU utilization[cite: 130]. [cite_start]It runs 5 strict tests, including testing equal weights, weighted priorities, and starvation prevention[cite: 135].

---

## 📚 Glossary of Terms
* [cite_start]**YOLOv11:** A state-of-the-art AI model used to instantly draw boxes around objects in a picture or video.
* **CV (Coefficient of Variation):** A math formula used to measure "fairness." [cite_start]A CV below 0.10 means all streams are getting almost exactly the same amount of processing time.
* [cite_start]**TensorRT FP16:** An Nvidia tool that simplifies the AI's math (using 16-bit numbers instead of 32-bit), making it run much faster without losing much accuracy[cite: 65].
* **CUDA Streams:** A way to make the GPU multitask. [cite_start]It allows the GPU to do math while simultaneously moving computer memory around[cite: 67].

---

## 🧪 Validation Tests
[cite_start]To verify our success metrics, run `test_runner.py` which executes the following scenarios[cite: 147]:
1. [cite_start]**Equal Weights:** All 10 streams at 1.0 weight (Checks CV < 0.10)[cite: 135].
2. [cite_start]**Weighted Priority:** Two streams set to 2.0 weight (Verifies they receive 2x frames)[cite: 135].
3. [cite_start]**Starvation Prevention:** Pausing one stream (Verifies the other 9 remain unaffected)[cite: 135].
4. [cite_start]**Latency Check:** Verifies 95th percentile latency remains under 200 ms[cite: 135].
5. [cite_start]**Throughput Check:** Confirms the total system output exceeds 100 FPS[cite: 135].
