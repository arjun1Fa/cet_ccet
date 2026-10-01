import time
import threading

import numpy as np


class MetricsTracker:
    """Thread-safe throughput and latency tracker for asynchronous inference."""

    def __init__(self, num_streams=10, window_size=None):
        self.num_streams = num_streams
        self.total_frames = 0
        self.per_stream_frames = {i: 0 for i in range(num_streams)}
        self.latencies = []
        self.max_latency_history = 1000
        self.start_time = time.time()
        self.last_display_time = self.start_time
        self._lock = threading.RLock()
        self.window_size = window_size

    def update(self, batch, results):
        """Record the frames represented by a completed inference batch."""
        current_time = time.time()
        with self._lock:
            for payload in batch:
                stream_id = payload["stream_id"]
                timestamps = payload.get("timestamps", [])
                num_frames = len(payload.get("frames", []))
                if not num_frames:
                    num_frames = len(timestamps)
                if not num_frames:
                    continue

                self.total_frames += num_frames
                if stream_id in self.per_stream_frames:
                    self.per_stream_frames[stream_id] += num_frames

                for ts in timestamps:
                    if isinstance(ts, (int, float)):
                        self.latencies.append(max(0.0, (current_time - ts) * 1000.0))

            if len(self.latencies) > self.max_latency_history:
                self.latencies = self.latencies[-self.max_latency_history :]

    def get_total_fps(self):
        with self._lock:
            elapsed = time.time() - self.start_time
            return self.total_frames / elapsed if elapsed > 0 else 0.0

    def get_per_stream_fps(self):
        with self._lock:
            elapsed = time.time() - self.start_time
            if elapsed <= 0:
                return {i: 0.0 for i in range(self.num_streams)}
            return {
                stream_id: count / elapsed
                for stream_id, count in self.per_stream_frames.items()
            }

    def get_latency_stats(self):
        with self._lock:
            if not self.latencies:
                return {"p50": 0.0, "p95": 0.0, "mean": 0.0}
            values = np.asarray(self.latencies, dtype=float)
            return {
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
                "mean": float(np.mean(values)),
            }

    def reset(self):
        """Reset counters and latency history without replacing the tracker."""
        with self._lock:
            self.total_frames = 0
            self.per_stream_frames = {i: 0 for i in range(self.num_streams)}
            self.latencies.clear()
            self.start_time = time.time()
            self.last_display_time = self.start_time

    def print_console_display(self, scheduler_stats=None, gpu_stats=None, queue_stats=None):
        """Print a once-per-second console snapshot."""
        current_time = time.time()
        with self._lock:
            if current_time - self.last_display_time < 1.0:
                return
            self.last_display_time = current_time

        total_fps = self.get_total_fps()
        latency = self.get_latency_stats()

        print(chr(27) + "[2J" + chr(27) + "[H")
        print("=" * 60)
        print("🚀 SYSTEM ARCHITECT & VALIDATION | PIPELINE METRICS 🚀")
        print("=" * 60)
        print(f"Total Throughput : {total_fps:.1f} FPS")
        print(
            f"Latency (p95)    : {latency['p95']:.1f} ms | "
            f"(p50): {latency['p50']:.1f} ms"
        )

        if gpu_stats:
            print(
                f"GPU Utilization  : {gpu_stats.get('utilization', 0)}% | "
                f"Mem: {gpu_stats.get('memory_usage_gb', 0):.1f} GB"
            )

        print("-" * 60)
        print("Per-Stream Fairness & Throughput:")
        stream_fps = self.get_per_stream_fps()
        for i in range(self.num_streams):
            fps = stream_fps.get(i, 0.0)
            weight_str = ""
            if scheduler_stats and i in scheduler_stats:
                weight_str = f"[Wt: {scheduler_stats[i].get('weight', 1.0):.1f}]"
            queue_str = ""
            if queue_stats and i in queue_stats:
                queue_str = f"| Q: {queue_stats[i].get('depth', 0):2d}"
            print(f"  Stream {i:02d} {weight_str:10} : {fps:5.1f} FPS  {queue_str}")
        print("=" * 60)
