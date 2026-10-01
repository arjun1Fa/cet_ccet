"""Weighted deficit round-robin scheduler for video inference payloads."""

import queue
import time
from typing import Dict, List

import numpy as np

from shared import config


class FairScheduler:
    """Select queued stream payloads according to configurable weights."""

    def __init__(self, stream_queues: List[queue.Queue], num_streams: int):
        if len(stream_queues) != num_streams:
            raise ValueError("stream_queues length must equal num_streams")
        self.stream_queues = stream_queues
        self.num_streams = num_streams
        self.credits = {i: 0.0 for i in range(num_streams)}
        self.weights = {
            i: max(0.0, float(config.STREAM_WEIGHTS.get(i, 1.0)))
            for i in range(num_streams)
        }
        self.stream_frames = {i: 0 for i in range(num_streams)}
        self.stream_start_time = time.time()

    def get_fair_batch(self, batch_size: int | None = None) -> List[Dict]:
        """Return up to ``batch_size`` payloads using weighted deficit credits."""
        if batch_size is None:
            batch_size = self.num_streams
        if batch_size <= 0:
            return []

        for stream_id in range(self.num_streams):
            self.credits[stream_id] += self.weights[stream_id]

        batch: List[Dict] = []
        while len(batch) < batch_size:
            made_progress = False
            for stream_id in sorted(self.credits, key=self.credits.get, reverse=True):
                if len(batch) >= batch_size or self.credits[stream_id] < 1.0:
                    continue
                try:
                    payload = self.stream_queues[stream_id].get_nowait()
                except queue.Empty:
                    continue
                batch.append(payload)
                self.credits[stream_id] -= 1.0
                self.stream_frames[stream_id] += len(payload.get("frames", []))
                made_progress = True
            if not made_progress:
                break
        return batch

    def get_adaptive_batch_size(self) -> int:
        """Choose a batch payload count from current average queue pressure."""
        if self.num_streams == 0:
            return 0
        avg_depth = sum(q.qsize() for q in self.stream_queues) / self.num_streams
        if avg_depth > config.QUEUE_DEPTH_HIGH:
            return self.num_streams * 2
        return self.num_streams

    def set_weight(self, stream_id: int, weight: float) -> None:
        """Set a non-negative stream weight."""
        if stream_id not in self.weights:
            raise KeyError(f"Unknown stream_id: {stream_id}")
        if weight < 0:
            raise ValueError("weight must be non-negative")
        self.weights[stream_id] = float(weight)

    def get_weights(self) -> Dict[int, float]:
        return self.weights.copy()

    def get_per_stream_stats(self) -> Dict[int, Dict]:
        elapsed = time.time() - self.stream_start_time
        stats = {}
        for stream_id in range(self.num_streams):
            frames = self.stream_frames[stream_id]
            stats[stream_id] = {
                "frames": frames,
                "fps": frames / elapsed if elapsed > 0 else 0.0,
                "weight": self.weights[stream_id],
            }
        return stats

    def calculate_fairness_cv(self) -> float:
        """Coefficient of variation of weight-normalized stream throughput."""
        stats = self.get_per_stream_stats()
        normalized = [
            s["fps"] / max(s["weight"], 0.001)
            for s in stats.values()
            if s["weight"] > 0
        ]
        if len(normalized) < 2:
            return 0.0
        mean = np.mean(normalized)
        return float(np.std(normalized) / mean) if mean > 0 else 0.0

    def reset_stats(self) -> None:
        self.stream_frames = {i: 0 for i in range(self.num_streams)}
        self.credits = {i: 0.0 for i in range(self.num_streams)}
        self.stream_start_time = time.time()
