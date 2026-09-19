#round robin schedeule with weights/credit system


# /person1_scheduler/fair_scheduler.py

import queue
import time
import numpy as np
from typing import List, Dict
import sys
sys.path.append('..')
from shared import config

class FairScheduler:
    """
    Weighted Round-Robin Scheduler with Deficit Compensation
    
    Ensures fair frame distribution across multiple video streams
    based on configurable priority weights.
    """
    
    def __init__(self, stream_queues: List[queue.Queue], num_streams: int):
        """
        Initialize the scheduler
        
        Args:
            stream_queues: List of queue.Queue objects from Person 3
            num_streams: Total number of streams
        """
        # TODO: Initialize these variables
        self.stream_queues = stream_queues
        self.num_streams = num_streams
        
        # Credit accounting (Deficit Round-Robin)
        self.credits = {i: 0.0 for i in range(num_streams)}
        
        # Priority weights
        self.weights = {i: config.STREAM_WEIGHTS.get(i, 1.0) 
                       for i in range(num_streams)}
        
        # Statistics tracking
        self.stream_frames = {i: 0 for i in range(num_streams)}
        self.stream_start_time = time.time()
    
    
    def get_fair_batch(self, batch_size: int = None) -> List[Dict]:
        """
        Weighted Round-Robin with multi-payload pulling.

        If a stream has high credits (due to weight or backlog), we can pull
        multiple payloads from it in a single cycle if batch_size allows.
        """
        if batch_size is None:
            batch_size = self.num_streams

        # --- Phase 1: Credit top-up (proportional to weight) ---
        for stream_id in range(self.num_streams):
            self.credits[stream_id] += self.weights[stream_id]

        # --- Phase 2: Pull in credit-order (highest first) ---
        sorted_streams = sorted(
            range(self.num_streams),
            key=lambda sid: self.credits[sid],
            reverse=True
        )

        batch = []
        # Loop over streams multiple times if needed to fill batch_size
        # (This allows draining backlogs if batch_size > num_streams)
        while len(batch) < batch_size:
            added_any = False
            for stream_id in sorted_streams:
                if len(batch) >= batch_size:
                    break
                if self.credits[stream_id] < 1.0:
                    continue

                try:
                    payload = self.stream_queues[stream_id].get_nowait()
                    batch.append(payload)
                    self.credits[stream_id] -= 1.0
                    self.stream_frames[stream_id] += config.BATCH_SIZE
                    added_any = True
                except queue.Empty:
                    pass

            if not added_any:
                break

        return batch
    
    
    def get_adaptive_batch_size(self) -> int:
        """
        Calculate total target payload count based on average queue depth.
        If total backlog exceeds HIGH threshold, we double the consumption rate.
        """
        # Average queue depth per stream
        avg_depth = sum(q.qsize() for q in self.stream_queues) / self.num_streams
        
        if avg_depth > config.QUEUE_DEPTH_HIGH:
            # Backlog detected: Pull 2 rounds of payloads per cycle
            return self.num_streams * 2
        elif avg_depth < config.QUEUE_DEPTH_LOW:
            # Shallow queues: Standard 1 round
            return self.num_streams
        else:
            return self.num_streams
    
    
    def set_weight(self, stream_id: int, weight: float):
        """
        Set priority weight for a stream
        
        Args:
            stream_id: Stream to adjust
            weight: Priority weight (1.0 = normal, 2.0 = high, 0.5 = low)
        """
        # TODO: Set weight for stream
        self.weights[stream_id] = weight
    
    
    def get_weights(self) -> Dict[int, float]:
        """
        Get current priority weights
        
        Returns:
            Dict mapping stream_id to weight
        """
        return self.weights.copy()
    
    
    def get_per_stream_stats(self) -> Dict[int, Dict]:
        """
        Get statistics for each stream (for Person 4)
        
        Returns:
            Dict mapping stream_id to stats dict
        """
        # TODO: Calculate per-stream FPS
        
        elapsed = time.time() - self.stream_start_time
        
        stats = {}
        for stream_id in range(self.num_streams):
            frames = self.stream_frames[stream_id]
            fps = frames / elapsed if elapsed > 0 else 0
            
            stats[stream_id] = {
                'frames': frames,
                'fps': fps,
                'weight': self.weights[stream_id]
            }
        
        return stats
    
    
    def calculate_fairness_cv(self) -> float:
        """
        Calculate Coefficient of Variation (fairness metric)
        Compares weight-normalised FPS so that a stream with weight=2.0
        getting 2x the throughput is still counted as 'fair'.

        CV = std_dev / mean of (fps / weight)
        Lower is better. Target: < 0.10
        """
        stats = self.get_per_stream_stats()

        if len(stats) < 2:
            return 0.0

        # Weight-normalize: each stream's fps divided by its weight
        normalised = [s['fps'] / max(s['weight'], 0.001) for s in stats.values()]

        mean = np.mean(normalised)
        std  = np.std(normalised)

        return float(std / mean) if mean > 0 else 0.0
    
    
    def reset_stats(self):
        """
        Reset statistics counters
        """
        self.stream_frames = {i: 0 for i in range(self.num_streams)}
        self.stream_start_time = time.time()