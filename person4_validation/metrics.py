import time
import numpy as np

class MetricsTracker:
    def __init__(self, num_streams=10):
        self.num_streams = num_streams
        self.total_frames = 0
        self.per_stream_frames = {i: 0 for i in range(num_streams)}
        
        self.latencies = [] # Store recent latencies in ms
        self.max_latency_history = 1000 # Prevent memory bloat
        
        self.start_time = time.time()
        self.last_display_time = self.start_time
        
    def update(self, batch, results):
        """
        Called after each GPU processing step.
        batch: list of pre-batched payloads from Person 3:
               [{'stream_id': int, 'frames': np.array(8,3,H,W), 'timestamps': [t1..t8]}, ...]
        results: list of result dicts (1:1 with batch payloads)
        """
        current_time = time.time()
        
        for payload in batch:
            stream_id = payload['stream_id']
            timestamps = payload.get('timestamps', [])
            num_frames = len(timestamps) if timestamps else 8
            
            # Count all frames in this payload
            self.total_frames += num_frames
            if stream_id in self.per_stream_frames:
                self.per_stream_frames[stream_id] += num_frames
            
            # Calculate latency for each individual frame timestamp
            for ts in timestamps:
                latency_ms = (current_time - ts) * 1000.0
                self.latencies.append(latency_ms)
            
        # Trim latencies to prevent memory leaks
        if len(self.latencies) > self.max_latency_history:
            self.latencies = self.latencies[-self.max_latency_history:]
            
    def get_total_fps(self):
        elapsed = time.time() - self.start_time
        if elapsed <= 0: return 0.0
        return self.total_frames / elapsed
        
    def get_per_stream_fps(self):
        elapsed = time.time() - self.start_time
        if elapsed <= 0:
            return {i: 0.0 for i in range(self.num_streams)}
            
        return {stream_id: count / elapsed 
                for stream_id, count in self.per_stream_frames.items()}
                
    def get_latency_stats(self):
        if not self.latencies:
            return {'p50': 0.0, 'p95': 0.0, 'mean': 0.0}
            
        return {
            'p50': float(np.percentile(self.latencies, 50)),
            'p95': float(np.percentile(self.latencies, 95)),
            'mean': float(np.mean(self.latencies))
        }

    def print_console_display(self, scheduler_stats=None, gpu_stats=None, queue_stats=None):
        """
        Simple console display updating every second.
        """
        current_time = time.time()
        if (current_time - self.last_display_time) < 1.0:
            return # Update at most once per second
            
        self.last_display_time = current_time
        
        total_fps = self.get_total_fps()
        latency = self.get_latency_stats()
        
        # Clear screen and print
        print(chr(27) + "[2J" + chr(27) + "[H") # ANSI clear screen
        print("="*60)
        print(f"🚀 SYSTEM ARCHITECT & VALIDATION | PIPELINE METRICS 🚀")
        print("="*60)
        print(f"Total Throughput : {total_fps:.1f} FPS")
        print(f"Latency (p95)    : {latency['p95']:.1f} ms | (p50): {latency['p50']:.1f} ms")
        
        if gpu_stats:
            print(f"GPU Utilization  : {gpu_stats.get('utilization', 0)}% | Mem: {gpu_stats.get('memory_usage_gb', 0):.1f} GB")
            
        print("-"*60)
        print("Per-Stream Fairness & Throughput:")
        stream_fps = self.get_per_stream_fps()
        for i in range(self.num_streams):
            fps = stream_fps.get(i, 0.0)
            
            # Incorporate scheduler weight info if provided
            weight_str = ""
            if scheduler_stats and i in scheduler_stats:
                weight = scheduler_stats[i].get('weight', 1.0)
                weight_str = f"[Wt: {weight:.1f}]"
                
            # Queue depth info
            queue_str = ""
            if queue_stats and i in queue_stats:
                depth = queue_stats[i].get('depth', 0)
                queue_str = f"| Q: {depth:2d}"
                
            print(f"  Stream {i:02d} {weight_str:10} : {fps:5.1f} FPS  {queue_str}")
            
        print("="*60)
