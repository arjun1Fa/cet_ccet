# /main.py

import time
import sys
import signal

from person3_video.video_capture import MultiStreamCapture  # Person 3
from person3_video.queue_manager import StreamQueueManager  # Person 3 Queues
from shared.fair_scheduler import FairScheduler   # me
from person2_gpu.gpu_processor import GPUProcessor           # Person 2
from person4_validation.metrics import MetricsTracker # Person 4
from person4_validation.dashboard import Dashboard           # Person 4
from shared import config

class SystemController:
    """
    Main system controller - integrates all 4 modules
    """
    
    def __init__(self):
        print("="*60)
        print("GPU VIDEO INFERENCE OPTIMIZER")
        print("="*60)
        
        # Initialize all modules
        self.initialize_modules()
        
        # Graceful shutdown handler
        signal.signal(signal.SIGINT, self.shutdown)
    
    
    def initialize_modules(self):
        """Initialize all 4 team members' modules"""
        
        print("\n[1/4] Initializing Video Capture (Person 3)...")
        self.queue_manager = StreamQueueManager(
            num_streams=config.NUM_STREAMS,
            maxsize=config.QUEUE_MAX_SIZE
        )
        
        video_sources_dict = {i: path for i, path in enumerate(config.VIDEO_PATHS)}
        
        self.video_capture = MultiStreamCapture(
            queue_manager=self.queue_manager,
            video_sources=video_sources_dict,
            target_fps=config.TARGET_CAPTURE_FPS,
            use_gpu=True
        )
        self.video_capture.start()
        
        # Wait for queues to fill
        print("      Waiting for frames to buffer...")
        time.sleep(2)
        
        print("\n[2/4] Initializing Fair Scheduler (Person 1 - YOU)...")
        self.scheduler = FairScheduler(
            stream_queues=self.queue_manager.get_queues(),
            num_streams=config.NUM_STREAMS
        )
        
        print("\n[3/4] Initializing GPU Processor (Person 2)...")
        self.gpu_processor = GPUProcessor(
            device_id=config.CUDA_DEVICE,
            use_tensorrt=config.USE_TENSORRT,
            use_fp16=config.USE_FP16
        )
        
        print("\n[4/4] Initializing Metrics & Dashboard (Person 4)...")
        self.metrics = MetricsTracker(num_streams=config.NUM_STREAMS)
        self.dashboard = Dashboard(
            metrics_tracker=self.metrics,
            scheduler=self.scheduler,
            queue_manager=self.queue_manager,
            gpu_processor=self.gpu_processor
        )
        
        print("\n✓ All modules initialized successfully")
        print("="*60)
    
    
    def run(self):
        """
        Main processing loop
        """
        
        print("\nStarting main processing loop...")
        print("Press Ctrl+C to stop\n")
        
        batch_count = 0
        
        try:
            while True:
                # STEP 1: Process exactly 1 stream's payload (8 frames) at a time
                if config.ADAPTIVE_BATCH:
                    batch_size = self.scheduler.get_adaptive_batch_size()
                else:
                    batch_size = 1  # 1 payload per API round-trip

                # Pull the fair weighted rounded batch
                batch = self.scheduler.get_fair_batch(batch_size=batch_size)

                # STEP 2: Handle empty batch (all queues empty — wait briefly)
                if not batch:
                    time.sleep(0.005)
                    continue

                # STEP 3: Dispatch payloads to the GPU daemon asynchronously.
                # The processor background thread will natively call self.metrics.update 
                # as a callback exactly when the HTTPS response finishes.
                self.gpu_processor.process_batch_async(batch, callback=self.metrics.update)

                # STEP 5: Update dashboard periodically
                batch_count += 1
                if batch_count % 50 == 0:
                    self.dashboard.update()
                
        except KeyboardInterrupt:
            print("\n\nShutdown requested...")
            self.shutdown()
    
    
    def shutdown(self, signum=None, frame=None):
        """
        Graceful shutdown
        """
        print("\n" + "="*60)
        print("SHUTTING DOWN")
        print("="*60)
        
        # Stop video capture
        print("Stopping video capture...")
        self.video_capture.stop()
        
        # Print final statistics
        print("\nFINAL STATISTICS:")
        print("-"*60)
        
        # Total FPS
        fps = self.metrics.get_total_fps()
        print(f"Total FPS: {fps:.1f}")
        
        # Fairness
        cv = self.scheduler.calculate_fairness_cv()
        print(f"Fairness CV: {cv:.4f} {'✓' if cv < 0.10 else '✗'}")
        
        # Per-stream
        print("\nPer-Stream FPS:")
        stats = self.scheduler.get_per_stream_stats()
        for stream_id, stream_stats in sorted(stats.items()):
            print(f"  Stream {stream_id}: {stream_stats['fps']:.1f} FPS "
                  f"(weight={stream_stats['weight']})")
        
        # Latency
        latency_stats = self.metrics.get_latency_stats()
        if latency_stats:
            print(f"\nLatency:")
            print(f"  p50: {latency_stats['p50']:.1f}ms")
            print(f"  p95: {latency_stats['p95']:.1f}ms")
        
        print("="*60)
        sys.exit(0)


def main():
    """Entry point"""
    
    # Create system controller
    controller = SystemController()
    
    # Run main loop
    controller.run()


if __name__ == '__main__':
    main()