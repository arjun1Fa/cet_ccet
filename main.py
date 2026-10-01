# /main.py

import signal
import sys
import time

from person3_video.video_capture import MultiStreamCapture
from person3_video.queue_manager import StreamQueueManager
from shared.fair_scheduler import FairScheduler
from person2_gpu.gpu_processor import GPUProcessor
from person4_validation.metrics import MetricsTracker
from person4_validation.dashboard import Dashboard
from shared import config


class SystemController:
    """Main system controller integrating capture, scheduling, inference and metrics."""

    def __init__(self):
        config.validate()

        print("=" * 60)
        print("GPU VIDEO INFERENCE OPTIMIZER")
        print("=" * 60)

        self.initialize_modules()
        signal.signal(signal.SIGINT, self.shutdown)

    def initialize_modules(self):
        print("\n[1/4] Initializing Video Capture...")
        self.queue_manager = StreamQueueManager(
            num_streams=config.NUM_STREAMS,
            maxsize=config.QUEUE_MAX_SIZE,
        )

        video_sources_dict = {i: path for i, path in enumerate(config.VIDEO_PATHS)}
        self.video_capture = MultiStreamCapture(
            queue_manager=self.queue_manager,
            video_sources=video_sources_dict,
            target_fps=config.TARGET_CAPTURE_FPS,
            use_gpu=True,
        )
        self.video_capture.start()

        print("      Waiting for frames to buffer...")
        time.sleep(2)

        print("\n[2/4] Initializing Fair Scheduler...")
        self.scheduler = FairScheduler(
            stream_queues=self.queue_manager.get_queues(),
            num_streams=config.NUM_STREAMS,
        )

        print("\n[3/4] Initializing GPU Processor...")
        self.gpu_processor = GPUProcessor(
            device_id=config.CUDA_DEVICE,
            use_tensorrt=config.USE_TENSORRT,
            use_fp16=config.USE_FP16,
        )

        print("\n[4/4] Initializing Metrics & Dashboard...")
        self.metrics = MetricsTracker(num_streams=config.NUM_STREAMS)
        self.dashboard = Dashboard(
            metrics_tracker=self.metrics,
            scheduler=self.scheduler,
            queue_manager=self.queue_manager,
            gpu_processor=self.gpu_processor,
        )

        print("\n✓ All modules initialized successfully")
        print("=" * 60)

    def run(self):
        print("\nStarting main processing loop...")
        print("Press Ctrl+C to stop\n")
        batch_count = 0

        try:
            while True:
                batch_size = (
                    self.scheduler.get_adaptive_batch_size()
                    if config.ADAPTIVE_BATCH
                    else 1
                )

                batch = self.scheduler.get_fair_batch(batch_size=batch_size)
                if not batch:
                    time.sleep(0.005)
                    continue

                self.gpu_processor.process_batch_async(
                    batch,
                    callback=self.metrics.update,
                )

                batch_count += 1
                if batch_count % 50 == 0:
                    self.dashboard.update()

        except KeyboardInterrupt:
            print("\n\nShutdown requested...")
            self.shutdown()

    def shutdown(self, signum=None, frame=None):
        print("\n" + "=" * 60)
        print("SHUTTING DOWN")
        print("=" * 60)

        print("Stopping video capture...")
        self.video_capture.stop()

        print("\nFINAL STATISTICS:")
        print("-" * 60)

        fps = self.metrics.get_total_fps()
        print(f"Total FPS: {fps:.1f}")

        cv = self.scheduler.calculate_fairness_cv()
        print(
            f"Fairness CV: {cv:.4f} "
            f"{'✓' if cv < config.FAIRNESS_CV_THRESHOLD else '✗'}"
        )

        print("\nPer-Stream FPS:")
        stats = self.scheduler.get_per_stream_stats()
        for stream_id, stream_stats in sorted(stats.items()):
            print(
                f"  Stream {stream_id}: {stream_stats['fps']:.1f} FPS "
                f"(weight={stream_stats['weight']})"
            )

        latency_stats = self.metrics.get_latency_stats()
        print("\nLatency:")
        print(f"  p50: {latency_stats['p50']:.1f}ms")
        print(f"  p95: {latency_stats['p95']:.1f}ms")
        print("=" * 60)
        sys.exit(0)


def main():
    controller = SystemController()
    controller.run()


if __name__ == "__main__":
    main()
