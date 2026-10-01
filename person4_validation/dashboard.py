"""Console dashboard adapter."""


class Dashboard:
    def __init__(self, metrics_tracker, scheduler=None, queue_manager=None, gpu_processor=None):
        self.metrics = metrics_tracker
        self.scheduler = scheduler
        self.queue_manager = queue_manager
        self.gpu_processor = gpu_processor

    def update(self):
        self.metrics.print_console_display(
            scheduler_stats=self.scheduler.get_per_stream_stats() if self.scheduler else None,
            gpu_stats=None,
            queue_stats=self.queue_manager.get_queue_stats() if self.queue_manager else None,
        )
