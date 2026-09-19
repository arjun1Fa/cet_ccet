import queue

class StreamQueueManager:
    """Manages thread-safe queues for video streams."""
    def __init__(self, num_streams=10, maxsize=50):
        self.num_streams = num_streams
        self.maxsize = maxsize
        self.queues = [queue.Queue(maxsize=self.maxsize) for _ in range(self.num_streams)]
        
    def get_queues(self):
        """Returns the list of 10 queue.Queue objects."""
        return self.queues
        
    def get_queue_stats(self):
        """Returns a summary of queue depths for monitoring."""
        stats = {}
        for stream_id, q in enumerate(self.queues):
            depth = q.qsize()
            capacity = self.maxsize
            utilization = (depth / capacity) * 100 if capacity > 0 else 0
            stats[stream_id] = {
                'depth': depth,
                'capacity': capacity,
                'utilization': utilization
            }
        return stats
