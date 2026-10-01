import threading
import time
import unittest

import numpy as np

from person4_validation.metrics import MetricsTracker


class TestMetricsTracker(unittest.TestCase):
    def test_counts_frames_from_payload(self):
        tracker = MetricsTracker(num_streams=2)
        batch = [{"stream_id": 0, "frames": [np.zeros((2, 2, 3))] * 4, "timestamps": [time.time()] * 4}]
        tracker.update(batch, [])
        self.assertEqual(tracker.total_frames, 4)
        self.assertEqual(tracker.per_stream_frames[0], 4)

    def test_negative_latency_is_clamped(self):
        tracker = MetricsTracker(num_streams=1)
        tracker.update([{"stream_id": 0, "frames": [np.zeros((1, 1, 3))], "timestamps": [time.time() + 10]}], [])
        self.assertEqual(tracker.get_latency_stats()["p95"], 0.0)

    def test_concurrent_updates_do_not_corrupt_counts(self):
        tracker = MetricsTracker(num_streams=1)
        payload = [{"stream_id": 0, "frames": [object()] * 5, "timestamps": []}]

        threads = [threading.Thread(target=tracker.update, args=(payload, [])) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(tracker.total_frames, 100)
        self.assertEqual(tracker.per_stream_frames[0], 100)

    def test_reset_clears_state(self):
        tracker = MetricsTracker(num_streams=2)
        tracker.update([{"stream_id": 1, "frames": [object()] * 3, "timestamps": []}], [])
        tracker.reset()
        self.assertEqual(tracker.total_frames, 0)
        self.assertEqual(tracker.per_stream_frames, {0: 0, 1: 0})
        self.assertEqual(tracker.latencies, [])


if __name__ == "__main__":
    unittest.main()
