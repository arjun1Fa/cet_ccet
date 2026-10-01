import queue
import unittest

from shared.fair_scheduler import FairScheduler


class TestFairScheduler(unittest.TestCase):
    def make_scheduler(self, weights=None):
        queues = [queue.Queue(maxsize=10) for _ in range(3)]
        from shared import config
        original = config.STREAM_WEIGHTS.copy()
        config.STREAM_WEIGHTS = weights or {0: 1.0, 1: 1.0, 2: 1.0}
        try:
            scheduler = FairScheduler(queues, 3)
        finally:
            config.STREAM_WEIGHTS = original
        return scheduler, queues

    @staticmethod
    def payload(stream_id, frames=4):
        return {
            "stream_id": stream_id,
            "frames": [object() for _ in range(frames)],
            "timestamps": [0.0] * frames,
        }

    def test_empty_queues_return_empty_batch(self):
        scheduler, _ = self.make_scheduler()
        self.assertEqual(scheduler.get_fair_batch(3), [])

    def test_consumed_frame_count_matches_payload(self):
        scheduler, queues = self.make_scheduler()
        queues[0].put(self.payload(0, frames=6))

        batch = scheduler.get_fair_batch(1)

        self.assertEqual(len(batch), 1)
        self.assertEqual(scheduler.get_per_stream_stats()[0]["frames"], 6)

    def test_non_negative_weight_is_accepted(self):
        scheduler, _ = self.make_scheduler()
        scheduler.set_weight(1, 2.5)
        self.assertEqual(scheduler.get_weights()[1], 2.5)

    def test_negative_weight_is_rejected(self):
        scheduler, _ = self.make_scheduler()
        with self.assertRaises(ValueError):
            scheduler.set_weight(1, -1)

    def test_unknown_stream_is_rejected(self):
        scheduler, _ = self.make_scheduler()
        with self.assertRaises(KeyError):
            scheduler.set_weight(99, 1)

    def test_reset_clears_credits_and_statistics(self):
        scheduler, queues = self.make_scheduler()
        queues[0].put(self.payload(0, frames=4))
        scheduler.get_fair_batch(1)
        scheduler.reset_stats()

        self.assertTrue(all(v == 0 for v in scheduler.stream_frames.values()))
        self.assertTrue(all(v == 0 for v in scheduler.credits.values()))


if __name__ == "__main__":
    unittest.main()
