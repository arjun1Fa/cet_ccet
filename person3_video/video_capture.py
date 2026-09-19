import cv2
import queue
import threading
import time
import subprocess
import os
import shutil
import numpy as np


class StreamWorker(threading.Thread):
    def __init__(self, stream_id, video_path, out_queue, target_fps=30, max_retries=3, use_gpu=False, batch_size=8):
        super().__init__(daemon=True)
        self.stream_id = stream_id
        self.video_path = video_path
        self.out_queue = out_queue
        self.target_fps = target_fps
        self.max_retries = max_retries
        self.use_gpu = use_gpu
        self.batch_size = batch_size

        self.stop_event = threading.Event()
        self.retry_count = 0
        self.frame_number = 0

    def run(self):
        while not self.stop_event.is_set() and self.retry_count <= self.max_retries:
            try:
                success = self._run_cv2_loop()
                if success:
                    self.retry_count = 0
                else:
                    self.retry_count += 1
                    print(f"[Stream {self.stream_id}] Loop ended, retrying ({self.retry_count}/{self.max_retries})...")
                    time.sleep(1.0)
            except Exception as e:
                print(f"[Stream {self.stream_id}] Exception: {e}")
                self.retry_count += 1
                time.sleep(1.0)

        if self.retry_count > self.max_retries:
            print(f"[Stream {self.stream_id}] Fatal: Max retries exceeded.")

    def _run_cv2_loop(self):
        """Standard OpenCV capture loop"""
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            print(f"[Stream {self.stream_id}] Failed to open {self.video_path}")
            return False

        frame_time = 1.0 / self.target_fps
        frame_buffer = []
        timestamp_buffer = []

        while not self.stop_event.is_set():
            start_t = time.time()
            ret, frame = cap.read()

            if not ret:
                break

            # Back-pressure: drop frame if queue is nearly full
            if self.out_queue.qsize() > int(self.out_queue.maxsize * 0.95):
                continue

            frame_buffer.append(frame)
            timestamp_buffer.append(time.time())
            self.frame_number += 1

            if len(frame_buffer) == self.batch_size:
                # Simple list of raw BGR numpy frames — exactly what Ultralytics expects
                payload = {
                    'stream_id': self.stream_id,
                    'frames': list(frame_buffer),
                    'timestamps': list(timestamp_buffer),
                    'frame_numbers': list(range(self.frame_number - self.batch_size, self.frame_number))
                }

                # Drop oldest if full
                if self.out_queue.full():
                    try:
                        self.out_queue.get_nowait()
                    except queue.Empty:
                        pass

                try:
                    self.out_queue.put_nowait(payload)
                except queue.Full:
                    pass

                frame_buffer.clear()
                timestamp_buffer.clear()

            elapsed = time.time() - start_t
            sleep_t = frame_time - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        cap.release()
        return True

    def stop(self):
        self.stop_event.set()


class MultiStreamCapture:
    """Manages N video capture threads"""
    def __init__(self, queue_manager, video_sources, target_fps=30, use_gpu=False, batch_size=8):
        self.queue_manager = queue_manager
        self.workers = []

        queues = self.queue_manager.get_queues()

        for stream_id, path in video_sources.items():
            if stream_id < len(queues):
                worker = StreamWorker(
                    stream_id=stream_id,
                    video_path=path,
                    out_queue=queues[stream_id],
                    target_fps=target_fps,
                    max_retries=3,
                    use_gpu=use_gpu,
                    batch_size=batch_size
                )
                self.workers.append(worker)

    def start(self):
        print(f"Starting {len(self.workers)} video capture threads natively batching {self.workers[0].batch_size} frames...")
        for worker in self.workers:
            worker.start()

    def stop(self, timeout=5.0):
        print("Stopping video capture threads gracefully...")
        for worker in self.workers:
            worker.stop_event.set()
        for worker in self.workers:
            worker.join(timeout=timeout)
            if worker.is_alive():
                print(f"Warning: Stream {worker.stream_id} thread join timed out.")
        print("All capture threads stopped.")
