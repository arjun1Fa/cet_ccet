import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import base64
import cv2
import sys
import os
import time

# Ensure local imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from person2_gpu.gpu_processor import GPUProcessor
from shared import config

class TestRunPodGPUProcessor(unittest.TestCase):
    def setUp(self):
        # Mocking config before initialization
        self.processor = GPUProcessor(device_id=0)

    @patch('requests.post')
    @patch('requests.get')
    def test_process_batch_success(self, mock_get, mock_post):
        # 1. Setup Mock Responses
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"id": "test-job-id"}
        
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "status": "COMPLETED",
            "output": {
                "detections": [
                    [{"bbox": [10, 10, 100, 100], "confidence": 0.95, "class_id": 0}]
                ]
            }
        }
        
        # 2. Create dummy input batch (1 frame)
        dummy_frame = np.zeros((3, 640, 640), dtype=np.uint8)
        batch = [{
            'stream_id': 0,
            'frames': np.expand_dims(dummy_frame, axis=0),
            'timestamps': [time.time()]
        }]
        
        # 3. Call process_batch
        results = self.processor.process_batch(batch)
        
        # 4. Assertions
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['stream_id'], 0)
        self.assertEqual(len(results[0]['detections']), 1)
        self.assertEqual(results[0]['detections'][0]['boxes'][0], [10, 10, 100, 100])
        self.assertEqual(results[0]['detections'][0]['scores'][0], 0.95)

    @patch('requests.post')
    def test_process_batch_timeout(self, mock_post):
        # 1. Setup Mock for job submission
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"id": "timeout-job-id"}
        
        # 2. Setup Mock for polling (always "IN_PROGRESS")
        with patch('requests.get') as mock_get:
            mock_get.return_value.json.return_value = {"status": "IN_PROGRESS"}
            
            dummy_frame = np.zeros((3, 640, 640), dtype=np.uint8)
            batch = [{'stream_id': 0, 'frames': np.expand_dims(dummy_frame, axis=0)}]
            
            # Use short internal timeout for testing if possible, or just mock the timer
            # Here it will naturally timeout after its internal 10s loop
            results = self.processor.process_batch(batch)
            
            self.assertEqual(results, [])

    @patch('requests.post')
    def test_malformed_response(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"id": "malformed-job-id"}
        
        with patch('requests.get') as mock_get:
            mock_get.return_value.json.return_value = {"status": "COMPLETED", "output": {"junk": "data"}}
            
            dummy_frame = np.zeros((3, 640, 640), dtype=np.uint8)
            batch = [{'stream_id': 0, 'frames': np.expand_dims(dummy_frame, axis=0)}]
            
            results = self.processor.process_batch(batch)
            self.assertEqual(results, [])

    def test_frame_to_b64_conversion(self):
        # Verify that frame conversion correctly produces a base64 string
        dummy_frame = np.zeros((3, 100, 100), dtype=np.uint8)
        b64_str = self.processor._frame_to_b64(dummy_frame)
        self.assertIsInstance(b64_str, str)
        self.assertTrue(len(b64_str) > 0)
        
        # Decode back to verify
        img_data = base64.b64decode(b64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        self.assertEqual(img.shape, (100, 100, 3))

if __name__ == '__main__':
    unittest.main()
