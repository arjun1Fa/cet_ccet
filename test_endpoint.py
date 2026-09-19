import base64
import cv2
import requests
import time
import os
import sys
import numpy as np

# Import config for RunPod variables
sys.path.append(os.path.dirname(__file__))
from shared import config

def test_endpoint():
    print("="*60)
    print("RUNPOD T4 ENDPOINT INDEPENDENT TEST")
    print("="*60)
    
    # 1. Prepare Test Frame
    test_file_path = "data/stream_ (1).mp4"
    if os.path.exists(test_file_path):
        cap = cv2.VideoCapture(test_file_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"Warning: Could not read {test_file_path}. Creating a blank frame.")
            frame = np.zeros((640, 640, 3), dtype=np.uint8)
    else:
        print(f"Warning: {test_file_path} not found. Creating a blank 640x640 frame.")
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        
    _, buffer = cv2.imencode('.jpg', frame)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    
    # 2. Submit Job
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "frames": [b64_str]
    }
    
    run_url = f"{config.RUNPOD_BASE_URL}/predict"
    
    print(f"Submitting job to {run_url}...")
    try:
        start_t = time.time()
        response = requests.post(run_url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        
        results = response.json()
        duration = time.time() - start_t
        
        if results and isinstance(results, list):
            print("\nTEST PASSED!")
            print(f"Inference returned in {duration:.2f}s")
            print("Output Detections:")
            print(results)
            return True
        else:
            print("\nTEST FAILED!")
            print(f"Invalid or empty response: {results}")
            return False
            
    except Exception as e:
        print(f"\nTEST FAILED: Exception: {e}")
        return False

if __name__ == "__main__":
    success = test_endpoint()
    if success:
        sys.exit(0)
    else:
        sys.exit(1)
