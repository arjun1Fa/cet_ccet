import runpod
from ultralytics import YOLO
import base64
import cv2
import numpy as np
import traceback

# Load globally on container startup
print("Loading YOLOv11L model globally...")
model = YOLO("yolo11l.pt")
print("Model loaded successfully.")

def handler(job):
    try:
        job_input = job.get('input', {})
        frames_b64 = job_input.get('frames', [])
        
        if not frames_b64:
            return {"error": "No frames provided in input"}
            
        all_detections = []
        
        for idx, b64_frame in enumerate(frames_b64):
            try:
                # Decode base64 string to bytes
                frame_bytes = base64.b64decode(b64_frame)
                
                # Convert to numpy array and decode image
                nparr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    print(f"Frame {idx} decode failed.")
                    all_detections.append([])
                    continue
                    
                # Perform inference
                results = model(frame, device=0, verbose=False)
                
                frame_detections = []
                for r in results:
                    if len(r.boxes) > 0:
                        data = r.boxes.data.cpu().numpy()
                        for box_data in data:
                            xyxy = box_data[:4].tolist()
                            conf = float(box_data[4])
                            cls = int(box_data[5])
                            
                            frame_detections.append({
                                "bbox": xyxy,
                                "confidence": conf,
                                "class_id": cls
                            })
                
                all_detections.append(frame_detections)
                
            except Exception as e:
                print(f"Error processing frame {idx}: {traceback.format_exc()}")
                all_detections.append([])
                
        return {"detections": all_detections}
    except Exception as e:
        return {"error": str(e)}

if __name__ == '__main__':
    runpod.serverless.start({"handler": handler})
