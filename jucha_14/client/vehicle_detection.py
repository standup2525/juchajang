import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Make sure this model is in the project folder

# Create folder for vehicle images
os.makedirs('images', exist_ok=True)

def detect_vehicle(frame):
    results = model.predict(frame, imgsz=640, conf=0.4)
    annotated_frame = frame.copy()
    captured = False
    save_path = None

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            if label in ['car', 'truck', 'bus', 'motorbike']:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1
                area = w * h
                screen_area = frame.shape[0] * frame.shape[1]

                if area > 0.2 * screen_area:
                    print(f"[Detection] {label} detected, large enough: {area/screen_area:.2%}")
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    save_path = f'images/vehicle_{timestamp}.jpg'
                    cv2.imwrite(save_path, frame)
                    captured = True
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    break  # Only process first large vehicle

    return annotated_frame, captured, save_path 