# This code detects vehicles using YOLOv8 and saves images if a vehicle is large enough

import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Ensure this file is present

# Check if images folder exists
os.makedirs('images', exist_ok=True)

def detect_vehicle(frame):
    results = model.predict(frame, imgsz=640, conf=0.4)
    annotated_frame = frame.copy()
    captured = False
    save_path = None

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            if model.names[cls] in ['car', 'truck', 'bus', 'motorbike']:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1
                area = w * h
                total_area = frame.shape[0] * frame.shape[1]

                # Check if object is larger than 20% of screen
                if area > 0.2 * total_area:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    save_path = f'images/vehicle_{timestamp}.jpg'
                    cv2.imwrite(save_path, frame)
                    captured = True
                    break  # Save only one large vehicle per frame

    return annotated_frame, captured, save_path 