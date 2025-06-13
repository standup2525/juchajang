# Detects vehicles and saves zoomed-in vehicle region if large enough

import cv2
from ultralytics import YOLO
from datetime import datetime
import os

model = YOLO('yolov8n.pt')  # 작은 경량 모델 사용
os.makedirs('images', exist_ok=True)

def detect_vehicle(frame):
    results = model(frame)
    boxes = results[0].boxes
    annotated_frame = frame.copy()

    captured = False
    img_path = ""

    for box in boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        if label not in ['car', 'truck', 'bus']:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1
        area_ratio = (w * h) / (frame.shape[0] * frame.shape[1]) * 100

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"{label} {area_ratio:.2f}%", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if area_ratio > 20:
            print(f"[Detection] {label} detected, large enough: {area_ratio:.2f}%")
            cropped = frame[y1:y2, x1:x2]
            enlarged = cv2.resize(cropped, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_LINEAR)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_path = f"images/vehicle_{timestamp}.jpg"
            cv2.imwrite(img_path, enlarged)
            captured = True
            break

    return annotated_frame, captured, img_path
