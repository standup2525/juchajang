import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime

# YOLOv8 모델 로드
model = YOLO('yolov8n.pt')

# 이미지 저장 디렉토리 생성
os.makedirs('images', exist_ok=True)

def detect_vehicle(frame, debug_mode=True):
    """차량 감지 함수 (디버그 모드 포함)"""
    # YOLOv8로 객체 감지
    results = model(frame)
    
    # 디버그용 이미지 생성
    debug_img = frame.copy()
    
    # 결과 처리
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # 클래스 확인 (차량 관련 클래스만)
            cls = int(box.cls[0])
            class_name = result.names[cls]
            
            if class_name in ['car', 'truck', 'bus', 'motorbike']:
                # 바운딩 박스 좌표
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 디버그 이미지에 바운딩 박스 그리기
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(debug_img, class_name, (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # 차량 영역 추출
                vehicle = frame[y1:y2, x1:x2]
                
                # 차량 크기가 충분히 큰 경우에만 처리
                height, width = vehicle.shape[:2]
                if width * height > (frame.shape[0] * frame.shape[1] * 0.2):  # 프레임의 20% 이상
                    # 현재 시간으로 파일명 생성
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"images/vehicle_{timestamp}.jpg"
                    
                    # 이미지 저장
                    cv2.imwrite(filename, vehicle)
                    
                    # 디버그 이미지 저장
                    if debug_mode:
                        debug_filename = f"images/debug_detection_{timestamp}.jpg"
                        cv2.imwrite(debug_filename, debug_img)
                        print(f"[DEBUG] Saved detection visualization to {debug_filename}")
                    
                    return filename
    
    # 디버그 이미지 저장 (차량이 감지되지 않은 경우)
    if debug_mode:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_filename = f"images/debug_no_detection_{timestamp}.jpg"
        cv2.imwrite(debug_filename, debug_img)
        print(f"[DEBUG] No vehicle detected. Saved debug image to {debug_filename}")
    
    return None 