#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for vehicle and license plate detection
This script tests the core detection functionality without server communication
"""

import cv2
import numpy as np
import logging
from ultralytics import YOLO
import pytesseract
import subprocess
import time

# ================================
# Configuration
# ================================

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 360
CAMERA_FPS = 30

# Detection settings
YOLO_CONFIDENCE = 0.5    # minimum confidence for YOLO detection

# ================================
# Logging Setup
# ================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================
# Helper Functions
# ================================

def start_stream():
    """카메라 스트림 시작"""
    return subprocess.Popen([
        "libcamera-vid",
        "--width", str(CAMERA_WIDTH),
        "--height", str(CAMERA_HEIGHT),
        "--framerate", str(CAMERA_FPS),
        "--codec", "mjpeg",
        "--timeout", "0",
        "--nopreview",
        "-o", "-"
    ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)

def detect_vehicles(frame, model):
    """
    Detect vehicles in the frame using YOLO
    Args:
        frame: Input frame
        model: YOLO model
    Returns:
        list: List of detected vehicle bounding boxes
    """
    results = model(frame)
    return results[0].boxes.data.cpu().numpy()

def recognize_plate(frame, box):
    """
    Recognize license plate in the given bounding box
    Args:
        frame: Input frame
        box: Bounding box (x1, y1, x2, y2)
    Returns:
        str: Recognized plate number or None
    """
    x1, y1, x2, y2 = map(int, box[:4])
    plate_img = frame[y1:y2, x1:x2]
    
    # 이미지 전처리
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # OCR 수행
    try:
        text = pytesseract.image_to_string(thresh, lang='kor+eng')
        return text.strip()
    except Exception as e:
        logger.error(f"OCR 오류: {str(e)}")
        return None

def main():
    """
    Main detection loop
    """
    try:
        # YOLO 모델 로드
        model = YOLO('yolov8n.pt')
        
        # 카메라 스트림 시작
        proc = start_stream()
        buf = b''
        
        logger.info("카메라 스트림 시작")
        
        while True:
            # 프레임 읽기
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
                
            buf += chunk
            start = buf.find(b'\xff\xd8')
            end = buf.find(b'\xff\xd9', start+2)
            
            if start != -1 and end != -1:
                jpg = buf[start:end+2]
                buf = buf[end+2:]
                frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue
                
                # 차량 감지
                boxes = detect_vehicles(frame, model)
                
                # 결과 표시
                for box in boxes:
                    x1, y1, x2, y2, conf, cls = box
                    if conf > YOLO_CONFIDENCE:  # 신뢰도 50% 이상
                        # 박스 그리기
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        
                        # 번호판 인식
                        plate_text = recognize_plate(frame, box)
                        if plate_text:
                            cv2.putText(frame, plate_text, (int(x1), int(y1)-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # 결과 표시
                cv2.imshow('Vehicle Detection', frame)
                
                # 'q' 키를 누르면 종료
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
    except Exception as e:
        logger.error(f"오류 발생: {str(e)}")
    finally:
        cv2.destroyAllWindows()
        proc.terminate()

if __name__ == "__main__":
    main() 