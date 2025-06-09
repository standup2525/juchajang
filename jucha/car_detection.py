import cv2
import numpy as np
import torch
from ultralytics import YOLO
import pytesseract
from PIL import Image
import time
import subprocess
import logging
from queue import Queue
import threading

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('car_detection.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 카메라 설정
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 360
CAMERA_FPS = 30

class CarDetector:
    def __init__(self):
        # YOLO 모델 로드
        self.model = YOLO('yolov8n.pt')
        # 자동차 클래스 ID (COCO 데이터셋 기준)
        self.car_class_id = 2  # COCO 데이터셋에서 자동차 클래스 ID
        
    def detect_cars(self, frame):
        # YOLO로 객체 감지
        results = self.model(frame)
        car_boxes = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 자동차 클래스인 경우만 처리
                if int(box.cls) == self.car_class_id:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    car_boxes.append((x1, y1, x2, y2))
        
        return car_boxes

class LicensePlateRecognizer:
    def __init__(self):
        # 번호판 인식을 위한 파라미터
        self.MIN_AREA = 210
        self.MIN_WIDTH, self.MIN_HEIGHT = 8, 16
        self.MIN_RATIO, self.MAX_RATIO = 0.4, 1.0
        
    def preprocess_image(self, image):
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 가우시안 블러 적용
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # 적응형 이진화
        thresh = cv2.adaptiveThreshold(
            blurred,
            255.0,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            19,
            9
        )
        return thresh

    def find_license_plate(self, image):
        # 이미지 전처리
        thresh = self.preprocess_image(image)
        
        # 윤곽선 찾기
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 가능한 번호판 윤곽선 필터링
        possible_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            ratio = w / h if h != 0 else 0
            
            if (area > self.MIN_AREA and
                w > self.MIN_WIDTH and h > self.MIN_HEIGHT and
                self.MIN_RATIO < ratio < self.MAX_RATIO):
                possible_contours.append({
                    'contour': contour,
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'cx': x + w/2, 'cy': y + h/2
                })
        
        return possible_contours

    def recognize_text(self, plate_image):
        # 이미지 크기 조정
        plate_image = cv2.resize(plate_image, None, fx=1.6, fy=1.6)
        # 이진화
        _, plate_image = cv2.threshold(plate_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # OCR 수행
        text = pytesseract.image_to_string(plate_image, lang='kor+eng')
        return text.strip()

def start_camera():
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

def read_frame(proc):
    """카메라에서 프레임 읽기"""
    try:
        # JPEG 헤더 찾기
        while True:
            buf = proc.stdout.read(2)
            if buf[0] == 0xFF and buf[1] == 0xD8:
                break
        
        # JPEG 데이터 읽기
        frame_data = bytearray()
        frame_data.extend(buf)
        
        while True:
            buf = proc.stdout.read(2)
            frame_data.extend(buf)
            if buf[0] == 0xFF and buf[1] == 0xD9:
                break
        
        # JPEG 데이터를 이미지로 변환
        frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        logger.error(f"Error reading frame: {e}")
        return None

def main():
    try:
        # 카메라 초기화
        logger.info("Starting camera...")
        proc = start_camera()
        
        # 객체 생성
        car_detector = CarDetector()
        plate_recognizer = LicensePlateRecognizer()
        
        logger.info("Starting detection system...")
        logger.info("Press 'q' to quit")
        
        while True:
            # 프레임 읽기
            frame = read_frame(proc)
            if frame is None:
                continue
                
            # 자동차 감지
            car_boxes = car_detector.detect_cars(frame)
            
            # 각 자동차에 대해 번호판 인식 시도
            for (x1, y1, x2, y2) in car_boxes:
                # 자동차 영역 추출
                car_roi = frame[y1:y2, x1:x2]
                
                # 번호판 찾기
                possible_plates = plate_recognizer.find_license_plate(car_roi)
                
                for plate in possible_plates:
                    # 번호판 영역 추출
                    plate_x = plate['x']
                    plate_y = plate['y']
                    plate_w = plate['w']
                    plate_h = plate['h']
                    
                    # 번호판 이미지 추출
                    plate_image = car_roi[plate_y:plate_y+plate_h, plate_x:plate_x+plate_w]
                    
                    # 번호판 텍스트 인식
                    plate_text = plate_recognizer.recognize_text(plate_image)
                    
                    if plate_text:
                        # 번호판 영역 표시
                        cv2.rectangle(frame, 
                                    (x1 + plate_x, y1 + plate_y),
                                    (x1 + plate_x + plate_w, y1 + plate_y + plate_h),
                                    (0, 255, 0), 2)
                        
                        # 인식된 텍스트 표시
                        cv2.putText(frame, plate_text,
                                  (x1 + plate_x, y1 + plate_y - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                  (0, 255, 0), 2)
                
                # 자동차 영역 표시
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # 결과 표시
            cv2.imshow('Car Detection and License Plate Recognition', frame)
            
            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
    finally:
        # 정리
        if 'proc' in locals():
            proc.terminate()
        cv2.destroyAllWindows()
        logger.info("System stopped")

if __name__ == "__main__":
    main() 