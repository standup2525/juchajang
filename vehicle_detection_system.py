#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
라즈베리파이5 + AI HAT 차량 번호판 인식 시스템
실시간으로 카메라를 모니터링하여 차량을 감지하고 번호판을 인식하여 웹서버에 전송
"""

import cv2
import numpy as np
import time
import requests
import json
import logging
from datetime import datetime
from threading import Thread, Lock
import pytesseract
from ultralytics import YOLO
import os

# ================================
# 시스템 설정 파라미터
# ================================

# 카메라 설정 (고해상도, 고프레임)
CAMERA_INDEX = 0                    # 카메라 인덱스 (0: 기본 카메라)
CAMERA_WIDTH = 1920                 # 카메라 해상도 너비 (Full HD)
CAMERA_HEIGHT = 1080                # 카메라 해상도 높이 (Full HD)
CAMERA_FPS = 60                     # 카메라 프레임 레이트 (고프레임)

# YOLO 모델 설정 (YOLOv8 사용)
YOLO_MODEL_PATH = "yolov8n.pt"      # YOLOv8 nano 모델 (빠른 추론)
YOLO_CONFIDENCE = 0.5               # YOLO 신뢰도 임계값
CAR_DETECTION_AREA = 0.3            # 화면에서 차량이 차지해야 하는 최소 비율 (30%)

# 번호판 인식 설정
MIN_AREA = 210                      # 문자 최소 면적
MIN_WIDTH, MIN_HEIGHT = 8, 16       # 문자 최소 너비, 높이
MIN_RATIO, MAX_RATIO = 0.4, 1.0     # 문자 가로세로 비율 범위
PLATE_WIDTH_PADDING = 1.3           # 번호판 너비 패딩
PLATE_HEIGHT_PADDING = 1.5          # 번호판 높이 패딩

# 문자 매칭 설정
MAX_DIAG_MULTIPLYER = 4             # 대각선 거리 배수
MAX_ANGLE_DIFF = 12.0               # 최대 각도 차이
MAX_AREA_DIFF = 0.25                # 최대 면적 차이
MAX_WIDTH_DIFF = 0.6                # 최대 너비 차이
MAX_HEIGHT_DIFF = 0.15              # 최대 높이 차이
MIN_N_MATCHED = 4                   # 최소 매칭 문자 수

# 입출차 구분 설정 (중앙 기준, 비율 조정 가능)
SCREEN_CENTER = 0.5                 # 화면 중앙 기준점 (50%)
LEFT_ZONE_WIDTH = 0.15              # 좌측 영역 너비 (중앙에서 15% 범위)
RIGHT_ZONE_WIDTH = 0.15             # 우측 영역 너비 (중앙에서 15% 범위)

# 계산된 경계값
SCREEN_LEFT_BOUNDARY = SCREEN_CENTER - LEFT_ZONE_WIDTH    # 좌측 경계 (35%)
SCREEN_RIGHT_BOUNDARY = SCREEN_CENTER + RIGHT_ZONE_WIDTH  # 우측 경계 (65%)

# 시스템 동작 설정
DETECTION_INTERVAL = 5              # 감지 간격 (5초)
DUPLICATE_PREVENTION_TIME = 30      # 중복 방지 시간 (초)

# 웹서버 설정 (로컬 Flask 서버)
WEB_SERVER_URL = "http://localhost:5000/api/vehicle"  # 로컬 웹서버 URL
API_TIMEOUT = 10                    # API 요청 타임아웃 (초)
API_RETRY_COUNT = 3                 # API 재시도 횟수

# 로그 설정
LOG_LEVEL = logging.INFO            # 로그 레벨
LOG_FILE = "vehicle_detection.log"  # 로그 파일명

# Tesseract 설정
TESSERACT_CONFIG = '--psm 7 --oem 3'  # Tesseract OCR 설정

# 성능 최적화 설정
FRAME_SKIP = 2                      # 프레임 스킵 (매 2프레임마다 처리)
MAX_PROCESSING_TIME = 8             # 최대 프레임 처리 시간 (초)

# ================================
# 전역 변수
# ================================

# 시스템 상태
system_running = True
detection_lock = Lock()
recent_detections = {}  # 최근 감지된 차량 번호판 저장 (중복 방지용)

# 로깅 설정
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================================
# 번호판 인식 클래스
# ================================

class LicensePlateRecognizer:
    """번호판 인식을 담당하는 클래스"""
    
    def __init__(self):
        """초기화"""
        logger.info("번호판 인식기 초기화 중...")
        
    def preprocess_image(self, image):
        """이미지 전처리"""
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 가우시안 블러 적용
        blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
        
        # 적응형 임계값 적용
        thresh = cv2.adaptiveThreshold(
            blurred,
            maxValue=255.0,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,
            blockSize=19,
            C=9
        )
        
        return thresh
    
    def find_contours(self, thresh_image):
        """윤곽선 찾기"""
        contours, _ = cv2.findContours(
            thresh_image,
            mode=cv2.RETR_LIST,
            method=cv2.CHAIN_APPROX_SIMPLE
        )
        
        contours_dict = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            contours_dict.append({
                'contour': contour,
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'cx': x + (w/2),
                'cy': y + (h/2)
            })
        
        return contours_dict
    
    def filter_contours(self, contours_dict):
        """문자 후보 윤곽선 필터링"""
        possible_contours = []
        cnt = 0
        
        for d in contours_dict:
            area = d['w'] * d['h']
            ratio = d['w'] / d['h']
            
            if (area > MIN_AREA and 
                d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT and
                MIN_RATIO < ratio < MAX_RATIO):
                d['idx'] = cnt
                cnt += 1
                possible_contours.append(d)
        
        return possible_contours
    
    def find_chars(self, contour_list):
        """문자 그룹 찾기 (재귀 함수)"""
        matched_result_idx = []
        
        for d1 in contour_list:
            matched_contours_idx = []
            for d2 in contour_list:
                if d1['idx'] == d2['idx']:
                    continue
                
                dx = abs(d1['cx'] - d2['cx'])
                dy = abs(d1['cy'] - d2['cy'])
                
                diagonal_length1 = np.sqrt(d1['w']**2 + d1['h']**2)
                distance = np.linalg.norm(
                    np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']])
                )
                
                if dx == 0:
                    angle_diff = 90
                else:
                    angle_diff = np.degrees(np.arctan(dy/dx))
                
                area_diff = abs(d1['w']*d1['h'] - d2['w']*d2['h']) / (d1['w']*d1['h'])
                width_diff = abs(d1['w'] - d2['w']) / d1['w']
                height_diff = abs(d1['h'] - d2['h']) / d1['h']
                
                if (distance < diagonal_length1 * MAX_DIAG_MULTIPLYER and
                    angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF and
                    width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF):
                    matched_contours_idx.append(d2['idx'])
            
            matched_contours_idx.append(d1['idx'])
            
            if len(matched_contours_idx) < MIN_N_MATCHED:
                continue
            
            matched_result_idx.append(matched_contours_idx)
            
            # 재귀 호출을 위한 미매칭 윤곽선 찾기
            unmatched_contour_idx = []
            for d4 in contour_list:
                if d4['idx'] not in matched_contours_idx:
                    unmatched_contour_idx.append(d4['idx'])
            
            if unmatched_contour_idx:
                unmatched_contour = np.take(contour_list, unmatched_contour_idx)
                recursive_contour_list = self.find_chars(unmatched_contour)
                
                for idx in recursive_contour_list:
                    matched_result_idx.append(idx)
            
            break
        
        return matched_result_idx
    
    def extract_plate_region(self, image, thresh_image, matched_chars):
        """번호판 영역 추출"""
        sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])
        
        # 번호판 중심점 계산
        plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
        plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2
        
        # 번호판 크기 계산
        plate_width = ((sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) 
                      * PLATE_WIDTH_PADDING)
        
        sum_height = sum(d['h'] for d in sorted_chars)
        plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)
        
        # 회전 각도 계산
        triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
        triangle_hypotenus = np.linalg.norm(
            np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) - 
            np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
        )
        
        if triangle_hypotenus > 0:
            angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))
        else:
            angle = 0
        
        # 이미지 회전
        height, width = thresh_image.shape
        rotation_matrix = cv2.getRotationMatrix2D(
            center=(plate_cx, plate_cy), angle=angle, scale=1.0
        )
        img_rotated = cv2.warpAffine(thresh_image, M=rotation_matrix, dsize=(width, height))
        
        # 번호판 영역 추출
        img_cropped = cv2.getRectSubPix(
            img_rotated,
            patchSize=(int(plate_width), int(plate_height)),
            center=(int(plate_cx), int(plate_cy))
        )
        
        return img_cropped, {
            'x': int(plate_cx - plate_width / 2),
            'y': int(plate_cy - plate_height / 2),
            'w': int(plate_width),
            'h': int(plate_height)
        }
    
    def recognize_text(self, plate_image):
        """번호판 텍스트 인식"""
        # 이미지 크기 조정
        plate_image = cv2.resize(plate_image, dsize=(0, 0), fx=1.6, fy=1.6)
        
        # 이진화
        _, plate_image = cv2.threshold(
            plate_image, thresh=0.0, maxval=255.0, 
            type=cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        
        # 윤곽선 다시 찾기
        contours, _ = cv2.findContours(
            plate_image, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 번호판 영역 경계 찾기
        plate_min_x, plate_min_y = plate_image.shape[1], plate_image.shape[0]
        plate_max_x, plate_max_y = 0, 0
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            ratio = w / h
            
            if (area > MIN_AREA and w > MIN_WIDTH and h > MIN_HEIGHT and
                MIN_RATIO < ratio < MAX_RATIO):
                plate_min_x = min(plate_min_x, x)
                plate_min_y = min(plate_min_y, y)
                plate_max_x = max(plate_max_x, x + w)
                plate_max_y = max(plate_max_y, y + h)
        
        # 번호판 영역 자르기
        img_result = plate_image[plate_min_y:plate_max_y, plate_min_x:plate_max_x]
        
        # 추가 전처리
        img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0)
        _, img_result = cv2.threshold(
            img_result, thresh=0.0, maxval=255.0, 
            type=cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        img_result = cv2.copyMakeBorder(
            img_result, top=10, bottom=10, left=10, right=10, 
            borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        
        # OCR 수행
        try:
            chars = pytesseract.image_to_string(
                img_result, lang='kor', config=TESSERACT_CONFIG
            )
            
            # 결과 정제
            result_chars = ''
            has_digit = False
            for c in chars:
                if ord('가') <= ord(c) <= ord('힣') or c.isdigit():
                    if c.isdigit():
                        has_digit = True
                    result_chars += c
            
            return result_chars if has_digit and len(result_chars) >= 4 else None
            
        except Exception as e:
            logger.error(f"OCR 처리 중 오류 발생: {e}")
            return None
    
    def recognize_license_plate(self, image):
        """번호판 인식 메인 함수"""
        try:
            start_time = time.time()
            
            # 이미지 전처리
            thresh_image = self.preprocess_image(image)
            
            # 윤곽선 찾기
            contours_dict = self.find_contours(thresh_image)
            
            # 문자 후보 필터링
            possible_contours = self.filter_contours(contours_dict)
            
            if not possible_contours:
                return None, None
            
            # 문자 그룹 찾기
            result_idx = self.find_chars(possible_contours)
            
            if not result_idx:
                return None, None
            
            # 매칭된 문자 그룹들
            matched_result = []
            for idx_list in result_idx:
                matched_result.append(np.take(possible_contours, idx_list))
            
            best_plate_text = None
            best_plate_info = None
            longest_text_length = 0
            
            # 각 번호판 후보에 대해 텍스트 인식
            for matched_chars in matched_result:
                try:
                    # 처리 시간 체크
                    if time.time() - start_time > MAX_PROCESSING_TIME:
                        logger.warning("번호판 인식 처리 시간 초과")
                        break
                    
                    plate_image, plate_info = self.extract_plate_region(
                        image, thresh_image, matched_chars
                    )
                    
                    plate_text = self.recognize_text(plate_image)
                    
                    if plate_text and len(plate_text) > longest_text_length:
                        best_plate_text = plate_text
                        best_plate_info = plate_info
                        longest_text_length = len(plate_text)
                        
                except Exception as e:
                    logger.warning(f"번호판 처리 중 오류: {e}")
                    continue
            
            processing_time = time.time() - start_time
            if best_plate_text:
                logger.info(f"번호판 인식 완료: {best_plate_text} (처리시간: {processing_time:.2f}초)")
            
            return best_plate_text, best_plate_info
            
        except Exception as e:
            logger.error(f"번호판 인식 중 오류 발생: {e}")
            return None, None

# ================================
# 차량 감지 클래스
# ================================

class VehicleDetector:
    """YOLOv8을 사용한 차량 감지 클래스"""
    
    def __init__(self):
        """초기화"""
        logger.info("YOLOv8 차량 감지기 초기화 중...")
        try:
            self.model = YOLO(YOLO_MODEL_PATH)
            logger.info(f"YOLOv8 모델 로드 완료: {YOLO_MODEL_PATH}")
            
            # 모델 워밍업
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model(dummy_image, verbose=False)
            logger.info("YOLOv8 모델 워밍업 완료")
            
        except Exception as e:
            logger.error(f"YOLOv8 모델 로드 실패: {e}")
            raise
    
    def detect_vehicles(self, image):
        """이미지에서 차량 감지"""
        try:
            start_time = time.time()
            
            # YOLOv8 추론 실행
            results = self.model(image, conf=YOLO_CONFIDENCE, verbose=False)
            
            vehicles = []
            height, width = image.shape[:2]
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # 클래스 ID 확인 (COCO 데이터셋에서 car=2, truck=7, bus=5, motorcycle=3)
                        class_id = int(box.cls[0])
                        if class_id in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf[0])
                            
                            # 차량 영역 크기 계산
                            vehicle_area = (x2 - x1) * (y2 - y1)
                            image_area = width * height
                            area_ratio = vehicle_area / image_area
                            
                            # 30% 이상 차지하는 차량만 선택
                            if area_ratio >= CAR_DETECTION_AREA:
                                center_x_normalized = (x1 + x2) / 2 / width
                                
                                vehicles.append({
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                    'confidence': confidence,
                                    'area_ratio': area_ratio,
                                    'center_x': center_x_normalized,
                                    'class_id': class_id,
                                    'class_name': self.get_class_name(class_id)
                                })
            
            processing_time = time.time() - start_time
            if vehicles:
                logger.debug(f"차량 감지 완료: {len(vehicles)}대 (처리시간: {processing_time:.3f}초)")
            
            return vehicles
            
        except Exception as e:
            logger.error(f"차량 감지 중 오류 발생: {e}")
            return []
    
    def get_class_name(self, class_id):
        """클래스 ID를 클래스 이름으로 변환"""
        class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        return class_names.get(class_id, 'vehicle')

# ================================
# 웹서버 통신 클래스
# ================================

class WebServerClient:
    """웹서버와의 통신을 담당하는 클래스"""
    
    def __init__(self):
        """초기화"""
        self.session = requests.Session()
        self.session.timeout = API_TIMEOUT
        logger.info(f"웹서버 클라이언트 초기화: {WEB_SERVER_URL}")
    
    def send_vehicle_data(self, plate_number, direction, timestamp, confidence=None, image_path=None):
        """차량 정보를 웹서버에 전송"""
        data = {
            'plate_number': plate_number,
            'direction': direction,  # 'entry' 또는 'exit'
            'timestamp': timestamp,
            'device_id': 'raspberry_pi_5'
        }
        
        if confidence is not None:
            data['confidence'] = confidence
        
        files = None
        if image_path and os.path.exists(image_path):
            try:
                files = {'image': open(image_path, 'rb')}
            except Exception as e:
                logger.warning(f"이미지 파일 열기 실패: {e}")
        
        for attempt in range(API_RETRY_COUNT):
            try:
                response = self.session.post(
                    WEB_SERVER_URL,
                    data=data,
                    files=files,
                    timeout=API_TIMEOUT
                )
                
                if files and 'image' in files:
                    files['image'].close()
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get('success'):
                        logger.info(f"차량 정보 전송 성공: {plate_number} ({direction})")
                        return True
                    else:
                        logger.warning(f"서버 처리 오류: {result.get('error', 'Unknown error')}")
                else:
                    logger.warning(f"서버 응답 오류: {response.status_code} - {response.text}")
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"웹서버 통신 오류 (시도 {attempt + 1}/{API_RETRY_COUNT}): {e}")
                if attempt < API_RETRY_COUNT - 1:
                    time.sleep(2)
            except Exception as e:
                logger.error(f"예상치 못한 오류: {e}")
                break
        
        return False

# ================================
# 메인 시스템 클래스
# ================================

class VehicleDetectionSystem:
    """차량 번호판 인식 시스템 메인 클래스"""
    
    def __init__(self):
        """초기화"""
        logger.info("차량 감지 시스템 초기화 중...")
        logger.info(f"설정 정보:")
        logger.info(f"  - 카메라 해상도: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
        logger.info(f"  - 프레임 레이트: {CAMERA_FPS}fps")
        logger.info(f"  - 프레임 스킵: 매 {FRAME_SKIP}프레임마다 처리")
        print()
        print(f"YOLO 설정:")
        print(f"  - 모델: {YOLO_MODEL_PATH}")
        print(f"  - 신뢰도 임계값: {YOLO_CONFIDENCE}")
        print(f"  - 차량 감지 최소 영역: {CAR_DETECTION_AREA * 100}%")
        print()
        print(f"입출차 구분 설정:")
        print(f"  - 화면 중앙: {SCREEN_CENTER * 100}%")
        print(f"  - 좌측 영역: ~{SCREEN_LEFT_BOUNDARY * 100:.1f}% (입차)")
        print(f"  - 중앙 영역: {SCREEN_LEFT_BOUNDARY * 100:.1f}%~{SCREEN_RIGHT_BOUNDARY * 100:.1f}% (미확정)")
        print(f"  - 우측 영역: {SCREEN_RIGHT_BOUNDARY * 100:.1f}%~ (출차)")
        print()
        print(f"시스템 동작:")
        print(f"  - 감지 간격: {DETECTION_INTERVAL}초")
        print(f"  - 중복 방지 시간: {DUPLICATE_PREVENTION_TIME}초")
        print(f"  - 최대 처리 시간: {MAX_PROCESSING_TIME}초")
        print()
        print(f"웹서버:")
        print(f"  - URL: {WEB_SERVER_URL}")
        print(f"  - 타임아웃: {API_TIMEOUT}초")
        print(f"  - 재시도 횟수: {API_RETRY_COUNT}회")
        print("=" * 60)
        
        # 컴포넌트 초기화
        self.vehicle_detector = VehicleDetector()
        self.plate_recognizer = LicensePlateRecognizer()
        self.web_client = WebServerClient()
        
        # 카메라 초기화
        self.camera = cv2.VideoCapture(CAMERA_INDEX)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        
        # 카메라 버퍼 크기 설정 (지연 최소화)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.camera.isOpened():
            raise Exception("카메라를 열 수 없습니다.")
        
        # 실제 카메라 설정 확인
        actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.camera.get(cv2.CAP_PROP_FPS))
        
        logger.info(f"실제 카메라 설정: {actual_width}x{actual_height} @ {actual_fps}fps")
        
        # 프레임 카운터
        self.frame_count = 0
        
        logger.info("시스템 초기화 완료")
    
    def determine_direction(self, center_x):
        """차량의 위치를 기반으로 입출차 방향 결정 (중앙 기준)"""
        if center_x < SCREEN_LEFT_BOUNDARY:
            return 'entry'  # 입차 (화면 왼쪽)
        elif center_x > SCREEN_RIGHT_BOUNDARY:
            return 'exit'   # 출차 (화면 오른쪽)
        else:
            return 'unknown'  # 중앙 영역 (방향 미확정)
    
    def is_duplicate_detection(self, plate_number):
        """중복 감지 확인"""
        current_time = time.time()
        
        with detection_lock:
            if plate_number in recent_detections:
                last_detection_time = recent_detections[plate_number]
                if current_time - last_detection_time < DUPLICATE_PREVENTION_TIME:
                    return True
            
            recent_detections[plate_number] = current_time
            
            # 오래된 기록 정리 (메모리 절약)
            expired_plates = [
                plate for plate, detection_time in recent_detections.items()
                if current_time - detection_time > DUPLICATE_PREVENTION_TIME
            ]
            for plate in expired_plates:
                del recent_detections[plate]
        
        return False
    
    def save_detection_image(self, image, plate_number, direction):
        """감지된 차량 이미지 저장"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 파일명에서 특수문자 제거
            safe_plate_number = ''.join(c for c in plate_number if c.isalnum())
            filename = f"{safe_plate_number}_{direction}_{timestamp}.jpg"
            filepath = os.path.join("detections", filename)
            
            # 디렉토리 생성
            os.makedirs("detections", exist_ok=True)
            
            # 이미지 품질 설정하여 저장
            cv2.imwrite(filepath, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            logger.debug(f"이미지 저장 완료: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"이미지 저장 실패: {e}")
            return None
    
    def process_frame(self, frame):
        """프레임 처리"""
        try:
            start_time = time.time()
            
            # 차량 감지
            vehicles = self.vehicle_detector.detect_vehicles(frame)
            
            if not vehicles:
                return
            
            logger.info(f"{len(vehicles)}대의 차량 감지됨")
            
            for i, vehicle in enumerate(vehicles):
                try:
                    # 차량 영역 추출
                    x1, y1, x2, y2 = vehicle['bbox']
                    vehicle_image = frame[y1:y2, x1:x2]
                    
                    if vehicle_image.size == 0:
                        continue
                    
                    logger.debug(f"차량 {i+1}: {vehicle['class_name']} (신뢰도: {vehicle['confidence']:.2f}, 위치: {vehicle['center_x']:.2f})")
                    
                    # 번호판 인식
                    plate_text, plate_info = self.plate_recognizer.recognize_license_plate(vehicle_image)
                    
                    if plate_text:
                        logger.info(f"번호판 인식됨: {plate_text}")
                        
                        # 중복 감지 확인
                        if self.is_duplicate_detection(plate_text):
                            logger.info(f"중복 감지 무시: {plate_text}")
                            continue
                        
                        # 입출차 방향 결정
                        direction = self.determine_direction(vehicle['center_x'])
                        
                        if direction != 'unknown':
                            # 이미지 저장
                            image_path = self.save_detection_image(frame, plate_text, direction)
                            
                            # 웹서버에 전송
                            timestamp = datetime.now().isoformat()
                            success = self.web_client.send_vehicle_data(
                                plate_text, direction, timestamp, 
                                confidence=vehicle['confidence'], 
                                image_path=image_path
                            )
                            
                            if success:
                                logger.info(f"차량 정보 처리 완료: {plate_text} ({direction}) - {vehicle['class_name']}")
                            else:
                                logger.error(f"차량 정보 전송 실패: {plate_text}")
                        else:
                            logger.info(f"방향 미확정: {plate_text} (중앙 영역, 위치: {vehicle['center_x']:.2f})")
                    
                except Exception as e:
                    logger.warning(f"차량 {i+1} 처리 중 오류: {e}")
                    continue
            
            processing_time = time.time() - start_time
            logger.debug(f"프레임 처리 완료 (총 처리시간: {processing_time:.2f}초)")
                
        except Exception as e:
            logger.error(f"프레임 처리 중 오류: {e}")
    
    def run(self):
        """시스템 실행"""
        logger.info("차량 감지 시스템 시작")
        logger.info("시스템 종료: Ctrl+C")
        
        try:
            last_detection_time = 0
            
            while system_running:
                ret, frame = self.camera.read()
                if not ret:
                    logger.error("카메라에서 프레임을 읽을 수 없습니다.")
                    time.sleep(1)
                    continue
                
                self.frame_count += 1
                current_time = time.time()
                
                # 프레임 스킵 (성능 최적화)
                if self.frame_count % FRAME_SKIP != 0:
                    continue
                
                # 감지 간격 체크
                if current_time - last_detection_time >= DETECTION_INTERVAL:
                    logger.debug(f"프레임 처리 시작 (프레임 #{self.frame_count})")
                    self.process_frame(frame)
                    last_detection_time = current_time
                
                # 짧은 대기 (CPU 사용률 조절)
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            logger.info("사용자에 의해 시스템 종료")
        except Exception as e:
            logger.error(f"시스템 실행 중 오류: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """시스템 정리"""
        logger.info("시스템 정리 중...")
        global system_running
        system_running = False
        
        if hasattr(self, 'camera'):
            self.camera.release()
        cv2.destroyAllWindows()
        logger.info("시스템 종료 완료")

# ================================
# 설정 출력 함수
# ================================

def print_system_config():
    """시스템 설정 정보 출력"""
    print("=" * 60)
    print("차량 번호판 인식 시스템 설정")
    print("=" * 60)
    print(f"카메라 설정:")
    print(f"  - 해상도: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    print(f"  - 프레임 레이트: {CAMERA_FPS}fps")
    print(f"  - 프레임 스킵: 매 {FRAME_SKIP}프레임마다 처리")
    print()
    print(f"YOLO 설정:")
    print(f"  - 모델: {YOLO_MODEL_PATH}")
    print(f"  - 신뢰도 임계값: {YOLO_CONFIDENCE}")
    print(f"  - 차량 감지 최소 영역: {CAR_DETECTION_AREA * 100}%")
    print()
    print(f"입출차 구분 설정:")
    print(f"  - 화면 중앙: {SCREEN_CENTER * 100}%")
    print(f"  - 좌측 영역: ~{SCREEN_LEFT_BOUNDARY * 100:.1f}% (입차)")
    print(f"  - 중앙 영역: {SCREEN_LEFT_BOUNDARY * 100:.1f}%~{SCREEN_RIGHT_BOUNDARY * 100:.1f}% (미확정)")
    print(f"  - 우측 영역: {SCREEN_RIGHT_BOUNDARY * 100:.1f}%~ (출차)")
    print()
    print(f"시스템 동작:")
    print(f"  - 감지 간격: {DETECTION_INTERVAL}초")
    print(f"  - 중복 방지 시간: {DUPLICATE_PREVENTION_TIME}초")
    print(f"  - 최대 처리 시간: {MAX_PROCESSING_TIME}초")
    print()
    print(f"웹서버:")
    print(f"  - URL: {WEB_SERVER_URL}")
    print(f"  - 타임아웃: {API_TIMEOUT}초")
    print(f"  - 재시도 횟수: {API_RETRY_COUNT}회")
    print("=" * 60)

# ================================
# 메인 실행부
# ================================

def main():
    """메인 함수"""
    try:
        # 설정 정보 출력
        print_system_config()
        
        # 시스템 초기화 및 실행
        system = VehicleDetectionSystem()
        system.run()
        
    except Exception as e:
        logger.error(f"시스템 초기화 실패: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 