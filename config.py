# -*- coding: utf-8 -*-
"""
차량 번호판 인식 시스템 설정 파일
모든 시스템 파라미터를 이 파일에서 관리합니다.
"""

import logging

# ================================
# 카메라 설정
# ================================
CAMERA_INDEX = 0                    # 카메라 인덱스 (0: 기본 카메라)
CAMERA_WIDTH = 1280                 # 카메라 해상도 너비
CAMERA_HEIGHT = 720                 # 카메라 해상도 높이
CAMERA_FPS = 30                     # 카메라 프레임 레이트

# ================================
# YOLO 모델 설정
# ================================
YOLO_MODEL_PATH = "yolov8n.pt"      # YOLO 모델 파일 경로
YOLO_CONFIDENCE = 0.5               # YOLO 신뢰도 임계값 (0.0 ~ 1.0)
CAR_DETECTION_AREA = 0.3            # 화면에서 차량이 차지해야 하는 최소 비율 (30%)

# COCO 데이터셋 클래스 ID (차량 관련)
VEHICLE_CLASS_IDS = [2, 5, 7]       # car=2, bus=5, truck=7

# ================================
# 번호판 인식 설정
# ================================
# 문자 필터링 파라미터
MIN_AREA = 210                      # 문자 최소 면적
MIN_WIDTH, MIN_HEIGHT = 8, 16       # 문자 최소 너비, 높이
MIN_RATIO, MAX_RATIO = 0.4, 1.0     # 문자 가로세로 비율 범위

# 번호판 영역 추출 파라미터
PLATE_WIDTH_PADDING = 1.3           # 번호판 너비 패딩 배수
PLATE_HEIGHT_PADDING = 1.5          # 번호판 높이 패딩 배수

# 문자 매칭 파라미터
MAX_DIAG_MULTIPLYER = 4             # 대각선 거리 배수
MAX_ANGLE_DIFF = 12.0               # 최대 각도 차이 (도)
MAX_AREA_DIFF = 0.25                # 최대 면적 차이 비율
MAX_WIDTH_DIFF = 0.6                # 최대 너비 차이 비율
MAX_HEIGHT_DIFF = 0.15              # 최대 높이 차이 비율
MIN_N_MATCHED = 4                   # 최소 매칭 문자 수

# 이미지 전처리 파라미터
GAUSSIAN_BLUR_KERNEL = (5, 5)       # 가우시안 블러 커널 크기
ADAPTIVE_THRESH_BLOCK_SIZE = 19     # 적응형 임계값 블록 크기
ADAPTIVE_THRESH_C = 9               # 적응형 임계값 상수

# OCR 설정
TESSERACT_CONFIG = '--psm 7 --oem 3'  # Tesseract OCR 설정
MIN_PLATE_TEXT_LENGTH = 4           # 유효한 번호판 텍스트 최소 길이

# ================================
# 입출차 구분 설정
# ================================
SCREEN_LEFT_BOUNDARY = 0.4          # 화면 좌측 경계 (40% 지점)
SCREEN_RIGHT_BOUNDARY = 0.6         # 화면 우측 경계 (60% 지점)

# 방향 정의
DIRECTION_ENTRY = 'entry'           # 입차
DIRECTION_EXIT = 'exit'             # 출차
DIRECTION_UNKNOWN = 'unknown'       # 미확정

# ================================
# 시스템 동작 설정
# ================================
DETECTION_INTERVAL = 3              # 감지 간격 (초)
DUPLICATE_PREVENTION_TIME = 30      # 중복 방지 시간 (초)
MAX_PROCESSING_TIME = 10            # 최대 프레임 처리 시간 (초)

# ================================
# 웹서버 설정
# ================================
WEB_SERVER_URL = "http://your-server.com/api/vehicle"  # 웹서버 URL
API_TIMEOUT = 10                    # API 요청 타임아웃 (초)
API_RETRY_COUNT = 3                 # API 재시도 횟수
API_RETRY_DELAY = 2                 # 재시도 간격 (초)

# 전송 데이터 필드
API_FIELDS = {
    'plate_number': 'plate_number',
    'direction': 'direction',
    'timestamp': 'timestamp',
    'device_id': 'device_id',
    'confidence': 'confidence',
    'image': 'image'
}

# 디바이스 식별자
DEVICE_ID = 'raspberry_pi_5'

# ================================
# 파일 및 디렉토리 설정
# ================================
# 로그 설정
LOG_LEVEL = logging.INFO            # 로그 레벨
LOG_FILE = "vehicle_detection.log"  # 로그 파일명
LOG_MAX_SIZE = 10 * 1024 * 1024     # 로그 파일 최대 크기 (10MB)
LOG_BACKUP_COUNT = 5                # 로그 백업 파일 수

# 이미지 저장 설정
DETECTION_IMAGE_DIR = "detections"  # 감지 이미지 저장 디렉토리
IMAGE_QUALITY = 95                  # JPEG 이미지 품질 (0-100)
SAVE_DETECTION_IMAGES = True        # 감지 이미지 저장 여부

# 모델 및 데이터 디렉토리
MODEL_DIR = "models"                # 모델 파일 디렉토리
DATA_DIR = "data"                   # 데이터 디렉토리

# ================================
# 성능 최적화 설정
# ================================
# 멀티스레딩 설정
USE_THREADING = True                # 멀티스레딩 사용 여부
MAX_WORKER_THREADS = 2              # 최대 워커 스레드 수

# 메모리 관리
MAX_FRAME_BUFFER_SIZE = 5           # 최대 프레임 버퍼 크기
GARBAGE_COLLECTION_INTERVAL = 100   # 가비지 컬렉션 간격 (프레임 수)

# GPU 사용 설정 (AI HAT)
USE_GPU = True                      # GPU 사용 여부
GPU_MEMORY_LIMIT = 512              # GPU 메모리 제한 (MB)

# ================================
# 디버그 및 개발 설정
# ================================
DEBUG_MODE = False                  # 디버그 모드
SAVE_DEBUG_IMAGES = False           # 디버그 이미지 저장 여부
SHOW_DETECTION_WINDOW = False       # 실시간 감지 창 표시 여부
VERBOSE_LOGGING = False             # 상세 로깅 여부

# 테스트 모드 설정
TEST_MODE = False                   # 테스트 모드
TEST_IMAGE_PATH = "test_images"     # 테스트 이미지 경로
TEST_VIDEO_PATH = "test_videos"     # 테스트 비디오 경로

# ================================
# 알림 설정
# ================================
ENABLE_NOTIFICATIONS = False        # 알림 기능 사용 여부
NOTIFICATION_WEBHOOK = ""           # 알림 웹훅 URL (Slack, Discord 등)

# 이메일 알림 설정
EMAIL_NOTIFICATIONS = False         # 이메일 알림 사용 여부
SMTP_SERVER = ""                    # SMTP 서버
SMTP_PORT = 587                     # SMTP 포트
EMAIL_USERNAME = ""                 # 이메일 사용자명
EMAIL_PASSWORD = ""                 # 이메일 비밀번호
EMAIL_RECIPIENTS = []               # 수신자 목록

# ================================
# 보안 설정
# ================================
# API 인증
USE_API_KEY = False                 # API 키 사용 여부
API_KEY = ""                        # API 키
API_KEY_HEADER = "X-API-Key"        # API 키 헤더명

# 데이터 암호화
ENCRYPT_STORED_DATA = False         # 저장 데이터 암호화 여부
ENCRYPTION_KEY = ""                 # 암호화 키

# ================================
# 시스템 모니터링 설정
# ================================
ENABLE_SYSTEM_MONITORING = True     # 시스템 모니터링 사용 여부
CPU_USAGE_THRESHOLD = 80            # CPU 사용률 임계값 (%)
MEMORY_USAGE_THRESHOLD = 80         # 메모리 사용률 임계값 (%)
DISK_USAGE_THRESHOLD = 90           # 디스크 사용률 임계값 (%)

# 성능 메트릭
TRACK_PERFORMANCE_METRICS = True    # 성능 메트릭 추적 여부
METRICS_SAVE_INTERVAL = 300         # 메트릭 저장 간격 (초)

# ================================
# 설정 검증 함수
# ================================

def validate_config():
    """설정 값들의 유효성을 검증합니다."""
    errors = []
    
    # 카메라 설정 검증
    if CAMERA_WIDTH <= 0 or CAMERA_HEIGHT <= 0:
        errors.append("카메라 해상도는 0보다 커야 합니다.")
    
    if CAMERA_FPS <= 0:
        errors.append("카메라 FPS는 0보다 커야 합니다.")
    
    # YOLO 설정 검증
    if not (0.0 <= YOLO_CONFIDENCE <= 1.0):
        errors.append("YOLO 신뢰도는 0.0과 1.0 사이여야 합니다.")
    
    if not (0.0 <= CAR_DETECTION_AREA <= 1.0):
        errors.append("차량 감지 영역 비율은 0.0과 1.0 사이여야 합니다.")
    
    # 번호판 인식 설정 검증
    if MIN_AREA <= 0:
        errors.append("최소 면적은 0보다 커야 합니다.")
    
    if MIN_WIDTH <= 0 or MIN_HEIGHT <= 0:
        errors.append("최소 너비와 높이는 0보다 커야 합니다.")
    
    if MIN_RATIO >= MAX_RATIO:
        errors.append("최소 비율은 최대 비율보다 작아야 합니다.")
    
    # 입출차 구분 설정 검증
    if not (0.0 <= SCREEN_LEFT_BOUNDARY <= 1.0):
        errors.append("화면 좌측 경계는 0.0과 1.0 사이여야 합니다.")
    
    if not (0.0 <= SCREEN_RIGHT_BOUNDARY <= 1.0):
        errors.append("화면 우측 경계는 0.0과 1.0 사이여야 합니다.")
    
    if SCREEN_LEFT_BOUNDARY >= SCREEN_RIGHT_BOUNDARY:
        errors.append("좌측 경계는 우측 경계보다 작아야 합니다.")
    
    # 시스템 동작 설정 검증
    if DETECTION_INTERVAL <= 0:
        errors.append("감지 간격은 0보다 커야 합니다.")
    
    if DUPLICATE_PREVENTION_TIME <= 0:
        errors.append("중복 방지 시간은 0보다 커야 합니다.")
    
    # 웹서버 설정 검증
    if API_TIMEOUT <= 0:
        errors.append("API 타임아웃은 0보다 커야 합니다.")
    
    if API_RETRY_COUNT < 0:
        errors.append("API 재시도 횟수는 0 이상이어야 합니다.")
    
    return errors

def print_config_summary():
    """현재 설정의 요약을 출력합니다."""
    print("=== 차량 번호판 인식 시스템 설정 요약 ===")
    print(f"카메라 해상도: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS}fps")
    print(f"YOLO 모델: {YOLO_MODEL_PATH} (신뢰도: {YOLO_CONFIDENCE})")
    print(f"차량 감지 영역: {CAR_DETECTION_AREA * 100}%")
    print(f"감지 간격: {DETECTION_INTERVAL}초")
    print(f"중복 방지 시간: {DUPLICATE_PREVENTION_TIME}초")
    print(f"웹서버 URL: {WEB_SERVER_URL}")
    print(f"로그 레벨: {LOG_LEVEL}")
    print(f"디버그 모드: {DEBUG_MODE}")
    print("=" * 50)

if __name__ == "__main__":
    # 설정 검증
    errors = validate_config()
    if errors:
        print("설정 오류:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("설정이 유효합니다.")
    
    # 설정 요약 출력
    print_config_summary() 