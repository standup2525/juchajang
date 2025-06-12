import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract
from PIL import Image
import time
import subprocess
import logging
from queue import Queue
import threading
from openvino.runtime import Core
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/car_detection.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 360
CAMERA_FPS = 30

class CarDetector:
    def __init__(self):
        self.model = YOLO('models/yolov8n.pt', task='detect')
        logger.info("YOLO model loaded successfully")
        self.car_class_id = 2  # Car class ID in COCO dataset

    def detect_cars(self, frame):
        results = self.model(frame)
        car_boxes = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if int(box.cls) == self.car_class_id:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    car_boxes.append((x1, y1, x2, y2))
        return car_boxes

class LicensePlateRecognizer:
    def __init__(self):
        # 번호판 검출을 위한 상수
        self.MIN_AREA = 210
        self.MIN_WIDTH, self.MIN_HEIGHT = 8, 16
        self.MIN_RATIO, self.MAX_RATIO = 0.4, 1.0
        
        # 번호판 문자 그룹화를 위한 상수
        self.MAX_DIAG_MULTIPLYER = 4
        self.MAX_ANGLE_DIFF = 12.0
        self.MAX_AREA_DIFF = 0.25
        self.MAX_WIDTH_DIFF = 0.6
        self.MAX_HEIGHT_DIFF = 0.15
        self.MIN_N_MATCHED = 4
        
        # 번호판 영역 보정을 위한 상수
        self.PLATE_WIDTH_PADDING = 1.3
        self.PLATE_HEIGHT_PADDING = 1.5
        self.MIN_PLATE_RATIO = 3
        self.MAX_PLATE_RATIO = 10

    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred,
            255.0,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            19,
            9
        )
        return thresh

    def find_chars(self, contour_list):
        matched_result_idx = []

        for d1 in contour_list:
            matched_contours_idx = []
            for d2 in contour_list:
                if d1['idx'] == d2['idx']:
                    continue

                dx = abs(d1['cx'] - d2['cx'])
                dy = abs(d1['cy'] - d2['cy'])

                diagonal_length1 = np.sqrt(d1['w']**2 + d1['h']**2)

                distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
                if dx == 0:
                    angle_diff = 90
                else:
                    angle_diff = np.degrees(np.arctan(dy/dx))
                area_diff = abs(d1['w']*d1['h']-d2['w']*d2['h'])/(d1['w']*d1['h'])
                width_diff = abs(d1['w']-d2['w'])/d1['w']
                height_diff = abs(d1['h']-d2['h'])/d1['h']

                if (distance < diagonal_length1*self.MAX_DIAG_MULTIPLYER
                    and angle_diff < self.MAX_ANGLE_DIFF 
                    and area_diff < self.MAX_AREA_DIFF
                    and width_diff < self.MAX_WIDTH_DIFF 
                    and height_diff < self.MAX_HEIGHT_DIFF):
                    matched_contours_idx.append(d2['idx'])

            matched_contours_idx.append(d1['idx'])

            if len(matched_contours_idx) < self.MIN_N_MATCHED:
                continue

            matched_result_idx.append(matched_contours_idx)

            unmatched_contour_idx = []
            for d4 in contour_list:
                if d4['idx'] not in matched_contours_idx:
                    unmatched_contour_idx.append(d4['idx'])

            unmatched_contour = np.take(contour_list, unmatched_contour_idx)

            recursive_contour_list = self.find_chars(unmatched_contour)

            for idx in recursive_contour_list:
                matched_result_idx.append(idx)

            break

        return matched_result_idx

    def find_license_plate(self, image):
        thresh = self.preprocess_image(image)
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        contours_dict = []
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            ratio = w / h if h != 0 else 0
            
            if (area > self.MIN_AREA and
                w > self.MIN_WIDTH and h > self.MIN_HEIGHT and
                self.MIN_RATIO < ratio < self.MAX_RATIO):
                contours_dict.append({
                    'contour': contour,
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'cx': x + w/2, 'cy': y + h/2,
                    'idx': i
                })
        
        if not contours_dict:
            return []

        result_idx = self.find_chars(contours_dict)
        matched_result = []
        for idx_list in result_idx:
            matched_result.append(np.take(contours_dict, idx_list))

        plate_imgs = []
        plate_infos = []

        for matched_chars in matched_result:
            sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

            plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx'])/2
            plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy'])/2

            plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * self.PLATE_WIDTH_PADDING

            sum_height = 0
            for d in sorted_chars:
                sum_height += d['h']

            plate_height = int(sum_height / len(sorted_chars) * self.PLATE_HEIGHT_PADDING)

            triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
            triangle_hypotenus = np.linalg.norm(
                np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) - 
                np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
            )

            angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))

            rotation_matrix = cv2.getRotationMatrix2D(
                center=(plate_cx, plate_cy), 
                angle=angle, 
                scale=1.0
            )

            img_rotated = cv2.warpAffine(
                thresh, 
                M=rotation_matrix, 
                dsize=(image.shape[1], image.shape[0])
            )

            img_cropped = cv2.getRectSubPix(
                img_rotated,
                patchSize=(int(plate_width), int(plate_height)),
                center=(int(plate_cx), int(plate_cy))
            )

            plate_imgs.append(img_cropped)
            plate_infos.append({
                'x': int(plate_cx - plate_width / 2),
                'y': int(plate_cy - plate_height / 2),
                'w': int(plate_width),
                'h': int(plate_height)
            })

        return plate_infos

    def recognize_text(self, plate_image):
        plate_image = cv2.resize(plate_image, None, fx=1.6, fy=1.6)
        _, plate_image = cv2.threshold(plate_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(plate_image, lang='kor+eng')
        return text.strip()

class ParkingSystem:
    def __init__(self):
        self.car_detector = CarDetector()
        self.plate_recognizer = LicensePlateRecognizer()
        self.parked_vehicles = {}  # Dictionary to store parked vehicles
        self.total_spots = 10
        self.available_spots = list(range(1, self.total_spots + 1))
        
    def process_frame(self, frame):
        car_boxes = self.car_detector.detect_cars(frame)
        logger.info(f"[진단] Detected {len(car_boxes)} cars in frame.")
        detected_plates = []
        
        for (x1, y1, x2, y2) in car_boxes:
            logger.info(f"[진단] Car box: ({x1},{y1},{x2},{y2})")
            car_roi = frame[y1:y2, x1:x2]
            plate_infos = self.plate_recognizer.find_license_plate(car_roi)
            logger.info(f"[진단]  - Found {len(plate_infos)} plate candidates in car ROI.")
            
            for plate_info in plate_infos:
                plate_x = plate_info['x']
                plate_y = plate_info['y']
                plate_w = plate_info['w']
                plate_h = plate_info['h']
                
                # 번호판 영역이 이미지 경계를 벗어나지 않도록 조정
                plate_x = max(0, min(plate_x, car_roi.shape[1] - plate_w))
                plate_y = max(0, min(plate_y, car_roi.shape[0] - plate_h))
                
                plate_image = car_roi[plate_y:plate_y+plate_h, plate_x:plate_x+plate_w]
                plate_text = self.plate_recognizer.recognize_text(plate_image)
                logger.info(f"[진단]    - Plate candidate text: '{plate_text}'")
                
                if plate_text:
                    detected_plates.append({
                        'plate': plate_text,
                        'box': (x1 + plate_x, y1 + plate_y, x1 + plate_x + plate_w, y1 + plate_y + plate_h)
                    })
                    
                    # Draw plate box
                    cv2.rectangle(frame, 
                                (x1 + plate_x, y1 + plate_y),
                                (x1 + plate_x + plate_w, y1 + plate_y + plate_h),
                                (0, 255, 0), 2)
                    
                    # Show plate text
                    cv2.putText(frame, plate_text,
                              (x1 + plate_x, y1 + plate_y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                              (0, 255, 0), 2)
            
            # Draw car box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        return frame, detected_plates

    def update_parking_status(self, detected_plates):
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Check for new vehicles
        for plate_info in detected_plates:
            plate = plate_info['plate']
            if plate not in self.parked_vehicles and self.available_spots:
                spot = self.available_spots.pop(0)
                self.parked_vehicles[plate] = {
                    'entry_time': current_time,
                    'spot': f'A{spot}'
                }
        
        # Update parking data
        parking_data = {
            'total_spots': self.total_spots,
            'available_spots': len(self.available_spots),
            'occupied_spots': len(self.parked_vehicles),
            'vehicles': [
                {
                    'plate': plate,
                    'entry_time': info['entry_time'],
                    'spot': info['spot']
                }
                for plate, info in self.parked_vehicles.items()
            ]
        }
        
        return parking_data

def start_camera():
    """Start camera stream using libcamera"""
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
    """Read one frame from camera"""
    try:
        while True:
            buf = proc.stdout.read(2)
            if buf[0] == 0xFF and buf[1] == 0xD8:
                break
        
        frame_data = bytearray()
        frame_data.extend(buf)
        
        while True:
            buf = proc.stdout.read(2)
            frame_data.extend(buf)
            if buf[0] == 0xFF and buf[1] == 0xD9:
                break
        
        frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        logger.error(f"Error reading frame: {e}")
        return None

# Global parking system instance
parking_system = ParkingSystem()

def get_parking_data():
    """Get current parking status"""
    return {
        'total_spots': parking_system.total_spots,
        'available_spots': len(parking_system.available_spots),
        'occupied_spots': len(parking_system.parked_vehicles),
        'vehicles': [
            {
                'plate': plate,
                'entry_time': info['entry_time'],
                'spot': info['spot']
            }
            for plate, info in parking_system.parked_vehicles.items()
        ]
    } 