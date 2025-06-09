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
        # Initialize OpenVINO
        self.core = Core()
        self.device = "MYRIAD" if "MYRIAD" in self.core.available_devices else "CPU"
        logger.info(f"Using device: {self.device}")
        
        # Load YOLO model
        self.model = YOLO('models/yolov8n.pt')
        self.ov_model = self.model.export(format="openvino")
        self.compiled_model = self.core.compile_model(self.ov_model, self.device)
        
        self.car_class_id = 2  # Car class ID in COCO dataset
        
    def detect_cars(self, frame):
        results = self.model(frame, device=self.device)
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
        self.MIN_AREA = 210
        self.MIN_WIDTH, self.MIN_HEIGHT = 8, 16
        self.MIN_RATIO, self.MAX_RATIO = 0.4, 1.0
        
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

    def find_license_plate(self, image):
        thresh = self.preprocess_image(image)
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
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
        detected_plates = []
        
        for (x1, y1, x2, y2) in car_boxes:
            car_roi = frame[y1:y2, x1:x2]
            possible_plates = self.plate_recognizer.find_license_plate(car_roi)
            
            for plate in possible_plates:
                plate_x = plate['x']
                plate_y = plate['y']
                plate_w = plate['w']
                plate_h = plate['h']
                
                plate_image = car_roi[plate_y:plate_y+plate_h, plate_x:plate_x+plate_w]
                plate_text = self.plate_recognizer.recognize_text(plate_image)
                
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