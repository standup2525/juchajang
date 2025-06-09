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

# Setup logging - save messages to file and show on screen
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/car_detection.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Camera settings - size and speed of video capture
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 360
CAMERA_FPS = 30

class CarDetector:
    def __init__(self):
        # Start OpenVINO - for faster AI processing
        self.core = Core()
        # Check if AIHAT is connected
        self.device = "MYRIAD" if "MYRIAD" in self.core.available_devices else "CPU"
        logger.info(f"Using device: {self.device}")
        
        # Load YOLO model and convert for OpenVINO
        self.model = YOLO('models/yolov8n.pt')
        self.ov_model = self.model.export(format="openvino")
        self.compiled_model = self.core.compile_model(self.ov_model, self.device)
        
        # Car class ID from COCO dataset
        self.car_class_id = 2  # 2 means car in COCO dataset
        
    def detect_cars(self, frame):
        # Use OpenVINO to find cars in the image
        results = self.model(frame, device=self.device)
        car_boxes = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Only process if it's a car
                if int(box.cls) == self.car_class_id:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    car_boxes.append((x1, y1, x2, y2))
        
        return car_boxes

class LicensePlateRecognizer:
    def __init__(self):
        # Settings for finding license plates
        self.MIN_AREA = 210
        self.MIN_WIDTH, self.MIN_HEIGHT = 8, 16
        self.MIN_RATIO, self.MAX_RATIO = 0.4, 1.0
        
    def preprocess_image(self, image):
        # Convert image to black and white
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Make image less noisy
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Make text more clear
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
        # Make image ready for plate detection
        thresh = self.preprocess_image(image)
        
        # Find shapes in the image
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Look for shapes that could be license plates
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
        # Make image bigger for better text reading
        plate_image = cv2.resize(plate_image, None, fx=1.6, fy=1.6)
        # Make text more clear
        _, plate_image = cv2.threshold(plate_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Read text from image
        text = pytesseract.image_to_string(plate_image, lang='kor+eng')
        return text.strip()

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
        # Find start of JPEG image
        while True:
            buf = proc.stdout.read(2)
            if buf[0] == 0xFF and buf[1] == 0xD8:
                break
        
        # Read JPEG data
        frame_data = bytearray()
        frame_data.extend(buf)
        
        while True:
            buf = proc.stdout.read(2)
            frame_data.extend(buf)
            if buf[0] == 0xFF and buf[1] == 0xD9:
                break
        
        # Convert JPEG data to image
        frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        logger.error(f"Error reading frame: {e}")
        return None

def main():
    try:
        # Start camera
        logger.info("Starting camera...")
        proc = start_camera()
        
        # Create objects for car and plate detection
        car_detector = CarDetector()
        plate_recognizer = LicensePlateRecognizer()
        
        logger.info("Starting detection system...")
        logger.info("Press 'q' to quit")
        
        while True:
            # Read frame from camera
            frame = read_frame(proc)
            if frame is None:
                continue
                
            # Find cars in frame
            car_boxes = car_detector.detect_cars(frame)
            
            # Try to find license plates for each car
            for (x1, y1, x2, y2) in car_boxes:
                # Get image of car
                car_roi = frame[y1:y2, x1:x2]
                
                # Look for license plate
                possible_plates = plate_recognizer.find_license_plate(car_roi)
                
                for plate in possible_plates:
                    # Get plate area
                    plate_x = plate['x']
                    plate_y = plate['y']
                    plate_w = plate['w']
                    plate_h = plate['h']
                    
                    # Get plate image
                    plate_image = car_roi[plate_y:plate_y+plate_h, plate_x:plate_x+plate_w]
                    
                    # Read plate text
                    plate_text = plate_recognizer.recognize_text(plate_image)
                    
                    if plate_text:
                        # Draw box around plate
                        cv2.rectangle(frame, 
                                    (x1 + plate_x, y1 + plate_y),
                                    (x1 + plate_x + plate_w, y1 + plate_y + plate_h),
                                    (0, 255, 0), 2)
                        
                        # Show plate text
                        cv2.putText(frame, plate_text,
                                  (x1 + plate_x, y1 + plate_y - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                  (0, 255, 0), 2)
                
                # Draw box around car
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Show result
            cv2.imshow('Car Detection and License Plate Recognition', frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
    finally:
        # Clean up
        if 'proc' in locals():
            proc.terminate()
        cv2.destroyAllWindows()
        logger.info("System stopped")

if __name__ == "__main__":
    main() 