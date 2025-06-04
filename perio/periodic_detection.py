#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Periodic Vehicle License Plate Detection System
This system periodically captures camera frames, detects vehicles using YOLO,
recognizes license plates, and sends data to the Flask server.
"""

import cv2
import numpy as np
import time
import requests
import logging
import os
from datetime import datetime
from ultralytics import YOLO
import threading
from queue import Queue
import base64
import subprocess
import pytesseract

# ================================
# Configuration
# ================================

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 360
CAMERA_FPS = 30

# Detection settings
DETECTION_INTERVAL = 5  # seconds
MIN_VEHICLE_AREA = 0.15  # minimum vehicle area as percentage of frame
MAX_VEHICLE_AREA = 0.8   # maximum vehicle area as percentage of frame
YOLO_CONFIDENCE = 0.5    # minimum confidence for YOLO detection
PLATE_CONFIDENCE = 0.7   # minimum confidence for plate recognition

# Direction detection settings
SCREEN_CENTER = 0.5      # center point of screen (50%)
LEFT_ZONE_WIDTH = 0.2    # width of left zone (20%)
RIGHT_ZONE_WIDTH = 0.2   # width of right zone (20%)

# Server settings
SERVER_URL = "http://localhost:5000/api/detections"
SERVER_TIMEOUT = 10      # seconds
DUPLICATE_PREVENTION_TIME = 30  # seconds

# Streaming settings
STREAM_QUEUE_SIZE = 10
STREAM_FPS = 15

# ================================
# Logging Setup
# ================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('periodic_detection.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================================
# Global Variables
# ================================

# Initialize YOLO model
model = YOLO('yolov8n.pt')

# Store last detection time for each plate
last_detections = {}

# Frame queue for streaming
frame_queue = Queue(maxsize=STREAM_QUEUE_SIZE)

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
    try:
        results = model(frame, conf=YOLO_CONFIDENCE)
        vehicles = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Check if detected object is a vehicle
                if box.cls in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    vehicles.append((x1, y1, x2, y2))
        
        return vehicles
    except Exception as e:
        logger.error(f"Error in vehicle detection: {e}")
        return []

def recognize_plate(frame, bbox):
    """
    Recognize license plate in the given bounding box
    Args:
        frame: Input frame
        bbox: Bounding box (x1, y1, x2, y2)
    Returns:
        str: Recognized plate number or None
    """
    try:
        x1, y1, x2, y2 = bbox
        plate_region = frame[y1:y2, x1:x2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Recognize text
        plate_text = pytesseract.image_to_string(thresh, lang='kor+eng')
        
        # Clean and validate plate number
        plate_text = ''.join(c for c in plate_text if c.isalnum() or c in '가나다라마바사아자차카타파하')
        
        if len(plate_text) >= 7:  # Minimum length for Korean plates
            return plate_text
        return None
    except Exception as e:
        logger.error(f"Error in plate recognition: {e}")
        return None

def determine_direction(bbox, frame_width):
    """
    Determine vehicle direction based on position
    Args:
        bbox: Bounding box (x1, y1, x2, y2)
        frame_width: Width of the frame
    Returns:
        str: 'in' or 'out'
    """
    x1, _, x2, _ = bbox
    center_x = (x1 + x2) / 2
    return 'in' if center_x < frame_width / 2 else 'out'

def send_to_server(plate_number, direction, frame):
    """
    Send detection data to server
    Args:
        plate_number: Detected plate number
        direction: Vehicle direction
        frame: Captured frame
    """
    try:
        # Check for duplicate detection
        current_time = time.time()
        if plate_number in last_detections:
            if current_time - last_detections[plate_number] < DUPLICATE_PREVENTION_TIME:
                logger.info(f"Skipping duplicate detection for {plate_number}")
                return
        
        # Update last detection time
        last_detections[plate_number] = current_time
        
        # Convert frame to base64
        _, buffer = cv2.imencode('.jpg', frame)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Prepare data
        data = {
            'plate_number': plate_number,
            'direction': direction,
            'image': image_base64
        }
        
        # Send to server
        response = requests.post(SERVER_URL, json=data)
        response.raise_for_status()
        
        logger.info(f"Data sent successfully: {plate_number}")
    except Exception as e:
        logger.error(f"Error sending data to server: {e}")

def stream_frame(frame_queue):
    """
    Stream frames to web interface
    Args:
        frame_queue: Queue for frames
    """
    while True:
        try:
            if not frame_queue.empty():
                frame = frame_queue.get()
                _, buffer = cv2.imencode('.jpg', frame)
                frame_queue.task_done()
                time.sleep(1/STREAM_FPS)
        except Exception as e:
            logger.error(f"Error in stream_frame: {e}")
            time.sleep(1)

# ================================
# Main Detection Loop
# ================================

def main():
    """
    Main detection loop
    """
    try:
        # Initialize camera
        proc = start_stream()
        buf = b''
        
        # Initialize YOLO model
        logger.info("Loading YOLO model...")
        model = YOLO('yolov8n.pt')
        logger.info("YOLO model loaded successfully")
        
        # Initialize frame queue for streaming
        frame_queue = Queue(maxsize=STREAM_QUEUE_SIZE)
        
        # Start streaming thread
        stream_thread = threading.Thread(target=stream_frame, args=(frame_queue,), daemon=True)
        stream_thread.start()
        
        logger.info("Starting detection system...")
        logger.info("Press 'q' to quit")
        
        while True:
            # Get frame
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
                
                # Detect vehicles
                vehicles = detect_vehicles(frame, model)
                
                # Process each detected vehicle
                for bbox in vehicles:
                    # Recognize plate
                    plate_number = recognize_plate(frame, bbox)
                    
                    if plate_number:
                        # Determine direction
                        direction = determine_direction(bbox, frame.shape[1])
                        
                        # Draw bounding box
                        x1, y1, x2, y2 = bbox
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Add plate number and direction
                        cv2.putText(frame, f"{plate_number} ({direction})", 
                                  (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
                        # Send to server
                        send_to_server(plate_number, direction, frame)
                
                # Update stream queue
                if not frame_queue.full():
                    frame_queue.put(frame.copy())
                
                # Show frame
                cv2.imshow("Vehicle Detection", frame)
                
                # Check for exit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
    except KeyboardInterrupt:
        logger.info("Stopping detection system")
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
    finally:
        cv2.destroyAllWindows()
        proc.terminate()

if __name__ == "__main__":
    main() 