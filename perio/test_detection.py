#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for vehicle and license plate detection
This script tests the core detection functionality without server communication
"""

import cv2
import numpy as np
import time
import logging
from ultralytics import YOLO
import pytesseract

# ================================
# Configuration
# ================================

# Camera settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# Detection settings
YOLO_CONFIDENCE = 0.5    # minimum confidence for YOLO detection

# ================================
# Logging Setup
# ================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================================
# Helper Functions
# ================================

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

def main():
    """
    Main detection loop
    """
    # Initialize camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        logger.error("Failed to open camera")
        return
    
    # Initialize YOLO model
    logger.info("Loading YOLO model...")
    model = YOLO('yolov8n.pt')
    logger.info("YOLO model loaded successfully")
    
    # Create window
    cv2.namedWindow("Vehicle Detection", cv2.WINDOW_NORMAL)
    
    logger.info("Starting detection system...")
    logger.info("Press 'q' to quit")
    
    try:
        while True:
            # Get frame
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to capture frame")
                time.sleep(1)
                continue
            
            # Detect vehicles
            vehicles = detect_vehicles(frame, model)
            
            # Process each detected vehicle
            for bbox in vehicles:
                # Recognize plate
                plate_number = recognize_plate(frame, bbox)
                
                # Draw bounding box
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add plate number if recognized
                if plate_number:
                    cv2.putText(frame, plate_number, 
                              (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
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
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 