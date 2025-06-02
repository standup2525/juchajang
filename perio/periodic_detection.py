#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Periodic Vehicle License Plate Detection System for Raspberry Pi
This program captures images at regular intervals, detects vehicles using YOLO,
recognizes license plates, and sends data to Flask server via JSON.
"""

import cv2  # OpenCV library for computer vision tasks
import numpy as np  # NumPy library for numerical calculations
import time  # Time library for measuring processing speed and intervals
import logging  # Logging library for recording program events
from datetime import datetime  # DateTime library for timestamps
import pytesseract  # Tesseract OCR library for text recognition
from ultralytics import YOLO  # YOLO library for object detection
import requests  # Requests library for HTTP communication with server
import json  # JSON library for data serialization
import os  # OS library for file operations
import threading  # Threading library for parallel processing
from queue import Queue  # Queue library for data storage

# ================================
# Configuration Parameters (Settings)
# ================================

# Camera Settings - These control how the camera works
CAMERA_INDEX = 0                    # Camera number (0 = first camera)
CAMERA_WIDTH = 1280                 # Camera image width in pixels
CAMERA_HEIGHT = 720                 # Camera image height in pixels
CAMERA_FPS = 30                     # Frames per second (how many pictures per second)

# Detection Interval Settings - These control when detection happens
DETECTION_INTERVAL = 5              # Detection interval in seconds (detect every 5 seconds)
PROCESSING_TIMEOUT = 10             # Maximum time to spend on one detection cycle (seconds)

# YOLO Settings - These control vehicle detection
YOLO_MODEL_PATH = "yolov8n.pt"      # Path to YOLO model file
YOLO_CONFIDENCE = 0.5               # Confidence threshold (0.0 to 1.0, higher = more sure)
MIN_VEHICLE_AREA = 0.15             # Minimum vehicle size (15% of screen area)
MAX_VEHICLE_AREA = 0.8              # Maximum vehicle size (80% of screen area)

# License Plate Recognition Settings - These control how we find text on plates
MIN_AREA = 200                      # Minimum character area in pixels
MIN_WIDTH, MIN_HEIGHT = 8, 16       # Minimum character width and height in pixels
MIN_RATIO, MAX_RATIO = 0.2, 1.5     # Character width/height ratio range (shape limits)
PLATE_WIDTH_PADDING = 1.3           # Extra space around plate width (30% more)
PLATE_HEIGHT_PADDING = 1.5          # Extra space around plate height (50% more)

# Character Matching Settings - These help group characters together
MAX_DIAG_MULTIPLYER = 5             # Maximum distance between characters (diagonal multiplier)
MAX_ANGLE_DIFF = 20.0               # Maximum angle difference between characters (degrees)
MAX_AREA_DIFF = 0.4                 # Maximum area difference between characters (40%)
MAX_WIDTH_DIFF = 0.8                # Maximum width difference between characters (80%)
MAX_HEIGHT_DIFF = 0.3               # Maximum height difference between characters (30%)
MIN_N_MATCHED = 3                   # Minimum number of characters to make a plate

# Direction Detection Settings - These determine entry/exit based on position
SCREEN_CENTER = 0.5                 # Screen center point (50%)
LEFT_ZONE_WIDTH = 0.2               # Left zone width (20% from center)
RIGHT_ZONE_WIDTH = 0.2              # Right zone width (20% from center)
SCREEN_LEFT_BOUNDARY = SCREEN_CENTER - LEFT_ZONE_WIDTH   # Left boundary (30%)
SCREEN_RIGHT_BOUNDARY = SCREEN_CENTER + RIGHT_ZONE_WIDTH # Right boundary (70%)

# Server Communication Settings - These control connection to Flask server
SERVER_URL = "http://localhost:5000/api/vehicle"  # Flask server URL
SERVER_TIMEOUT = 10                 # HTTP request timeout (seconds)
SERVER_RETRY_COUNT = 3              # Number of retry attempts for failed requests
DEVICE_ID = "raspberry_pi_camera_01" # Unique identifier for this camera device

# File Storage Settings - These control where images are saved
IMAGE_SAVE_DIR = "detected_vehicles"  # Directory to save detected vehicle images
IMAGE_QUALITY = 90                    # JPEG quality (0-100, higher = better quality)

# Duplicate Prevention Settings - These prevent sending same vehicle multiple times
DUPLICATE_PREVENTION_TIME = 30      # Seconds to wait before detecting same plate again
RECENT_DETECTIONS_CLEANUP_INTERVAL = 60  # Seconds between cleanup of old detections

# Tesseract OCR Settings - These control text recognition
TESSERACT_CONFIG = '--psm 7 --oem 3'  # OCR configuration for license plates

# Logging Settings - These control program messages
LOG_LEVEL = logging.INFO            # Logging level (DEBUG, INFO, WARNING, ERROR)
LOG_FILE = "periodic_detection.log" # Log file name

# ================================
# Logging Setup
# ================================

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
# Global Variables
# ================================

# System state variables
system_running = True                # Control flag for main loop
detection_lock = threading.Lock()    # Thread lock for shared data
recent_detections = {}               # Store recent detections to prevent duplicates

# ================================
# License Plate Recognition Class
# ================================

class LicensePlateRecognizer:
    """
    License Plate Recognition Class
    This class handles finding and reading license plates from vehicle images.
    Uses computer vision techniques to locate text and OCR to read it.
    """
    
    def __init__(self):
        """
        Initialize the plate recognizer
        This function runs when we create a new LicensePlateRecognizer object
        """
        logger.info("License plate recognizer initialized successfully")
    
    def preprocess_image(self, image):
        """
        Preprocess image for better text detection
        This function prepares the image to make it easier to find text
        
        Args:
            image: Input image (can be color or grayscale)
            
        Returns:
            thresh: Binary image (black and white only) ready for text detection
        """
        # Convert to grayscale (black and white) if image is in color
        if len(image.shape) == 3:  # Check if image has 3 channels (color)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image  # Already grayscale
        
        # Apply Gaussian blur to reduce noise (make image smoother)
        blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
        
        # Apply adaptive threshold to create binary image (black and white only)
        # This makes text stand out from background
        thresh = cv2.adaptiveThreshold(
            blurred,                                    # Input image
            maxValue=255.0,                            # Maximum pixel value (white)
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Method for calculating threshold
            thresholdType=cv2.THRESH_BINARY_INV,       # Invert colors (text becomes white)
            blockSize=19,                              # Size of area for threshold calculation
            C=9                                        # Constant subtracted from mean
        )
        
        return thresh
    
    def find_and_filter_contours(self, thresh_image):
        """
        Find and filter contours (shapes) that might be characters
        This function looks for shapes in the image that could be letters or numbers
        
        Args:
            thresh_image: Binary (black and white) image
            
        Returns:
            possible_contours: List of shapes that might be characters
        """
        # Find all contours (shapes) in the image
        contours, _ = cv2.findContours(
            thresh_image,                    # Input binary image
            mode=cv2.RETR_LIST,             # Retrieve all contours
            method=cv2.CHAIN_APPROX_SIMPLE  # Compress contour by removing redundant points
        )
        
        possible_contours = []  # List to store potential character shapes
        
        # Check each contour to see if it could be a character
        for i, contour in enumerate(contours):
            # Get bounding rectangle (box that fits around the shape)
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h  # Calculate area (size) of the shape
            ratio = w / h if h > 0 else 0  # Calculate width/height ratio (shape proportion)
            
            # Filter contours based on size and shape
            # Keep only shapes that are big enough and have reasonable proportions
            if (area > MIN_AREA and 
                w > MIN_WIDTH and h > MIN_HEIGHT and
                MIN_RATIO < ratio < MAX_RATIO):
                
                # Store information about this potential character
                possible_contours.append({
                    'contour': contour,         # The actual shape data
                    'x': x, 'y': y,            # Top-left corner position
                    'w': w, 'h': h,            # Width and height
                    'cx': x + (w/2),           # Center X coordinate
                    'cy': y + (h/2),           # Center Y coordinate
                    'idx': i,                  # Index number
                    'area': area,              # Area size
                    'ratio': ratio             # Width/height ratio
                })
        
        return possible_contours
    
    def find_char_groups(self, contour_list):
        """
        Find groups of characters that belong together
        This function groups nearby characters that might form a license plate
        
        Args:
            contour_list: List of potential character contours
            
        Returns:
            matched_groups: List of character groups that might be license plates
        """
        # Need at least minimum number of characters to make a group
        if len(contour_list) < MIN_N_MATCHED:
            return []
        
        matched_groups = []    # List to store groups of characters
        used_indices = set()   # Keep track of which characters we've already used
        
        # Try each character as a starting point for a group
        for i, d1 in enumerate(contour_list):
            if i in used_indices:  # Skip if this character is already used
                continue
                
            group = [d1]           # Start new group with this character
            group_indices = {i}    # Track which characters are in this group
            
            # Try to add other characters to this group
            for j, d2 in enumerate(contour_list):
                if j <= i or j in used_indices:  # Skip if same character or already used
                    continue
                
                # Check if these two characters should be grouped together
                # Calculate distance and similarity between characters
                dx = abs(d1['cx'] - d2['cx'])  # Distance in X direction
                dy = abs(d1['cy'] - d2['cy'])  # Distance in Y direction
                
                diagonal_length1 = np.sqrt(d1['w']**2 + d1['h']**2)  # Character size reference
                distance = np.sqrt(dx**2 + dy**2)  # Actual distance between centers
                
                # Only consider characters that are close enough
                if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER:
                    # Calculate similarity metrics
                    angle_diff = np.degrees(np.arctan2(dy, dx + 1e-6))  # Angle between characters
                    area_diff = abs(d1['area'] - d2['area']) / max(d1['area'], d2['area'])  # Size difference
                    width_diff = abs(d1['w'] - d2['w']) / max(d1['w'], d2['w'])  # Width difference
                    height_diff = abs(d1['h'] - d2['h']) / max(d1['h'], d2['h'])  # Height difference
                    
                    # Check if characters are similar enough to be in same plate
                    if (abs(angle_diff) < MAX_ANGLE_DIFF and 
                        area_diff < MAX_AREA_DIFF and
                        width_diff < MAX_WIDTH_DIFF and 
                        height_diff < MAX_HEIGHT_DIFF):
                        
                        group.append(d2)      # Add character to group
                        group_indices.add(j)  # Mark character as used
            
            # Keep groups that have enough characters
            if len(group) >= MIN_N_MATCHED:
                matched_groups.append(group)
                used_indices.update(group_indices)  # Mark all characters in group as used
        
        return matched_groups
    
    def extract_plate_region(self, image, matched_chars):
        """
        Extract the license plate region from image
        This function cuts out the area containing the license plate
        
        Args:
            image: Input image
            matched_chars: List of characters that form a license plate
            
        Returns:
            plate_region: Image containing just the license plate
            plate_info: Dictionary with plate position and size information
        """
        if not matched_chars:  # Return None if no characters provided
            return None, None
        
        # Sort characters from left to right based on center X coordinate
        sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])
        
        # Find the boundaries of all characters (the area they cover)
        min_x = min(d['x'] for d in sorted_chars)              # Leftmost edge
        max_x = max(d['x'] + d['w'] for d in sorted_chars)     # Rightmost edge
        min_y = min(d['y'] for d in sorted_chars)              # Topmost edge
        max_y = max(d['y'] + d['h'] for d in sorted_chars)     # Bottommost edge
        
        # Add padding around the characters to get full plate
        plate_width = int((max_x - min_x) * PLATE_WIDTH_PADDING)
        plate_height = int((max_y - min_y) * PLATE_HEIGHT_PADDING)
        
        # Calculate center point of the plate
        plate_cx = (min_x + max_x) // 2  # Center X
        plate_cy = (min_y + max_y) // 2  # Center Y
        
        # Make sure we don't go outside image boundaries
        h, w = image.shape[:2]  # Get image height and width
        x1 = max(0, plate_cx - plate_width // 2)   # Left edge (not less than 0)
        y1 = max(0, plate_cy - plate_height // 2)  # Top edge (not less than 0)
        x2 = min(w, plate_cx + plate_width // 2)   # Right edge (not more than image width)
        y2 = min(h, plate_cy + plate_height // 2)  # Bottom edge (not more than image height)
        
        # Check if the region is big enough to be useful
        if x2 - x1 < 30 or y2 - y1 < 15:
            return None, None
        
        # Cut out the plate region from the image
        plate_region = image[y1:y2, x1:x2]
        
        # Store plate position and size information
        plate_info = {'x': x1, 'y': y1, 'w': x2-x1, 'h': y2-y1}
        
        return plate_region, plate_info
    
    def recognize_text(self, plate_image):
        """
        Text recognition using OCR (Optical Character Recognition)
        This function reads the text from a license plate image
        
        Args:
            plate_image: Image containing the license plate
            
        Returns:
            result: String containing recognized text, or None if no text found
        """
        # Check if image is valid
        if plate_image is None or plate_image.size == 0:
            return None
        
        try:
            # Resize image for better OCR results
            plate_image = cv2.resize(plate_image, dsize=(0, 0), fx=1.6, fy=1.6)
            
            # Convert to grayscale if image is in color
            if len(plate_image.shape) == 3:
                plate_image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            
            # Apply binary threshold to make text clearer
            _, plate_image = cv2.threshold(
                plate_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
            )
            
            # Find contours to refine plate boundaries
            contours, _ = cv2.findContours(
                plate_image, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Find actual text boundaries
            plate_min_x, plate_min_y = plate_image.shape[1], plate_image.shape[0]
            plate_max_x, plate_max_y = 0, 0
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                ratio = w / h if h > 0 else 0
                
                if (area > MIN_AREA and w > MIN_WIDTH and h > MIN_HEIGHT and
                    MIN_RATIO < ratio < MAX_RATIO):
                    plate_min_x = min(plate_min_x, x)
                    plate_min_y = min(plate_min_y, y)
                    plate_max_x = max(plate_max_x, x + w)
                    plate_max_y = max(plate_max_y, y + h)
            
            # Extract refined text region
            if plate_max_x > plate_min_x and plate_max_y > plate_min_y:
                img_result = plate_image[plate_min_y:plate_max_y, plate_min_x:plate_max_x]
            else:
                img_result = plate_image
            
            # Additional preprocessing for better OCR
            img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0)
            _, img_result = cv2.threshold(
                img_result, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
            )
            
            # Add padding around text for better OCR results
            img_result = cv2.copyMakeBorder(
                img_result, top=10, bottom=10, left=10, right=10, 
                borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
            
            # Perform OCR (text recognition)
            text = pytesseract.image_to_string(
                img_result, lang='kor', config=TESSERACT_CONFIG
            ).strip()  # Remove extra spaces
            
            # Clean up the result - keep only Korean characters and numbers
            result = ''
            has_digit = False
            for char in text:
                if '가' <= char <= '힣' or char.isdigit():  # Korean characters or digits
                    if char.isdigit():
                        has_digit = True
                    result += char
            
            # Return result only if we found enough characters including at least one digit
            return result if has_digit and len(result) >= 4 else None
            
        except Exception as e:
            logger.debug(f"OCR error: {e}")
            return None
    
    def recognize_plate(self, vehicle_image):
        """
        Main license plate recognition function
        This function coordinates all the steps to find and read a license plate
        
        Args:
            vehicle_image: Image of a vehicle
            
        Returns:
            plate_text: Recognized license plate text
            plate_info: Position and size information of the plate
        """
        try:
            start_time = time.time()  # Record start time for performance monitoring
            
            # Step 1: Preprocess image to make text easier to find
            thresh_image = self.preprocess_image(vehicle_image)
            
            # Step 2: Find potential character shapes
            possible_contours = self.find_and_filter_contours(thresh_image)
            
            # Step 3: Check if we have enough potential characters
            if len(possible_contours) < MIN_N_MATCHED:
                return None, None
            
            # Step 4: Group characters that might belong to the same plate
            char_groups = self.find_char_groups(possible_contours)
            
            # Step 5: Check if we found any valid character groups
            if not char_groups:
                return None, None
            
            # Step 6: Process each potential license plate
            best_plate_text = None
            best_plate_info = None
            longest_text_length = 0
            
            for matched_chars in char_groups:
                # Check processing time limit
                if time.time() - start_time > PROCESSING_TIMEOUT:
                    logger.warning("Processing timeout reached")
                    break
                
                try:
                    # Extract the license plate region from the image
                    plate_region, plate_info = self.extract_plate_region(thresh_image, matched_chars)
                    
                    # Check if we successfully extracted a plate region
                    if plate_region is None:
                        continue
                    
                    # Recognize text using OCR
                    plate_text = self.recognize_text(plate_region)
                    
                    # Keep the longest valid text found
                    if plate_text and len(plate_text) > longest_text_length:
                        best_plate_text = plate_text
                        best_plate_info = plate_info
                        longest_text_length = len(plate_text)
                        
                except Exception as e:
                    logger.debug(f"Error processing plate candidate: {e}")
                    continue
            
            processing_time = time.time() - start_time
            if best_plate_text:
                logger.info(f"License plate recognized: {best_plate_text} (processing time: {processing_time:.2f}s)")
            
            return best_plate_text, best_plate_info
            
        except Exception as e:
            logger.error(f"License plate recognition error: {e}")
            return None, None

# ================================
# Vehicle Detection Class
# ================================

class VehicleDetector:
    """
    Vehicle Detection Class
    This class uses YOLO (You Only Look Once) AI model to detect vehicles in images.
    YOLO is a fast and accurate object detection algorithm.
    """
    
    def __init__(self):
        """
        Initialize the vehicle detector
        This function loads the YOLO model and prepares it for use
        """
        logger.info("Loading YOLOv8 model...")
        try:
            self.model = YOLO(YOLO_MODEL_PATH)  # Load the YOLO model from file
            
            # Warm up the model by running it once on a dummy image
            # This makes the first real detection faster
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)  # Create fake image
            self.model(dummy_image, verbose=False)  # Run model on fake image
            logger.info("YOLOv8 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def detect_vehicles(self, image):
        """
        Detect vehicles in an image using YOLO
        This function finds all vehicles and returns their positions
        
        Args:
            image: Input image to search for vehicles
            
        Returns:
            vehicles: List of detected vehicles with their positions and information
        """
        try:
            # Run YOLO detection on the image
            results = self.model(image, conf=YOLO_CONFIDENCE, verbose=False)
            
            vehicles = []  # List to store detected vehicles
            height, width = image.shape[:2]  # Get image dimensions
            
            # Process each detection result
            for result in results:
                boxes = result.boxes  # Get bounding boxes of detected objects
                if boxes is not None:
                    # Check each detected object
                    for box in boxes:
                        class_id = int(box.cls[0])  # Get object class ID
                        
                        # Filter for vehicle classes only
                        # COCO dataset classes: car=2, motorcycle=3, bus=5, truck=7
                        if class_id in [2, 3, 5, 7]:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf[0])  # Get detection confidence
                            
                            # Calculate vehicle area as percentage of image
                            vehicle_area = (x2 - x1) * (y2 - y1)  # Vehicle area in pixels
                            image_area = width * height            # Total image area
                            area_ratio = vehicle_area / image_area # Percentage of image
                            
                            # Only keep vehicles that are within size limits
                            if MIN_VEHICLE_AREA <= area_ratio <= MAX_VEHICLE_AREA:
                                # Calculate center position for direction detection
                                center_x = (x1 + x2) / 2
                                center_x_normalized = center_x / width
                                
                                vehicles.append({
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],  # Bounding box coordinates
                                    'confidence': confidence,                        # How sure YOLO is
                                    'area_ratio': area_ratio,                       # Size as percentage of image
                                    'center_x': center_x_normalized,                # Normalized center X position
                                    'class_id': class_id,                           # Object type ID
                                    'class_name': self.get_class_name(class_id)     # Object type name
                                })
            
            return vehicles
            
        except Exception as e:
            logger.error(f"Vehicle detection error: {e}")
            return []
    
    def get_class_name(self, class_id):
        """
        Convert class ID number to readable name
        This function translates YOLO's number codes into words
        
        Args:
            class_id: Numeric class identifier from YOLO
            
        Returns:
            class_name: Human-readable name of the object type
        """
        class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        return class_names.get(class_id, 'vehicle')

# ================================
# Server Communication Class
# ================================

class ServerCommunicator:
    """
    Server Communication Class
    This class handles sending vehicle detection data to the Flask server via HTTP requests
    """
    
    def __init__(self):
        """
        Initialize the server communicator
        This function sets up HTTP session for server communication
        """
        self.session = requests.Session()  # Create HTTP session for connection reuse
        self.session.timeout = SERVER_TIMEOUT  # Set timeout for requests
        logger.info(f"Server communicator initialized. Target URL: {SERVER_URL}")
    
    def send_vehicle_data(self, plate_number, direction, timestamp, confidence=None, image_path=None):
        """
        Send vehicle detection data to Flask server
        This function sends license plate information via HTTP POST request
        
        Args:
            plate_number: Detected license plate text
            direction: Vehicle direction ("entry" or "exit")
            timestamp: When the detection occurred (ISO format string)
            confidence: Detection confidence level (optional)
            image_path: Path to saved vehicle image (optional)
            
        Returns:
            success: True if data was sent successfully, False otherwise
        """
        # Prepare form data for HTTP request
        data = {
            'plate_number': plate_number,
            'direction': direction,
            'timestamp': timestamp,
            'device_id': DEVICE_ID
        }
        
        # Add confidence if provided
        if confidence is not None:
            data['confidence'] = confidence
        
        # Prepare image file if provided
        files = None
        if image_path and os.path.exists(image_path):
            try:
                files = {'image': open(image_path, 'rb')}
            except Exception as e:
                logger.warning(f"Failed to open image file: {e}")
        
        # Try sending data with retry mechanism
        for attempt in range(SERVER_RETRY_COUNT):
            try:
                # Send HTTP POST request to Flask server
                response = self.session.post(
                    SERVER_URL,
                    data=data,
                    files=files,
                    timeout=SERVER_TIMEOUT
                )
                
                # Close image file if it was opened
                if files and 'image' in files:
                    files['image'].close()
                
                # Check response status
                if response.status_code == 200:
                    try:
                        result = response.json()  # Parse JSON response
                        if result.get('success'):
                            logger.info(f"Vehicle data sent successfully: {plate_number} ({direction})")
                            return True
                        else:
                            logger.warning(f"Server processing error: {result.get('error', 'Unknown error')}")
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON response from server: {response.text}")
                else:
                    logger.warning(f"Server response error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.Timeout:
                logger.error(f"Server request timeout (attempt {attempt + 1}/{SERVER_RETRY_COUNT})")
            except requests.exceptions.ConnectionError:
                logger.error(f"Server connection error (attempt {attempt + 1}/{SERVER_RETRY_COUNT})")
            except Exception as e:
                logger.error(f"Unexpected error sending data (attempt {attempt + 1}/{SERVER_RETRY_COUNT}): {e}")
            
            # Wait before retry (except on last attempt)
            if attempt < SERVER_RETRY_COUNT - 1:
                time.sleep(2 ** attempt)  # Exponential backoff: 2, 4, 8 seconds
        
        logger.error(f"Failed to send vehicle data after {SERVER_RETRY_COUNT} attempts")
        return False

# ================================
# Utility Functions
# ================================

def determine_direction(center_x):
    """
    Determine vehicle direction based on position on screen
    This function analyzes where the vehicle is positioned to determine entry/exit
    
    Args:
        center_x: Normalized X position of vehicle center (0.0 to 1.0)
        
    Returns:
        direction: "entry", "exit", or "unknown"
    """
    if center_x < SCREEN_LEFT_BOUNDARY:
        return 'entry'  # Vehicle on left side = entering
    elif center_x > SCREEN_RIGHT_BOUNDARY:
        return 'exit'   # Vehicle on right side = exiting
    else:
        return 'unknown'  # Vehicle in center = direction uncertain

def is_duplicate_detection(plate_number):
    """
    Check if this plate was recently detected to prevent duplicates
    This function prevents sending the same vehicle multiple times
    
    Args:
        plate_number: License plate text to check
        
    Returns:
        is_duplicate: True if this plate was recently detected, False otherwise
    """
    current_time = time.time()
    
    with detection_lock:
        # Check if plate was recently detected
        if plate_number in recent_detections:
            last_detection_time = recent_detections[plate_number]
            if current_time - last_detection_time < DUPLICATE_PREVENTION_TIME:
                return True  # This is a duplicate
        
        # Record this detection
        recent_detections[plate_number] = current_time
        
        # Clean up old detections to save memory
        expired_plates = [
            plate for plate, detection_time in recent_detections.items()
            if current_time - detection_time > DUPLICATE_PREVENTION_TIME
        ]
        for plate in expired_plates:
            del recent_detections[plate]
    
    return False  # This is not a duplicate

def save_vehicle_image(image, plate_number, direction):
    """
    Save detected vehicle image to disk
    This function stores the vehicle image for record keeping
    
    Args:
        image: Vehicle image to save
        plate_number: License plate text for filename
        direction: Vehicle direction for filename
        
    Returns:
        image_path: Path to saved image file, or None if save failed
    """
    try:
        # Create save directory if it doesn't exist
        os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
        
        # Create safe filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_plate_number = ''.join(c for c in plate_number if c.isalnum())
        filename = f"{safe_plate_number}_{direction}_{timestamp}.jpg"
        image_path = os.path.join(IMAGE_SAVE_DIR, filename)
        
        # Save image with specified quality
        cv2.imwrite(image_path, image, [cv2.IMWRITE_JPEG_QUALITY, IMAGE_QUALITY])
        logger.debug(f"Vehicle image saved: {image_path}")
        return image_path
        
    except Exception as e:
        logger.error(f"Failed to save vehicle image: {e}")
        return None

# ================================
# Main Detection System Class
# ================================

class PeriodicDetectionSystem:
    """
    Periodic Detection System
    This is the main class that coordinates everything:
    - Captures images from camera at regular intervals
    - Detects vehicles using YOLO
    - Recognizes license plates
    - Sends data to Flask server
    """
    
    def __init__(self):
        """
        Initialize the detection system
        This function sets up the camera, AI models, and server communication
        """
        logger.info("Initializing periodic detection system...")
        
        # Initialize AI components
        self.vehicle_detector = VehicleDetector()      # Vehicle detection AI
        self.plate_recognizer = LicensePlateRecognizer()  # License plate recognition AI
        self.server_communicator = ServerCommunicator()   # Server communication handler
        
        # Initialize camera
        self.camera = cv2.VideoCapture(CAMERA_INDEX)  # Connect to camera
        
        # Set camera properties for best performance
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)   # Set image width
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT) # Set image height
        self.camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)             # Set frame rate
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)               # Minimize delay
        
        # Check if camera opened successfully
        if not self.camera.isOpened():
            raise Exception("Cannot open camera!")
        
        # Get actual camera settings (might be different from requested)
        actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Camera settings: {actual_width}x{actual_height}")
        
        # Initialize counters
        self.detection_count = 0  # Count total detections performed
        
        logger.info("Periodic detection system initialized successfully")
    
    def capture_frame(self):
        """
        Capture a single frame from camera
        This function gets one image from the camera for processing
        
        Returns:
            frame: Captured image, or None if capture failed
        """
        try:
            # Read frame from camera
            ret, frame = self.camera.read()
            if not ret:
                logger.error("Failed to capture frame from camera")
                return None
            
            return frame
            
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return None
    
    def process_detection_cycle(self):
        """
        Process one complete detection cycle
        This function handles one round of vehicle detection and plate recognition
        
        Returns:
            success: True if cycle completed successfully, False otherwise
        """
        try:
            cycle_start_time = time.time()
            self.detection_count += 1
            
            logger.info(f"Starting detection cycle #{self.detection_count}")
            
            # Step 1: Capture frame from camera
            frame = self.capture_frame()
            if frame is None:
                return False
            
            # Step 2: Detect vehicles using YOLO
            vehicles = self.vehicle_detector.detect_vehicles(frame)
            
            if not vehicles:
                logger.debug("No vehicles detected in this cycle")
                return True  # No vehicles is not an error
            
            logger.info(f"Detected {len(vehicles)} vehicles")
            
            # Step 3: Process each detected vehicle
            for i, vehicle in enumerate(vehicles):
                try:
                    # Check processing time limit
                    if time.time() - cycle_start_time > PROCESSING_TIMEOUT:
                        logger.warning("Detection cycle timeout reached")
                        break
                    
                    # Extract vehicle information
                    x1, y1, x2, y2 = vehicle['bbox']
                    confidence = vehicle['confidence']
                    center_x = vehicle['center_x']
                    class_name = vehicle['class_name']
                    
                    logger.debug(f"Processing vehicle {i+1}: {class_name} at position {center_x:.2f} with confidence {confidence:.2f}")
                    
                    # Extract vehicle image for plate recognition
                    vehicle_image = frame[y1:y2, x1:x2]
                    if vehicle_image.size == 0:
                        logger.warning(f"Empty vehicle image for vehicle {i+1}")
                        continue
                    
                    # Step 4: Recognize license plate
                    plate_text, plate_info = self.plate_recognizer.recognize_plate(vehicle_image)
                    
                    if not plate_text:
                        logger.debug(f"No license plate detected in vehicle {i+1}")
                        continue
                    
                    logger.info(f"License plate recognized: {plate_text}")
                    
                    # Step 5: Check for duplicate detection
                    if is_duplicate_detection(plate_text):
                        logger.info(f"Duplicate detection ignored: {plate_text}")
                        continue
                    
                    # Step 6: Determine vehicle direction
                    direction = determine_direction(center_x)
                    
                    if direction == 'unknown':
                        logger.info(f"Direction uncertain for {plate_text} (center position: {center_x:.2f})")
                        continue
                    
                    logger.info(f"Vehicle direction determined: {plate_text} -> {direction}")
                    
                    # Step 7: Save vehicle image
                    image_path = save_vehicle_image(frame, plate_text, direction)
                    
                    # Step 8: Send data to server
                    timestamp = datetime.now().isoformat()
                    success = self.server_communicator.send_vehicle_data(
                        plate_text, direction, timestamp, confidence, image_path
                    )
                    
                    if success:
                        logger.info(f"Vehicle data sent successfully: {plate_text} ({direction})")
                    else:
                        logger.error(f"Failed to send vehicle data: {plate_text}")
                
                except Exception as e:
                    logger.error(f"Error processing vehicle {i+1}: {e}")
                    continue
            
            cycle_time = time.time() - cycle_start_time
            logger.info(f"Detection cycle #{self.detection_count} completed in {cycle_time:.2f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"Error in detection cycle: {e}")
            return False
    
    def run(self):
        """
        Main execution loop
        This function runs continuously, performing detection at regular intervals
        """
        logger.info("Starting periodic detection system")
        logger.info(f"Detection interval: {DETECTION_INTERVAL} seconds")
        logger.info(f"Server URL: {SERVER_URL}")
        logger.info("Press Ctrl+C to stop")
        
        try:
            while system_running:
                # Perform one detection cycle
                self.process_detection_cycle()
                
                # Wait for next detection interval
                logger.debug(f"Waiting {DETECTION_INTERVAL} seconds until next detection...")
                time.sleep(DETECTION_INTERVAL)
                
        except KeyboardInterrupt:
            logger.info("User interrupted (Ctrl+C)")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """
        Clean up system resources
        This function properly closes camera and sessions when program ends
        """
        logger.info("Cleaning up system...")
        global system_running
        system_running = False
        
        # Release camera resource
        if hasattr(self, 'camera'):
            self.camera.release()
        
        # Close server session
        if hasattr(self, 'server_communicator') and hasattr(self.server_communicator, 'session'):
            self.server_communicator.session.close()
        
        logger.info("System cleanup completed")

# ================================
# Configuration Display Function
# ================================

def print_system_config():
    """
    Print system configuration information
    This function displays all the important settings for reference
    """
    print("=" * 80)
    print("Periodic Vehicle License Plate Detection System Configuration")
    print("=" * 80)
    print(f"Camera Settings:")
    print(f"  - Resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    print(f"  - FPS: {CAMERA_FPS}")
    print()
    print(f"Detection Settings:")
    print(f"  - Detection interval: {DETECTION_INTERVAL} seconds")
    print(f"  - Processing timeout: {PROCESSING_TIMEOUT} seconds")
    print(f"  - Vehicle size range: {MIN_VEHICLE_AREA*100:.1f}% - {MAX_VEHICLE_AREA*100:.1f}%")
    print()
    print(f"YOLO Settings:")
    print(f"  - Model: {YOLO_MODEL_PATH}")
    print(f"  - Confidence threshold: {YOLO_CONFIDENCE}")
    print()
    print(f"Direction Detection:")
    print(f"  - Left zone (entry): 0% - {SCREEN_LEFT_BOUNDARY*100:.1f}%")
    print(f"  - Center zone (unknown): {SCREEN_LEFT_BOUNDARY*100:.1f}% - {SCREEN_RIGHT_BOUNDARY*100:.1f}%")
    print(f"  - Right zone (exit): {SCREEN_RIGHT_BOUNDARY*100:.1f}% - 100%")
    print()
    print(f"Server Communication:")
    print(f"  - URL: {SERVER_URL}")
    print(f"  - Timeout: {SERVER_TIMEOUT} seconds")
    print(f"  - Retry count: {SERVER_RETRY_COUNT}")
    print(f"  - Device ID: {DEVICE_ID}")
    print()
    print(f"File Storage:")
    print(f"  - Image directory: {IMAGE_SAVE_DIR}")
    print(f"  - Image quality: {IMAGE_QUALITY}%")
    print()
    print(f"Duplicate Prevention:")
    print(f"  - Prevention time: {DUPLICATE_PREVENTION_TIME} seconds")
    print("=" * 80)

# ================================
# Main Execution Section
# ================================

def main():
    """
    Main function - entry point of the program
    This function starts everything and handles any errors
    """
    # Print configuration information
    print_system_config()
    
    try:
        # Create and run the detection system
        system = PeriodicDetectionSystem()
        system.run()
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        return 1  # Return error code
    
    return 0  # Return success code

# Program entry point - this runs when you execute the file
if __name__ == "__main__":
    exit(main())  # Run main function and exit with its return code 