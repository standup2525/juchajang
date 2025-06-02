#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time Vehicle License Plate Recognition System for Raspberry Pi
This program captures video from camera, detects vehicles using YOLO,
recognizes license plates, and displays results on screen in real-time.
"""

import cv2  # OpenCV library for computer vision tasks
import numpy as np  # NumPy library for numerical calculations
import time  # Time library for measuring processing speed
import logging  # Logging library for recording program events
from datetime import datetime  # DateTime library for timestamps
import pytesseract  # Tesseract OCR library for text recognition
from ultralytics import YOLO  # YOLO library for object detection
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

# YOLO Settings - These control vehicle detection
YOLO_MODEL_PATH = "yolov8n.pt"      # Path to YOLO model file
YOLO_CONFIDENCE = 0.5               # Confidence threshold (0.0 to 1.0, higher = more sure)
CAR_DETECTION_AREA = 0.1            # Minimum vehicle size (10% of screen, low for real-time)

# License Plate Recognition Settings - These control how we find text on plates
MIN_AREA = 150                      # Minimum character area in pixels (adjusted for real-time)
MIN_WIDTH, MIN_HEIGHT = 6, 12       # Minimum character width and height in pixels
MIN_RATIO, MAX_RATIO = 0.3, 1.2     # Character width/height ratio range (shape limits)
PLATE_WIDTH_PADDING = 1.2           # Extra space around plate width (20% more)
PLATE_HEIGHT_PADDING = 1.4          # Extra space around plate height (40% more)

# Character Matching Settings - These help group characters together
MAX_DIAG_MULTIPLYER = 5             # Maximum distance between characters (diagonal multiplier)
MAX_ANGLE_DIFF = 15.0               # Maximum angle difference between characters (degrees)
MAX_AREA_DIFF = 0.3                 # Maximum area difference between characters (30%)
MAX_WIDTH_DIFF = 0.7                # Maximum width difference between characters (70%)
MAX_HEIGHT_DIFF = 0.2               # Maximum height difference between characters (20%)
MIN_N_MATCHED = 3                   # Minimum number of characters to make a plate (adjusted for real-time)

# Display Settings - These control what you see on screen
WINDOW_NAME = "Vehicle License Plate Recognition System"  # Window title
DISPLAY_WIDTH = 1024                # Display window width in pixels
DISPLAY_HEIGHT = 576                # Display window height in pixels
SHOW_FPS = True                     # Whether to show FPS (frames per second) on screen
FONT_SCALE = 0.6                    # Text size (bigger number = bigger text)
THICKNESS = 2                       # Line thickness for boxes and text

# Color Settings (BGR format - Blue, Green, Red values 0-255)
COLOR_VEHICLE = (0, 255, 0)         # Vehicle bounding box color (green)
COLOR_PLATE = (0, 0, 255)           # License plate bounding box color (red)
COLOR_TEXT = (255, 255, 255)        # Text color (white)
COLOR_FPS = (0, 255, 255)           # FPS text color (yellow)

# Tesseract OCR Settings - These control text recognition
TESSERACT_CONFIG = '--psm 8 --oem 3'  # OCR configuration (optimized for real-time processing)

# Performance Settings - These control processing speed
PLATE_DETECTION_INTERVAL = 10       # Process plates every N frames (10 = every 10th frame)
MAX_PROCESSING_TIME = 0.5           # Maximum time to spend processing one frame (seconds)

# Logging Settings - These control program messages
logging.basicConfig(level=logging.INFO)  # Set logging level to INFO
logger = logging.getLogger(__name__)      # Create logger for this program

# ================================
# License Plate Recognition Class
# ================================

class RealtimePlateRecognizer:
    """
    Real-time License Plate Recognition Class
    This class handles finding and reading license plates from vehicle images.
    It uses computer vision techniques to locate text and OCR to read it.
    """
    
    def __init__(self):
        """
        Initialize the plate recognizer
        This function runs when we create a new RealtimePlateRecognizer object
        """
        self.frame_count = 0  # Keep track of how many frames we've processed
        logger.info("Real-time plate recognizer initialized successfully")
    
    def preprocess_image(self, image):
        """
        Preprocess image for better text detection (optimized for speed)
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
        blurred = cv2.GaussianBlur(gray, ksize=(3, 3), sigmaX=0)
        
        # Apply adaptive threshold to create binary image (black and white only)
        # This makes text stand out from background
        thresh = cv2.adaptiveThreshold(
            blurred,                                    # Input image
            maxValue=255.0,                            # Maximum pixel value (white)
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Method for calculating threshold
            thresholdType=cv2.THRESH_BINARY_INV,       # Invert colors (text becomes white)
            blockSize=15,                              # Size of area for threshold calculation (smaller = faster)
            C=7                                        # Constant subtracted from mean
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
        Find groups of characters that belong together (optimized for speed)
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
        Extract the license plate region from image (optimized for speed)
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
        if x2 - x1 < 20 or y2 - y1 < 10:
            return None, None
        
        # Cut out the plate region from the image
        plate_region = image[y1:y2, x1:x2]
        
        # Store plate position and size information
        plate_info = {'x': x1, 'y': y1, 'w': x2-x1, 'h': y2-y1}
        
        return plate_region, plate_info
    
    def recognize_text_fast(self, plate_image):
        """
        Fast text recognition using OCR (Optical Character Recognition)
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
            # Resize image if it's too small (OCR works better on larger images)
            height, width = plate_image.shape[:2]
            if width < 100:
                scale = 100 / width  # Calculate scaling factor
                new_width = int(width * scale)
                new_height = int(height * scale)
                plate_image = cv2.resize(plate_image, (new_width, new_height))
            
            # Convert to grayscale if image is in color
            if len(plate_image.shape) == 3:
                plate_image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            
            # Apply binary threshold to make text clearer
            _, plate_image = cv2.threshold(
                plate_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
            )
            
            # Add padding around image to help OCR
            plate_image = cv2.copyMakeBorder(
                plate_image, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=0
            )
            
            # Perform OCR (text recognition)
            text = pytesseract.image_to_string(
                plate_image, lang='kor', config=TESSERACT_CONFIG
            ).strip()  # Remove extra spaces
            
            # Clean up the result - keep only Korean characters and numbers
            result = ''
            for char in text:
                if '가' <= char <= '힣' or char.isdigit():  # Korean characters or digits
                    result += char
            
            # Return result only if we found enough characters
            return result if len(result) >= 2 else None
            
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
            
            # Step 6: Choose the largest group (most likely to be a license plate)
            best_group = max(char_groups, key=len)
            
            # Step 7: Check if we're taking too long (timeout for real-time performance)
            if time.time() - start_time > MAX_PROCESSING_TIME:
                return None, None
            
            # Step 8: Extract the license plate region from the image
            plate_region, plate_info = self.extract_plate_region(thresh_image, best_group)
            
            # Step 9: Check if we successfully extracted a plate region
            if plate_region is None:
                return None, None
            
            # Step 10: Recognize text using OCR
            plate_text = self.recognize_text_fast(plate_region)
            
            return plate_text, plate_info
            
        except Exception as e:
            logger.debug(f"License plate recognition error: {e}")
            return None, None

# ================================
# Real-time Vehicle Detection Class
# ================================

class RealtimeVehicleDetector:
    """
    Real-time Vehicle Detection Class
    This class uses YOLO (You Only Look Once) AI model to detect vehicles in images.
    YOLO is a fast and accurate object detection algorithm.
    """
    
    def __init__(self):
        """
        Initialize the vehicle detector
        This function loads the YOLO model and prepares it for use
        """
        logger.info("Loading YOLOv8 model...")
        self.model = YOLO(YOLO_MODEL_PATH)  # Load the YOLO model from file
        
        # Warm up the model by running it once on a dummy image
        # This makes the first real detection faster
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)  # Create fake image
        self.model(dummy_image, verbose=False)  # Run model on fake image
        logger.info("YOLOv8 model loaded successfully")
    
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
                            
                            # Only keep vehicles that are big enough
                            if area_ratio >= CAR_DETECTION_AREA:
                                vehicles.append({
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],  # Bounding box coordinates
                                    'confidence': confidence,                        # How sure YOLO is
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
# Main System Class
# ================================

class RealtimeDetectionSystem:
    """
    Real-time Detection System
    This is the main class that coordinates everything:
    - Captures video from camera
    - Detects vehicles using YOLO
    - Recognizes license plates
    - Displays results on screen
    """
    
    def __init__(self):
        """
        Initialize the detection system
        This function sets up the camera, AI models, and display
        """
        logger.info("Initializing real-time detection system...")
        
        # Initialize AI components
        self.vehicle_detector = RealtimeVehicleDetector()  # Vehicle detection AI
        self.plate_recognizer = RealtimePlateRecognizer()  # License plate recognition AI
        
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
        
        # Initialize counters and timers
        self.frame_count = 0        # Count total frames processed
        self.fps_counter = 0        # Count frames for FPS calculation
        self.fps_start_time = time.time()  # Start time for FPS calculation
        self.current_fps = 0        # Current frames per second
        
        # Initialize data storage
        self.last_detections = []   # Store recent detection results
        self.detection_history = Queue(maxsize=30)  # Store last 30 frames of results
        
        logger.info("Real-time detection system initialized successfully")
    
    def calculate_fps(self):
        """
        Calculate frames per second (FPS)
        This function measures how fast the system is running
        """
        self.fps_counter += 1  # Count this frame
        
        # Calculate FPS every 30 frames
        if self.fps_counter >= 30:
            current_time = time.time()
            elapsed_time = current_time - self.fps_start_time  # Time for 30 frames
            self.current_fps = self.fps_counter / elapsed_time  # FPS = frames / time
            
            # Reset counters for next calculation
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def draw_detection_info(self, frame, vehicles, processing_time):
        """
        Draw detection information on the screen
        This function adds text showing system performance and status
        
        Args:
            frame: Image to draw on
            vehicles: List of detected vehicles
            processing_time: How long processing took (in seconds)
        """
        # Show FPS (frames per second) if enabled
        if SHOW_FPS:
            fps_text = f"FPS: {self.current_fps:.1f}"
            cv2.putText(frame, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLOR_FPS, THICKNESS)
        
        # Show processing time in milliseconds
        time_text = f"Processing: {processing_time*1000:.1f}ms"
        cv2.putText(frame, time_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLOR_FPS, THICKNESS)
        
        # Show number of detected vehicles
        count_text = f"Vehicles: {len(vehicles)}"
        cv2.putText(frame, count_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLOR_FPS, THICKNESS)
        
        # Show current frame number
        frame_text = f"Frame: {self.frame_count}"
        cv2.putText(frame, frame_text, (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLOR_FPS, THICKNESS)
    
    def draw_vehicles_and_plates(self, frame, vehicles):
        """
        Draw vehicles and license plates on the screen
        This function adds colored boxes and text to show detected objects
        
        Args:
            frame: Image to draw on
            vehicles: List of detected vehicles
        """
        # Process each detected vehicle
        for i, vehicle in enumerate(vehicles):
            x1, y1, x2, y2 = vehicle['bbox']      # Get vehicle position
            confidence = vehicle['confidence']     # Get detection confidence
            class_name = vehicle['class_name']     # Get vehicle type name
            
            # Draw vehicle bounding box (green rectangle)
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_VEHICLE, THICKNESS)
            
            # Prepare vehicle information text
            vehicle_text = f"{class_name}: {confidence:.2f}"
            
            # Calculate text size to create background rectangle
            text_size = cv2.getTextSize(vehicle_text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, THICKNESS)[0]
            
            # Draw background rectangle for text (for better readability)
            cv2.rectangle(frame, (x1, y1-25), (x1+text_size[0], y1), COLOR_VEHICLE, -1)
            
            # Draw vehicle information text
            cv2.putText(frame, vehicle_text, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLOR_TEXT, THICKNESS)
            
            # Try to recognize license plate (only every N frames for performance)
            if self.frame_count % PLATE_DETECTION_INTERVAL == 0:
                # Extract vehicle image from the frame
                vehicle_image = frame[y1:y2, x1:x2]
                
                # Check if vehicle image is valid
                if vehicle_image.size > 0:
                    # Run license plate recognition
                    plate_text, plate_info = self.plate_recognizer.recognize_plate(vehicle_image)
                    
                    # If we found a license plate
                    if plate_text and plate_info:
                        # Convert plate coordinates from vehicle image to full frame coordinates
                        plate_x1 = x1 + plate_info['x']  # Adjust X position
                        plate_y1 = y1 + plate_info['y']  # Adjust Y position
                        plate_x2 = plate_x1 + plate_info['w']  # Calculate right edge
                        plate_y2 = plate_y1 + plate_info['h']  # Calculate bottom edge
                        
                        # Draw license plate bounding box (red rectangle)
                        cv2.rectangle(frame, (plate_x1, plate_y1), (plate_x2, plate_y2), COLOR_PLATE, THICKNESS)
                        
                        # Prepare license plate text for display
                        plate_display_text = f"Plate: {plate_text}"
                        
                        # Calculate text size for background rectangle
                        text_size = cv2.getTextSize(plate_display_text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, THICKNESS)[0]
                        
                        # Draw background rectangle for plate text
                        cv2.rectangle(frame, (plate_x1, plate_y2), (plate_x1+text_size[0], plate_y2+25), COLOR_PLATE, -1)
                        
                        # Draw license plate text
                        cv2.putText(frame, plate_display_text, (plate_x1, plate_y2+20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLOR_TEXT, THICKNESS)
                        
                        # Log the recognition result
                        logger.info(f"License plate recognized: {plate_text}")
    
    def process_frame(self, frame):
        """
        Process a single frame from the camera
        This function does all the AI processing for one image
        
        Args:
            frame: Single image from camera
            
        Returns:
            frame: Same image with detection results drawn on it
        """
        start_time = time.time()  # Record start time for performance measurement
        
        # Step 1: Detect vehicles using YOLO AI
        vehicles = self.vehicle_detector.detect_vehicles(frame)
        
        # Step 2: Draw detection results on the image
        self.draw_vehicles_and_plates(frame, vehicles)
        
        # Step 3: Calculate how long processing took
        processing_time = time.time() - start_time
        
        # Step 4: Draw performance information on screen
        self.draw_detection_info(frame, vehicles, processing_time)
        
        return frame
    
    def run(self):
        """
        Main execution loop
        This function runs continuously, processing camera frames and showing results
        """
        logger.info("Starting real-time detection system")
        logger.info("Exit: Press ESC key or 'q' key")
        
        # Create window to display results
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        
        try:
            # Main loop - runs until user presses exit key
            while True:
                # Step 1: Read one frame from camera
                ret, frame = self.camera.read()
                if not ret:  # Check if frame was captured successfully
                    logger.error("Cannot read frame from camera!")
                    break
                
                # Step 2: Count this frame
                self.frame_count += 1
                
                # Step 3: Process the frame (detect vehicles and plates)
                processed_frame = self.process_frame(frame)
                
                # Step 4: Resize frame for display if needed
                if processed_frame.shape[1] != DISPLAY_WIDTH:
                    processed_frame = cv2.resize(processed_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                
                # Step 5: Show the processed frame on screen
                cv2.imshow(WINDOW_NAME, processed_frame)
                
                # Step 6: Calculate and update FPS
                self.calculate_fps()
                
                # Step 7: Check for user input (keyboard keys)
                key = cv2.waitKey(1) & 0xFF  # Wait 1ms for key press
                
                if key == 27 or key == ord('q'):  # ESC key (27) or 'q' key
                    break  # Exit the main loop
                elif key == ord('s'):  # 's' key for screenshot
                    # Save current frame as image file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    logger.info(f"Screenshot saved: {filename}")
                
        except KeyboardInterrupt:
            logger.info("User interrupted (Ctrl+C)")
        except Exception as e:
            logger.error(f"Error during execution: {e}")
        finally:
            self.cleanup()  # Clean up resources
    
    def cleanup(self):
        """
        Clean up system resources
        This function properly closes camera and windows when program ends
        """
        logger.info("Cleaning up system...")
        
        # Release camera resource
        if hasattr(self, 'camera'):
            self.camera.release()
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        
        logger.info("System cleanup completed")

# ================================
# Main Execution Section
# ================================

def main():
    """
    Main function - entry point of the program
    This function starts everything and handles any errors
    """
    # Print program information
    print("=" * 60)
    print("Raspberry Pi Real-time Vehicle License Plate Recognition System")
    print("=" * 60)
    print(f"Camera resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    print(f"Display resolution: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
    print(f"YOLO model: {YOLO_MODEL_PATH}")
    print("=" * 60)
    print("Controls:")
    print("  ESC or 'q': Exit program")
    print("  's': Save screenshot")
    print("=" * 60)
    
    try:
        # Create and run the detection system
        system = RealtimeDetectionSystem()
        system.run()
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        return 1  # Return error code
    
    return 0  # Return success code

# Program entry point - this runs when you execute the file
if __name__ == "__main__":
    exit(main())  # Run main function and exit with its return code 