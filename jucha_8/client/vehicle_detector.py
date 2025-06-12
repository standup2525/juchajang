import cv2
import os
import torch
from ultralytics import YOLO
from datetime import datetime
import config

class VehicleDetector:
    def __init__(self):
        """
        Initialize vehicle detector with YOLOv8 model
        """
        # Add safe globals for model loading
        torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])
        
        # Load YOLOv8 model
        self.model = YOLO('yolov8n.pt')
        
        # Create output directory if it doesn't exist
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    def calculate_size_ratio(self, bbox, frame_shape):
        """
        Calculate the ratio of vehicle size to frame size
        """
        x1, y1, x2, y2 = bbox
        vehicle_area = (x2 - x1) * (y2 - y1)
        frame_area = frame_shape[0] * frame_shape[1]
        return vehicle_area / frame_area
    
    def detect_vehicles(self, frame):
        """
        Detect vehicles in the frame and check their size
        """
        # Run YOLOv8 inference
        results = self.model(frame, conf=config.CONFIDENCE_THRESHOLD)
        
        vehicles = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Check if detected object is a vehicle
                if int(box.cls) in config.VEHICLE_CLASSES:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    bbox = (x1, y1, x2, y2)
                    
                    # Calculate size ratio
                    size_ratio = self.calculate_size_ratio(bbox, frame.shape[:2])
                    
                    # Only include vehicles that are large enough
                    if size_ratio >= config.MIN_VEHICLE_SIZE_RATIO:
                        vehicles.append({
                            'bbox': bbox,
                            'confidence': float(box.conf[0]),
                            'size_ratio': size_ratio
                        })
        
        return vehicles
    
    def save_detected_image(self, frame, vehicles):
        """
        Save image with detected vehicles
        """
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Draw bounding boxes and size ratio
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add size ratio label
            label = f"Size: {vehicle['size_ratio']:.2%}"
            cv2.putText(frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save image
        output_path = os.path.join(config.OUTPUT_DIR, f"detected_{timestamp}.jpg")
        cv2.imwrite(output_path, frame)
        
        return timestamp 