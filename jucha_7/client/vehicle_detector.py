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
    
    def detect_vehicles(self, frame):
        """
        Detect vehicles in the frame
        """
        # Run YOLOv8 inference
        results = self.model(frame, conf=config.CONFIDENCE_THRESHOLD)
        
        vehicles = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Check if detected object is a vehicle
                if box.cls in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    vehicles.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': float(box.conf[0])
                    })
        
        return vehicles
    
    def save_detected_image(self, frame, vehicles):
        """
        Save image with detected vehicles
        """
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Draw bounding boxes
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Save image
        output_path = os.path.join(config.OUTPUT_DIR, f"detected_{timestamp}.jpg")
        cv2.imwrite(output_path, frame)
        
        return timestamp 