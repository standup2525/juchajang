import os
import sys
import subprocess
import cv2
import time
import requests
from datetime import datetime
import config
from vehicle_detector import VehicleDetector
from plate_recognizer import PlateRecognizer

def activate_venv():
    """
    Activate virtual environment
    """
    venv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'venv')
    if os.path.exists(venv_path):
        activate_script = os.path.join(venv_path, 'bin', 'activate')
        if os.path.exists(activate_script):
            activate_cmd = f"source {activate_script} && exec python {__file__} {' '.join(sys.argv[1:])}"
            os.execvp('bash', ['bash', '-c', activate_cmd])

def send_to_server(plate_text, timestamp):
    """
    Send plate information to the server
    """
    data = {
        'plate_number': plate_text,
        'timestamp': timestamp
    }
    
    for attempt in range(config.MAX_RETRIES):
        try:
            response = requests.post(config.API_ENDPOINT, json=data)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error sending data to server (attempt {attempt + 1}/{config.MAX_RETRIES}): {e}")
            if attempt < config.MAX_RETRIES - 1:
                time.sleep(config.RETRY_DELAY)
    
    return False

def main():
    # Activate virtual environment if not already activated
    if not hasattr(sys, 'real_prefix') and not hasattr(sys, 'base_prefix'):
        activate_venv()
        return  # This line will not be reached as exec will replace the process
    
    # Initialize camera
    cap = cv2.VideoCapture(config.CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    
    # Initialize detector and recognizer
    detector = VehicleDetector()
    recognizer = PlateRecognizer()
    
    last_detection_time = 0
    
    print("Starting vehicle detection and plate recognition...")
    
    try:
        while True:
            # Read frame from camera
            ret, frame = cap.read()
            if not ret:
                print("Error reading from camera")
                break
            
            current_time = time.time()
            
            # Check if it's time for next detection
            if current_time - last_detection_time >= config.DETECTION_INTERVAL:
                # Detect vehicles
                vehicles = detector.detect_vehicles(frame)
                
                if vehicles:
                    # Save detected image
                    timestamp = detector.save_detected_image(frame, vehicles)
                    
                    # Process each detected vehicle
                    for vehicle in vehicles:
                        # Recognize plate
                        plate_text, plate_img = recognizer.recognize_plate(
                            frame, vehicle['bbox']
                        )
                        
                        # Draw plate information
                        frame = recognizer.draw_plate_info(
                            frame, vehicle['bbox'], plate_text
                        )
                        
                        # Send to server
                        if send_to_server(plate_text, timestamp):
                            print(f"Successfully processed plate: {plate_text}")
                        else:
                            print(f"Failed to send plate data: {plate_text}")
                
                last_detection_time = current_time
            
            # Display the frame
            cv2.imshow('Vehicle Detection', frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 