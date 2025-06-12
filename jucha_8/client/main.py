import os
import sys
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
    
    print(f"\n[Server] Sending data: {data}")
    
    for attempt in range(config.MAX_RETRIES):
        try:
            response = requests.post(config.API_ENDPOINT, json=data)
            response.raise_for_status()
            print(f"[Server] Success! Response: {response.json()}")
            return True
        except requests.exceptions.RequestException as e:
            print(f"[Server] Error (attempt {attempt + 1}/{config.MAX_RETRIES}): {e}")
            if attempt < config.MAX_RETRIES - 1:
                time.sleep(config.RETRY_DELAY)
    
    return False

def save_detection_image(frame, timestamp, vehicle_bbox, plate_bbox, plate_text):
    """
    Save image with vehicle and plate bounding boxes
    """
    # Create a copy of the frame for drawing
    marked_frame = frame.copy()
    
    # Draw vehicle bounding box
    x1, y1, x2, y2 = vehicle_bbox
    cv2.rectangle(marked_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(marked_frame, "Vehicle", (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw plate bounding box and text
    if plate_bbox and plate_text:
        px1, py1, px2, py2 = plate_bbox
        cv2.rectangle(marked_frame, (px1, py1), (px2, py2), (255, 0, 0), 2)
        cv2.putText(marked_frame, plate_text, (px1, py2+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Save the marked image
    output_path = os.path.join('images', f"detected_{timestamp}.jpg")
    cv2.imwrite(output_path, marked_frame)
    print(f"[Image] Saved marked image: {output_path}")

def main():
    # Activate virtual environment if not already activated
    if not hasattr(sys, 'real_prefix') and not hasattr(sys, 'base_prefix'):
        activate_venv()
        return  # This line will not be reached as exec will replace the process
    
    print("\n=== Vehicle Detection and Plate Recognition System ===")
    print("Initializing camera...")
    
    # Initialize camera
    cap = cv2.VideoCapture(config.CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Initializing YOLOv8 model...")
    detector = VehicleDetector()
    
    print("Initializing plate recognizer...")
    recognizer = PlateRecognizer()
    
    print("\nSystem ready! Press 'q' to quit.")
    print("Waiting for vehicles...")
    
    last_detection_time = 0
    
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
                    print(f"\n[Detection] Found {len(vehicles)} vehicles")
                    
                    # Create timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # Process each detected vehicle
                    for i, vehicle in enumerate(vehicles, 1):
                        print(f"\n[Vehicle {i}] Size ratio: {vehicle['size_ratio']:.2%}")
                        
                        # Recognize plate
                        plate_text, plate_img, plate_bbox = recognizer.recognize_plate(
                            frame, vehicle['bbox']
                        )
                        
                        if plate_text:
                            print(f"[Vehicle {i}] Recognized plate: {plate_text}")
                            
                            # Draw plate information
                            frame = recognizer.draw_plate_info(
                                frame, vehicle['bbox'], plate_bbox, plate_text
                            )
                            
                            # Save detection image
                            save_detection_image(
                                frame, timestamp, vehicle['bbox'], plate_bbox, plate_text
                            )
                            
                            # Send to server
                            if send_to_server(plate_text, timestamp):
                                print(f"[Vehicle {i}] Successfully processed")
                            else:
                                print(f"[Vehicle {i}] Failed to process")
                        else:
                            print(f"[Vehicle {i}] No plate detected")
                            # Save image even if no plate is detected
                            save_detection_image(
                                frame, timestamp, vehicle['bbox'], None, None
                            )
                
                last_detection_time = current_time
            
            # Display the frame
            cv2.imshow('Vehicle Detection', frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nQuitting...")
                break
    
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print("\nSystem shutdown complete.")

if __name__ == "__main__":
    main() 