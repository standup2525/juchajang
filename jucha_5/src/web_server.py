from flask import Flask, render_template, jsonify, Response
import logging
from datetime import datetime
import threading
import cv2
import numpy as np
from car_detection import parking_system, start_camera, read_frame
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/web_server.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
camera_proc = None
camera_thread = None
is_camera_running = False
camera_error = None

def camera_loop():
    """Camera processing loop"""
    global camera_proc, is_camera_running, camera_error
    try:
        camera_proc = start_camera()
        is_camera_running = True
        camera_error = None
        
        while is_camera_running:
            frame = read_frame(camera_proc)
            if frame is not None:
                # Process frame and update parking status
                processed_frame, detected_plates = parking_system.process_frame(frame)
                parking_system.update_parking_status(detected_plates)
                
                # Show frame on local display
                cv2.imshow('Car Detection', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                logger.warning("[Warning] Failed to read camera frame.")
            time.sleep(1)  # Process 1 frame per second
    except Exception as e:
        logger.error(f"[Error] Camera loop error: {e}")
        camera_error = str(e)
    finally:
        if camera_proc:
            camera_proc.terminate()
        cv2.destroyAllWindows()
        is_camera_running = False

def start_camera_thread():
    """Start camera processing in a separate thread"""
    global camera_thread
    if not is_camera_running:
        camera_thread = threading.Thread(target=camera_loop)
        camera_thread.daemon = True
        camera_thread.start()

@app.route('/')
def index():
    """Main page showing parking status and camera status"""
    return render_template('index.html', data=parking_system.get_parking_data(), camera_error=camera_error)

@app.route('/api/parking-status')
def parking_status():
    """API endpoint for parking status"""
    return jsonify(parking_system.get_parking_data())

@app.route('/history')
def history():
    """Page showing parking history"""
    return render_template('history.html')

if __name__ == '__main__':
    # Start camera processing in a separate thread
    start_camera_thread()
    # Start web server (always available even if camera fails)
    app.run(host='0.0.0.0', port=5000, debug=True) 