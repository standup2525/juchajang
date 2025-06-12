import cv2
import os
import time
from ultralytics import YOLO
from plate_recognizer import recognize_plate

CAPTURE_DIR = os.path.join(os.path.dirname(__file__), '..', 'static', 'captures')
if not os.path.exists(CAPTURE_DIR):
    os.makedirs(CAPTURE_DIR)

class Detector:
    def __init__(self):
        self.model = YOLO(os.path.join(os.path.dirname(__file__), 'yolov8n.pt'))
        self.latest_frame_path = None
        self.latest_plate_info = None
        self.last_capture_time = 0
        self.capture_interval = 1  # seconds
        self.camera = cv2.VideoCapture(0)

    def run(self):
        while True:
            ret, frame = self.camera.read()
            if not ret:
                print('[Warning] Failed to read camera frame.')
                time.sleep(1)
                continue
            results = self.model(frame)
            found_vehicle = False
            for result in results:
                for box in result.boxes:
                    if int(box.cls) in [2, 5, 7]:  # car, bus, truck
                        found_vehicle = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # Save frame if vehicle detected and enough time has passed
            if found_vehicle and (time.time() - self.last_capture_time > self.capture_interval):
                filename = f'vehicle_{int(time.time())}.jpg'
                save_path = os.path.join(CAPTURE_DIR, filename)
                cv2.imwrite(save_path, frame)
                self.latest_frame_path = f'captures/{filename}'
                # Plate recognition
                plate_text = recognize_plate(frame)
                self.latest_plate_info = {'plate': plate_text, 'img': self.latest_frame_path}
                print(f'[Info] Vehicle detected. Plate: {plate_text}')
                self.last_capture_time = time.time()
            time.sleep(0.1)

    def get_latest_frame_path(self):
        return self.latest_frame_path

    def get_latest_plate_info(self):
        return self.latest_plate_info

    def generate_mjpeg(self):
        while True:
            ret, frame = self.camera.read()
            if not ret:
                continue
            _, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n') 