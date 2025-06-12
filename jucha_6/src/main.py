from flask import Flask, render_template, Response, send_from_directory
import threading
import os
from detector import Detector

app = Flask(__name__)

detector = Detector()

def start_detection():
    detector.run()

detection_thread = threading.Thread(target=start_detection, daemon=True)
detection_thread.start()

@app.route('/')
def index():
    # Show latest frame, detected vehicles, and plates
    latest_frame = detector.get_latest_frame_path()
    latest_plate = detector.get_latest_plate_info()
    return render_template('index.html', frame_path=latest_frame, plate_info=latest_plate)

@app.route('/video')
def video():
    # MJPEG video streaming
    return Response(detector.generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/captures/<filename>')
def captures(filename):
    # Serve captured images
    return send_from_directory(os.path.join('..', 'static', 'captures'), filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 