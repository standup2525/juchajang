# This is the main control file. It runs the camera, detects vehicles, reads plates, and sends data to the server.

import cv2
import subprocess
import numpy as np
from vehicle_detection import detect_vehicle
from plate_reader import read_plate
from send_to_server import send_plate_text

def start_stream():
    return subprocess.Popen([
        "libcamera-vid", "--width", "640", "--height", "360",
        "--framerate", "30", "--codec", "mjpeg",
        "--timeout", "0", "--nopreview", "-o", "-"
    ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)

def main():
    proc = start_stream()
    buf = b""
    print("Camera started. Press ESC to exit.")

    while True:
        chunk = proc.stdout.read(4096)
        if not chunk:
            break
        buf += chunk
        start = buf.find(b'\xff\xd8')
        end = buf.find(b'\xff\xd9', start + 2)
        if start != -1 and end != -1:
            jpg = buf[start:end + 2]
            buf = buf[end + 2:]
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # Vehicle detection
            annotated, captured, img_path = detect_vehicle(frame)

            # Plate reading and server send
            if captured and img_path:
                plate_text = read_plate(img_path)
                if plate_text:
                    print("Plate found:", plate_text)
                    send_plate_text(plate_text)
                else:
                    print("No plate detected.")

            # Show annotated frame
            cv2.imshow("Vehicle Detection", annotated)
            if cv2.waitKey(1) == 27:
                break

    proc.terminate()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 