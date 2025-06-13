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
    print("[Main] Camera started. Press ESC to exit.")

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

            annotated, captured, img_path = detect_vehicle(frame)

            if captured and img_path:
                print(f"[Main] Image saved at {img_path}")
                plate_text = read_plate(img_path)
                if plate_text:
                    send_plate_text(plate_text)
                else:
                    print("[Main] No valid plate found.")
            else:
                print("[Main] No large vehicle captured.")

            cv2.imshow("Live Feed", annotated)
            if cv2.waitKey(1) == 27:
                print("[Main] ESC pressed. Exiting.")
                break

    proc.terminate()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 