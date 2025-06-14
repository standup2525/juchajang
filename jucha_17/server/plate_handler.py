import csv
import os
import time
from difflib import get_close_matches

# ✅ 등록된 차량 화이트리스트
WHITELIST = ["200태0323", "203서1215", "203지0513"]

# ✅ 서버 상태
STATUS_FILE = "status.txt"
CSV_FILE = "plate_log.csv"

def get_closest_plate(plate):
    matches = get_close_matches(plate, WHITELIST, n=1, cutoff=0.4)
    return matches[0] if matches else None

def get_current_status():
    if not os.path.exists(STATUS_FILE):
        return "입차 대기"
    with open(STATUS_FILE, "r") as f:
        return f.read().strip()

def set_status(new_status):
    with open(STATUS_FILE, "w") as f:
        f.write(new_status)

def reset_status():
    set_status("입차 대기")

def log_plate(plate, action):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, plate, action])

def handle_plate(plate_text):
    matched = get_closest_plate(plate_text)
    if not matched:
        return {"status": "거부", "plate": plate_text}

    current = get_current_status()
    if current == "입차 대기":
        log_plate(matched, "입차")
        set_status("출차 대기")
        return {"status": "입차 허가", "plate": matched}
    elif current == "출차 대기":
        log_plate(matched, "출차")
        set_status("입차 대기")
        return {"status": "출차 허가", "plate": matched}
    else:
        return {"status": "알 수 없음", "plate": matched} 