import csv
import os
from datetime import datetime

CSV_FILE = "plate_log.csv"
STATUS_FILE = "status.txt"

WHITELIST = ["200태0323", "203서1215", "203지0513"]

def read_status():
    if not os.path.exists(STATUS_FILE):
        return "idle", None
    with open(STATUS_FILE, "r") as f:
        line = f.read().strip()
        if line.startswith("awaiting_exit:"):
            return "awaiting_exit", line.split(":")[1]
        else:
            return "idle", None

def write_status(status, plate=None):
    with open(STATUS_FILE, "w") as f:
        if status == "idle":
            f.write("idle")
        elif status == "awaiting_exit" and plate:
            f.write(f"awaiting_exit:{plate}")

def log_plate(plate, action):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([now, plate, action])

def get_current_status():
    status, plate = read_status()
    if status == "idle":
        return "Waiting for vehicle"
    elif status == "awaiting_exit":
        return f"Vehicle {plate} inside (awaiting exit)"

def reset_status():
    write_status("idle")

def match_plate_to_whitelist(plate):
    def similarity(a, b):
        return sum(x == y for x, y in zip(a, b))
    plate = plate.replace(" ", "").upper()
    scores = [(w, similarity(plate, w)) for w in WHITELIST]
    best, score = max(scores, key=lambda x: x[1])
    return best if score >= 5 else None  # 최소 유사도 기준 5자 이상

def handle_plate(plate_text):
    plate = plate_text.strip().replace(" ", "").upper()

    if plate == "NO PLATE":
        status, current_plate = read_status()
        if status == "awaiting_exit" and current_plate:
            log_plate(current_plate, "EXIT")
            write_status("idle")
            return {"status": "Vehicle exited", "plate": current_plate}
        else:
            return {"status": "Idle - no vehicle"}
    
    matched_plate = match_plate_to_whitelist(plate)
    if not matched_plate:
        return {"status": "Unknown plate", "plate": plate}

    status, current_plate = read_status()
    if status == "idle":
        log_plate(matched_plate, "ENTRY")
        write_status("awaiting_exit", matched_plate)
        return {"status": "ENTRY GRANTED", "plate": matched_plate}
    
    elif status == "awaiting_exit":
        if matched_plate == current_plate:
            return {"status": "Already inside", "plate": matched_plate}
        else:
            return {"status": "Another vehicle detected, ignored", "plate": matched_plate}