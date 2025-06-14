from flask import Flask, request, render_template, redirect
from plate_handler import handle_plate, reset_status, get_current_status
from sense_controller import show_entry, show_exit, show_denied, show_waiting
import csv
import os

app = Flask(__name__)

CSV_FILE = "plate_log.csv"

@app.route("/")
def index():
    logs = []
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, newline="") as f:
            reader = csv.reader(f)
            logs = list(reader)[-10:]  # 최근 10개만
    current = get_current_status()
    return render_template("index.html", logs=logs, status=current)

@app.route("/plate", methods=["POST"])
def plate():
    data = request.get_json()
    plate_text = data.get("plate", "")
    print(f"[Server] Plate received: {plate_text}")
    result = handle_plate(plate_text)
    status = result["status"]

    if status == "입차 허가":
        show_entry()
    elif status == "출차 허가":
        show_exit()
    else:
        show_denied()

    return result

@app.route("/reset", methods=["POST"])
def reset():
    reset_status()
    show_waiting()
    return {"status": "초기화 완료"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000) 