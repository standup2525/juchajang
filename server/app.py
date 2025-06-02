from flask import Flask, request, jsonify, render_template
from models import db, ParkingLog
from datetime import datetime
import os

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# ========================================
# 최초 요청 시 DB 테이블 생성
# ========================================
@app.before_first_request
def create_tables():
    db.create_all()

# ========================================
# 기본 페이지 (index.html 렌더링용)
# ========================================
@app.route('/')
def home():
    return render_template("index.html")

# ========================================
# 수동 입출차 확인용 (모바일 or 테스트 클라이언트)
# ========================================
@app.route('/plate', methods=['POST'])
def process_plate():
    plate = request.json.get("plate_number")
    now = datetime.now()

    record = ParkingLog.query.filter_by(plate_number=plate, exit_time=None).first()
    if record:
        # 출차 처리
        record.exit_time = now
        minutes = (now - record.entry_time).seconds // 60
        fee = minutes * 100
        record.fee = fee
        db.session.commit()
        return jsonify({"message": "출차 처리됨", "fee": fee})
    else:
        # 입차 처리
        new_log = ParkingLog(plate_number=plate, entry_time=now)
        db.session.add(new_log)
        db.session.commit()
        return jsonify({"message": "입차 처리됨"})

# ========================================
# 실시간 차량 정보 확인용 (웹 UI 연동 가능)
# ========================================
@app.route('/check_car', methods=['POST'])
def check_car():
    plate = request.json.get("plate_number")
    now = datetime.now()

    record = ParkingLog.query.filter_by(plate_number=plate).order_by(ParkingLog.id.desc()).first()
    if not record:
        return jsonify({"status": "not_found"})

    if record.exit_time is None:
        duration = (now - record.entry_time).seconds // 60
        fee = duration * 100
        return jsonify({
            "status": "입차",
            "entry_time": record.entry_time.strftime("%Y-%m-%d %H:%M:%S"),
            "fee": fee
        })
    else:
        return jsonify({
            "status": "출차",
            "entry_time": record.entry_time.strftime("%Y-%m-%d %H:%M:%S"),
            "exit_time": record.exit_time.strftime("%Y-%m-%d %H:%M:%S"),
            "fee": record.fee
        })

# ========================================
# YOLO 차량 인식 시스템 → Flask 서버로 전송되는 API
# ========================================
@app.route('/api/vehicle', methods=['POST'])
def receive_vehicle_data():
    plate = request.form.get('plate_number')
    direction = request.form.get('direction')
    timestamp = request.form.get('timestamp')
    device_id = request.form.get('device_id')
    confidence = request.form.get('confidence')
    image = request.files.get('image')

    # 이미지 저장
    if image:
        save_dir = 'static/uploads'
        os.makedirs(save_dir, exist_ok=True)
        safe_plate = ''.join(c for c in plate if c.isalnum())
        safe_time = timestamp.replace(':', '-')
        filename = f"{safe_plate}_{direction}_{safe_time}.jpg"
        image_path = os.path.join(save_dir, filename)
        image.save(image_path)

    # 입차 처리
    if direction == "entry":
        new_log = ParkingLog(
            plate_number=plate,
            entry_time=datetime.fromisoformat(timestamp)
        )
        db.session.add(new_log)
        db.session.commit()
        return jsonify({"success": True, "message": "입차 기록됨"})

    # 출차 처리
    elif direction == "exit":
        record = ParkingLog.query.filter_by(plate_number=plate, exit_time=None).first()
        if record:
            record.exit_time = datetime.fromisoformat(timestamp)
            minutes = (record.exit_time - record.entry_time).seconds // 60
            fee = minutes * 100
            record.fee = fee
            db.session.commit()
            return jsonify({"success": True, "message": "출차 기록됨", "fee": fee})
        else:
            return jsonify({"success": False, "error": "입차 기록 없음"})

    return jsonify({"success": False, "error": "잘못된 direction 값"})

# ========================================
# 서버 실행
# ========================================
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')  # 외부 접속 가능하게 설정
