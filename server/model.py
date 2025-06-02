from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class ParkingLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    plate_number = db.Column(db.String(20), nullable=False)
    entry_time = db.Column(db.DateTime, nullable=False)
    exit_time = db.Column(db.DateTime, nullable=True)
    fee = db.Column(db.Integer, nullable=True)
