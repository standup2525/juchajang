import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Camera settings - size and speed of video capture
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 360
CAMERA_FPS = 30

# YOLO settings - path to model and confidence level
YOLO_MODEL_PATH = 'models/yolov8n.pt'
YOLO_CONFIDENCE = 0.5

# Server settings - where to send data
SERVER_URL = os.getenv('SERVER_URL', 'http://localhost:5000')
API_KEY = os.getenv('API_KEY', '')

# Database settings - where to store data
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'parking_system')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', '')

# Hardware settings - pins for LED and barrier control
USE_RFID = os.getenv('USE_RFID', 'False').lower() == 'true'
LED_ENTRY = 17      # Green LED for entry
LED_EXIT = 27      # Green LED for exit
BARRIER_ENTRY = 22  # Motor control for entry barrier
BARRIER_EXIT = 23   # Motor control for exit barrier
RFID_RST = 24      # Reset pin for RFID reader
RFID_SDA = 25      # Data pin for RFID reader 