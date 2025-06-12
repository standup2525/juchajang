import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Camera settings
CAMERA_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Detection settings
CONFIDENCE_THRESHOLD = 0.5
MIN_VEHICLE_SIZE_RATIO = 0.2  # 20% of frame size
DETECTION_INTERVAL = 1.0  # seconds

# Server settings
API_ENDPOINT = os.getenv('API_ENDPOINT', 'http://localhost:5000/api/plates')
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds

# Output settings
OUTPUT_DIR = 'output'
SAVE_IMAGES = True

# Vehicle classes in COCO dataset
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck 