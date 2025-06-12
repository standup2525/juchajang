# Server settings
SERVER_IP = "192.168.0.100"  # Change this to your server's IP address
SERVER_PORT = 5000
API_ENDPOINT = f"http://{SERVER_IP}:{SERVER_PORT}/api/parking"

# Camera settings
CAMERA_ID = 0  # Change this if using a different camera
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Detection settings
DETECTION_INTERVAL = 1.0  # Time between detections in seconds
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for vehicle detection
VEHICLE_CLASSES = [2, 3, 5, 7]  # YOLO class IDs for cars, motorcycles, buses, and trucks

# Image settings
SAVE_IMAGES = True  # Whether to save detected images
IMAGE_QUALITY = 95  # JPEG quality for saved images

# Plate recognition settings
PLATE_MIN_WIDTH = 60  # Minimum width of plate in pixels
PLATE_MIN_HEIGHT = 20  # Minimum height of plate in pixels

# Error handling
MAX_RETRIES = 3  # Maximum number of retries for server communication
RETRY_DELAY = 1.0  # Delay between retries in seconds 