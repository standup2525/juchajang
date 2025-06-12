#!/bin/bash

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Download YOLOv8n model if not exists
if [ ! -f "src/yolov8n.pt" ]; then
    wget -O src/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
fi

# Run Flask server
export FLASK_APP=src/main.py
flask run --host=0.0.0.0 --port=5000 