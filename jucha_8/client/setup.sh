#!/bin/bash

# Exit on error
set -e

echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3-opencv \
    tesseract-ocr \
    tesseract-ocr-kor \
    python3-numpy \
    python3-pip \
    python3-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-full

echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing Python packages..."
python -m pip install --no-cache-dir \
    ultralytics==8.1.0 \
    opencv-python==4.9.0.80 \
    numpy==1.26.4 \
    requests==2.31.0 \
    python-dotenv==1.0.1 \
    pytesseract==0.3.10

echo "Setup completed successfully!"
echo "To activate the virtual environment, run: source venv/bin/activate" 