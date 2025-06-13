#!/bin/bash

echo "Installing Python packages..."
pip install opencv-python numpy matplotlib pytesseract requests ultralytics

echo "Installing Tesseract OCR..."
sudo apt-get update
sudo apt-get install -y tesseract-ocr
sudo apt-get install -y tesseract-ocr-kor

echo "Setup complete!" 