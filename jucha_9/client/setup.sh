#!/bin/bash

# Simple shell script to install dependencies (assuming virtualenv is already activated)

echo "Installing Python packages..."
pip install opencv-python numpy matplotlib pytesseract requests ultralytics

echo "Installing Tesseract OCR..."
sudo apt-get update
sudo apt-get install -y tesseract-ocr

echo "Setup complete." 