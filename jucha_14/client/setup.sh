#!/bin/bash

# Dependency installation for the client (assumes venv is already activated)

echo "[Setup] Installing Python packages..."
pip install opencv-python numpy matplotlib pytesseract requests ultralytics

echo "[Setup] Installing system package for OCR..."
sudo apt-get update
sudo apt-get install -y tesseract-ocr

echo "[Setup] Setup complete." 