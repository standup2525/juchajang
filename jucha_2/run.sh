#!/bin/bash

# 로그 디렉토리 생성
mkdir -p logs

# 가상환경이 없으면 생성
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# 가상환경 활성화
source venv/bin/activate

# 필요한 패키지 설치
echo "Installing required packages..."
pip install -r requirements.txt

# Tesseract OCR 한국어 데이터 확인 및 다운로드
if [ ! -f "/usr/share/tesseract-ocr/4.00/tessdata/kor.traineddata" ]; then
    echo "Downloading Korean language data for Tesseract OCR..."
    sudo wget -O /usr/share/tesseract-ocr/4.00/tessdata/kor.traineddata https://github.com/tesseract-ocr/tessdata_best/raw/main/kor.traineddata
    sudo ldconfig
fi

# YOLO 모델 다운로드 확인
if [ ! -f "models/yolov8n.pt" ]; then
    echo "Downloading YOLOv8 model..."
    mkdir -p models
    wget -O models/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
fi

# 웹 서버 실행
echo "Starting web server..."
python src/web_server.py 