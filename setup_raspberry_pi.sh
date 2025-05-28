#!/bin/bash

# 라즈베리파이5 + AI HAT 차량 번호판 인식 시스템 설정 스크립트

echo "==================================="
echo "라즈베리파이5 시스템 설정 시작"
echo "==================================="

# 시스템 업데이트
echo "시스템 패키지 업데이트 중..."
sudo apt update && sudo apt upgrade -y

# 필수 시스템 패키지 설치
echo "필수 시스템 패키지 설치 중..."
sudo apt install -y \
    python3-pip \
    python3-venv \
    tesseract-ocr \
    tesseract-ocr-kor \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-alsa \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 \
    gstreamer1.0-pulseaudio

# 카메라 활성화 확인
echo "카메라 설정 확인 중..."
if ! grep -q "camera_auto_detect=1" /boot/config.txt; then
    echo "카메라 자동 감지 활성화..."
    echo "camera_auto_detect=1" | sudo tee -a /boot/config.txt
fi

# GPU 메모리 설정 (AI HAT 사용을 위해)
echo "GPU 메모리 설정 중..."
if ! grep -q "gpu_mem=128" /boot/config.txt; then
    echo "gpu_mem=128" | sudo tee -a /boot/config.txt
fi

# Python 가상환경 생성
echo "Python 가상환경 생성 중..."
python3 -m venv venv
source venv/bin/activate

# Python 패키지 설치
echo "Python 패키지 설치 중..."
pip install --upgrade pip
pip install -r requirements.txt

# YOLO 모델 다운로드
echo "YOLO 모델 다운로드 중..."
python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
print('YOLO 모델 다운로드 완료')
"

# 디렉토리 생성
echo "필요한 디렉토리 생성 중..."
mkdir -p detections
mkdir -p logs

# 권한 설정
echo "권한 설정 중..."
chmod +x vehicle_detection_system.py

# 서비스 파일 생성 (선택사항)
echo "시스템 서비스 파일 생성 중..."
sudo tee /etc/systemd/system/vehicle-detection.service > /dev/null <<EOF
[Unit]
Description=Vehicle Detection System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin
ExecStart=$(pwd)/venv/bin/python $(pwd)/vehicle_detection_system.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 서비스 활성화 (선택사항)
echo "서비스 등록 중..."
sudo systemctl daemon-reload
sudo systemctl enable vehicle-detection.service

echo "==================================="
echo "설정 완료!"
echo "==================================="
echo ""
echo "사용법:"
echo "1. 수동 실행: python3 vehicle_detection_system.py"
echo "2. 서비스 시작: sudo systemctl start vehicle-detection"
echo "3. 서비스 상태 확인: sudo systemctl status vehicle-detection"
echo "4. 로그 확인: tail -f vehicle_detection.log"
echo ""
echo "주의사항:"
echo "- vehicle_detection_system.py 파일에서 웹서버 URL을 수정하세요"
echo "- 카메라 설정을 확인하세요"
echo "- 재부팅 후 카메라가 정상 작동하는지 확인하세요"
echo ""
echo "재부팅이 필요할 수 있습니다: sudo reboot" 