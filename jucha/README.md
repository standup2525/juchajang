# 자동차 번호판 인식 시스템

이 프로젝트는 라즈베리파이 5에서 카메라를 통해 자동차를 감지하고 번호판을 인식하는 시스템입니다.

## 기능

- YOLO를 사용한 실시간 자동차 감지
- 감지된 자동차에서 번호판 영역 추출
- OCR을 통한 번호판 텍스트 인식
- 실시간 결과 표시

## 요구사항

- Python 3.8 이상
- 라즈베리파이 5
- 카메라 모듈
- Tesseract OCR

## 설치 방법

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

2. Tesseract OCR 설치:
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-kor
```

3. YOLO 모델 다운로드:
```bash
# YOLOv8n 모델이 자동으로 다운로드됩니다.
```

## 사용 방법

1. 프로그램 실행:
```bash
python car_detection.py
```

2. 프로그램 종료:
- 'q' 키를 눌러 프로그램을 종료합니다.

## 주의사항

- 카메라가 올바르게 연결되어 있어야 합니다.
- 충분한 조명이 필요합니다.
- 번호판이 카메라에 잘 보이도록 설치해야 합니다.

## 향후 개선 사항

- 자동차만을 인식하도록 YOLO 모델 커스터마이징
- 번호판 인식 정확도 향상
- 실시간 처리 속도 개선 