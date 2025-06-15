# AI 기반 주차관리 시스템 (AI-Based Parking Management System)

## 📋 시스템 개요 (System Overview)

### 목표 (Objectives)
- 차량의 입차·출차 시점을 자동으로 인식
- 체류 시간 기반 요금 자동 계산
- 실시간 주차 상태 모니터링

### 주요 기능 (Key Features)
- 실시간 차량 인식 및 번호판 OCR
- 자동 입출차 시간 기록
- 체류 시간 기반 요금 계산
- 사용자 알림 시스템

## 🔄 시스템 구성 흐름 (System Flow)

### 1. 차량 진입 감지 (Vehicle Entry Detection)
- YOLOv8 기반 차량 객체 탐지
- OpenCV를 활용한 번호판 영역 추출
- Tesseract OCR을 통한 번호판 인식

### 2. 입출차 판단 (Entry/Exit Determination)
- 서버 입출차 구분
- 입차기록 유무 -> 자동 입출차 판단

### 3. 시간 기록 (Time Recording)
- 번호판을 키로 한 입출차 시간 기록
- 데이터베이스 저장 및 관리

### 4. 요금 계산 (Fee Calculation)
- 체류 시간 기반 요금 산정
- 설정 가능한 요금 정책 (예: 10분당 500원)
- 유연한 요금제 적용

### 5. 사용자 알림 (User Notification)
- 출차 시 요금 안내
- 결제 처리 및 출차 승인

## 🛠 기술 스택 (Technical Stack)

### 핵심 기술 (Core Technologies)
- **차량 인식**: YOLOv8, OpenCV
- **번호판 인식**: OpenCV, Tesseract OCR
- **데이터베이스**: SQLite 고려 중
- **백엔드**: Python
- **프론트엔드**: HTTP

### 시스템 요구사항 (System Requirements)
- Python 3.8 이상
- CUDA 지원 GPU (선택적)
- 웹캠 또는 IP 카메라
- 인터넷 연결

## 💻 설치 및 실행 (Installation & Setup)

### 필수 패키지 설치 (Required Packages)
```bash
pip install -r requirements.txt
```

### 환경 설정 (Environment Setup)
1. 카메라 연결 및 설정
2. 데이터베이스 초기화
3. 요금 정책 설정

### 서버는 가상환경에서 실행해요~!!! -서영이-

### 회의 밥 목록
-더하고 부대찌개(만두 부대찌개)
-이태윤 협찬 성심당 튀김소보로
-동대문엽기떡볶이 마라로제 착한맛
-와 AI 기반 주차관리 시스템 (AI-Based Parking Management System)

## 📋 시스템 개요 (System Overview)

### 목표 (Objectives)
- 차량의 입차·출차 시점을 자동으로 인식
- 체류 시간 기반 요금 자동 계산
- 실시간 주차 상태 모니터링

### 주요 기능 (Key Features)
- 실시간 차량 인식 및 번호판 OCR
- 자동 입출차 시간 기록
- 체류 시간 기반 요금 계산
- 사용자 알림 시스템

## 🔄 시스템 구성 흐름 (System Flow)

### 1. 차량 진입 감지 (Vehicle Entry Detection)
- YOLOv8 기반 차량 객체 탐지
- OpenCV를 활용한 번호판 영역 추출
- Tesseract OCR을 통한 번호판 인식

### 2. 입출차 판단 (Entry/Exit Determination)
- 서버 입출차 구분
- 입차기록 유무 -> 자동 입출차 판단

### 3. 시간 기록 (Time Recording)
- 번호판을 키로 한 입출차 시간 기록
- 데이터베이스 저장 및 관리

### 4. 요금 계산 (Fee Calculation)
- 체류 시간 기반 요금 산정
- 설정 가능한 요금 정책 (예: 10분당 500원)
- 유연한 요금제 적용

### 5. 사용자 알림 (User Notification)
- 출차 시 요금 안내
- 결제 처리 및 출차 승인

## 🛠 기술 스택 (Technical Stack)

### 핵심 기술 (Core Technologies)
- **차량 인식**: YOLOv8, OpenCV
- **번호판 인식**: OpenCV, Tesseract OCR
- **데이터베이스**: SQLite 고려 중
- **백엔드**: Python
- **프론트엔드**: HTTP

### 시스템 요구사항 (System Requirements)
- Python 3.8 이상
- CUDA 지원 GPU (선택적)
- 웹캠 또는 IP 카메라
- 인터넷 연결

## 💻 설치 및 실행 (Installation & Setup)

### 필수 패키지 설치 (Required Packages)
```bash
pip install -r requirements.txt
```

### 환경 설정 (Environment Setup)
1. 카메라 연결 및 설정
2. 데이터베이스 초기화
3. 요금 정책 설정


### 회의 밥 목록
- 더하고 부대찌개(만두 부대찌개)
- 이태윤 협찬 성심당 튀김소보로
- 동대문엽기떡볶이 마라로제 착한맛
- 수제버거 아메리칸 치즈버거(이태윤이랑 나) 뭔가채소많이들어간버거(지나)
"왜 국물이 있냐?" -이태윤-
- 짜장면 앤 간짜장 앤 볶짜면 앤 탕수육(볶음밥최악)
- 마라탕! 앤 꿔바로우
### SPECIAL THANKS 기계관 Emart 24!

