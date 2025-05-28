#!/bin/bash

# 차량 번호판 인식 시스템 시작 스크립트

echo "==================================="
echo "차량 번호판 인식 시스템 시작"
echo "==================================="

# 가상환경 활성화
if [ -d "venv" ]; then
    echo "Python 가상환경 활성화 중..."
    source venv/bin/activate
else
    echo "경고: Python 가상환경이 없습니다. setup_raspberry_pi.sh를 먼저 실행하세요."
fi

# 웹서버와 감지 시스템을 백그라운드에서 실행
echo "웹서버 시작 중..."
python3 web_server.py &
WEB_SERVER_PID=$!

# 웹서버가 시작될 때까지 대기
echo "웹서버 시작 대기 중..."
sleep 5

# 웹서버 상태 확인
if curl -s http://localhost:5000 > /dev/null; then
    echo "웹서버 시작 완료: http://localhost:5000"
else
    echo "웹서버 시작 실패"
    kill $WEB_SERVER_PID 2>/dev/null
    exit 1
fi

echo "차량 감지 시스템 시작 중..."
echo "시스템 종료: Ctrl+C"
echo ""

# 차량 감지 시스템 실행 (포그라운드)
python3 vehicle_detection_system.py

# 시스템 종료 시 웹서버도 종료
echo ""
echo "시스템 종료 중..."
kill $WEB_SERVER_PID 2>/dev/null
echo "시스템 종료 완료" 