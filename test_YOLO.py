from ultralytics import YOLO
import cv2

# YOLO 모델 로드
model = YOLO('/path/to/your/best.pt')  # 모델 경로 지정 (best.pt 또는 last.pt)

# 라즈베리파이 카메라 연결
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()  # 프레임 읽기
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # YOLO 모델로 객체 탐지 수행
    results = model(frame)

    # 예측 결과 렌더링 (객체의 바운딩 박스와 라벨을 이미지에 그리기)
    annotated_frame = results.render()[0]

    # 결과 이미지를 화면에 표시
    cv2.imshow('YOLO Detection', annotated_frame)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # 카메라 리소스 해제
cv2.destroyAllWindows()  # 모든 창 닫기
