import cv2
import time
from vehicle_detection import detect_vehicle
from plate_reader import read_plate
from send_to_server import send_to_server
import os

def main():
    # 디버그 모드 설정
    DEBUG_MODE = True
    USE_SERVER = False  # 서버 연결 비활성화
    
    # 카메라 초기화
    print("Initializing camera...")
    
    # 라즈베리파이 카메라 설정
    camera_index = 0  # 기본 카메라 인덱스
    cap = cv2.VideoCapture(camera_index)
    
    # 카메라가 열리지 않으면 다른 인덱스 시도
    if not cap.isOpened():
        print("Trying alternative camera index...")
        cap = cv2.VideoCapture(1)  # 두 번째 카메라 인덱스 시도
    
    # 카메라 연결 상태 확인
    if not cap.isOpened():
        print("Error: Could not open camera")
        print("Please check if:")
        print("1. The camera is properly connected")
        print("2. No other application is using the camera")
        print("3. The camera is not disabled in system settings")
        print("4. Try running 'sudo raspi-config' and enable the camera")
        return
    
    # 카메라 속성 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # 카메라 속성 확인
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera initialized successfully:")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")

    # 이미지 저장 디렉토리 생성
    os.makedirs('images', exist_ok=True)
    if DEBUG_MODE:
        os.makedirs('debug_images', exist_ok=True)

    print("\n=== Program Configuration ===")
    print(f"Debug Mode: {'Enabled' if DEBUG_MODE else 'Disabled'}")
    print(f"Server Connection: {'Enabled' if USE_SERVER else 'Disabled'}")
    print("===========================\n")

    print("Starting vehicle detection and plate recognition...")
    print("Press 'q' to quit")

    frame_count = 0
    error_count = 0
    max_errors = 5
    last_detection_time = 0
    detection_interval = 2  # 2초 간격으로 감지

    while True:
        ret, frame = cap.read()
        if not ret:
            error_count += 1
            print(f"Error: Could not read frame (Attempt {error_count}/{max_errors})")
            if error_count >= max_errors:
                print("Too many consecutive errors. Please check camera connection.")
                print("Try the following:")
                print("1. Run 'sudo raspi-config' and enable the camera")
                print("2. Check if the camera is properly connected")
                print("3. Try rebooting the Raspberry Pi")
                break
            time.sleep(1)  # Wait a bit before trying again
            continue
        
        error_count = 0  # Reset error count on successful frame read
        frame_count += 1

        # 현재 시간 확인
        current_time = time.time()
        
        # 일정 간격으로만 차량 감지 수행
        if current_time - last_detection_time >= detection_interval:
            print("\n=== Starting New Detection Cycle ===")
            # 차량 감지
            img_path = detect_vehicle(frame)
            
            if img_path:
                print(f"Vehicle detected! Image saved to {img_path}")
                last_detection_time = current_time
                
                # 번호판 인식
                plate_text = read_plate(img_path, debug_mode=DEBUG_MODE)
                if plate_text:
                    print(f"License plate detected: {plate_text}")
                    
                    # 서버 연결이 활성화된 경우에만 전송
                    if USE_SERVER:
                        if send_to_server(plate_text, img_path):
                            print("Data sent to server successfully")
                        else:
                            print("Failed to send data to server")
                    else:
                        print("Server connection disabled - skipping data transmission")
                else:
                    print("No license plate detected")
            else:
                print("No vehicle detected in this frame")

        # 화면에 표시
        cv2.imshow('Vehicle Detection', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 정리
    print("\nCleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    print("Program terminated.")

if __name__ == "__main__":
    main() 