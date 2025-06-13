import cv2
import time

def test_camera():
    print("Testing camera connection...")
    
    # 카메라 초기화 (여러 카메라 인덱스 시도)
    for camera_index in [0, 1, 2]:
        print(f"\nTrying camera index {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Failed to open camera {camera_index}")
            continue
            
        # 카메라 속성 출력
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Camera {camera_index} properties:")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps}")
        
        # 프레임 읽기 시도
        ret, frame = cap.read()
        if ret:
            print(f"Successfully read frame from camera {camera_index}")
            # 테스트 이미지 저장
            cv2.imwrite(f"camera_test_{camera_index}.jpg", frame)
            print(f"Saved test image as camera_test_{camera_index}.jpg")
        else:
            print(f"Failed to read frame from camera {camera_index}")
        
        cap.release()
        time.sleep(1)  # 카메라 해제 후 잠시 대기

if __name__ == "__main__":
    test_camera() 