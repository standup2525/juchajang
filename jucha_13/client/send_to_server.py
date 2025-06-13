import requests
import json
import time
import os
from datetime import datetime

def send_to_server(plate_text, image_path):
    server_url = "http://localhost:5000/plate"
    
    # 이미지 파일이 존재하는지 확인
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return False
    
    # 현재 시간을 ISO 형식으로 변환
    current_time = datetime.now().isoformat()
    
    # 이미지 파일을 바이너리로 읽기
    with open(image_path, 'rb') as img_file:
        files = {
            'image': (os.path.basename(image_path), img_file, 'image/jpeg')
        }
        
        data = {
            'plate_text': plate_text,
            'timestamp': current_time
        }
        
        try:
            # 서버에 POST 요청 보내기
            response = requests.post(server_url, files=files, data=data)
            
            # 응답 확인
            if response.status_code == 200:
                print(f"Successfully sent plate data to server: {plate_text}")
                return True
            else:
                print(f"Failed to send data to server. Status code: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"Error sending data to server: {str(e)}")
            return False 