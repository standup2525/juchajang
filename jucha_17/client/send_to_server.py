import requests

def send_plate_text(text):
    url = 'http://192.168.118.14:5000/plate'  # Replace if used
    try:
        response = requests.post(url, json={'plate': text})
        if response.status_code == 200:
            print("[Server] Plate sent successfully.")
        else:
            print(f"[Server] Failed to send. Status code: {response.status_code}")
    except Exception as e:
        print(f"[Server] No server available. Simulated send: '{text}'") 