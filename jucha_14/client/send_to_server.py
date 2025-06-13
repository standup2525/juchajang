import requests

def send_plate_text(text):
    url = 'http://<YOUR_SERVER_IP>:5000/receive_plate'  # Replace if used
    try:
        response = requests.post(url, json={'plate': text})
        if response.status_code == 200:
            print("[Server] Plate sent successfully.")
        else:
            print(f"[Server] Failed to send. Status code: {response.status_code}")
    except Exception as e:
        print(f"[Server] No server available. Simulated send: '{text}'") 