# This file sends recognized plate text to Flask server

import requests

def send_plate_text(text):
    url = 'http://<YOUR_SERVER_IP>:5000/receive_plate'  # Replace with actual Flask server IP
    try:
        response = requests.post(url, json={'plate': text})
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print("Failed to send to server:", e)
        return False 