import cv2
import pytesseract
import numpy as np

def recognize_plate(image):
    # Convert to gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Threshold
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Find possible plate regions
    plate_img = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        ratio = w / h if h != 0 else 0
        if area > 210 and w > 8 and h > 16 and 0.4 < ratio < 1.0:
            plate_img = gray[y:y+h, x:x+w]
            break
    if plate_img is not None:
        plate_img = cv2.resize(plate_img, None, fx=1.6, fy=1.6)
        _, plate_img = cv2.threshold(plate_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(plate_img, lang='eng')
        return text.strip()
    return '' 