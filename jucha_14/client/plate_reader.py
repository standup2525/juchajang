import cv2
import numpy as np
import pytesseract
import os

cv2.setNumThreads(4)
os.makedirs('results', exist_ok=True)

def read_plate(image_path):
    print(f"[PlateReader] Reading plate from {image_path}")
    img_ori = cv2.imread(image_path)
    if img_ori is None:
        print("[PlateReader] Failed to read image.")
        return ""

    img_ori = cv2.resize(img_ori, (640, 480))
    height, width = img_ori.shape[:2]
    gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
    img_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    img_thresh = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 19, 9)

    contours, _ = cv2.findContours(img_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    MIN_AREA, MIN_WIDTH, MIN_HEIGHT = 210, 8, 16
    MIN_RATIO, MAX_RATIO = 0.4, 1.0
    possible_contours = []
    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > MIN_AREA and w > MIN_WIDTH and h > MIN_HEIGHT and MIN_RATIO < w / h < MAX_RATIO:
            possible_contours.append({'contour': contour, 'x': x, 'y': y, 'w': w, 'h': h,
                                      'cx': x + w/2, 'cy': y + h/2, 'idx': idx})

    def find_chars(contours_list):
        matched_result_idx = []
        for d1 in contours_list:
            matched = []
            for d2 in contours_list:
                if d1['idx'] == d2['idx']:
                    continue
                dx, dy = abs(d1['cx'] - d2['cx']), abs(d1['cy'] - d2['cy'])
                dist = np.linalg.norm([dx, dy])
                diag = np.sqrt(d1['w']**2 + d1['h']**2)
                if dx == 0:
                    angle_diff = 90
                else:
                    angle_diff = np.degrees(np.arctan(dy / dx))
                area_diff = abs(d1['w']*d1['h'] - d2['w']*d2['h']) / (d1['w']*d1['h'])
                if dist < diag*4 and angle_diff < 12 and area_diff < 0.25:
                    matched.append(d2['idx'])
            matched.append(d1['idx'])
            if len(matched) >= 4:
                matched_result_idx.append(matched)
                break
        return matched_result_idx

    result_idx = find_chars(possible_contours)
    matched_result = [[possible_contours[i] for i in idx_list] for idx_list in result_idx]

    PLATE_WIDTH_PADDING, PLATE_HEIGHT_PADDING = 1.3, 1.5
    plate_chars = ""
    for chars in matched_result:
        sorted_chars = sorted(chars, key=lambda x: x['cx'])
        cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
        cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2
        plate_w = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING
        plate_h = int(np.mean([d['h'] for d in sorted_chars]) * PLATE_HEIGHT_PADDING)
        angle = np.degrees(np.arcsin((sorted_chars[-1]['cy'] - sorted_chars[0]['cy']) /
                                     np.linalg.norm([sorted_chars[-1]['cx'] - sorted_chars[0]['cx'],
                                                     sorted_chars[-1]['cy'] - sorted_chars[0]['cy']])))
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        img_rotated = cv2.warpAffine(img_thresh, M, (width, height))
        img_cropped = cv2.getRectSubPix(img_rotated, (int(plate_w), int(plate_h)), (int(cx), int(cy)))

        if img_cropped is None or img_cropped.shape[0] == 0 or img_cropped.shape[1] == 0:
            continue

        img_cropped = cv2.resize(img_cropped, (0, 0), fx=1.6, fy=1.6)
        _, img_bin = cv2.threshold(img_cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_bin = cv2.GaussianBlur(img_bin, (3, 3), 0)
        img_bin = cv2.copyMakeBorder(img_bin, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        chars_raw = pytesseract.image_to_string(img_bin, lang='kor', config='--psm 7 --oem 3')
        chars_clean = ''.join([c for c in chars_raw if c.isdigit() or ('가' <= c <= '힣')])
        if any(c.isdigit() for c in chars_clean) and len(chars_clean) > len(plate_chars):
            plate_chars = chars_clean
            result_image = img_ori.copy()
            cv2.putText(result_image, plate_chars, (int(cx - plate_w/2), int(cy - plate_h/2) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imwrite(f"results/result_{plate_chars}.jpg", result_image)

    print(f"[PlateReader] Result: {plate_chars or 'No plate detected'}")
    return plate_chars 