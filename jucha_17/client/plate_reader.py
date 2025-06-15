import cv2
import numpy as np
import pytesseract
import os
from send_to_server import send_plate_text
from time import sleep

# 디버그 이미지 저장 폴더
os.makedirs('results', exist_ok=True)

# 번호판 후보 조건
MIN_AREA = 210
MIN_WIDTH, MIN_HEIGHT = 8, 16
MIN_RATIO, MAX_RATIO = 0.4, 1.0
MAX_DIAG_MULTIPLYER = 4
MAX_ANGLE_DIFF = 12.0
MAX_AREA_DIFF = 0.25
MAX_WIDTH_DIFF = 0.6
MAX_HEIGHT_DIFF = 0.15
MIN_N_MATCHED = 7
PLATE_WIDTH_PADDING = 1.55
PLATE_HEIGHT_PADDING = 1.5
MIN_PLATE_RATIO = 3
MAX_PLATE_RATIO = 10

def find_chars(contour_list):
    matched_result_idx = []

    for d1 in contour_list:
        matched_contours_idx = []
        for d2 in contour_list:
            if d1['idx'] == d2['idx']:
                continue
            dx = abs(d1['cx'] - d2['cx'])
            dy = abs(d1['cy'] - d2['cy'])
            diagonal_length1 = np.sqrt(d1['w']**2 + d1['h']**2)
            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
            if dx == 0:
                angle_diff = 90
            else:
                angle_diff = np.degrees(np.arctan(dy / dx))
            area_diff = abs(d1['w']*d1['h'] - d2['w']*d2['h']) / (d1['w']*d1['h'])
            width_diff = abs(d1['w'] - d2['w']) / d1['w']
            height_diff = abs(d1['h'] - d2['h']) / d1['h']

            if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER and angle_diff < MAX_ANGLE_DIFF and \
               area_diff < MAX_AREA_DIFF and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                matched_contours_idx.append(d2['idx'])

        matched_contours_idx.append(d1['idx'])

        if len(matched_contours_idx) < MIN_N_MATCHED:
            continue

        matched_result_idx.append(matched_contours_idx)

        unmatched_contour_idx = [d['idx'] for d in contour_list if d['idx'] not in matched_contours_idx]
        unmatched_contour = [d for d in contour_list if d['idx'] in unmatched_contour_idx]

        recursive = find_chars(unmatched_contour)
        matched_result_idx.extend(recursive)
        break

    return matched_result_idx

def read_plate(img_path):
    print(f"[PlateReader] Reading plate from {img_path}")
    img_ori = cv2.imread(img_path)
    if img_ori is None:
        print("[PlateReader] Image not found.")
        return ""

    height, width, _ = img_ori.shape
    gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
    img_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 19, 9)
    cv2.imwrite('results/1_thresh.jpg', img_thresh)

    contours, _ = cv2.findContours(img_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours_dict = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        contours_dict.append({'contour': contour, 'x': x, 'y': y, 'w': w, 'h': h,
                              'cx': x + w/2, 'cy': y + h/2})

    possible_contours = []
    cnt = 0
    for d in contours_dict:
        area = d['w'] * d['h']
        ratio = d['w'] / d['h']
        if area > MIN_AREA and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT and MIN_RATIO < ratio < MAX_RATIO:
            d['idx'] = cnt
            cnt += 1
            possible_contours.append(d)

    result_idx = find_chars(possible_contours)
    matched_result = []
    for idx_list in result_idx:
        group = [possible_contours[i] for i in idx_list if i < len(possible_contours)]
        if group:
            matched_result.append(group)

    for i, matched_chars in enumerate(matched_result):
        sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])
        plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
        plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2
        plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING
        sum_height = sum(d['h'] for d in sorted_chars)
        plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)
        triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
        triangle_hypotenus = np.linalg.norm(
            np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) -
            np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']]))
        angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))
        rotation_matrix = cv2.getRotationMatrix2D((plate_cx, plate_cy), angle, 1.0)
        img_rotated = cv2.warpAffine(img_thresh, rotation_matrix, (width, height))
        img_cropped = cv2.getRectSubPix(img_rotated, (int(plate_width), int(plate_height)), (plate_cx, plate_cy))
        cv2.imwrite(f'results/2_plate_candidate_{i}.jpg', img_cropped)

        img_result = cv2.resize(img_cropped, dsize=(0, 0), fx=1.6, fy=1.6)
        _, img_result = cv2.threshold(img_result, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img_result = cv2.copyMakeBorder(img_result, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        cv2.imwrite(f'results/3_ocr_input_{i}.jpg', img_result)

        chars = pytesseract.image_to_string(img_result, lang='kor', config='--psm 7 --oem 3')
        result_chars = ''
        has_digit = False
        for c in chars:
            if ord('가') <= ord(c) <= ord('힣') or c.isdigit():
                if c.isdigit():
                    has_digit = True
                result_chars += c

        if has_digit and len(result_chars) >= 4:
            print(f"[PlateReader] Plate detected: {result_chars}")
            send_plate_text(result_chars)
            sleep(10)  # 10초 대기
            return result_chars

    print("[PlateReader] No plate detected.")
    return ""
