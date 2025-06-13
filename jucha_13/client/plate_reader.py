import cv2
import numpy as np
import pytesseract
import os
from datetime import datetime

cv2.setNumThreads(4)

def save_debug_image(img, name, debug_dir='debug_images'):
    """디버그 이미지를 저장하는 함수"""
    os.makedirs(debug_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{debug_dir}/{name}_{timestamp}.jpg"
    cv2.imwrite(filename, img)
    return filename

def read_plate(image_path, debug_mode=True):
    """번호판 인식 함수 (디버그 모드 포함)"""
    print(f"\n[DEBUG] Starting plate recognition for image: {image_path}")
    
    # 원본 이미지 로드
    img_ori = cv2.imread(image_path)
    if img_ori is None:
        print("[ERROR] Failed to load image")
        return None
    
    img_ori = cv2.resize(img_ori, (640, 480))
    if debug_mode:
        save_debug_image(img_ori, "1_original")

    # 그레이스케일 변환
    gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
    if debug_mode:
        save_debug_image(gray, "2_gray")

    # 블러 처리
    img_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    if debug_mode:
        save_debug_image(img_blurred, "3_blurred")

    # 이진화
    img_thresh = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 19, 9)
    if debug_mode:
        save_debug_image(img_thresh, "4_threshold")

    # 윤곽선 검출
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print(f"[DEBUG] Found {len(contours)} contours")

    # 윤곽선 시각화
    contour_img = img_ori.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    if debug_mode:
        save_debug_image(contour_img, "5_all_contours")

    # 가능한 번호판 윤곽선 필터링
    MIN_AREA, MIN_WIDTH, MIN_HEIGHT = 210, 8, 16
    MIN_RATIO, MAX_RATIO = 0.4, 1.0
    possible_contours = []
    
    for cnt, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        ratio = w / h
        
        if area > MIN_AREA and w > MIN_WIDTH and h > MIN_HEIGHT and MIN_RATIO < ratio < MAX_RATIO:
            possible_contours.append({
                'contour': contour,
                'x': x, 'y': y, 'w': w, 'h': h,
                'cx': x + w / 2, 'cy': y + h / 2,
                'idx': cnt
            })
    
    print(f"[DEBUG] Found {len(possible_contours)} possible plate contours")
    
    # 가능한 번호판 윤곽선 시각화
    possible_contour_img = img_ori.copy()
    for contour in possible_contours:
        x, y, w, h = contour['x'], contour['y'], contour['w'], contour['h']
        cv2.rectangle(possible_contour_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if debug_mode:
        save_debug_image(possible_contour_img, "6_possible_contours")

    def find_chars(contour_list):
        MAX_DIAG_MULTIPLYER = 4
        MAX_ANGLE_DIFF = 12.0
        MAX_AREA_DIFF = 0.25
        MAX_WIDTH_DIFF = 0.6
        MAX_HEIGHT_DIFF = 0.15
        MIN_N_MATCHED = 4
        matched_result_idx = []

        for d1 in contour_list:
            matched_contours_idx = []
            for d2 in contour_list:
                if d1['idx'] == d2['idx']: continue
                
                dx = abs(d1['cx'] - d2['cx'])
                dy = abs(d1['cy'] - d2['cy'])
                diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)
                distance = np.linalg.norm([d1['cx'] - d2['cx'], d1['cy'] - d2['cy']])
                angle_diff = 90 if dx == 0 else np.degrees(np.arctan(dy / dx))
                area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
                width_diff = abs(d1['w'] - d2['w']) / d1['w']
                height_diff = abs(d1['h'] - d2['h']) / d1['h']
                
                if (distance < diagonal_length1 * MAX_DIAG_MULTIPLYER and 
                    angle_diff < MAX_ANGLE_DIFF and 
                    area_diff < MAX_AREA_DIFF and 
                    width_diff < MAX_WIDTH_DIFF and 
                    height_diff < MAX_HEIGHT_DIFF):
                    if d2['idx'] < len(contour_list):
                        matched_contours_idx.append(d2['idx'])
            
            matched_contours_idx.append(d1['idx'])
            if len(matched_contours_idx) >= MIN_N_MATCHED:
                matched_result_idx.append(matched_contours_idx)
                break
        
        return matched_result_idx

    result_idx = find_chars(possible_contours)
    print(f"[DEBUG] Found {len(result_idx)} potential plate character groups")
    
    matched_result = [np.take(possible_contours, idx_list) for idx_list in result_idx]
    PLATE_WIDTH_PADDING, PLATE_HEIGHT_PADDING = 1.3, 1.5
    plate_imgs, plate_infos = [], []
    
    for i, matched_chars in enumerate(matched_result):
        sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])
        plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
        plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2
        plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING
        sum_height = sum(d['h'] for d in sorted_chars)
        plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)
        
        angle = np.degrees(np.arcsin((sorted_chars[-1]['cy'] - sorted_chars[0]['cy']) /
                                   np.linalg.norm([sorted_chars[0]['cx'] - sorted_chars[-1]['cx'],
                                                  sorted_chars[0]['cy'] - sorted_chars[-1]['cy']])))
        
        matrix = cv2.getRotationMatrix2D((plate_cx, plate_cy), angle, 1.0)
        img_rotated = cv2.warpAffine(img_thresh, matrix, (img_ori.shape[1], img_ori.shape[0]))
        img_cropped = cv2.getRectSubPix(img_rotated, (int(plate_width), int(plate_height)), 
                                      (int(plate_cx), int(plate_cy)))
        
        if debug_mode:
            save_debug_image(img_cropped, f"7_plate_candidate_{i}")
        
        plate_imgs.append(img_cropped)

    plate_chars = ""
    for i, plate_img in enumerate(plate_imgs):
        # 이미지 전처리
        plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)
        _, plate_img = cv2.threshold(plate_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        plate_img = cv2.GaussianBlur(plate_img, (3, 3), 0)
        plate_img = cv2.copyMakeBorder(plate_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        if debug_mode:
            save_debug_image(plate_img, f"8_processed_plate_{i}")
        
        # OCR 수행
        chars = pytesseract.image_to_string(plate_img, lang='kor', config='--psm 7 --oem 3')
        result = ''.join([c for c in chars if c.isdigit() or ('가' <= c <= '힣')])
        
        print(f"[DEBUG] Candidate {i} OCR result: {result}")
        
        if any(c.isdigit() for c in result) and len(result) > len(plate_chars):
            plate_chars = result

    if plate_chars:
        print(f"[SUCCESS] Final plate text: {plate_chars}")
    else:
        print("[WARNING] No valid plate text found")
    
    return plate_chars 