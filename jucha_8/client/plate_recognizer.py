import cv2
import numpy as np
import pytesseract
import config
import os

class PlateRecognizer:
    def __init__(self):
        """
        Initialize plate recognizer with constants from 번호판.py
        """
        # Constants for character detection
        self.MIN_AREA = 210
        self.MIN_WIDTH, self.MIN_HEIGHT = 8, 16
        self.MIN_RATIO, self.MAX_RATIO = 0.4, 1.0
        
        # Constants for character matching
        self.MAX_DIAG_MULTIPLYER = 4
        self.MAX_ANGLE_DIFF = 12.0
        self.MAX_AREA_DIFF = 0.25
        self.MAX_WIDTH_DIFF = 0.6
        self.MAX_HEIGHT_DIFF = 0.15
        self.MIN_N_MATCHED = 4
        
        # Constants for plate detection
        self.PLATE_WIDTH_PADDING = 1.3
        self.PLATE_HEIGHT_PADDING = 1.5
        self.MIN_PLATE_RATIO = 3
        self.MAX_PLATE_RATIO = 10
        
        # Tesseract configuration for Korean license plates
        self.tesseract_config = '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주그느드르므브스으즈기니디리미비시이지'
        
        # Create output directory
        os.makedirs('images', exist_ok=True)
    
    def find_chars(self, contour_list):
        """
        Find characters that form a license plate
        """
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
                    angle_diff = np.degrees(np.arctan(dy/dx))
                area_diff = abs(d1['w']*d1['h']-d2['w']*d2['h'])/(d1['w']*d1['h'])
                width_diff = abs(d1['w']-d2['w'])/d1['w']
                height_diff = abs(d1['h']-d2['h'])/d1['h']
                
                if distance < diagonal_length1*self.MAX_DIAG_MULTIPLYER\
                and angle_diff < self.MAX_ANGLE_DIFF and area_diff < self.MAX_AREA_DIFF\
                and width_diff < self.MAX_WIDTH_DIFF and height_diff < self.MAX_HEIGHT_DIFF:
                    matched_contours_idx.append(d2['idx'])
            
            matched_contours_idx.append(d1['idx'])
            
            if len(matched_contours_idx) < self.MIN_N_MATCHED:
                continue
            
            matched_result_idx.append(matched_contours_idx)
            
            unmatched_contour_idx = []
            for d4 in contour_list:
                if d4['idx'] not in matched_contours_idx:
                    unmatched_contour_idx.append(d4['idx'])
            
            unmatched_contour = np.take(contour_list, unmatched_contour_idx)
            
            recursive_contour_list = self.find_chars(unmatched_contour)
            
            for idx in recursive_contour_list:
                matched_result_idx.append(idx)
            
            break
        
        return matched_result_idx
    
    def recognize_plate(self, frame, bbox):
        """
        Recognize license plate from the vehicle bounding box using 번호판.py algorithm
        """
        x1, y1, x2, y2 = bbox
        plate_img = frame[y1:y2, x1:x2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
        
        # Apply adaptive thresholding
        img_thresh = cv2.adaptiveThreshold(
            img_blurred,
            maxValue=255.0,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,
            blockSize=19,
            C=9
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            img_thresh,
            mode=cv2.RETR_LIST,
            method=cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Create dictionary of contours
        contours_dict = []
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            contours_dict.append({
                'contour': contour,
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'cx': x + (w/2),
                'cy': y + (h/2),
                'idx': i
            })
        
        # Filter possible contours
        possible_contours = []
        for d in contours_dict:
            area = d['w'] * d['h']
            ratio = d['w'] / d['h']
            
            if area > self.MIN_AREA\
            and d['w'] > self.MIN_WIDTH and d['h'] > self.MIN_HEIGHT\
            and self.MIN_RATIO < ratio < self.MAX_RATIO:
                possible_contours.append(d)
        
        # Find characters
        result_idx = self.find_chars(possible_contours)
        
        # Process found characters
        matched_result = []
        for idx_list in result_idx:
            matched_result.append(np.take(possible_contours, idx_list))
        
        # Extract and process plate images
        plate_imgs = []
        plate_infos = []
        
        for matched_chars in matched_result:
            sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])
            
            plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx'])/2
            plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy'])/2
            
            plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * self.PLATE_WIDTH_PADDING
            
            sum_height = 0
            for d in sorted_chars:
                sum_height += d['h']
            
            plate_height = int(sum_height / len(sorted_chars) * self.PLATE_HEIGHT_PADDING)
            
            triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
            triangle_hypotenus = np.linalg.norm(
                np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) - 
                np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
            )
            
            angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))
            
            rotation_matrix = cv2.getRotationMatrix2D(
                center=(plate_cx, plate_cy), 
                angle=angle, 
                scale=1.0
            )
            
            img_rotated = cv2.warpAffine(
                img_thresh, 
                M=rotation_matrix, 
                dsize=(plate_img.shape[1], plate_img.shape[0])
            )
            
            img_cropped = cv2.getRectSubPix(
                img_rotated,
                patchSize=(int(plate_width), int(plate_height)),
                center=(int(plate_cx), int(plate_cy))
            )
            
            plate_imgs.append(img_cropped)
            plate_infos.append({
                'x': int(plate_cx - plate_width / 2),
                'y': int(plate_cy - plate_height / 2),
                'w': int(plate_width),
                'h': int(plate_height)
            })
        
        # Process each plate image
        longest_idx, longest_text = -1, 0
        plate_chars = []
        best_plate_info = None
        
        for i, plate_img in enumerate(plate_imgs):
            plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)
            _, plate_img = cv2.threshold(
                plate_img, 
                thresh=0.0, 
                maxval=255.0, 
                type=cv2.THRESH_BINARY | cv2.THRESH_OTSU
            )
            
            # Perform OCR
            chars = pytesseract.image_to_string(
                plate_img, 
                config=self.tesseract_config
            ).strip()
            
            if len(chars) > longest_text:
                longest_idx = i
                longest_text = len(chars)
                plate_chars = chars
                best_plate_info = plate_infos[i]
        
        if plate_chars and best_plate_info:
            # Convert plate coordinates to original image coordinates
            plate_box = (
                x1 + best_plate_info['x'],
                y1 + best_plate_info['y'],
                x1 + best_plate_info['x'] + best_plate_info['w'],
                y1 + best_plate_info['y'] + best_plate_info['h']
            )
            return plate_chars, plate_img, plate_box
        return None, plate_img, None
    
    def draw_plate_info(self, frame, vehicle_bbox, plate_bbox, plate_text):
        """
        Draw plate information on the frame
        """
        # Draw vehicle bounding box
        x1, y1, x2, y2 = vehicle_bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Vehicle", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw plate bounding box and text
        if plate_bbox and plate_text:
            px1, py1, px2, py2 = plate_bbox
            cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 2)
            cv2.putText(frame, plate_text, (px1, py2+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return frame 