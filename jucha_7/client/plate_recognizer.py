import cv2
import numpy as np
import os
from datetime import datetime
import config
import pytesseract

class PlateRecognizer:
    def __init__(self):
        # Create plates directory if it doesn't exist
        os.makedirs('images/plates', exist_ok=True)
        
        # Constants for plate detection
        self.MIN_AREA = 210
        self.MIN_WIDTH, self.MIN_HEIGHT = 8, 16
        self.MIN_RATIO, self.MAX_RATIO = 0.4, 1.0
        self.MAX_DIAG_MULTIPLYER = 4
        self.MAX_ANGLE_DIFF = 12.0
        self.MAX_AREA_DIFF = 0.25
        self.MAX_WIDTH_DIFF = 0.6
        self.MAX_HEIGHT_DIFF = 0.15
        self.MIN_N_MATCHED = 4
        self.PLATE_WIDTH_PADDING = 1.3
        self.PLATE_HEIGHT_PADDING = 1.5
        self.MIN_PLATE_RATIO = 3
        self.MAX_PLATE_RATIO = 10
        
    def preprocess_image(self, img):
        """
        Preprocess the image for plate detection
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
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
        
        return img_thresh
    
    def find_chars(self, contour_list):
        """
        Find characters in the image using contour analysis
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
                
                if (distance < diagonal_length1*self.MAX_DIAG_MULTIPLYER and
                    angle_diff < self.MAX_ANGLE_DIFF and
                    area_diff < self.MAX_AREA_DIFF and
                    width_diff < self.MAX_WIDTH_DIFF and
                    height_diff < self.MAX_HEIGHT_DIFF):
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
    
    def recognize_plate(self, frame, vehicle_bbox):
        """
        Recognize the license plate from the vehicle bounding box
        Returns: recognized plate text and plate image
        """
        x1, y1, x2, y2 = vehicle_bbox
        
        # Extract vehicle region
        vehicle_img = frame[y1:y2, x1:x2]
        
        # Preprocess image
        img_thresh = self.preprocess_image(vehicle_img)
        
        # Find contours
        contours, _ = cv2.findContours(
            img_thresh,
            mode=cv2.RETR_LIST,
            method=cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Create contour dictionary
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
            
            if (area > self.MIN_AREA and
                d['w'] > self.MIN_WIDTH and d['h'] > self.MIN_HEIGHT and
                self.MIN_RATIO < ratio < self.MAX_RATIO):
                possible_contours.append(d)
        
        # Find characters
        result_idx = self.find_chars(possible_contours)
        
        # Process found characters
        plate_text = ""
        plate_img = None
        
        if result_idx:
            matched_result = []
            for idx_list in result_idx:
                matched_result.append(np.take(possible_contours, idx_list))
            
            # Get the first matched result
            chars = matched_result[0]
            sorted_chars = sorted(chars, key=lambda x: x['cx'])
            
            # Calculate plate position and size
            plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
            plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2
            plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * self.PLATE_WIDTH_PADDING
            
            sum_height = sum(d['h'] for d in sorted_chars)
            plate_height = int(sum_height / len(sorted_chars) * self.PLATE_HEIGHT_PADDING)
            
            # Calculate rotation angle
            triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
            triangle_hypotenus = np.linalg.norm(
                np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) -
                np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
            )
            angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))
            
            # Rotate and crop plate image
            rotation_matrix = cv2.getRotationMatrix2D(
                center=(plate_cx, plate_cy),
                angle=angle,
                scale=1.0
            )
            img_rotated = cv2.warpAffine(
                img_thresh,
                M=rotation_matrix,
                dsize=(vehicle_img.shape[1], vehicle_img.shape[0])
            )
            
            plate_img = cv2.getRectSubPix(
                img_rotated,
                patchSize=(int(plate_width), int(plate_height)),
                center=(int(plate_cx), int(plate_cy))
            )
            
            # Resize and threshold plate image
            plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)
            _, plate_img = cv2.threshold(
                plate_img,
                thresh=0.0,
                maxval=255.0,
                type=cv2.THRESH_BINARY | cv2.THRESH_OTSU
            )
            
            # Recognize text using Tesseract
            plate_text = pytesseract.image_to_string(
                plate_img,
                lang='kor+eng',
                config='--psm 7 --oem 0'
            ).strip()
        
        # Save plate image if enabled
        if config.SAVE_IMAGES and plate_img is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f'images/plates/{timestamp}.jpg', plate_img)
        
        return plate_text, plate_img
    
    def draw_plate_info(self, frame, vehicle_bbox, plate_text):
        """
        Draw the recognized plate information on the frame
        """
        x1, y1, x2, y2 = vehicle_bbox
        
        # Draw plate text
        cv2.putText(frame, plate_text, (x1, y2+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame 