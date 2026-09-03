import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import filedialog
import time
import threading
import math 

import pytesseract
from textblob import TextBlob # USING TEXTBLOB NOW
from PIL import Image, ImageDraw, ImageFont

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

root = tk.Tk()
root.withdraw()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.85)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 

if not cap.isOpened():
    print("ERROR: Cannot open the webcam.")
    exit()

canvas = None           
text_canvas = None      
bg_image = None  
xp, yp = 0, 0  
draw_color = (255, 0, 0)
brush_thickness = 5
eraser_thickness = 50 
last_click_time = 0 
palette_open = False
is_ai_processing = False

frames_lost = 0 
MAX_FRAMES_LOST = 3 

tray_width, tray_height = 400, 60
tray_hsv = np.zeros((tray_height, tray_width, 3), dtype=np.uint8)
hues = np.linspace(0, 179, tray_width, dtype=np.uint8)
tray_hsv[:, :, 0] = hues        
tray_hsv[:, :, 1] = 255         
tray_hsv[:, :, 2] = 255         
color_tray_bgr = cv2.cvtColor(tray_hsv, cv2.COLOR_HSV2BGR)

def process_handwriting_with_ai(image_array):
    global is_ai_processing, canvas, text_canvas
    try:
        # 1. Isolate the drawn lines
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # --- NEW: Morphological Dilation ---
        # Thicken the lines to simulate bold, printed text for the OCR engine
        kernel = np.ones((5,5), np.uint8)
        thick_drawing = cv2.dilate(thresh, kernel, iterations=2)
        
        # Invert for Tesseract (It expects black text on a white background)
        thresh_inv = cv2.bitwise_not(thick_drawing)
        
        # --- NEW: Tesseract Whitelisting ---
        # Force the AI to ONLY guess letters, eliminating numbers or weird punctuation
        custom_config = r'-c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 7'
        raw_text = pytesseract.image_to_string(thresh_inv, config=custom_config).strip()
        
        if raw_text and len(raw_text) > 1:
            # Fix spelling
            corrected_text = str(TextBlob(raw_text).correct())
            render_text_with_pil(corrected_text)
            canvas = np.zeros_like(canvas)
            
    except Exception as e:
        print("Local AI Error:", e)
    is_ai_processing = False

def render_text_with_pil(text):
    global text_canvas
    img_pil = Image.fromarray(cv2.cvtColor(text_canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    try: 
        font = ImageFont.truetype("segoesc.ttf", 55) # Increased font size
    except: 
        font = ImageFont.load_default()

    y_offset = 150 
    margin = 50
    for line in text.split('\n'):
        # --- NEW: Professional Drop Shadow ---
        # Draw the shadow slightly offset
        draw.text((margin+4, y_offset+4), line, font=font, fill=(0, 0, 0))
        # Draw the main solid text on top
        draw.text((margin, y_offset), line, font=font, fill=(255, 255, 255))
        y_offset += 70 
        
    text_canvas = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
while True:
    success, frame = cap.read()
    if not success: break
        
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    if canvas is None or canvas.shape != (h, w, 3):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        text_canvas = np.zeros((h, w, 3), dtype=np.uint8)

    if bg_image is not None:
        display_frame = bg_image.copy()
        pip_w, pip_h = int(w * 0.25), int(h * 0.25)
        mini_cam = cv2.resize(frame, (pip_w, pip_h))
        display_frame[h-pip_h:h, w-pip_w:w] = mini_cam
    else:
        display_frame = frame.copy()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    
    drew_this_frame = False
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            
            x1, y1 = lmList[8][1], lmList[8][2] 
            x2, y2 = lmList[12][1], lmList[12][2] 
            x0, y0 = lmList[0][1], lmList[0][2] 
            
            # Euclidean distance logic
            dist_index = math.hypot(x1 - x0, y1 - y0)
            dist_middle = math.hypot(x2 - x0, y2 - y0)
            tip_distance = math.hypot(x2 - x1, y2 - y1)
            
            is_index_up = dist_index > 100 
            is_middle_up = dist_middle > 100

            # GESTURE 1: HOVER MODE 
            if is_index_up and is_middle_up and tip_distance < 60:
                xp, yp = 0, 0 
                frames_lost = 0
                
                if 20 < y1 < 70:
                    if 20 < x1 < 120:     palette_open = True
                    elif 140 < x1 < 240:  
                        draw_color = (0, 0, 0)
                        palette_open = False
                    elif 260 < x1 < 360:  
                        if time.time() - last_click_time > 2: 
                            filepath = filedialog.askopenfilename(title="Select Presentation", filetypes=[("Images", "*.jpg *.png")])
                            if filepath:
                                bg_image = cv2.resize(cv2.imread(filepath), (w, h))
                                canvas = np.zeros((h, w, 3), dtype=np.uint8)
                                text_canvas = np.zeros((h, w, 3), dtype=np.uint8)
                            last_click_time = time.time()
                    elif 380 < x1 < 480:  
                        canvas = np.zeros((h, w, 3), dtype=np.uint8)
                        text_canvas = np.zeros((h, w, 3), dtype=np.uint8)
                        bg_image = None 
                    elif 500 < x1 < 600:
                        if time.time() - last_click_time > 3 and not is_ai_processing:
                            is_ai_processing = True
                            threading.Thread(target=process_handwriting_with_ai, args=(canvas.copy(),)).start()
                            last_click_time = time.time()

                if palette_open:
                    if 80 <= y1 < 140 and 20 <= x1 < 20+tray_width:
                        tray_x, tray_y = x1 - 20, y1 - 80
                        picked_color = color_tray_bgr[tray_y, tray_x].tolist()
                        draw_color = tuple(int(c) for c in picked_color)
                    elif y1 > 170 or x1 > 450:
                        palette_open = False

                cv2.circle(display_frame, (x1, y1), 10, draw_color if draw_color != (0,0,0) else (255,255,255), 2)
                
            # GESTURE 2: DRAW MODE
            elif is_index_up and (not is_middle_up or tip_distance >= 60):
                palette_open = False 
                drew_this_frame = True
                frames_lost = 0 
                
                if y1 > 80:
                    if xp == 0 and yp == 0: xp, yp = x1, y1
                    
                    if draw_color == (0, 0, 0): 
                        cv2.line(canvas, (xp, yp), (x1, y1), draw_color, eraser_thickness)
                    else: 
                        cv2.line(canvas, (xp, yp), (x1, y1), draw_color, brush_thickness, cv2.LINE_AA)
                        
                    xp, yp = x1, y1
                cv2.circle(display_frame, (x1, y1), 10, draw_color if draw_color != (0,0,0) else (255,255,255), cv2.FILLED)
                
    if not drew_this_frame:
        frames_lost += 1
        if frames_lost > MAX_FRAMES_LOST:
            xp, yp = 0, 0 

    combined_overlays = cv2.add(canvas, text_canvas)

    imgGray = cv2.cvtColor(combined_overlays, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(imgGray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    
    display_frame_bg = cv2.bitwise_and(display_frame, display_frame, mask=mask_inv)
    merged_frame = cv2.add(display_frame_bg, combined_overlays)

    cv2.rectangle(merged_frame, (20, 20), (120, 70), draw_color if draw_color != (0,0,0) else (150,150,150), cv2.FILLED)
    cv2.putText(merged_frame, "COLOR", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    
    cv2.rectangle(merged_frame, (140, 20), (240, 70), (255, 255, 255), cv2.FILLED)
    cv2.putText(merged_frame, "ERASE", (155, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    
    cv2.rectangle(merged_frame, (260, 20), (360, 70), (200, 200, 200), cv2.FILLED)
    cv2.putText(merged_frame, "LOAD", (285, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    
    cv2.rectangle(merged_frame, (380, 20), (480, 70), (50, 50, 50), cv2.FILLED)
    cv2.putText(merged_frame, "CLEAR", (395, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    if is_ai_processing:
        cv2.rectangle(merged_frame, (500, 20), (660, 70), (0, 165, 255), cv2.FILLED)
        cv2.putText(merged_frame, "THINKING...", (510, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        cv2.rectangle(merged_frame, (500, 20), (600, 70), (0, 215, 255), cv2.FILLED)
        cv2.putText(merged_frame, "AI FIX", (515, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    if palette_open:
        merged_frame[80:140, 20:20+tray_width] = color_tray_bgr
        cv2.rectangle(merged_frame, (20, 80), (20+tray_width, 140), (255, 255, 255), 2)

    cv2.imshow("Air Canvas Pro - EdTech Edition", merged_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()