import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import filedialog
import time
import threading
import math 
import os
import subprocess
import wave
from datetime import datetime

import pytesseract
import sounddevice as sd
import imageio_ffmpeg
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
white_canvas = False
xp, yp = 0, 0  
shape_start = None
draw_color = (255, 0, 0)
brush_thickness = 5
brush_presets = [3, 5, 8, 12]
brush_preset_index = brush_presets.index(brush_thickness)
eraser_thickness = 50 
last_click_time = 0 
last_pen_toggle_time = 0
last_undo_time = 0
last_size_change_time = 0
last_shape_action_time = 0
last_sheet_toggle_time = 0
last_record_toggle_time = 0
record_button_armed = True
pen_enabled = True
canvas_history = []
video_writer = None
audio_stream = None
audio_frames = []
is_recording = False
recording_path = None
video_temp_path = None
audio_temp_path = None
shape_mode = "freehand"
shape_menu_open = False
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

    margin = 40
    max_text_width = int(img_pil.width * 0.65)
    y_offset = 100
    for line in text.split('\n'):
        font_size = 42
        try:
            font = ImageFont.truetype("segoesc.ttf", font_size)
            text_width = draw.textbbox((0, 0), line, font=font)[2]
            if text_width > max_text_width:
                font_size = max(24, int(font_size * max_text_width / text_width))
                font = ImageFont.truetype("segoesc.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()

        draw.text((margin+4, y_offset+4), line, font=font, fill=(0, 0, 0))
        draw.text((margin, y_offset), line, font=font, fill=(255, 255, 255))
        y_offset += font_size + 12
        
    text_canvas = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def draw_shape(image, start_point, end_point):
    start_x, start_y = start_point
    end_x, end_y = end_point
    color = (255, 255, 255) if draw_color == (0, 0, 0) else draw_color
    thickness = eraser_thickness if draw_color == (0, 0, 0) else brush_thickness

    if shape_mode == "square":
        side = max(abs(end_x - start_x), abs(end_y - start_y))
        corner_x = start_x + side if end_x >= start_x else start_x - side
        corner_y = start_y + side if end_y >= start_y else start_y - side
        cv2.rectangle(image, (start_x, start_y), (corner_x, corner_y), color, thickness)
    elif shape_mode == "rectangle":
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y), color, thickness)
    elif shape_mode == "circle":
        radius = int(math.hypot(end_x - start_x, end_y - start_y))
        cv2.circle(image, (start_x, start_y), radius, color, thickness)
    elif shape_mode == "triangle":
        points = np.array([
            [start_x, end_y],
            [end_x, end_y],
            [(start_x + end_x) // 2, start_y],
        ], dtype=np.int32)
        cv2.polylines(image, [points], True, color, thickness)
    elif shape_mode == "line":
        cv2.line(image, (start_x, start_y), (end_x, end_y), color, thickness, cv2.LINE_AA)

def audio_callback(indata, frames, callback_time, status):
    if status:
        print(f"Microphone status: {status}")
    audio_frames.append(indata.copy())

def start_recording(frame_size):
    global video_writer, audio_stream, audio_frames, is_recording
    global recording_path, video_temp_path, audio_temp_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    recording_path = f"air_canvas_{timestamp}.mp4"
    video_temp_path = f"air_canvas_{timestamp}_video.mp4"
    audio_temp_path = f"air_canvas_{timestamp}_audio.wav"
    video_writer = cv2.VideoWriter(
        video_temp_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        20.0,
        frame_size,
    )
    if not video_writer.isOpened():
        video_writer.release()
        video_writer = None
        return False

    audio_frames = []
    try:
        audio_stream = sd.InputStream(
            samplerate=44100,
            channels=1,
            dtype="int16",
            callback=audio_callback,
        )
        audio_stream.start()
    except Exception as error:
        print(f"Microphone unavailable: {error}")
        video_writer.release()
        video_writer = None
        audio_stream = None
        return False

    is_recording = True
    return True

def stop_recording():
    global video_writer, audio_stream, audio_frames, is_recording
    global recording_path, video_temp_path, audio_temp_path

    is_recording = False
    if audio_stream is not None:
        audio_stream.stop()
        audio_stream.close()
        audio_stream = None
    if video_writer is not None:
        video_writer.release()
        video_writer = None

    if not audio_frames or not video_temp_path:
        return

    audio_data = np.concatenate(audio_frames, axis=0)
    with wave.open(audio_temp_path, "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(44100)
        audio_file.writeframes(audio_data.tobytes())

    try:
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run(
            [ffmpeg_path, "-y", "-i", video_temp_path, "-i", audio_temp_path,
             "-c:v", "copy", "-c:a", "aac", "-shortest", recording_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.remove(video_temp_path)
        os.remove(audio_temp_path)
        print(f"Recording saved: {recording_path}")
    except Exception as error:
        print(f"Audio merge failed; video remains at {video_temp_path}: {error}")

while True:
    success, frame = cap.read()
    if not success: break
        
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    if canvas is None or canvas.shape != (h, w, 3):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        text_canvas = np.zeros((h, w, 3), dtype=np.uint8)

    if white_canvas or bg_image is not None:
        if white_canvas:
            display_frame = np.full((h, w, 3), 255, dtype=np.uint8)
        else:
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
            record_button_hover = 160 < x1 < 230 and 60 < y1 < 105
            if not record_button_hover:
                record_button_armed = True

            # GESTURE 1: HOVER MODE 
            if is_index_up and is_middle_up and tip_distance < 60:
                xp, yp = 0, 0 
                shape_start = None
                frames_lost = 0
                
                if 10 < y1 < 55:
                    if 10 < x1 < 80:     palette_open = True
                    elif 85 < x1 < 155:  
                        draw_color = (0, 0, 0)
                        palette_open = False
                    elif 160 < x1 < 230:  
                        if time.time() - last_click_time > 2: 
                            filepath = filedialog.askopenfilename(title="Select Presentation", filetypes=[("Images", "*.jpg *.png")])
                            if filepath:
                                bg_image = cv2.resize(cv2.imread(filepath), (w, h))
                                white_canvas = False
                                canvas = np.zeros((h, w, 3), dtype=np.uint8)
                                text_canvas = np.zeros((h, w, 3), dtype=np.uint8)
                            last_click_time = time.time()
                    elif 230 < x1 < 300:  
                        canvas = np.zeros((h, w, 3), dtype=np.uint8)
                        text_canvas = np.zeros((h, w, 3), dtype=np.uint8)
                        bg_image = None 
                    elif 310 < x1 < 380:
                        if time.time() - last_click_time > 3 and not is_ai_processing:
                            is_ai_processing = True
                            threading.Thread(target=process_handwriting_with_ai, args=(canvas.copy(),)).start()
                            last_click_time = time.time()
                    elif 385 < x1 < 455:
                        if time.time() - last_pen_toggle_time > 1:
                            pen_enabled = not pen_enabled
                            last_pen_toggle_time = time.time()
                            xp, yp = 0, 0
                    elif 460 < x1 < 530:
                        if time.time() - last_undo_time > 1 and canvas_history:
                            canvas = canvas_history.pop()
                            last_undo_time = time.time()
                            xp, yp = 0, 0
                    elif 535 < x1 < 605:
                        if time.time() - last_size_change_time > 1:
                            brush_preset_index = (brush_preset_index + 1) % len(brush_presets)
                            brush_thickness = brush_presets[brush_preset_index]
                            last_size_change_time = time.time()
                elif 85 < x1 < 155 and 60 < y1 < 105:
                    if time.time() - last_sheet_toggle_time > 1:
                        white_canvas = not white_canvas
                        if white_canvas:
                            bg_image = None
                        last_sheet_toggle_time = time.time()
                        xp, yp = 0, 0
                        shape_start = None
                elif 160 < x1 < 230 and 60 < y1 < 105:
                    if record_button_armed and time.time() - last_record_toggle_time > 1:
                        if not is_recording:
                            start_recording((w, h))
                        else:
                            stop_recording()
                        last_record_toggle_time = time.time()
                        record_button_armed = False
                        xp, yp = 0, 0
                elif 10 < x1 < 80 and 60 < y1 < 105:
                    if time.time() - last_shape_action_time > 1:
                        shape_menu_open = not shape_menu_open
                        last_shape_action_time = time.time()
                        xp, yp = 0, 0
                        shape_start = None
                elif shape_menu_open and 110 < y1 < 155:
                    shape_ranges = [(10, 80, "square"), (85, 155, "circle"), (160, 230, "triangle"), (235, 305, "rectangle"), (310, 380, "line"), (385, 455, "freehand")]
                    for range_start, range_end, selected_shape in shape_ranges:
                        if range_start < x1 < range_end:
                            if time.time() - last_shape_action_time > 1:
                                shape_mode = selected_shape
                                shape_menu_open = False
                                last_shape_action_time = time.time()
                                xp, yp = 0, 0
                                shape_start = None
                            break

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
                drew_this_frame = pen_enabled
                frames_lost = 0 
                
                if pen_enabled and y1 > 80:
                    if xp == 0 and yp == 0:
                        if len(canvas_history) >= 10:
                            canvas_history.pop(0)
                        canvas_history.append(canvas.copy())
                        shape_start = (x1, y1)
                        xp, yp = x1, y1

                    if shape_mode == "freehand":
                        if draw_color == (0, 0, 0):
                            cv2.line(canvas, (xp, yp), (x1, y1), draw_color, eraser_thickness)
                        else:
                            cv2.line(canvas, (xp, yp), (x1, y1), draw_color, brush_thickness, cv2.LINE_AA)
                    else:
                        canvas = canvas_history[-1].copy()
                        draw_shape(canvas, shape_start, (x1, y1))
                        
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

    cv2.rectangle(merged_frame, (10, 10), (80, 55), draw_color if draw_color != (0,0,0) else (150,150,150), cv2.FILLED)
    cv2.putText(merged_frame, "COLOR", (14, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1)
    
    cv2.rectangle(merged_frame, (85, 10), (155, 55), (255, 255, 255), cv2.FILLED)
    cv2.putText(merged_frame, "ERASE", (91, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,0,0), 1)
    
    cv2.rectangle(merged_frame, (160, 10), (230, 55), (200, 200, 200), cv2.FILLED)
    cv2.putText(merged_frame, "LOAD", (177, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,0,0), 1)
    
    cv2.rectangle(merged_frame, (230, 10), (300, 55), (50, 50, 50), cv2.FILLED)
    cv2.putText(merged_frame, "CLEAR", (238, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1)

    if is_ai_processing:
        cv2.rectangle(merged_frame, (310, 10), (380, 55), (0, 165, 255), cv2.FILLED)
        cv2.putText(merged_frame, "WAIT", (328, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    else:
        cv2.rectangle(merged_frame, (310, 10), (380, 55), (0, 215, 255), cv2.FILLED)
        cv2.putText(merged_frame, "AI FIX", (321, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)

    pen_button_color = (0, 180, 0) if pen_enabled else (0, 0, 180)
    pen_button_label = "PEN ON" if pen_enabled else "PEN OFF"
    cv2.rectangle(merged_frame, (385, 10), (455, 55), pen_button_color, cv2.FILLED)
    cv2.putText(merged_frame, pen_button_label, (389, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1)

    cv2.rectangle(merged_frame, (460, 10), (530, 55), (80, 80, 80), cv2.FILLED)
    cv2.putText(merged_frame, "UNDO", (475, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

    cv2.rectangle(merged_frame, (535, 10), (605, 55), (100, 100, 100), cv2.FILLED)
    cv2.putText(merged_frame, f"SZ: {brush_thickness}", (544, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1)

    cv2.rectangle(merged_frame, (10, 60), (80, 105), (70, 70, 70), cv2.FILLED)
    shape_label = "FREE" if shape_mode == "freehand" else shape_mode.upper()[:5]
    cv2.putText(merged_frame, shape_label, (18, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

    sheet_button_color = (0, 150, 0) if white_canvas else (70, 70, 70)
    cv2.rectangle(merged_frame, (85, 60), (155, 105), sheet_button_color, cv2.FILLED)
    cv2.putText(merged_frame, "SHEET", (94, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)

    record_button_color = (0, 0, 220) if is_recording else (70, 70, 70)
    record_label = "STOP" if is_recording else "REC"
    cv2.rectangle(merged_frame, (160, 60), (230, 105), record_button_color, cv2.FILLED)
    cv2.putText(merged_frame, record_label, (177, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

    if shape_menu_open:
        shape_tiles = [(10, "SQ"), (85, "CIR"), (160, "TRI"), (235, "RECT"), (310, "LINE"), (385, "FREE")]
        for tile_x, tile_label in shape_tiles:
            cv2.rectangle(merged_frame, (tile_x, 110), (tile_x + 70, 155), (80, 80, 80), cv2.FILLED)
            cv2.putText(merged_frame, tile_label, (tile_x + 8, 137), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    if palette_open:
        merged_frame[80:140, 20:20+tray_width] = color_tray_bgr
        cv2.rectangle(merged_frame, (20, 80), (20+tray_width, 140), (255, 255, 255), 2)

    cv2.imshow("Air Canvas Pro - EdTech Edition", merged_frame)

    if is_recording and video_writer is not None:
        video_writer.write(merged_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if is_recording:
    stop_recording()
cap.release()
cv2.destroyAllWindows()