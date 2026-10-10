import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog
import time
import threading
import math
import os
import subprocess
import wave
import textwrap
from datetime import datetime

import pytesseract
import sounddevice as sd
import imageio_ffmpeg
from textblob import TextBlob
from PIL import Image, ImageDraw, ImageFont, ImageTk

# Set path for Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- Optional: TrOCR (transformer OCR trained on real handwriting, incl.
# cursive) is a much better fit than Tesseract for air-written text. It's
# optional so the app still runs with just Tesseract if these aren't
# installed. To enable it:
#     pip install torch transformers
# (use the correct torch command for your OS/GPU from https://pytorch.org
#  if you want GPU acceleration — plain CPU install works fine too, just
#  slower on the very first run while it downloads the model, ~1.3 GB,
#  which then caches locally for every run after that).
try:
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    TROCR_LIBS_AVAILABLE = True
except Exception as _trocr_import_error:  # ImportError, or OSError from a Windows DLL clash
    print("TrOCR libraries could not be imported:", _trocr_import_error)
    TROCR_LIBS_AVAILABLE = False


class ModernAirCanvas:
    def __init__(self):
        # --- 1. CustomTkinter Window Setup ---
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("Air Canvas Pro - EdTech Edition")
        self.root.geometry("1100x650")
        self.root.minsize(900, 600)
        self.root.state("zoomed")

        # --- Layout Frames ---
        self.main_container = ctk.CTkFrame(self.root, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=15, pady=15)

        # Video feed container
        self.video_frame = ctk.CTkFrame(self.main_container, corner_radius=15, fg_color="#1E1E1E")
        self.video_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.video_label = tk.Label(self.video_frame, bg="#1E1E1E")
        self.video_label.pack(expand=True, fill="both", padx=10, pady=10)

        # --- 2. OpenCV & Mediapipe Setup ---
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("ERROR: Cannot open the webcam.")
            exit()

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.85)

        # Get camera resolution
        success, frame = self.cap.read()
        self.h, self.w, _ = frame.shape

        # --- 3. App State Variables ---
        self.canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.text_canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.bg_image = None
        self.white_canvas = False
        self.xp, self.yp = 0, 0
        self.shape_start = None

        # Lock to protect canvas/text_canvas since the AI thread writes to them
        # while the main video loop reads/writes them ~66 times a second.
        self.canvas_lock = threading.Lock()

        # Modern Palette Colors (RGB for PIL)
        self.draw_color_rgb = (97, 175, 239)  # Soft Blue
        self.color_menu_open = False
        self.color_button_armed = True
        self.color_options = [
            ("BLUE", (97, 175, 239)),
            ("GREEN", (46, 160, 67)),
            ("RED", (248, 81, 73)),
            ("YELLOW", (210, 153, 34)),
            ("PURPLE", (180, 130, 220)),
            ("WHITE", (245, 245, 245)),
        ]
        self.brush_thickness = 5
        self.eraser_thickness = 50
        self.shape_mode = "freehand"

        self.pen_enabled = True
        self.canvas_history = []

        # Timers to prevent double-clicks
        self.last_action_time = 0
        self.sheet_button_armed = True
        self.frames_lost = 0

        # Recording State
        self.is_recording = False
        self.video_writer = None
        self.audio_stream = None
        self.audio_frames = []
        self.is_ai_processing = False
        self.shape_menu_open = False

        # TrOCR model is loaded lazily (on first AI FIX use) since it's slow
        # to load and would delay app startup otherwise.
        self.trocr_processor = None
        self.trocr_model = None
        self.trocr_load_attempted = False

        # Side Control Panel
        self.side_panel = ctk.CTkFrame(self.main_container, width=250, corner_radius=15)
        self.side_panel.pack(side="right", fill="y")
        self.build_side_panel()

        # Build Air UI Boundaries
        self.setup_air_buttons()

        # Start Main Loop
        self.update_frame()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()

    def build_side_panel(self):
        """Builds the physical CustomTkinter UI panel."""
        ctk.CTkLabel(self.side_panel, text="Controls", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(20, 10))

        self.pen_btn = ctk.CTkButton(self.side_panel, text="Pen: ON", fg_color="#2EA043",
                                      hover_color="#237A33", command=self.toggle_pen)
        self.pen_btn.pack(pady=10, padx=20, fill="x")

        ctk.CTkButton(self.side_panel, text="Clear Canvas", fg_color="#D29922",
                      hover_color="#A37516", command=self.clear_canvas).pack(pady=10, padx=20, fill="x")
        ctk.CTkButton(self.side_panel, text="Undo Last", command=self.undo).pack(pady=10, padx=20, fill="x")

        ctk.CTkLabel(self.side_panel, text="Brush Size").pack(pady=(20, 0))
        self.size_slider = ctk.CTkSlider(self.side_panel, from_=2, to=20, command=self.update_brush_size)
        self.size_slider.set(self.brush_thickness)
        self.size_slider.pack(pady=10, padx=20, fill="x")

        ctk.CTkButton(self.side_panel, text="Load Presentation", fg_color="#444C56",
                      command=self.load_bg).pack(pady=(20, 10), padx=20, fill="x")

        # --- NEW: AI recognized-text panel ---
        # This is the readable "clean" output of the AI FIX feature. The tiny
        # text rendered on the video canvas is easy to miss on a projector,
        # so we also surface it here, big and scrollable, for students.
        ctk.CTkLabel(self.side_panel, text="AI Recognized Text",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10, 5))
        self.ai_text_box = ctk.CTkTextbox(self.side_panel, height=160, wrap="word",
                                           font=ctk.CTkFont(size=15))
        self.ai_text_box.pack(pady=(0, 10), padx=20, fill="x")
        self.ai_text_box.insert("1.0", "Write on the canvas, then hover over AI FIX.")
        self.ai_text_box.configure(state="disabled")

        self.rec_btn = ctk.CTkButton(self.side_panel, text="Start Recording", fg_color="#F85149",
                                      hover_color="#B53932", command=self.toggle_recording)
        self.rec_btn.pack(side="bottom", pady=20, padx=20, fill="x")

    def setup_air_buttons(self):
        """Defines the hitboxes for the on-screen glassmorphic UI."""
        self.air_btns = {
            "COLOR": (15, 15, 85, 50),
            "ERASE": (95, 15, 165, 50),
            "CLEAR": (175, 15, 245, 50),
            "AI_FIX": (255, 15, 325, 50),
            "PEN": (335, 15, 405, 50),
            "UNDO": (415, 15, 485, 50),
            "SHEET": (495, 15, 565, 50),

            "SHAPE": (15, 60, 85, 95),
            "REC": (495, 60, 565, 95)
        }

        self.shape_btns = {
            "freehand": (95, 60, 165, 95),
            "line": (175, 60, 245, 95),
            "rectangle": (255, 60, 325, 95),
            "circle": (335, 60, 405, 95),
            "square": (415, 60, 485, 95),
            "triangle": (495, 60, 565, 95)
        }
        self.color_btns = {
            name: (95 + index * 80, 60, 165 + index * 80, 95)
            for index, (name, _) in enumerate(self.color_options)
        }

    # --- Interaction Logic Methods ---
    def toggle_pen(self):
        self.pen_enabled = not self.pen_enabled
        self.xp, self.yp = 0, 0
        self.pen_btn.configure(text=f"Pen: {'ON' if self.pen_enabled else 'OFF'}",
                                fg_color="#2EA043" if self.pen_enabled else "#444C56")

    def update_brush_size(self, val):
        self.brush_thickness = int(val)

    def clear_canvas(self):
        with self.canvas_lock:
            self.canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
            self.text_canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.bg_image = None
        self.xp, self.yp = 0, 0
        self.reset_ai_textbox()

    def undo(self):
        if self.canvas_history:
            with self.canvas_lock:
                self.canvas = self.canvas_history.pop()
        self.xp, self.yp = 0, 0

    def load_bg(self):
        filepath = filedialog.askopenfilename(title="Select Presentation", filetypes=[("Images", "*.jpg *.png")])
        if filepath:
            image = cv2.imread(filepath)
            if image is None:
                self.show_ai_status("Could not open the selected image.")
                return

            self.clear_canvas()
            self.bg_image = cv2.resize(image, (self.w, self.h))
            self.white_canvas = False

    def toggle_recording(self):
        if not self.is_recording:
            success = self.start_recording((self.w, self.h))
            if success:
                self.rec_btn.configure(text="Stop Recording", fg_color="#B53932")
        else:
            self.stop_recording()
            self.rec_btn.configure(text="Start Recording", fg_color="#F85149")

    # --- Glassmorphism UI Drawing ---
    def draw_glassmorphic_ui(self, overlay_img):
        """Draws modern, semi-transparent rounded buttons using PIL."""
        draw = ImageDraw.Draw(overlay_img)

        try:
            font = ImageFont.truetype("arial.ttf", 13)
        except Exception:
            font = ImageFont.load_default()

        dock_height = 105 if self.shape_menu_open or self.color_menu_open else 60
        draw.rounded_rectangle([(5, 5), (575, dock_height)], radius=15, fill=(30, 33, 40, 180))

        def draw_btn(box, text, active=False, accent_color=(97, 175, 239)):
            bg_color = accent_color + (220,) if active else (60, 64, 72, 180)
            draw.rounded_rectangle([box[0], box[1], box[2], box[3]], radius=8, fill=bg_color)

            try:
                text_bbox = draw.textbbox((0, 0), text, font=font)
            except Exception:
                text_bbox = [0, 0, len(text) * 6, 10]
            tw, th = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            cx, cy = box[0] + (box[2] - box[0]) // 2, box[1] + (box[3] - box[1]) // 2

            draw.text((cx - tw // 2, cy - th // 2 - 2), text, font=font, fill=(255, 255, 255, 255))

        draw_btn(self.air_btns["COLOR"], "COLOR", True, self.draw_color_rgb)
        draw_btn(self.air_btns["ERASE"], "ERASE", self.draw_color_rgb == (0, 0, 0), (200, 100, 100))
        draw_btn(self.air_btns["CLEAR"], "CLEAR")
        draw_btn(self.air_btns["AI_FIX"], "WAIT AI" if self.is_ai_processing else "AI FIX",
                 self.is_ai_processing, (210, 153, 34))
        draw_btn(self.air_btns["PEN"], "PEN ON" if self.pen_enabled else "PEN OFF", self.pen_enabled, (46, 160, 67))
        draw_btn(self.air_btns["UNDO"], "UNDO")
        draw_btn(self.air_btns["SHEET"], "SHEET", self.white_canvas, (100, 100, 100))

        if self.shape_menu_open:
            draw_btn(self.air_btns["SHAPE"], "CLOSE", True, (150, 150, 150))
            for mode, box in self.shape_btns.items():
                draw_btn(box, mode[:4].upper(), self.shape_mode == mode)
        elif self.color_menu_open:
            draw_btn(self.air_btns["SHAPE"], "SHAPES")
            for name, color in self.color_options:
                box = self.color_btns[name]
                draw.rounded_rectangle([box[0], box[1], box[2], box[3]], radius=8,
                                       fill=color + (230,))
                if color == self.draw_color_rgb:
                    draw.rounded_rectangle([box[0], box[1], box[2], box[3]], radius=8,
                                           outline=(255, 255, 255, 255), width=3)
                text_color = (20, 20, 20, 255) if sum(color) > 390 else (255, 255, 255, 255)
                label_box = draw.textbbox((0, 0), name, font=font)
                label_width = label_box[2] - label_box[0]
                label_height = label_box[3] - label_box[1]
                label_x = box[0] + (box[2] - box[0] - label_width) // 2
                label_y = box[1] + (box[3] - box[1] - label_height) // 2 - 2
                draw.text((label_x, label_y), name, font=font, fill=text_color)
        else:
            draw_btn(self.air_btns["SHAPE"], "SHAPES")

        draw_btn(self.air_btns["REC"], "REC ON" if self.is_recording else "REC OFF", self.is_recording, (248, 81, 73))

    def handle_air_click(self, x, y):
        """Processes logic when a finger hovers over a UI button."""
        now = time.time()
        def inside(box):
            return box[0] < x < box[2] and box[1] < y < box[3]

        on_color_button = inside(self.air_btns["COLOR"])
        if not on_color_button:
            self.color_button_armed = True
        elif not self.color_button_armed:
            return

        on_sheet_button = inside(self.air_btns["SHEET"])
        if not on_sheet_button:
            self.sheet_button_armed = True
        elif not self.sheet_button_armed:
            return

        if now - self.last_action_time < 1.0:
            return  # Prevent double clicks

        if inside(self.air_btns["COLOR"]):
            self.color_menu_open = not self.color_menu_open
            self.color_button_armed = False
            if self.color_menu_open:
                self.shape_menu_open = False
        elif inside(self.air_btns["ERASE"]):
            self.draw_color_rgb = (0, 0, 0)
            self.color_menu_open = False
        elif inside(self.air_btns["CLEAR"]):
            self.clear_canvas()
        elif inside(self.air_btns["AI_FIX"]) and not self.is_ai_processing:
            self.is_ai_processing = True
            threading.Thread(target=self.process_handwriting_with_ai, daemon=True).start()
        elif inside(self.air_btns["PEN"]):
            self.toggle_pen()
        elif inside(self.air_btns["UNDO"]):
            self.undo()
        elif inside(self.air_btns["SHEET"]):
            self.white_canvas = not self.white_canvas
            if self.white_canvas:
                self.bg_image = None
            self.sheet_button_armed = False
        elif inside(self.air_btns["REC"]):
            self.toggle_recording()
        elif inside(self.air_btns["SHAPE"]):
            self.shape_menu_open = not self.shape_menu_open
            if self.shape_menu_open:
                self.color_menu_open = False
        elif self.color_menu_open:
            for name, box in self.color_btns.items():
                if inside(box):
                    self.draw_color_rgb = dict(self.color_options)[name]
                    self.color_menu_open = False
                    break
        elif self.shape_menu_open:
            for mode, box in self.shape_btns.items():
                if inside(box):
                    self.shape_mode = mode
                    self.shape_menu_open = False
                    break

        self.last_action_time = now
        self.xp, self.yp = 0, 0
        self.shape_start = None

    # --- Main Video Loop ---
    def update_frame(self):
        success, frame = self.cap.read()
        if not success:
            self.root.after(15, self.update_frame)
            return

        frame = cv2.flip(frame, 1)

        # 1. Background Setup
        if self.white_canvas or self.bg_image is not None:
            display_frame = (np.full((self.h, self.w, 3), 255, dtype=np.uint8)
                              if self.white_canvas else self.bg_image.copy())
            pip_w, pip_h = int(self.w * 0.25), int(self.h * 0.25)
            mini_cam = cv2.resize(frame, (pip_w, pip_h))
            pip_y = self.h - pip_h - 12
            pip_x = 12
            display_frame[pip_y:pip_y + pip_h, pip_x:pip_x + pip_w] = mini_cam
            cv2.rectangle(display_frame, (pip_x, pip_y), (pip_x + pip_w, pip_y + pip_h), (255, 255, 255), 2)
        else:
            display_frame = frame.copy()

        # 2. Hand Tracking
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_frame)
        drew_this_frame = False

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                lmList = [[id, int(lm.x * self.w), int(lm.y * self.h)] for id, lm in enumerate(hand_landmarks.landmark)]
                x1, y1 = lmList[8][1], lmList[8][2]
                x2, y2 = lmList[12][1], lmList[12][2]
                x0, y0 = lmList[0][1], lmList[0][2]

                dist_index = math.hypot(x1 - x0, y1 - y0)
                dist_middle = math.hypot(x2 - x0, y2 - y0)
                tip_distance = math.hypot(x2 - x1, y2 - y1)

                is_index_up = dist_index > 100
                is_middle_up = dist_middle > 100

                draw_color_bgr = tuple(reversed(self.draw_color_rgb))

                # HOVER MODE (Index and Middle fingers up)
                if is_index_up and is_middle_up and tip_distance < 60:
                    self.xp, self.yp = 0, 0
                    self.shape_start = None
                    self.frames_lost = 0

                    self.handle_air_click(x1, y1)
                    cv2.circle(display_frame, (x1, y1), 10,
                               draw_color_bgr if draw_color_bgr != (0, 0, 0) else (255, 255, 255), 2)

                # DRAW MODE (Only Index finger up)
                elif is_index_up and (not is_middle_up or tip_distance >= 60):
                    drew_this_frame = self.pen_enabled
                    self.frames_lost = 0

                    if self.pen_enabled and y1 > 110:
                        if self.xp == 0 and self.yp == 0:
                            if len(self.canvas_history) >= 10:
                                self.canvas_history.pop(0)
                            with self.canvas_lock:
                                self.canvas_history.append(self.canvas.copy())
                            self.shape_start = (x1, y1)
                            self.xp, self.yp = x1, y1

                        with self.canvas_lock:
                            if self.shape_mode == "freehand":
                                thick = self.eraser_thickness if draw_color_bgr == (0, 0, 0) else self.brush_thickness
                                cv2.line(self.canvas, (self.xp, self.yp), (x1, y1), draw_color_bgr, thick, cv2.LINE_AA)
                            else:
                                self.canvas = self.canvas_history[-1].copy()
                                self.draw_shape(self.canvas, self.shape_start, (x1, y1), draw_color_bgr)

                        self.xp, self.yp = x1, y1
                    cv2.circle(display_frame, (x1, y1), 10,
                               draw_color_bgr if draw_color_bgr != (0, 0, 0) else (255, 255, 255), cv2.FILLED)

        if not drew_this_frame:
            self.frames_lost += 1
            if self.frames_lost > 3:
                self.xp, self.yp = 0, 0

        # 3. Merge Drawings onto Camera
        with self.canvas_lock:
            combined_overlays = cv2.add(self.canvas, self.text_canvas)
        imgGray = cv2.cvtColor(combined_overlays, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(imgGray, 10, 255, cv2.THRESH_BINARY)
        display_frame_bg = cv2.bitwise_and(display_frame, display_frame, mask=cv2.bitwise_not(mask))
        merged_frame = cv2.add(display_frame_bg, combined_overlays)

        # 4. Apply Glassmorphism UI (Using PIL)
        rgb_frame = cv2.cvtColor(merged_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame).convert("RGBA")

        ui_overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
        self.draw_glassmorphic_ui(ui_overlay)

        final_pil = Image.alpha_composite(pil_img, ui_overlay).convert("RGB")

        if self.is_recording and self.video_writer is not None:
            out_frame = cv2.cvtColor(np.array(final_pil), cv2.COLOR_RGB2BGR)
            self.video_writer.write(out_frame)

        label_width = self.video_label.winfo_width()
        label_height = self.video_label.winfo_height()
        if label_width > 100 and label_height > 100:
            scale = min(label_width / self.w, label_height / self.h)
            display_size = (int(self.w * scale), int(self.h * scale))
            display_pil = final_pil.resize(display_size, Image.Resampling.LANCZOS)
        else:
            display_pil = final_pil

        self.tk_image = ImageTk.PhotoImage(image=display_pil)
        self.video_label.configure(image=self.tk_image)

        self.root.after(15, self.update_frame)

    # --- Utility Methods ---
    def draw_shape(self, image, start_point, end_point, color):
        thickness = self.eraser_thickness if color == (0, 0, 0) else self.brush_thickness
        if color == (0, 0, 0):
            color = (255, 255, 255)

        sx, sy = start_point
        ex, ey = end_point
        if self.shape_mode == "square":
            side = max(abs(ex - sx), abs(ey - sy))
            corner_x = sx + side if ex >= sx else sx - side
            corner_y = sy + side if ey >= sy else sy - side
            cv2.rectangle(image, (sx, sy), (corner_x, corner_y), color, thickness)
        elif self.shape_mode == "triangle":
            points = np.array([
                [sx, ey],
                [ex, ey],
                [(sx + ex) // 2, sy],
            ], dtype=np.int32)
            cv2.polylines(image, [points], True, color, thickness, cv2.LINE_AA)
        elif self.shape_mode == "rectangle":
            cv2.rectangle(image, (sx, sy), (ex, ey), color, thickness)
        elif self.shape_mode == "circle":
            cv2.circle(image, (sx, sy), int(math.hypot(ex - sx, ey - sy)), color, thickness)
        elif self.shape_mode == "line":
            cv2.line(image, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)

    # =========================================================
    #                    AI FIX  (rewritten)
    # =========================================================
    # Goal: take whatever the student wrote in the air, find just that
    # drawn region (ignore empty space), clean it up so an OCR engine
    # can actually read it, run recognition + spell-correction, and
    # then present it TWO ways so it's unmistakably readable:
    #   1. As a large wrapped text block with a solid backdrop, drawn
    #      onto the canvas itself (visible on the video/recording).
    #   2. In the side-panel textbox, in full size, scrollable, and
    #      guaranteed not to be cut off by video resolution.
    # =========================================================

    def get_drawing_bbox(self, thresh):
        """Returns (x, y, w, h) bounding box of all non-zero (drawn) pixels,
        or None if the canvas is blank."""
        coords = cv2.findNonZero(thresh)
        if coords is None:
            return None
        return cv2.boundingRect(coords)

    # ---------------------------------------------------------------
    # TrOCR — a transformer model actually trained on handwriting
    # (the IAM dataset, which includes cursive). This is the accuracy
    # fix: Tesseract is a printed-font engine and will always struggle
    # with joined cursive strokes, no matter how good preprocessing is.
    # ---------------------------------------------------------------

    def ensure_trocr_loaded(self):
        """Loads the TrOCR model once and caches it. Returns True if the
        model is ready to use, False if unavailable (libraries missing or
        the download/load failed)."""
        if self.trocr_model is not None:
            return True
        if self.trocr_load_attempted:
            return False  # already tried and failed this session

        self.trocr_load_attempted = True
        if not TROCR_LIBS_AVAILABLE:
            print("TrOCR not available — install with: pip install torch transformers")
            self.show_ai_status("TrOCR not installed — using basic Tesseract. Run: pip install torch transformers")
            return False

        try:
            self.show_ai_status("Loading handwriting AI model (first run only)...")
            model_name = "microsoft/trocr-base-handwritten"
            self.trocr_processor = TrOCRProcessor.from_pretrained(model_name)
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained(model_name)
            self.trocr_model.eval()
            print("TrOCR model loaded successfully.")
            return True
        except Exception as e:
            print("Failed to load TrOCR model:", e)
            self.trocr_model = None
            return False

    def recognize_with_trocr(self, bitonal_image):
        """Runs TrOCR on a single-channel (grayscale/bitonal) numpy image:
        dark strokes on a light background. Returns (text, confidence_pct)."""
        rgb = cv2.cvtColor(bitonal_image, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(rgb)

        pixel_values = self.trocr_processor(images=pil_img, return_tensors="pt").pixel_values

        with torch.no_grad():
            output = self.trocr_model.generate(
                pixel_values,
                max_length=32,
                output_scores=True,
                return_dict_in_generate=True,
            )

        generated_ids = output.sequences
        text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        # Approximate a confidence percentage from the model's own token
        # probabilities (average of the top probability at each generated
        # step) — not identical to Tesseract's confidence scale, but useful
        # as a relative "how sure was it" signal.
        confidence = 100.0
        if output.scores:
            probs = []
            for step_logits in output.scores:
                step_probs = torch.softmax(step_logits, dim=-1)
                probs.append(torch.max(step_probs).item())
            if probs:
                confidence = (sum(probs) / len(probs)) * 100.0

        return text, confidence

    # Tesseract page-segmentation modes worth trying for short, handwritten
    # air-canvas input, in priority order:
    #   8  = treat image as a SINGLE WORD          (best for one cursive word)
    #  13  = raw line, bypass most of Tesseract's own layout heuristics
    #        (often best for messy/joined handwriting)
    #   7  = treat image as a SINGLE TEXT LINE      (short phrases)
    #   6  = treat image as a UNIFORM BLOCK of text (multi-line notes)
    _OCR_PSM_MODES = [8, 13, 7, 6]

    # Save the cropped/preprocessed image Tesseract actually sees, so you can
    # open it and check whether the problem is preprocessing or OCR itself.
    DEBUG_SAVE_OCR_INPUT = True

    def run_ocr_best_effort(self, ocr_ready_image):
        """Tries several PSM modes and keeps whichever gives the highest
        average confidence. Returns (raw_text, avg_confidence)."""
        whitelist = (
            r'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
            r'0123456789.,?!\'"- '
        )
        best_text, best_conf = "", -1.0

        for psm in self._OCR_PSM_MODES:
            config = rf'--oem 3 --psm {psm} -c tessedit_char_whitelist={whitelist}'
            try:
                data = pytesseract.image_to_data(ocr_ready_image, config=config,
                                                  output_type=pytesseract.Output.DICT)
            except pytesseract.TesseractError as e:
                print(f"Tesseract error (psm {psm}):", e)
                continue

            words, confs = [], []
            for i, word in enumerate(data['text']):
                word = word.strip()
                try:
                    conf = float(data['conf'][i])
                except (ValueError, TypeError):
                    conf = -1
                if word and conf >= 0:
                    words.append(word)
                    confs.append(conf)

            text = " ".join(words).strip()
            avg_conf = sum(confs) / len(confs) if confs else -1

            print(f"[OCR debug] psm={psm} -> '{text}' (avg conf {avg_conf:.1f})")

            if text and avg_conf > best_conf:
                best_text, best_conf = text, avg_conf

        return best_text, best_conf

    def process_handwriting_with_ai(self):
        try:
            with self.canvas_lock:
                canvas_snapshot = self.canvas.copy()

            gray = cv2.cvtColor(canvas_snapshot, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

            bbox = self.get_drawing_bbox(thresh)
            if bbox is None:
                self.show_ai_status("Nothing to recognize — write something on the canvas first.")
                return

            # Crop to just the drawn region (plus a little breathing room)
            x, y, w, h = bbox
            pad = 30
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(thresh.shape[1], x + w + pad), min(thresh.shape[0], y + h + pad)
            cropped = thresh[y0:y1, x0:x1]

            # Thicken the thin finger-drawn strokes — but LIGHTLY. Cursive
            # letters are already touching each other; heavy dilation can
            # fuse loops (like the bowl of an 'a') into solid blobs and make
            # things WORSE, not better. 1 iteration on a small kernel is
            # usually enough.
            thick = cv2.dilate(cropped, np.ones((2, 2), np.uint8), iterations=1)

            # Upscale — OCR accuracy drops sharply on small/low-res text
            scale = 4
            resized = cv2.resize(thick, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            resized = cv2.medianBlur(resized, 3)

            # Tesseract expects dark text on a light background
            ocr_ready = cv2.bitwise_not(resized)

            if self.DEBUG_SAVE_OCR_INPUT:
                debug_path = os.path.join(os.getcwd(), "ai_fix_debug_input.png")
                cv2.imwrite(debug_path, ocr_ready)
                print("AI FIX debug image saved to:", debug_path)

            # Prefer TrOCR (trained on real handwriting, including cursive)
            # when it's available; fall back to the Tesseract multi-PSM
            # attempt otherwise. Tesseract is kept as a guaranteed fallback
            # so the feature still works even without torch/transformers.
            raw_text, avg_conf, engine_used = "", -1.0, "tesseract"
            if self.ensure_trocr_loaded():
                try:
                    raw_text, avg_conf = self.recognize_with_trocr(ocr_ready)
                    engine_used = "trocr"
                except Exception as e:
                    print("TrOCR inference failed, falling back to Tesseract:", e)

            if not raw_text:
                raw_text, avg_conf = self.run_ocr_best_effort(ocr_ready)
                engine_used = "tesseract"

            print(f"[AI FIX] engine={engine_used} result='{raw_text}' conf={avg_conf:.1f}")

            if not raw_text:
                self.show_ai_status("Couldn't detect any letter shapes at all. Try writing "
                                     "bigger, and check ai_fix_debug_input.png to see what "
                                     "the AI is looking at.")
                return

            if engine_used == "tesseract":
                corrected_text = str(TextBlob(raw_text).correct())
            else:
                corrected_text = raw_text  # TrOCR output: don't let TextBlob 'fix' names/words

            # Still show a result even at low confidence — a wrong guess the
            # student can see and fix is more useful than a dead end. Just
            # label it clearly so they know to double check it.
            if avg_conf < 40:
                if engine_used == "tesseract":
                    note = (" (low confidence — Tesseract struggles with cursive; install "
                             "'torch transformers' for much better handwriting recognition, "
                             "or try writing in separated block letters)")
                else:
                    note = " (low confidence — try writing a bit bigger/slower)"
            else:
                note = ""

            self.render_ai_text(corrected_text)
            self.update_ai_textbox(raw_text, corrected_text + note, avg_conf, engine_used)

            # The handwriting has now been converted to clean text — clear the
            # messy strokes so only the readable version remains.
            with self.canvas_lock:
                self.canvas = np.zeros_like(self.canvas)

        except Exception as e:
            print("AI Error:", e)
            self.show_ai_status("AI recognition failed. Make sure Tesseract is installed "
                                 "and the path in tesseract_cmd is correct.")
        finally:
            self.is_ai_processing = False

    def render_ai_text(self, text, font_size=42, max_width_chars=34):
        """Draws the recognized text as a large, wrapped block with a solid
        backdrop onto text_canvas, so it's clearly legible on the video feed
        (not tiny, not overlapping old strokes, not clipped)."""
        with self.canvas_lock:
            base = self.text_canvas.copy()

        img_pil = Image.fromarray(cv2.cvtColor(base, cv2.COLOR_BGR2RGB)).convert("RGBA")
        draw = ImageDraw.Draw(img_pil)

        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

        wrapped_lines = textwrap.wrap(text, width=max_width_chars) or [text]

        line_height = font_size + 14
        block_width = min(self.w - 100, 900)
        block_height = line_height * len(wrapped_lines) + 30

        start_x = 50
        start_y = 130  # sits just below the button dock

        # Solid-ish backdrop so the text is readable over any camera/slide content
        draw.rounded_rectangle(
            [(start_x - 20, start_y - 20), (start_x + block_width, start_y + block_height)],
            radius=18, fill=(15, 17, 21, 225)
        )

        y = start_y
        for line in wrapped_lines:
            # subtle shadow + bright text for contrast on any backdrop
            draw.text((start_x + 2, y + 2), line, font=font, fill=(0, 0, 0, 180))
            draw.text((start_x, y), line, font=font, fill=(255, 255, 255, 255))
            y += line_height

        with self.canvas_lock:
            self.text_canvas = cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)

    # --- Thread-safe helpers for updating the Tkinter side panel ---
    # process_handwriting_with_ai() runs on a background thread, but Tkinter
    # widgets must only ever be touched from the main thread. root.after(0, ...)
    # safely schedules the update to run on the main thread instead.

    def update_ai_textbox(self, raw_text, corrected_text, confidence, engine=""):
        def _update():
            self.ai_text_box.configure(state="normal")
            self.ai_text_box.delete("1.0", "end")
            self.ai_text_box.insert(
                "1.0",
                f"{corrected_text}\n\n"
                f"[engine: {engine.upper()}] raw: \"{raw_text}\" · confidence: {confidence:.0f}%"
            )
            self.ai_text_box.configure(state="disabled")
        self.root.after(0, _update)

    def show_ai_status(self, message):
        def _update():
            self.ai_text_box.configure(state="normal")
            self.ai_text_box.delete("1.0", "end")
            self.ai_text_box.insert("1.0", message)
            self.ai_text_box.configure(state="disabled")
        self.root.after(0, _update)

    def reset_ai_textbox(self):
        self.show_ai_status("Write on the canvas, then hover over AI FIX.")

    # --- Recording logic (unchanged) ---
    def start_recording(self, frame_size):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_path = f"air_canvas_{timestamp}.mp4"
        self.video_temp_path = f"air_canvas_{timestamp}_video.mp4"
        self.audio_temp_path = f"air_canvas_{timestamp}_audio.wav"

        self.video_writer = cv2.VideoWriter(self.video_temp_path, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, frame_size)
        if not self.video_writer.isOpened():
            self.video_writer.release()
            self.video_writer = None
            print("Could not start video recording.")
            return False

        self.audio_frames = []

        def audio_callback(indata, frames, callback_time, status):
            self.audio_frames.append(indata.copy())

        try:
            self.audio_stream = sd.InputStream(samplerate=44100, channels=1, dtype="int16", callback=audio_callback)
            self.audio_stream.start()
            self.is_recording = True
            return True
        except Exception as e:
            print(f"Microphone unavailable: {e}")
            if self.audio_stream is not None:
                self.audio_stream.close()
                self.audio_stream = None
            self.video_writer.release()
            self.video_writer = None
            return False

    def stop_recording(self):
        self.is_recording = False
        if self.audio_stream is not None:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        if not self.audio_frames:
            print("No microphone audio was captured; check the selected input device and microphone permissions.")
            return

        audio_data = np.concatenate(self.audio_frames, axis=0)
        with wave.open(self.audio_temp_path, "wb") as audio_file:
            audio_file.setnchannels(1)
            audio_file.setsampwidth(2)
            audio_file.setframerate(44100)
            audio_file.writeframes(audio_data.tobytes())

        try:
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            subprocess.run(
                [
                    ffmpeg_path, "-y", "-i", self.video_temp_path, "-i", self.audio_temp_path,
                    "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "128k", "-shortest", "-movflags", "+faststart", self.recording_path,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            os.remove(self.video_temp_path)
            os.remove(self.audio_temp_path)
            print(f"Recording saved with microphone audio: {self.recording_path}")
        except (OSError, subprocess.CalledProcessError) as error:
            details = error.stderr.strip() if isinstance(error, subprocess.CalledProcessError) and error.stderr else str(error)
            print(f"Could not merge recording audio: {details}")
            print(f"Video and audio remain available at {self.video_temp_path} and {self.audio_temp_path}.")

    def on_close(self):
        if self.is_recording:
            self.stop_recording()
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    app = ModernAirCanvas()
