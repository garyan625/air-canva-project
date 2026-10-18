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

# Set path for Tesseract (only used as a last-resort fallback now)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- Qwen2.5-VL-3B-Instruct (vision-language model) -------------------------
# Install with:
#     pip install -U transformers accelerate bitsandbytes
# and make sure torch is a CUDA build (https://pytorch.org).
# To test ONLY the model loading + recognition (no webcam / GUI):
#     python air_canvas_pro.py --check
import gc
import re
import sys
import traceback

# Which model to use. Qwen2.5-VL-3B is the default. To try the newer, slightly
# stronger Qwen3-VL-4B (needs a recent transformers: pip install -U transformers)
# just change the id to "Qwen/Qwen3-VL-4B-Instruct".
VLM_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

VLM_IMPORT_ERROR = ""
VLM_CLASS = None
try:
    import torch
    import torchvision  # the Qwen image/video processors refuse to load without it
    from transformers import AutoProcessor, BitsAndBytesConfig
    try:
        if "Qwen3-VL" in VLM_MODEL_ID:
            from transformers import Qwen3VLForConditionalGeneration as VLM_CLASS
        elif "Qwen2.5-VL" in VLM_MODEL_ID:
            from transformers import Qwen2_5_VLForConditionalGeneration as VLM_CLASS
    except ImportError:
        VLM_CLASS = None
    if VLM_CLASS is None:
        # generic class: works for any image-text model the installed transformers knows
        from transformers import AutoModelForImageTextToText as VLM_CLASS
    VLM_LIBS_AVAILABLE = True
except Exception as _vlm_import_error:  # ImportError, or OSError from a Windows DLL clash
    VLM_IMPORT_ERROR = f"{type(_vlm_import_error).__name__}: {_vlm_import_error}"
    print("AI model libraries could not be imported ->", VLM_IMPORT_ERROR)
    VLM_LIBS_AVAILABLE = False

PRELOAD_VLM = True   # load the model in the background right after startup

# The same writing is rendered with three different pen thicknesses and the
# model reads each one; the answer most renderings agree on wins. This removes
# most one-off misreads. It stops early when the first two agree.
# Each variant = (pen thickness factor, curve-smoothing passes). A variant with 0
# smoothing was tried in the field: its jagged edges made the model answer UNREADABLE
# ('aryan') or misread ('mat' -> 'max'), so every variant keeps some smoothing.
VLM_RENDER_VARIANTS = ((1.0, 2), (0.8, 1), (1.3, 2))
PEN_FRACTION = 0.038   # pen width as a fraction of the writing height (real pen ink is thin)

TARGET_INK_H = 200   # height (px) the writing is scaled to before it is shown to the model
MAX_PAGE_W = 1500


# ---------------------------------------------------------------------------
# Turning finger strokes into something that looks like pen handwriting.
# The model was trained on real ink, not on 5 px jittery webcam lines, so we
# re-draw the writing cleanly: smooth curves, round pen, constant relative
# thickness, normalised size, white margins.
# ---------------------------------------------------------------------------

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _chaikin(pts, iterations=2):
    """Corner-cutting curve smoothing for an (N, 2) float32 polyline."""
    for _ in range(iterations):
        if len(pts) < 3:
            break
        q = 0.75 * pts[:-1] + 0.25 * pts[1:]
        r = 0.25 * pts[:-1] + 0.75 * pts[1:]
        mid = np.empty((2 * len(q), 2), dtype=np.float32)
        mid[0::2] = q
        mid[1::2] = r
        pts = np.vstack([pts[:1], mid, pts[-1:]])
    return pts


def _fit_scale(w, h):
    s = _clamp(TARGET_INK_H / max(h, 20.0), 0.6, 4.0)
    return min(s, MAX_PAGE_W / max(w, 1.0))


def _place_on_page(ink):
    """ink: uint8 (255 = ink) tight crop. Returns dark ink on a white page
    with generous margins and a line-like minimum aspect ratio."""
    h, w = ink.shape[:2]
    margin = max(24, int(h * 0.30))
    page_h = h + 2 * margin
    page_w = max(w + 2 * margin, int(page_h * 1.6))
    page = np.full((page_h, page_w), 255, dtype=np.uint8)
    x0 = (page_w - w) // 2
    page[margin:margin + h, x0:x0 + w] = 255 - ink
    return page


def render_strokes_page(strokes, thickness_scale=1.0, smooth_iters=2):
    """strokes: list of strokes, each a list of (x, y) fingertip points.
    Returns a single-channel image (dark ink on white) or None."""
    cleaned = []
    for stroke in strokes:
        arr = np.asarray(stroke, dtype=np.float32).reshape(-1, 2)
        if len(arr) == 0:
            continue
        if len(arr) > 1:
            step = np.hypot(arr[1:, 0] - arr[:-1, 0], arr[1:, 1] - arr[:-1, 1])
            arr = arr[np.concatenate([[True], step >= 1.0])]
        if len(arr) >= 3:
            arr = _chaikin(arr, smooth_iters) if smooth_iters else arr
        cleaned.append(arr)
    if not cleaned:
        return None

    allp = np.vstack(cleaned)
    mn, mx = allp.min(axis=0), allp.max(axis=0)
    w, h = float(mx[0] - mn[0]), float(mx[1] - mn[1])
    scale = _fit_scale(w, h)

    pen = max(3, int(round(max(h * scale, 60.0) * PEN_FRACTION * thickness_scale)))
    off = pen + 2
    canvas = np.zeros((int(h * scale) + 2 * off + 1, int(w * scale) + 2 * off + 1), dtype=np.uint8)
    for arr in cleaned:
        pts = ((arr - mn) * scale + off).astype(np.int32)
        if len(pts) > 1:
            cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False, 255, pen, cv2.LINE_AA)
        for px, py in pts:  # round joints and caps, like a real pen
            cv2.circle(canvas, (int(px), int(py)), pen // 2, 255, -1, cv2.LINE_AA)
    return _place_on_page(canvas)


def render_bitmap_page(mask, brush_px, thickness_scale=1.0):
    """Fallback when the stroke list can't be trusted (eraser / shapes used):
    works from the canvas bitmap. mask: uint8, 255 where something was drawn."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    mask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = mask.shape
    scale = _fit_scale(w, h)

    big = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    big = cv2.GaussianBlur(big, (0, 0), max(1.0, scale))      # smooth the staircase edges
    _, big = cv2.threshold(big, 70, 255, cv2.THRESH_BINARY)

    target_pen = max(3, int(round(max(big.shape[0], 60) * PEN_FRACTION * thickness_scale)))
    extra = target_pen - int(brush_px * scale)
    if extra >= 2:
        k = extra | 1
        big = cv2.copyMakeBorder(big, k, k, k, k, cv2.BORDER_CONSTANT, value=0)
        big = cv2.dilate(big, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    big = cv2.GaussianBlur(big, (3, 3), 0)
    return _place_on_page(big)


# ---------------------------------------------------------------------------
# Small text helpers: edit distance (used to compare readings) and an OPTIONAL
# word list. If a file called vocabulary.txt (one word per line) sits next to
# the script, a reading that is 1-2 letters away from exactly one word in the
# list is corrected to that word, and the app says so on screen. Typical use:
# the teacher's spelling words or the students' names for that lesson.
# ---------------------------------------------------------------------------

def levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


VOCAB_FILE = "vocabulary.txt"


def load_vocabulary(path=VOCAB_FILE):
    if not os.path.exists(path):
        return []
    words = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                words.append(line)
    print(f"[AI FIX] word list loaded from {path}: {len(words)} words")
    return words


def snap_to_vocabulary(text, vocab):
    """Returns (new_text, [(read_word, list_word), ...])."""
    lower = {w.lower(): w for w in vocab}
    out, changes = [], []
    for word in text.split(" "):
        lw = word.lower()
        if not word or lw in lower:
            out.append(word)
            continue
        scored = sorted((levenshtein(lw, k), k) for k in lower)
        max_d = 1 if len(word) <= 4 else 2
        if scored and scored[0][0] <= max_d and (len(scored) == 1 or scored[1][0] > scored[0][0]):
            new = lower[scored[0][1]]
            out.append(new)
            changes.append((word, new))
        else:
            out.append(word)
    return " ".join(out), changes


# ---------------------------------------------------------------------------
# Qwen2.5-VL reader: loading, inference, voting. No GUI code in here, so it
# can also be tested on its own with:  python air_canvas_pro.py --check
# ---------------------------------------------------------------------------

class QwenHandwritingReader:
    # Two different questions are asked about the same writing:
    #  - "spell": read it LETTER BY LETTER. This stops the model from replacing
    #    shaky handwriting with a similar-looking real word (its main failure).
    #  - "word": read it normally, as a cross-check.
    PROMPT_WORD = (
        "This image contains handwriting drawn in the air with a finger: "
        "the strokes are dark ink on a white page. "
        "Transcribe exactly what is written, character by character. "
        "It is usually a single word, a person's name, a number, a short phrase "
        "or simple maths written by a student. "
        "It may be a name or a made-up word, so do NOT autocorrect it and do NOT "
        "replace it with a more common English word. "
        "Keep the original capitalization and line breaks. "
        "Output only the transcribed text, with no quotes and no explanation. "
        "If nothing legible is written, output exactly: UNREADABLE"
    )
    PROMPT_SPELL = (
        "This image contains handwriting drawn in the air with a finger: "
        "the strokes are dark ink on a white page. "
        "Spell out exactly what is written, one letter or digit at a time, "
        "separated by single spaces, for example: a r y a n. "
        "If there are several words, put ' / ' between the words. "
        "Write every character you actually SEE, in its original upper or lower case, "
        "even if the result is not a real word - it may be a person's name or a "
        "made-up word. Never correct the spelling and never guess a familiar word. "
        "Output only the characters, with no explanation. "
        "If nothing legible is written, output exactly: UNREADABLE"
    )

    def __init__(self, on_status=None):
        self.model = None
        self.proc = None
        self.error = ""          # human-readable reason if loading failed
        self.attempts = 0
        self.on_cpu = False
        self.lock = threading.Lock()
        self.on_status = on_status or (lambda msg: None)

    def _status(self, msg):
        print("[Qwen]", msg)
        try:
            self.on_status(msg)
        except Exception:
            pass

    def _from_pretrained(self, **kw):
        try:
            return VLM_CLASS.from_pretrained(VLM_MODEL_ID, low_cpu_mem_usage=True, **kw)
        except TypeError:
            # newer transformers renamed torch_dtype -> dtype
            if "torch_dtype" in kw:
                kw = dict(kw)
                kw["dtype"] = kw.pop("torch_dtype")
                return VLM_CLASS.from_pretrained(VLM_MODEL_ID, low_cpu_mem_usage=True, **kw)
            raise

    def _load_processor(self, has_cuda, low_vram=False):
        # small pages are upscaled to at least min_pixels; max_pixels keeps
        # VRAM and latency under control (smaller on CPU, where it is slow)
        units = 360 if low_vram else (640 if has_cuda else 320)
        px = dict(min_pixels=256 * 28 * 28, max_pixels=units * 28 * 28)
        try:
            return AutoProcessor.from_pretrained(VLM_MODEL_ID, **px)
        except Exception as first:
            if "torchvision" in str(first).lower():
                raise
            return AutoProcessor.from_pretrained(VLM_MODEL_ID)   # model family without these kwargs

    def ensure_loaded(self):
        """Loads the model once. Thread-safe. Returns True when ready; otherwise
        False with the reason stored in self.error."""
        with self.lock:
            if self.model is not None:
                return True
            if not VLM_LIBS_AVAILABLE:
                self.error = (f"Libraries missing or broken ({VLM_IMPORT_ERROR}). "
                              "Run: pip install -U transformers accelerate bitsandbytes torchvision")
                return False
            if self.attempts >= 3:
                return False
            self.attempts += 1

            has_cuda = torch.cuda.is_available()
            self.on_cpu = not has_cuda
            low_vram = False
            if has_cuda:
                props = torch.cuda.get_device_properties(0)
                vram_gb = props.total_memory / 1024 ** 3
                low_vram = vram_gb < 5.5   # e.g. RTX 2050/3050 4 GB: needs the leanest settings
                print(f"[Qwen] GPU: {props.name}, {vram_gb:.1f} GB VRAM"
                      + (" (low-VRAM mode: full 4-bit + smaller images)" if low_vram else ""))
            else:
                print(f"[Qwen] WARNING: torch {torch.__version__} cannot see a CUDA GPU "
                      "(CPU-only torch build, or no NVIDIA GPU). This will be very slow. "
                      "Install the CUDA build from pytorch.org.")

            def bnb(skip_vision):
                kw = dict(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                          bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
                if skip_vision:
                    # keep the image encoder at full fp16 precision: it matters for reading
                    kw["llm_int8_skip_modules"] = ["visual", "lm_head"]
                return BitsAndBytesConfig(**kw)

            if has_cuda and low_vram:
                # 4 GB card: everything 4-bit, pinned to the GPU (no slow CPU offload).
                # Close other GPU-heavy apps (browser tabs, games) before starting.
                plans = [("4-bit, low VRAM", dict(quantization_config=bnb(False),
                                                  torch_dtype=torch.float16, device_map={"": 0}))]
            elif has_cuda:
                plans = [
                    ("4-bit, fp16 vision", dict(quantization_config=bnb(True),
                                                torch_dtype=torch.float16, device_map="auto")),
                    ("4-bit", dict(quantization_config=bnb(False),
                                   torch_dtype=torch.float16, device_map="auto")),
                    ("fp16", dict(torch_dtype=torch.float16, device_map="auto")),
                ]
            else:
                plans = [("CPU bf16", dict(torch_dtype=torch.bfloat16))]

            # Processor first: it is tiny and fails immediately if something like
            # torchvision is missing, instead of after minutes of loading weights.
            try:
                self._status(f"Preparing {VLM_MODEL_ID} ...")
                self.proc = self._load_processor(has_cuda, low_vram)
            except Exception as e:
                traceback.print_exc()
                self.error = f"processor: {type(e).__name__}: {str(e)[:300]}"
                return False

            errors, model = [], None
            for label, kw in plans:
                try:
                    self._status(f"Loading {VLM_MODEL_ID} ({label}). First run downloads ~7 GB - please wait...")
                    model = self._from_pretrained(**kw)
                    print(f"[Qwen] model loaded: {label}")
                    break
                except Exception as e:
                    traceback.print_exc()
                    errors.append(f"{label}: {type(e).__name__}: {str(e)[:200]}")
                    model = None
                    gc.collect()
                    if has_cuda:
                        torch.cuda.empty_cache()

            if model is None:
                self.error = " | ".join(errors)
                return False

            model.eval()
            self.model = model
            self.error = ""
            return True

    # -- inference ----------------------------------------------------
    @staticmethod
    def _parse_spelled(text):
        """'a r y a n'  ->  'aryan';  'h i / t h e r e'  ->  'hi there'.
        Also tolerates 'A-R-Y-A-N' or an unspaced answer."""
        if text.strip().upper().startswith("UNREADABLE"):
            return ""
        words = []
        for part in text.split("/"):
            tokens = re.sub(r"[,\-·•]", " ", part).split()
            word = "".join(tokens)
            if word:
                words.append(word)
        return " ".join(words)

    def _read_once(self, page, mode="word"):
        """page: single-channel image, dark ink on white.
        mode: 'word' (normal transcription) or 'spell' (letter by letter).
        Returns (text, token_confidence_pct)."""
        pil_img = Image.fromarray(cv2.cvtColor(page, cv2.COLOR_GRAY2RGB))
        prompt_text = self.PROMPT_SPELL if mode == "spell" else self.PROMPT_WORD
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt_text}]}]
        prompt = self.proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.proc(text=[prompt], images=[pil_img], padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)

        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=64, do_sample=False,
                                      output_scores=True, return_dict_in_generate=True)

        new_tokens = out.sequences[0, inputs.input_ids.shape[1]:]
        raw = self.proc.decode(new_tokens, skip_special_tokens=True).strip().strip('`"\'').strip()

        # Token confidence: average probability over the tokens that carry a
        # letter/digit (spaces and separators are trivially certain and would
        # inflate the number).
        special_ids = set(self.proc.tokenizer.all_special_ids)
        probs = []
        for i, step_logits in enumerate(out.scores):
            tok = new_tokens[i].item()
            if tok in special_ids:
                continue
            if not any(ch.isalnum() for ch in self.proc.tokenizer.decode([tok])):
                continue
            probs.append(torch.softmax(step_logits[0].float(), dim=-1)[tok].item())
        conf = (sum(probs) / len(probs) * 100.0) if probs else 0.0

        if mode == "spell":
            return self._parse_spelled(raw), conf
        if raw.upper().startswith("UNREADABLE"):
            return "", 0.0
        return raw, conf

    @staticmethod
    def _key(text):
        return re.sub(r"\s+", " ", text.strip().lower())

    # Order in which the readings are taken: (mode, which rendering). Letter
    # by letter is the main one; plain transcription is a cross-check.
    READ_PLAN = (("spell", 0), ("spell", 1), ("word", 0), ("spell", 2), ("word", 1))

    def read(self, pages):
        """Takes up to 5 readings and returns the consensus (or None if the
        writing was unreadable). Stops after 3 if they all agree exactly."""
        cands = []
        plan = self.READ_PLAN[:1] + (self.READ_PLAN[2:3] if self.on_cpu else self.READ_PLAN[1:])
        for mode, idx in plan:
            if idx >= len(pages):
                continue
            text, conf = self._read_once(pages[idx], mode)
            print(f"[Qwen] {mode:5s} / rendering {idx + 1}: '{text}' (token conf {conf:.0f}%)")
            cands.append({"text": text, "tok": conf, "mode": mode})
            if len(cands) == 3 and all(c["text"] for c in cands) \
                    and len({self._key(c["text"]) for c in cands}) == 1:
                break
        return self.consensus(cands)

    def consensus(self, cands):
        """Picks the reading that is closest (edit distance) to all the others,
        i.e. the 'median' reading, so one odd read can't win. Reported
        confidence = how strongly the independent readings agree with it."""
        valid = [c for c in cands if c["text"]]
        if not valid:
            return None

        def sim(a, b):
            ka, kb = self._key(a), self._key(b)
            return 1.0 - levenshtein(ka, kb) / max(len(ka), len(kb), 1)

        best, best_score = None, -1.0
        for c in valid:
            score = sum(sim(c["text"], o["text"]) for o in valid)
            if best is None or (round(score, 6), c["tok"]) > (round(best_score, 6), best["tok"]):
                best, best_score = c, score

        same = [o for o in valid if self._key(o["text"]) == self._key(best["text"])]
        shown = max(same, key=lambda o: o["tok"])["text"]          # keeps the best-supported capitalisation
        others = {}
        for o in valid:
            k = self._key(o["text"])
            if k != self._key(best["text"]):
                others.setdefault(k, o["text"])
        return {
            "text": shown,
            "confidence": best_score / len(valid) * 100.0,        # agreement between readings
            "token_conf": sum(o["tok"] for o in same) / len(same),
            "votes": len(same),
            "total": len(cands),
            "valid": len(valid),
            "empty": len(cands) - len(valid),
            "alternatives": list(others.values()),
        }


def run_diagnostics(image_path="ai_fix_debug_input.png"):
    """python air_canvas_pro.py --check [image.png]
    Tests library imports, model loading and recognition without the GUI."""
    print("=== Air Canvas AI FIX diagnostics ===")
    print("python:", sys.version.split()[0])
    if not VLM_LIBS_AVAILABLE:
        print("IMPORT FAILED:", VLM_IMPORT_ERROR)
        print("Fix: pip install -U transformers accelerate bitsandbytes torchvision")
        return
    import transformers
    print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())
    print("transformers:", transformers.__version__)
    print("torchvision:", torchvision.__version__)
    print("model id:", VLM_MODEL_ID)
    try:
        import bitsandbytes
        print("bitsandbytes:", bitsandbytes.__version__)
    except Exception as e:
        print("bitsandbytes: NOT usable ->", type(e).__name__, e)

    reader = QwenHandwritingReader(on_status=print)
    if not reader.ensure_loaded():
        print("\nMODEL LOAD FAILED:\n", reader.error)
        return
    print("\nModel loaded OK.")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(image_path) else None
    if img is None:
        print(f"No '{image_path}' found - use AI FIX once in the app to create it, then re-run.")
        return
    print("Recognition result:", reader.read([img]))


class ModernAirCanvas:
    # Fingertip smoothing: 1.0 = no smoothing, lower = smoother but laggier.
    SMOOTH_ALPHA = 0.6

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
        self.smooth_pt = None  # smoothed fingertip position (float x, y)

        # Recorded fingertip paths (one list of (x, y) per stroke). AI FIX re-draws
        # these as clean ink, which is far easier to read than the webcam bitmap.
        # stroke_marks stays parallel to canvas_history so UNDO can drop strokes too.
        # strokes_valid turns False once the eraser/shapes are used (the path no
        # longer matches the canvas); AI FIX then falls back to the bitmap.
        self.strokes = []
        self.stroke_marks = []
        self.strokes_valid = True

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

        # Qwen2.5-VL reader. Loaded in a background thread shortly after startup
        # (or on first AI FIX). show_ai_status is thread-safe.
        self.reader = QwenHandwritingReader(on_status=self.show_ai_status)
        self.vocabulary = load_vocabulary()   # optional vocabulary.txt next to the script

        # Side Control Panel
        self.side_panel = ctk.CTkFrame(self.main_container, width=250, corner_radius=15)
        self.side_panel.pack(side="right", fill="y")
        self.build_side_panel()

        # Build Air UI Boundaries
        self.setup_air_buttons()

        # Start Main Loop
        self.update_frame()
        if PRELOAD_VLM:
            # Scheduled via after() so the thread only starts once the Tk
            # main loop is actually running.
            self.root.after(1500, lambda: threading.Thread(
                target=self._preload_reader, daemon=True).start())
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

        # AI recognized-text panel
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
            self.canvas_history.clear()
            self.stroke_marks.clear()
            self.strokes.clear()
            self.strokes_valid = True
        self.bg_image = None
        self.xp, self.yp = 0, 0
        self.reset_ai_textbox()

    def undo(self):
        if self.canvas_history:
            with self.canvas_lock:
                self.canvas = self.canvas_history.pop()
                if self.stroke_marks:
                    del self.strokes[self.stroke_marks.pop():]
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

    # --- Fingertip smoothing ---
    def smooth_point(self, x, y):
        """Exponential moving average over the fingertip position. MediaPipe
        landmarks jitter by a few pixels every frame, which turns straight
        strokes into wobbly polylines that OCR models read badly. The filter
        restarts at the beginning of every new stroke so there's no streak
        from the previous position."""
        if (self.xp == 0 and self.yp == 0) or self.smooth_pt is None:
            self.smooth_pt = (float(x), float(y))
        else:
            a = self.SMOOTH_ALPHA
            px, py = self.smooth_pt
            self.smooth_pt = (px + a * (x - px), py + a * (y - py))
        return int(round(self.smooth_pt[0])), int(round(self.smooth_pt[1]))

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
                    self.smooth_pt = None
                    self.frames_lost = 0

                    self.handle_air_click(x1, y1)
                    cv2.circle(display_frame, (x1, y1), 10,
                               draw_color_bgr if draw_color_bgr != (0, 0, 0) else (255, 255, 255), 2)

                # DRAW MODE (Only Index finger up)
                elif is_index_up and (not is_middle_up or tip_distance >= 60):
                    drew_this_frame = self.pen_enabled
                    self.frames_lost = 0

                    # Smooth the raw fingertip position to remove landmark jitter
                    x1, y1 = self.smooth_point(x1, y1)

                    if self.pen_enabled and y1 > 110:
                        if self.xp == 0 and self.yp == 0:
                            with self.canvas_lock:
                                if len(self.canvas_history) >= 10:
                                    self.canvas_history.pop(0)
                                    if self.stroke_marks:
                                        self.stroke_marks.pop(0)
                                self.canvas_history.append(self.canvas.copy())
                                self.stroke_marks.append(len(self.strokes))
                                if self.shape_mode == "freehand" and draw_color_bgr != (0, 0, 0):
                                    self.strokes.append([(x1, y1)])
                                else:
                                    self.strokes_valid = False
                            self.shape_start = (x1, y1)
                            self.xp, self.yp = x1, y1

                        with self.canvas_lock:
                            if self.shape_mode == "freehand":
                                thick = self.eraser_thickness if draw_color_bgr == (0, 0, 0) else self.brush_thickness
                                cv2.line(self.canvas, (self.xp, self.yp), (x1, y1), draw_color_bgr, thick, cv2.LINE_AA)
                                if draw_color_bgr != (0, 0, 0) and self.strokes:
                                    self.strokes[-1].append((x1, y1))
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
                self.smooth_pt = None

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
    #                         AI FIX
    # =========================================================
    # Pipeline: the strokes you wrote are re-drawn as clean pen ink (3 pen
    # thicknesses) -> Qwen2.5-VL reads each one -> the consensus answer is
    # shown big on the canvas and in the side panel. Tesseract is only a
    # last-resort fallback if the Qwen model could not be loaded.
    # =========================================================

    def get_drawing_bbox(self, thresh):
        """Returns (x, y, w, h) bounding box of all non-zero (drawn) pixels,
        or None if the canvas is blank."""
        coords = cv2.findNonZero(thresh)
        if coords is None:
            return None
        return cv2.boundingRect(coords)

    # Tesseract fallback settings
    _OCR_PSM_MODES = [8, 13, 7, 6]

    # Saves the exact images the model sees. If YOU can't read them easily,
    # the model can't either - this is the best debugging tool for AI FIX.
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

    def _preload_reader(self):
        """Runs on a background thread shortly after startup."""
        if self.reader.ensure_loaded():
            if self.reader.on_cpu:
                self.show_ai_status("AI model ready but running on the CPU: each AI FIX will take "
                                    "a minute or more. Install the CUDA build of PyTorch for 2-3 s.")
            else:
                self.show_ai_status("AI model ready. Write on the canvas, then hover over AI FIX.")
        else:
            self.show_ai_status("Qwen AI model could NOT load, so AI FIX will fall back to weak "
                                "Tesseract.\nReason: " + self.reader.error +
                                "\n\nRun 'python air_canvas_pro.py --check' for full diagnostics.")

    def process_handwriting_with_ai(self):
        try:
            with self.canvas_lock:
                canvas_snapshot = self.canvas.copy()
                strokes_snapshot = [list(s) for s in self.strokes] if self.strokes_valid else []

            gray = cv2.cvtColor(canvas_snapshot, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

            if self.get_drawing_bbox(thresh) is None:
                self.show_ai_status("Nothing to recognize - write something on the canvas first.")
                return

            # Prefer the recorded fingertip path (clean vector data). If the
            # eraser or shapes were used, the path no longer matches the canvas,
            # so fall back to the bitmap.
            use_strokes = sum(len(s) for s in strokes_snapshot) >= 2
            pages = []
            for thickness, smooth_iters in VLM_RENDER_VARIANTS:
                if use_strokes:
                    page = render_strokes_page(strokes_snapshot, thickness, smooth_iters)
                else:
                    page = render_bitmap_page(thresh, self.brush_thickness, thickness)
                if page is not None:
                    pages.append(page)
            if not pages:
                self.show_ai_status("Nothing to recognize - write something on the canvas first.")
                return

            print(f"[AI FIX] input from {'recorded strokes' if use_strokes else 'canvas bitmap'}; "
                  f"{len(pages)} renderings, page {pages[0].shape[1]}x{pages[0].shape[0]}")

            if self.DEBUG_SAVE_OCR_INPUT:
                names = ["ai_fix_debug_input.png", "ai_fix_debug_input_thin.png", "ai_fix_debug_input_thick.png"]
                for name, page in zip(names, pages):
                    cv2.imwrite(os.path.join(os.getcwd(), name), page)
                print("AI FIX debug images saved to:", os.getcwd())

            result, vlm_answered = None, False
            if self.reader.ensure_loaded():
                self.show_ai_status("Reading your handwriting...")
                try:
                    result = self.reader.read(pages)
                    vlm_answered = True
                except Exception as e:
                    traceback.print_exc()
                    self.reader.error = f"inference failed: {type(e).__name__}: {str(e)[:200]}"
                    if VLM_LIBS_AVAILABLE and torch.cuda.is_available():
                        torch.cuda.empty_cache()

            if vlm_answered:
                engine_used = "qwen2.5-vl"
                raw_text = result["text"] if result else ""
                avg_conf = result["token_conf"] if result else -1.0   # model's own certainty
            else:
                # Qwen unavailable or crashed: weak Tesseract fallback
                engine_used = "tesseract"
                raw_text, avg_conf = self.run_ocr_best_effort(pages[0])

            print(f"[AI FIX] engine={engine_used} result='{raw_text}' conf={avg_conf:.1f}")

            if not raw_text:
                self.show_ai_status("Couldn't read that. Try writing bigger and slower with clear gaps "
                                    "between letters, and check ai_fix_debug_input.png to see what "
                                    "the AI is looking at."
                                    + ("" if vlm_answered else f"\n(Qwen unavailable: {self.reader.error})"))
                return

            # Autocorrect only for Tesseract; it would ruin names from the VLM.
            corrected_text = str(TextBlob(raw_text).correct()) if engine_used == "tesseract" else raw_text
            snaps = []
            if engine_used != "tesseract" and self.vocabulary:
                corrected_text, snaps = snap_to_vocabulary(corrected_text, self.vocabulary)

            note, extra = "", ""
            if engine_used == "tesseract":
                note = f" (Tesseract fallback - Qwen failed to load: {self.reader.error[:300]})"
            else:
                extra = (f"{result['votes']} of {result['valid']} readable readings gave this answer"
                         + (f" ({result['empty']} judged unreadable)" if result["empty"] else "")
                         + " - agreement is not the same as being correct")
                if result["alternatives"]:
                    extra += " · other readings: " + ", ".join(result["alternatives"])
                if snaps:
                    extra += " · matched to your word list: " + ", ".join(f"{a} -> {b}" for a, b in snaps)
                if avg_conf < 50:
                    note = " (low model certainty - write bigger/slower, with gaps between letters)"

            self.render_ai_text(corrected_text)
            self.update_ai_textbox(raw_text, corrected_text + note, avg_conf, engine_used, extra)

            # Handwriting is now clean text: clear the strokes, the recorded
            # stroke data and the undo history (UNDO would otherwise bring the
            # old messy strokes back on top of the clean text).
            with self.canvas_lock:
                self.canvas = np.zeros_like(self.canvas)
                self.canvas_history.clear()
                self.stroke_marks.clear()
                self.strokes.clear()
                self.strokes_valid = True

        except Exception as e:
            traceback.print_exc()
            self.show_ai_status(f"AI recognition failed: {type(e).__name__}: {e}")
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

        wrapped_lines = []
        for part in text.splitlines() or [text]:
            wrapped_lines.extend(textwrap.wrap(part, width=max_width_chars) or [""])
        if not wrapped_lines:
            wrapped_lines = [text]

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
    # The AI code runs on background threads, but Tkinter widgets must only
    # be touched from the main thread. root.after(0, ...) schedules the
    # update on the main thread instead.

    def update_ai_textbox(self, raw_text, corrected_text, confidence, engine="", extra=""):
        def _update():
            self.ai_text_box.configure(state="normal")
            self.ai_text_box.delete("1.0", "end")
            self.ai_text_box.insert(
                "1.0",
                f"{corrected_text}\n\n"
                f"[engine: {engine.upper()}] raw: \"{raw_text}\" · confidence: {confidence:.0f}%"
                + (f"\n{extra}" if extra else "")
            )
            self.ai_text_box.configure(state="disabled")
        try:
            self.root.after(0, _update)
        except RuntimeError:
            pass

    def show_ai_status(self, message):
        def _update():
            self.ai_text_box.configure(state="normal")
            self.ai_text_box.delete("1.0", "end")
            self.ai_text_box.insert("1.0", message)
            self.ai_text_box.configure(state="disabled")
        try:
            self.root.after(0, _update)
        except RuntimeError:
            pass

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
    if "--check" in sys.argv:
        args = [a for a in sys.argv[1:] if a != "--check"]
        run_diagnostics(args[0] if args else "ai_fix_debug_input.png")
    else:
        app = ModernAirCanvas()
