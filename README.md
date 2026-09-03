# 🎨 Air Canva (Virtual Air Canvas)

A computer vision-based application built with Python that allows users to draw in the air using hand gestures in real-time. By leveraging OpenCV and MediaPipe, the application tracks hand movements through a webcam and translates them into digital drawings on a virtual canvas.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## ✨ Features

- **Real-Time Hand Tracking:** Utilizes Google's MediaPipe framework for high-precision, low-latency landmark tracking of hands.
- **Gesture-Based Control:** Switch between drawing modes and color palettes seamlessly using simple finger configurations (e.g., index finger up for drawing, multiple fingers up for selection/hovering).
- **Multi-Color Support:** Select from a palette of colors (Blue, Green, Red, Yellow) directly on the screen interface.
- **Eraser Tool:** Easily clear mistakes or erase drawings using specific gesture commands.
- **Clean Canvas Control:** Quick shortcuts/gestures to clear the entire screen canvas.

---

## 🛠️ Tech Stack & Libraries

- **Language:** Python 3.x
- **Computer Vision:** OpenCV (`opencv-python`)
- **Hand Tracking / ML:** MediaPipe (`mediapipe`)
- **Math/Array Processing:** NumPy (`numpy`)

---
air-canva-project/
│
├── project.py              # Main execution script
├── modern_air_canva.py    # Project dependencies
├── README.md            # Project documentation
