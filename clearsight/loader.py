import os
import tempfile

import cv2
import numpy as np


def decode_image(file_bytes):
    data = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not read this file as an image.")
    return img


def limit_size(img, max_width=800):
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / w
    return cv2.resize(img, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)


def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def open_video(file_bytes, name):
    path = os.path.join(tempfile.gettempdir(), "clearsight_" + os.path.basename(name))
    with open(path, "wb") as f:
        f.write(file_bytes)
    return cv2.VideoCapture(path)


def open_webcam(index=0):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    return cap
