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