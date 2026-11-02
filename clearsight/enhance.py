import cv2
import numpy as np


def _on_lightness(img, func):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = func(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def gamma_correct(img, gamma):
    if gamma == 1.0:
        return img
    table = np.clip(((np.arange(256) / 255.0) ** gamma) * 255, 0, 255).astype(np.uint8)
    return _on_lightness(img, lambda l: cv2.LUT(l, table))


def equalize_global(img):
    return _on_lightness(img, cv2.equalizeHist)


def equalize_clahe(img, clip_limit=2.0, tile=8):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return _on_lightness(img, clahe.apply)