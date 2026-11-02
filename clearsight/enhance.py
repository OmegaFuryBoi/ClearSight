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

def denoise_gaussian(img, strength):
    k = 2 * strength + 1
    return cv2.GaussianBlur(img, (k, k), 0)


def denoise_median(img, strength):
    k = 2 * strength + 1
    return cv2.medianBlur(img, k)


def denoise_bilateral(img, strength):
    s = 20 * strength
    return cv2.bilateralFilter(img, 9, s, s)


def has_impulse_noise(img, threshold=0.0005):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    extreme = (g == 0) | (g == 255)
    spikes = cv2.absdiff(g, cv2.medianBlur(g, 3)) > 60
    return np.mean(extreme & spikes) > threshold


def denoise_auto(img, strength):
    if has_impulse_noise(img):
        img = denoise_median(img, 1)
    return denoise_bilateral(img, strength)


def denoise(img, method, strength):
    if method == "Gaussian":
        return denoise_gaussian(img, strength)
    if method == "Median":
        return denoise_median(img, strength)
    if method == "Bilateral":
        return denoise_bilateral(img, strength)
    if method == "Auto":
        return denoise_auto(img, strength)
    return img


def sharpen_unsharp(img, amount, radius=2.0):
    def f(l):
        blurred = cv2.GaussianBlur(l, (0, 0), radius)
        return cv2.addWeighted(l, 1 + amount, blurred, -amount, 0)
    return _on_lightness(img, f)


def sharpen_laplacian(img, amount):
    def f(l):
        l = l.astype(np.float32)
        lap = cv2.Laplacian(l, cv2.CV_32F, ksize=1)
        return np.clip(l - amount * lap, 0, 255).astype(np.uint8)
    return _on_lightness(img, f)


def sharpen(img, method, amount):
    if method == "Unsharp mask":
        return sharpen_unsharp(img, amount)
    if method == "Laplacian":
        return sharpen_laplacian(img, amount)
    return img

def auto_gamma(img):
    l = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0]
    mean = np.clip(l.mean() / 255.0, 0.01, 0.99)
    return float(np.clip(np.log(0.5) / np.log(mean), 0.25, 1.0))


def smooth_skin(img, mask, box, strength):
    if strength == 0 or box is None:
        return img
    x, y, w, h = box
    face = img[y:y + h, x:x + w]
    smooth = cv2.bilateralFilter(face, 9, 60, 60)
    smooth = cv2.bilateralFilter(smooth, 9, 60, 60)
    weight = cv2.GaussianBlur(mask[y:y + h, x:x + w], (0, 0), 3).astype(np.float32) / 255 * strength
    weight = weight[:, :, None]
    out = img.copy()
    out[y:y + h, x:x + w] = (face * (1 - weight) + smooth * weight).astype(np.uint8)
    return out