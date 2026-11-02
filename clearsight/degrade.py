import cv2
import numpy as np


def gaussian_blur(img, strength):
    if strength == 0:
        return img
    k = 2 * strength + 1                      # kernel size must be odd: 3, 5, 7...
    return cv2.GaussianBlur(img, (k, k), 0)


def lower_resolution(img, factor):
    if factor <= 1:
        return img
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // factor), max(1, h // factor)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def darken(img, gamma):
    if gamma == 1.0:
        return img
    table = np.clip(((np.arange(256) / 255.0) ** gamma) * 255, 0, 255).astype(np.uint8)
    return cv2.LUT(img, table)


def gaussian_noise(img, sigma, rng):
    if sigma == 0:
        return img
    noise = rng.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def salt_pepper_noise(img, amount, rng):
    if amount == 0:
        return img
    out = img.copy()
    r = rng.random(img.shape[:2])
    out[r < amount / 2] = 0
    out[r > 1 - amount / 2] = 255
    return out


def degrade(img, blur=0, scale=1, gamma=1.0, noise_sigma=0, sp_amount=0.0, seed=42):
    rng = np.random.default_rng(seed)
    out = gaussian_blur(img, blur)            # lens is out of focus
    out = lower_resolution(out, scale)        # camera sensor is low-res
    out = darken(out, gamma)                  # poor lighting
    out = gaussian_noise(out, noise_sigma, rng)   # sensor grain
    out = salt_pepper_noise(out, sp_amount, rng)  # faulty pixels
    return out