import cv2
import numpy as np


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def psnr(ref, img):
    return min(cv2.PSNR(ref, img), 99.0)


def ssim(ref, img):
    a = _gray(ref).astype(np.float64)
    b = _gray(img).astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a ** 2
    var_b = blur(b * b) - mu_b ** 2
    cov = blur(a * b) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    return float((num / den).mean())


def sharpness(img):
    return float(cv2.Laplacian(_gray(img), cv2.CV_64F).var())


def edge_density(img):
    return float(np.mean(cv2.Canny(_gray(img), 100, 200) > 0) * 100)


def lbp(img):
    g = _gray(img).astype(np.int16)
    c = g[1:-1, 1:-1]
    codes = np.zeros_like(c, dtype=np.uint8)
    neighbours = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for bit, (dy, dx) in enumerate(neighbours):
        n = g[1 + dy:g.shape[0] - 1 + dy, 1 + dx:g.shape[1] - 1 + dx]
        codes |= ((n >= c).astype(np.uint8) << bit)
    return codes


def lbp_hist(img):
    hist = np.bincount(lbp(img).ravel(), minlength=256).astype(np.float64)
    return hist / hist.sum()


def lbp_similarity(ref, img):
    return float(np.minimum(lbp_hist(ref), lbp_hist(img)).sum())


def evaluate(original, degraded, enhanced):
    rows = []
    for name, im in (("Original", original), ("Degraded", degraded), ("Enhanced", enhanced)):
        rows.append({
            "Image": name,
            "PSNR (dB)": round(psnr(original, im), 2),
            "SSIM": round(ssim(original, im), 3),
            "LBP similarity": round(lbp_similarity(original, im), 3),
            "Sharpness": round(sharpness(im), 1),
            "Edge density (%)": round(edge_density(im), 2),
        })
    return rows