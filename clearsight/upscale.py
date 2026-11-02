import cv2

from clearsight.enhance import sharpen_unsharp

INTERPOLATIONS = {
    "Nearest": cv2.INTER_NEAREST,
    "Bilinear": cv2.INTER_LINEAR,
    "Bicubic": cv2.INTER_CUBIC,
}


def enlarge(img, size, method="Bicubic", progressive=False, sharpen_amount=0.0):
    tw, th = size
    interp = INTERPOLATIONS[method]
    out = img
    if progressive:
        while out.shape[1] * 2 < tw:
            out = cv2.resize(out, None, fx=2, fy=2, interpolation=interp)
            if sharpen_amount:
                out = sharpen_unsharp(out, sharpen_amount / 2, radius=1.0)
    out = cv2.resize(out, (tw, th), interpolation=interp)
    if sharpen_amount:
        out = sharpen_unsharp(out, sharpen_amount / 2 if progressive else sharpen_amount, radius=1.0)
    return out