import cv2
import numpy as np

_hog = cv2.HOGDescriptor()
_hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def skin_mask_ycrcb(img, cr_min=133, cr_max=173, cb_min=77, cb_max=127):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, cr_min, cb_min], np.uint8)
    upper = np.array([255, cr_max, cb_max], np.uint8)
    return cv2.inRange(ycrcb, lower, upper)


def skin_mask_kmeans(img, k=3):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    small = cv2.resize(ycrcb, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    data = small[:, :, 1:].reshape(-1, 2).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    cv2.setRNGSeed(42)
    _, _, centers = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    skin_ref = np.array([150, 110], np.float32)
    skin_idx = np.argmin(np.linalg.norm(centers - skin_ref, axis=1))
    pixels = ycrcb[:, :, 1:].reshape(-1, 2).astype(np.float32)
    dists = np.linalg.norm(pixels[:, None, :] - centers[None, :, :], axis=2)
    labels = np.argmin(dists, axis=1)
    return ((labels == skin_idx).reshape(img.shape[:2]) * 255).astype(np.uint8)


def clean_mask(mask, size=7):
    return cv2.medianBlur(mask, size)


def detect_person(img, work_width=400):
    f = work_width / img.shape[1]
    small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    rects, weights = _hog.detectMultiScale(small, winStride=(4, 4), padding=(16, 16), scale=1.03)
    if len(rects) == 0:
        return None
    x, y, w, h = rects[int(np.argmax(weights))]
    H, W = img.shape[:2]
    x0, y0 = max(0, int(x / f)), max(0, int(y / f))
    x1, y1 = min(W, int((x + w) / f)), min(H, int((y + h) / f))
    return x0, y0, x1 - x0, y1 - y0


def head_region(person, top=0.25, side=0.2):
    x, y, w, h = person
    return x + int(w * side), y, int(w * (1 - 2 * side)), int(h * top)


def find_face_in_region(mask, region, min_area=0.01, pad_x=0.2, pad_top=0.4, max_aspect=1.3):
    x, y, w, h = region
    sub = mask[y:y + h, x:x + w]
    contours, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > min_area * w * h]
    if not boxes:
        return None
    main = min(boxes, key=lambda b: abs(b[0] + b[2] / 2 - w / 2))
    left, right = main[0], main[0] + main[2]
    group = [b for b in boxes if b[0] < right and b[0] + b[2] > left]
    x0 = min(b[0] for b in group)
    y0 = min(b[1] for b in group)
    x1 = max(b[0] + b[2] for b in group)
    y1 = max(b[1] + b[3] for b in group)
    bw, bh = x1 - x0, y1 - y0
    x0, x1 = max(0, x0 - int(bw * pad_x)), min(w, x1 + int(bw * pad_x))
    y0 = max(0, y0 - int(bh * pad_top))
    y1 = min(y1, y0 + int((x1 - x0) * max_aspect))
    return x + x0, y + y0, x1 - x0, y1 - y0


def locate_face(img, mask):
    person = detect_person(img)
    if person:
        region = head_region(person)
        return person, region, find_face_in_region(mask, region) or region
    H, W = mask.shape
    return None, None, find_face_in_region(mask, (0, 0, W, H))


def overlay_mask(img, mask, color=(0, 255, 0), alpha=0.4):
    tint = np.zeros_like(img)
    tint[:] = color
    blended = cv2.addWeighted(img, 1 - alpha, tint, alpha, 0)
    out = img.copy()
    out[mask > 0] = blended[mask > 0]
    return out


def draw_box(img, box, color=(0, 255, 0)):
    out = img.copy()
    if box:
        x, y, w, h = box
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
    return out


def crop(img, box):
    x, y, w, h = box
    return img[y:y + h, x:x + w]