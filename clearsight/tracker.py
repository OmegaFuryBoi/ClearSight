import cv2
import numpy as np


class FaceTracker:
    def __init__(self, redetect_every=30):
        self.redetect_every = redetect_every
        self.box = None
        self.hist = None
        self.frames = 0

    def active(self):
        return self.box is not None and self.frames < self.redetect_every

    def start(self, frame, box):
        x, y, w, h = box
        hsv = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2HSV)
        valid = cv2.inRange(hsv, np.array((0, 40, 40)), np.array((180, 255, 255)))
        self.hist = cv2.calcHist([hsv], [0], valid, [180], [0, 180])
        cv2.normalize(self.hist, self.hist, 0, 255, cv2.NORM_MINMAX)
        self.box = box
        self.frames = 0

    def update(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        back = cv2.calcBackProject([hsv], [0], self.hist, [0, 180], 1)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        _, (nx, ny, nw, nh) = cv2.CamShift(back, self.box, criteria)
        if nw * nh < 25:
            self.box = None
            return None
        x, y, w, h = self.box
        H, W = frame.shape[:2]
        cx, cy = nx + nw // 2, ny + nh // 2
        x = int(np.clip(cx - w // 2, 0, W - w))
        y = int(np.clip(cy - h // 2, 0, H - h))
        self.box = (x, y, w, h)
        self.frames += 1
        return self.box