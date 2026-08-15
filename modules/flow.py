import cv2
import numpy as np
from collections import defaultdict, deque


class SpeedEstimator:
    def __init__(self, fps: float):
        self.fps = fps

        self.lk_params = {
            'winSize':  (21, 21),   # large patch — robust on distant small vehicles
            'maxLevel': 3,          # 3 pyramid levels — handles large inter-frame displacement
            'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        }

        self.prev_centers = {}   # tid -> (cx, cy) from previous frame
        self.prev_gray    = None # grayscale frame from previous iteration

        # per vehicle: deque of dicts {speed, u, v} — maxlen=90 covers 3s at 30fps
        self.history = defaultdict(lambda: deque(maxlen=90))

    def update(self, frame: np.ndarray, tracks: list) -> list:
        """Compute optical flow speed for each tracked vehicle.

        Args:
            frame:  current BGR frame
            tracks: list of (x1, y1, x2, y2, tid, conf) from ReIDMatcher

        Returns:
            same list with speed info added:
            (x1, y1, x2, y2, tid, conf, speed, u, v)
        """
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result    = []

        for (x1, y1, x2, y2, tid, conf) in tracks:
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            speed, u, v = 0.0, 0.0, 0.0

            if self.prev_gray is not None and tid in self.prev_centers:
                prev_cx, prev_cy = self.prev_centers[tid]

                # LK expects shape (N, 1, 2) float32
                prev_pt  = np.array([[prev_cx, prev_cy]], dtype=np.float32).reshape(1, 1, 2)
                curr_pt, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, curr_gray, prev_pt, None, **self.lk_params
                )

                if status[0][0] == 1:
                    # status=1 means LK successfully tracked the point
                    new_cx, new_cy = curr_pt[0][0]
                    u     = float(new_cx - prev_cx)   # lateral displacement (px/frame)
                    v     = float(new_cy - prev_cy)   # longitudinal displacement (px/frame)
                    speed = float(np.sqrt(u**2 + v**2))

            self.history[tid].append({'speed': speed, 'u': u, 'v': v})
            self.prev_centers[tid] = (cx, cy)
            result.append((x1, y1, x2, y2, tid, conf, speed, u, v))

        self.prev_gray = curr_gray
        return result
