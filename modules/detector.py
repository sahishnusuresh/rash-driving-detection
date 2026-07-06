import cv2
import numpy as np
from ultralytics import YOLO

VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck


class Detector:
    def __init__(self, model_path: str, conf: float, frame_w: int, frame_h: int):
        self.model = YOLO(model_path)
        self.conf  = conf

        # trapezoid ROI — road-ahead region, excludes ego hood + side traffic
        self.roi = np.array([
            [int(0.30 * frame_w), int(0.35 * frame_h)],
            [int(0.70 * frame_w), int(0.35 * frame_h)],
            [int(0.90 * frame_w), int(0.72 * frame_h)],
            [int(0.10 * frame_w), int(0.72 * frame_h)],
        ])
        self.min_box_height = 0.025 * frame_h

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Run YOLO on frame, apply ROI + size filter.
        Returns array of shape (N, 6): [x1, y1, x2, y2, conf, cls]."""
        results = self.model(frame, classes=VEHICLE_CLASSES,
                             conf=self.conf, verbose=False)[0]
        dets = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls  = float(box.cls[0])
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            inside     = cv2.pointPolygonTest(self.roi, (cx, cy), False) >= 0
            tall_enough = (y2 - y1) > self.min_box_height
            if inside and tall_enough:
                dets.append([x1, y1, x2, y2, conf, cls])
        if dets:
            return np.array(dets, dtype=np.float32)
        return np.empty((0, 6), dtype=np.float32)
