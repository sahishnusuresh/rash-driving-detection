import cv2
import numpy as np


class Visualizer:
    def __init__(self, roi: np.ndarray):
        self.roi = roi

    def draw(self, frame: np.ndarray, tracks: list) -> np.ndarray:
        """Draw ROI and bounding boxes with IDs onto frame (in-place)."""
        cv2.polylines(frame, [self.roi], isClosed=True, color=(0, 255, 255), thickness=2)
        for (x1, y1, x2, y2, tid, conf) in tracks:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{tid} {conf:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame
