import cv2
import numpy as np


class Visualizer:
    def __init__(self, roi: np.ndarray):
        self.roi = roi

    def draw(self, frame: np.ndarray, tracks: list) -> np.ndarray:
        """Draw ROI and bounding boxes with IDs and speed onto frame (in-place)."""
        cv2.polylines(frame, [self.roi], isClosed=True, color=(0, 255, 255), thickness=2)
        for track in tracks:
            x1, y1, x2, y2, tid, conf = track[:6]
            speed = track[6] if len(track) > 6 else None
            label = f"ID:{tid} {conf:.2f}"
            if speed is not None:
                label += f" s:{speed:.1f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label,
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame
