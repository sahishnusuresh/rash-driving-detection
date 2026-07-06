import numpy as np
from boxmot.trackers import DeepOcSort
from boxmot.reid.core import ReID


class VehicleTracker:
    def __init__(self):
        reid = ReID()
        self._tracker = DeepOcSort(
            reid_model=reid.model,
            max_age=90,            # keep lost track alive for ~3s — covers overpass transitions
            min_hits=5,            # require 5 consecutive frames before confirming — filters brief intersection traffic
            w_association_emb=0.6, # balance appearance vs IoU — tuned for crowded intersections
        )

    def update(self, dets: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Update tracker with detections.
        Returns array of shape (M, 8): [x1, y1, x2, y2, tracker_id, conf, cls, idx]."""
        return self._tracker.update(dets, frame)
