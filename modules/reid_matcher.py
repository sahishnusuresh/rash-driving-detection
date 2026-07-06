import cv2
import numpy as np
from collections import deque


class ReIDMatcher:
    """Lighting-robust re-identification gate.

    Combines OSNet appearance embeddings (from the tracker's ReID backend)
    with Lab ab-channel colour histograms. The Lab ab component is invariant
    to luminance changes, which makes it robust to overpass / tunnel shadows
    where pure appearance-based ReID typically fails.

    When a new tracker_id appears, this class checks whether it matches any
    recently lost track using the combined distance. If a match is found the
    new id is silently remapped to the old canonical id.
    """

    REID_WEIGHT     = 0.6
    LAB_WEIGHT      = 0.4
    REMAP_THRESHOLD = 0.35
    LOST_TIMEOUT    = 120   # frames before a lost track is forgotten
    CACHE_MAXLEN    = 90    # covers ~3s at 30fps — longer than most overpasses
    GOOD_BOX_RATIO  = 0.06  # box must be > 6% of frame height to extract features

    def __init__(self, reid_backend, frame_w: int, frame_h: int):
        self._reid     = reid_backend
        self._w        = frame_w
        self._h        = frame_h

        self._emb_cache  = {}   # canonical_tid -> deque of OSNet embeddings
        self._lab_cache  = {}   # canonical_tid -> deque of Lab ab histograms
        self._id_remap   = {}   # raw_tid -> canonical_tid
        self._lost       = {}   # canonical_tid -> {reid, lab, last_frame}
        self._frame_idx  = 0

    DRIFT_THRESHOLD  = 0.45  # embedding drift above this = likely internal ID swap
    DRIFT_BOX_RATIO  = 0.03  # smaller threshold for drift check — catches small distant boxes

    # ------------------------------------------------------------------
    def update(self, tracks: np.ndarray, frame: np.ndarray) -> list:
        """Process tracker output. Returns list of (x1,y1,x2,y2,tid,conf)."""
        self._frame_idx += 1
        self._expire_lost()

        active_raw = set()
        final      = []

        for track in tracks:
            x1, y1, x2, y2, raw_tid, conf, cls, _ = track
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            raw_tid = int(raw_tid)
            active_raw.add(raw_tid)

            tid      = self._id_remap.get(raw_tid, raw_tid)
            reid_emb = None
            lab_hist = None
            box_h    = y2 - y1

            # drift check runs on any box large enough to extract features
            if box_h > self.DRIFT_BOX_RATIO * self._h:
                reid_emb, lab_hist = self._extract(frame, x1, y1, x2, y2)
                if reid_emb is not None:
                    tid = self._check_drift(raw_tid, tid, reid_emb, lab_hist)

            # only cache features from large well-lit boxes
            if box_h > self.GOOD_BOX_RATIO * self._h and reid_emb is not None:
                self._emb_cache.setdefault(tid, deque(maxlen=self.CACHE_MAXLEN)).append(reid_emb)
                self._lab_cache.setdefault(tid, deque(maxlen=self.CACHE_MAXLEN)).append(lab_hist)

            if raw_tid not in self._id_remap and raw_tid not in self._emb_cache:
                tid = self._try_remap(raw_tid, reid_emb, lab_hist, frame, x1, y1, x2, y2)

            final.append((x1, y1, x2, y2, tid, float(conf)))

        self._update_lost(active_raw)
        return final

    def _check_drift(self, raw_tid: int, tid: int,
                     reid_emb: np.ndarray, lab_hist: np.ndarray) -> int:
        """Detect tracker-internal ID swaps by measuring embedding drift.
        If drift exceeds threshold, save the old identity to lost_tracks and
        try to remap to the correct canonical id."""
        if tid not in self._emb_cache or len(self._emb_cache[tid]) < 2:
            return tid

        mean_reid = np.mean(self._emb_cache[tid], axis=0)
        mean_lab  = np.mean(self._lab_cache[tid],  axis=0)
        drift     = self._combined_dist(reid_emb, mean_reid, lab_hist, mean_lab)

        if drift <= self.DRIFT_THRESHOLD:
            return tid

        # preserve old identity before it's overwritten
        self._lost[tid] = {'reid': mean_reid, 'lab': mean_lab,
                           'last_frame': self._frame_idx}
        self._emb_cache.pop(tid, None)
        self._lab_cache.pop(tid, None)

        best_tid = self._find_match(reid_emb, lab_hist)
        if best_tid is not None and best_tid != tid:
            self._id_remap[raw_tid] = best_tid
            return best_tid

        return tid

    def _try_remap(self, raw_tid: int, reid_emb, lab_hist,
                   frame, x1, y1, x2, y2) -> int:
        """For brand-new tracker IDs, attempt to match against lost tracks."""
        if reid_emb is None:
            reid_emb, lab_hist = self._extract(frame, x1, y1, x2, y2)
        if reid_emb is None:
            return raw_tid

        best_tid = self._find_match(reid_emb, lab_hist)
        if best_tid is not None:
            self._id_remap[raw_tid] = best_tid
            return best_tid

        return raw_tid

    # ------------------------------------------------------------------
    def _extract(self, frame, x1, y1, x2, y2):
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self._w, x2), min(self._h, y2)
        if x2 <= x1 or y2 <= y1:
            return None, None

        xyxy     = np.array([[x1, y1, x2, y2]], dtype=np.float32)
        feats    = self._reid.get_features(xyxy, frame)
        reid_emb = feats[0] if feats is not None and len(feats) > 0 else None
        if reid_emb is None:
            return None, None

        lab_hist = self._lab_histogram(frame, x1, y1, x2, y2)
        return reid_emb, lab_hist

    def _lab_histogram(self, frame, x1, y1, x2, y2, bins=16):
        """Normalised ab histogram in Lab space — luminance-invariant colour descriptor."""
        crop = frame[y1:y2, x1:x2]
        lab  = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab)
        a_ch = lab[:, :, 1].ravel().astype(np.float32)
        b_ch = lab[:, :, 2].ravel().astype(np.float32)
        ha, _ = np.histogram(a_ch, bins=bins, range=(0, 255), density=True)
        hb, _ = np.histogram(b_ch, bins=bins, range=(0, 255), density=True)
        hist  = np.concatenate([ha, hb]).astype(np.float32)
        return hist / (np.linalg.norm(hist) + 1e-6)

    @staticmethod
    def _cosine_dist(a, b):
        a = a / (np.linalg.norm(a) + 1e-6)
        b = b / (np.linalg.norm(b) + 1e-6)
        return 1.0 - float(np.dot(a, b))

    def _combined_dist(self, reid_a, reid_b, lab_a, lab_b):
        return (self.REID_WEIGHT * self._cosine_dist(reid_a, reid_b) +
                self.LAB_WEIGHT  * self._cosine_dist(lab_a,  lab_b))

    def _find_match(self, reid_emb, lab_hist):
        best_tid, best_dist = None, self.REMAP_THRESHOLD
        for lost_tid, info in self._lost.items():
            dist = self._combined_dist(reid_emb, info['reid'], lab_hist, info['lab'])
            if dist < best_dist:
                best_dist = dist
                best_tid  = lost_tid
        return best_tid

    def _expire_lost(self):
        stale = [t for t, v in self._lost.items()
                 if self._frame_idx - v['last_frame'] > self.LOST_TIMEOUT]
        for t in stale:
            del self._lost[t]

    def _update_lost(self, active_raw: set):
        canonical_active = {self._id_remap.get(t, t) for t in active_raw}
        for tid in self._emb_cache.keys() - canonical_active:
            if tid not in self._lost:
                self._lost[tid] = {
                    'reid':       np.mean(self._emb_cache[tid], axis=0),
                    'lab':        np.mean(self._lab_cache[tid],  axis=0),
                    'last_frame': self._frame_idx,
                }
