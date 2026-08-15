from collections import defaultdict, deque
import numpy as np


class VehicleTracker:
    """Aggregates per-vehicle motion history from SpeedEstimator output.

    Maintains a trajectory (cx, cy, speed, u, v) per tracked vehicle and
    computes behavioural features used by the LSTM classifier:
      - avg_speed, max_speed, speed_variance
      - braking_count  (sharp speed drops over 3 consecutive frames)
      - weave_count    (lateral direction reversals in u component)
    """

    WINDOW         = 30    # frames needed before features are computed
    BRAKE_WINDOW   = 3     # frames over which speed drop is measured
    BRAKE_THRESH   = 3.0   # px/frame drop to count as a braking event
    WEAVE_WINDOW   = 15    # frames to look back for lateral reversals
    WEAVE_MIN      = 3     # minimum direction changes to count as weaving
    PRUNE_AFTER    = 90    # frames since last seen before track is removed

    def __init__(self):
        # tid -> deque of (cx, cy, speed, u, v)
        self._trajectories  = defaultdict(lambda: deque(maxlen=self.WINDOW))
        self._last_seen     = {}   # tid -> frame index
        self._frame_idx     = 0

    def update(self, frame_w: int, frame_h: int, tracks: list) -> None:
        """Ingest one frame of tracks from SpeedEstimator.

        Args:
            frame_w, frame_h: frame dimensions for normalising cx, cy
            tracks: list of (x1, y1, x2, y2, tid, conf, speed, u, v)
        """
        self._frame_idx += 1
        self._prune()

        for track in tracks:
            x1, y1, x2, y2, tid, _, speed, u, v = track
            cx = ((x1 + x2) / 2.0) / frame_w   # normalise to [0, 1]
            cy = ((y1 + y2) / 2.0) / frame_h
            self._trajectories[tid].append((cx, cy, float(speed), float(u), float(v)))
            self._last_seen[tid] = self._frame_idx

    def get_features(self, tid: int) -> dict | None:
        """Return behavioural features for a vehicle.

        Returns None if fewer than WINDOW frames have been tracked.
        """
        traj = self._trajectories.get(tid)
        if traj is None or len(traj) < self.WINDOW:
            return None

        speeds = [t[2] for t in traj]
        u_vals = [t[3] for t in traj]

        return {
            'avg_speed':      float(np.mean(speeds)),
            'max_speed':      float(np.max(speeds)),
            'speed_variance': float(np.var(speeds)),
            'braking_count':  self._count_braking(speeds),
            'weave_count':    self._count_weaving(u_vals),
        }

    def get_trajectory_sequence(self, tid: int) -> np.ndarray | None:
        """Return (WINDOW, 5) array of [cx, cy, speed, u, v] for LSTM input.

        Returns None if fewer than WINDOW frames have been tracked.
        """
        traj = self._trajectories.get(tid)
        if traj is None or len(traj) < self.WINDOW:
            return None
        return np.array(traj, dtype=np.float32)   # shape (30, 5)

    # ------------------------------------------------------------------
    def _count_braking(self, speeds: list) -> int:
        count = 0
        for i in range(len(speeds) - self.BRAKE_WINDOW):
            drop = speeds[i] - speeds[i + self.BRAKE_WINDOW]
            if drop > self.BRAKE_THRESH:
                count += 1
        return count

    def _count_weaving(self, u_vals: list) -> int:
        if len(u_vals) < self.WEAVE_WINDOW:
            return 0
        recent = u_vals[-self.WEAVE_WINDOW:]
        changes = sum(
            1 for i in range(1, len(recent) - 1)
            if (recent[i] - recent[i-1]) * (recent[i+1] - recent[i]) < 0
        )
        return changes

    def _prune(self) -> None:
        stale = [tid for tid, last in self._last_seen.items()
                 if self._frame_idx - last > self.PRUNE_AFTER]
        for tid in stale:
            del self._trajectories[tid]
            del self._last_seen[tid]
