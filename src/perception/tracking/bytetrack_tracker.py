"""ByteTrack tracker (Phase 2.1).

Drop-in replacement for DeepSORTTracker with the same update() signature.
Uses the ``supervision`` library for the ByteTrack algorithm.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

try:
    import supervision as sv
except ImportError:  # pragma: no cover
    sv = None

from src.types.detection import Detection
from src.types.track import Track


class ByteTrackTracker:
    """ByteTrack multi-object tracker via supervision."""

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
    ):
        if sv is None:
            raise ImportError("supervision>=0.18 is required for ByteTrackTracker")
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )
        self.prev_centers: dict[int, tuple[float, float]] = {}
        self.trajectories: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=30))
        self._class_map: dict[int, str] = {}

    def update(
        self, frame: np.ndarray, detections: list[Detection]
    ) -> tuple[list[Track], dict[int, list[tuple[int, int]]]]:
        if not detections:
            return [], {}

        xyxy = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=np.float32)
        confs = np.array([d.conf for d in detections], dtype=np.float32)
        class_ids = np.array([d.class_id for d in detections], dtype=int)

        sv_dets = sv.Detections(xyxy=xyxy, confidence=confs, class_id=class_ids)
        tracked = self.tracker.update_with_detections(sv_dets)

        # Build class name map
        for d in detections:
            self._class_map[d.class_id] = d.class_name

        out_tracks: list[Track] = []
        for i in range(len(tracked)):
            x1, y1, x2, y2 = map(int, tracked.xyxy[i])
            tid = int(tracked.tracker_id[i])
            cls_id = int(tracked.class_id[i]) if tracked.class_id is not None else 0
            cls_name = self._class_map.get(cls_id, "object")
            conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            vel = None
            if tid in self.prev_centers:
                px, py = self.prev_centers[tid]
                vel = (cx - px, cy - py)
            self.prev_centers[tid] = (cx, cy)
            self.trajectories[tid].append((int(cx), int(cy)))

            out_tracks.append(
                Track(
                    track_id=tid,
                    bbox_xyxy=(x1, y1, x2, y2),
                    class_name=cls_name,
                    conf=conf,
                    age=0,
                    is_confirmed=True,
                    velocity_px_per_frame=vel,
                )
            )

        traj_out = {tid: list(pts) for tid, pts in self.trajectories.items()}
        return out_tracks, traj_out
