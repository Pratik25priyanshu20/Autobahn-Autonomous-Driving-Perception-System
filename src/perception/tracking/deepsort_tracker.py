from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

from src.perception.detection.yolo import Detection
from src.perception.tracking.track import Track


@dataclass
class TrackerConfig:
    max_age: int = 25
    n_init: int = 2
    nms_max_overlap: float = 1.0
    max_iou_distance: float = 0.7
    max_cosine_distance: float = 0.2  # lower = stricter appearance match


class DeepSORTTracker:
    """
    Appearance-based multi-object tracking (DeepSORT).
    Uses an internal ReID embedder (runs CPU on macOS, which is OK).
    """

    def __init__(self, cfg: TrackerConfig | None = None):
        cfg = cfg or TrackerConfig()
        self.tracker = DeepSort(
            max_age=cfg.max_age,
            n_init=cfg.n_init,
            nms_max_overlap=cfg.nms_max_overlap,
            max_iou_distance=cfg.max_iou_distance,
            max_cosine_distance=cfg.max_cosine_distance,
            embedder="mobilenet",
            half=True,  # safe; set to False if issues arise
            bgr=True,  # OpenCV frames are BGR
        )
        self.prev_centers: Dict[int, Tuple[float, float]] = {}
        self.trajectories: Dict[int, Deque[Tuple[int, int]]] = defaultdict(lambda: deque(maxlen=30))

    def update(self, frame: np.ndarray, detections: List[Detection]) -> Tuple[List[Track], Dict[int, List[Tuple[int, int]]]]:
        """
        Returns:
          tracks: list of Track objects for this frame
          trajectories: dict(track_id -> list of center points)
        """
        ds_dets = [([d.x1, d.y1, d.x2, d.y2], float(d.conf), d.class_name) for d in detections]
        ds_tracks = self.tracker.update_tracks(ds_dets, frame=frame)

        out_tracks: List[Track] = []

        for t in ds_tracks:
            if not t.is_confirmed():
                continue

            ltrb = t.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            track_id = int(t.track_id)
            cls = getattr(t, "det_class", None) or "object"
            conf = float(getattr(t, "det_conf", 0.0) or 0.0)

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            vel = None
            if track_id in self.prev_centers:
                px, py = self.prev_centers[track_id]
                vel = (cx - px, cy - py)
            self.prev_centers[track_id] = (cx, cy)
            self.trajectories[track_id].append((int(cx), int(cy)))

            out_tracks.append(
                Track(
                    track_id=track_id,
                    bbox_xyxy=(x1, y1, x2, y2),
                    class_name=str(cls),
                    conf=conf,
                    age=int(getattr(t, "age", 0) or 0),
                    is_confirmed=True,
                    velocity_px_per_frame=vel,
                )
            )

        traj_out = {tid: list(points) for tid, points in self.trajectories.items()}
        return out_tracks, traj_out
