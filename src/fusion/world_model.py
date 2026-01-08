from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class RuntimeStats:
    fps: float = 0.0
    stages_ms: Dict[str, float] = field(default_factory=dict)


@dataclass
class DrivableArea:
    mask: Optional[np.ndarray] = None
    confidence: float = 0.0


@dataclass
class WorldModel:
    """
    Canonical per-frame state object.
    Everything writes into this. Everything reads from this.
    """

    frame_id: int
    frame: Any  # numpy.ndarray (OpenCV frame)
    detections: List[Any] = field(default_factory=list)  # List[Detection] from YOLO
    tracks: List[Any] = field(default_factory=list)  # populated in Phase 2 tracking
    trajectories: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)
    lanes: Dict[str, Any] = field(default_factory=dict)
    fcw: Dict[str, Any] = field(default_factory=dict)
    fcw_pre: Dict[str, Any] = field(default_factory=dict)
    safety: Dict[str, Any] = field(default_factory=dict)
    drivable_area: DrivableArea = field(default_factory=DrivableArea)
    warnings: List[str] = field(default_factory=list)
    runtime: RuntimeStats = field(default_factory=RuntimeStats)
    _prev: "WorldModel | None" = field(default=None, repr=False)
    _frame_idx: int = 0

    def snapshot(self):
        self._prev = deepcopy(self)
        self._frame_idx += 1

    @staticmethod
    def ema(prev: float | None, curr: float, alpha: float = 0.8) -> float:
        return curr if prev is None else alpha * curr + (1.0 - alpha) * prev

    def set_lane_geometry(self, lane: Dict[str, Any]):
        # Smooth ego offset to reduce flicker
        if self._prev and getattr(self._prev, "lanes", None):
            prev_lane = self._prev.lanes
            prev_off = prev_lane.get("ego_offset_px")
            curr_off = lane.get("ego_offset_px")
            if prev_off is not None and curr_off is not None:
                lane["ego_offset_px"] = self.ema(prev_off, curr_off, alpha=0.85)

        lane["stable"] = (lane.get("lane_confidence", 0.0) >= 0.7) and (abs(lane.get("ego_offset_px", 0.0)) <= 0.6)
        self.lanes = lane
        self.confidence["lane"] = lane.get("lane_confidence", 0.0)

    def set_objects(self, objects: List[Any]):
        if self._prev:
            prev_map = {getattr(o, "track_id", None): o for o in getattr(self._prev, "tracks", [])}
            for o in objects:
                p = prev_map.get(getattr(o, "track_id", None))
                if p is not None:
                    prev_conf = getattr(p, "conf", getattr(p, "confidence", None))
                    curr_conf = getattr(o, "conf", getattr(o, "confidence", None))
                    if curr_conf is not None:
                        smoothed = self.ema(prev_conf, curr_conf, alpha=0.6) if prev_conf is not None else curr_conf
                        if hasattr(o, "conf"):
                            o.conf = smoothed
                        else:
                            o.confidence = smoothed
        self.tracks = objects
        self.confidence["objects"] = (sum(getattr(o, "conf", getattr(o, "confidence", 0.0)) for o in objects) / len(objects)) if objects else 0.0

    def set_drivable_area(self, mask: np.ndarray, conf: float):
        self.drivable_area = DrivableArea(mask=mask, confidence=conf)
        self.confidence["drivable_area"] = conf

    def summary(self) -> str:
        return (
            f"frame={self.frame_id} "
            f"objects={len(self.tracks)} "
            f"lanes={'yes' if self.lanes else 'no'} "
            f"fcw={self.fcw.get('state', 'N/A') if self.fcw else 'N/A'} "
            f"ldw={self.lanes.get('lane_departure') if self.lanes else 'None'}"
        )
