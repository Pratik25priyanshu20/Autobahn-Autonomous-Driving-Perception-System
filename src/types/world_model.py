"""Canonical WorldModel and supporting types for APS++."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from src.fusion.lidar_camera_fusion import FusedDetection
    from src.fusion.temporal_predictor import PredictionPoint
    from src.prediction.interaction_model import InteractionEvent
    from src.safety.occupancy_grid import OccupancyGrid
    from src.types.pointcloud import BEVGrid, PointCloud
    from src.types.radar import RadarDetection


@dataclass
class RuntimeStats:
    fps: float = 0.0
    stages_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class DrivableArea:
    mask: np.ndarray | None = None
    confidence: float = 0.0


@dataclass
class WorldModel:
    """Canonical per-frame state object.

    Everything writes into this. Everything reads from this.
    """

    frame_id: int = 0
    frame: np.ndarray | None = None
    detections: list[Any] = field(default_factory=list)
    tracks: list[Any] = field(default_factory=list)
    trajectories: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    lanes: dict[str, Any] = field(default_factory=dict)
    fcw: dict[str, Any] = field(default_factory=dict)
    fcw_pre: dict[str, Any] = field(default_factory=dict)
    safety: dict[str, Any] = field(default_factory=dict)
    drivable_area: DrivableArea = field(default_factory=DrivableArea)
    warnings: list[str] = field(default_factory=list)
    runtime: RuntimeStats = field(default_factory=RuntimeStats)

    # Phase 1.3: depth map
    depth_map: np.ndarray | None = None

    # Phase 2.3: temporal predictions
    predictions: dict[int, list[PredictionPoint]] = field(default_factory=dict)

    # Phase 2.4: top-K trajectory hypotheses
    predictions_topk: dict[int, list[PredictionPoint]] = field(default_factory=dict)

    # Phase 3.1: occupancy grid
    occupancy: OccupancyGrid | None = None

    # LIDAR / fusion fields
    lidar_detections: list[Any] = field(default_factory=list)
    fused_detections: list[FusedDetection] = field(default_factory=list)
    point_cloud: PointCloud | None = None
    bev_grid: BEVGrid | None = None

    # Phase 6.2: control output
    control: dict[str, Any] | None = None

    # Round 2: Radar fusion
    radar_detections: list[RadarDetection] = field(default_factory=list)
    radar_fused: bool = False

    # Round 2: Sensor health
    sensor_health: dict[str, float] = field(default_factory=dict)

    # Round 2: Saliency / explainability
    saliency_map: np.ndarray | None = None

    # Round 2: Interaction model
    interactions: list[InteractionEvent] = field(default_factory=list)

    _prev: WorldModel | None = field(default=None, repr=False)
    _frame_idx: int = 0

    def snapshot(self):
        self._prev = deepcopy(self)
        self._frame_idx += 1

    @staticmethod
    def ema(prev: float | None, curr: float, alpha: float = 0.8) -> float:
        return curr if prev is None else alpha * curr + (1.0 - alpha) * prev

    def set_lane_geometry(self, lane: dict[str, Any]):
        if self._prev and getattr(self._prev, "lanes", None):
            prev_lane = self._prev.lanes
            prev_off = prev_lane.get("ego_offset_px")
            curr_off = lane.get("ego_offset_px")
            if prev_off is not None and curr_off is not None:
                lane["ego_offset_px"] = self.ema(prev_off, curr_off, alpha=0.85)

        lane["stable"] = (lane.get("lane_confidence", 0.0) >= 0.7) and (abs(lane.get("ego_offset_px", 0.0)) <= 0.6)
        self.lanes = lane

    def set_objects(self, objects: list[Any]):
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

    def set_drivable_area(self, mask: np.ndarray, conf: float):
        self.drivable_area = DrivableArea(mask=mask, confidence=conf)

    def summary(self) -> str:
        return (
            f"frame={self.frame_id} "
            f"objects={len(self.tracks)} "
            f"lanes={'yes' if self.lanes else 'no'} "
            f"fcw={self.fcw.get('state', 'N/A') if self.fcw else 'N/A'} "
            f"ldw={self.lanes.get('lane_departure') if self.lanes else 'None'}"
        )
