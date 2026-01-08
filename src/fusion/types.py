from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class EgoState:
    speed_mps: float = 0.0
    yaw_rate: float = 0.0
    acceleration: float = 0.0


@dataclass
class TrackedObject:
    track_id: int
    class_name: str
    bbox_xyxy: Tuple[int, int, int, int]
    confidence: float
    distance_m: Optional[float] = None
    relative_velocity_mps: Optional[float] = None


@dataclass
class LaneGeometry:
    left_lane: Optional[np.ndarray] = None
    right_lane: Optional[np.ndarray] = None
    ego_offset_m: Optional[float] = None
    confidence: float = 0.0
    stable: bool = False


@dataclass
class DrivableArea:
    mask: Optional[np.ndarray] = None
    confidence: float = 0.0


@dataclass
class SafetyState:
    state: str = "NORMAL"
    message: str = "System OK"
