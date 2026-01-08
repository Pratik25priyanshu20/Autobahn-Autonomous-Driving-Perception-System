from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import time


@dataclass
class EgoState:
    speed_mps: float = 0.0
    yaw_rate: float = 0.0
    accel_mps2: float = 0.0


@dataclass
class TrackedObject:
    track_id: int
    class_name: str
    bbox_xyxy: List[float]
    confidence: float = 1.0

    distance_m: Optional[float] = None
    relative_speed_mps: Optional[float] = None
    ttc_s: Optional[float] = None


@dataclass
class LaneState:
    left_detected: bool
    right_detected: bool
    center_offset_px: float
    jitter_px: float
    stable: bool
    confidence: float


@dataclass
class SafetyState:
    fcw_state: str = "NORMAL"
    ldw_state: Optional[str] = None
    risk_score: float = 0.0


@dataclass
class WorldModel:
    frame_id: int
    timestamp: float = field(default_factory=time.time)

    ego: EgoState = field(default_factory=EgoState)
    objects: List[TrackedObject] = field(default_factory=list)
    lanes: Optional[LaneState] = None
    safety: SafetyState = field(default_factory=SafetyState)

    confidence: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"[WORLD] frame={self.frame_id} "
            f"objects={len(self.objects)} "
            f"lanes={'yes' if self.lanes else 'no'} "
            f"fcw={self.safety.fcw_state} "
            f"ldw={self.safety.ldw_state}"
        )
