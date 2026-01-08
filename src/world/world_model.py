"""Deprecated — re-exports from canonical types.  Use src.types instead."""
from src.types.ego import EgoState
from src.types.lanes import LaneState
from src.types.safety import SafetyState
from src.types.world_model import WorldModel

# Keep TrackedObject definition for backward compat with world module consumers
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TrackedObject:
    track_id: int
    class_name: str
    bbox_xyxy: List[float]
    confidence: float = 1.0
    distance_m: Optional[float] = None
    relative_speed_mps: Optional[float] = None
    ttc_s: Optional[float] = None


__all__ = ["EgoState", "LaneState", "SafetyState", "TrackedObject", "WorldModel"]
