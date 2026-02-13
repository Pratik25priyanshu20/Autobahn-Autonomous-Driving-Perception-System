"""Deprecated — re-exports from canonical types.  Use src.types instead."""
# Keep TrackedObject definition for backward compat with world module consumers
from dataclasses import dataclass

from src.types.ego import EgoState
from src.types.lanes import LaneState
from src.types.safety import SafetyState
from src.types.world_model import WorldModel


@dataclass
class TrackedObject:
    track_id: int
    class_name: str
    bbox_xyxy: list[float]
    confidence: float = 1.0
    distance_m: float | None = None
    relative_speed_mps: float | None = None
    ttc_s: float | None = None


__all__ = ["EgoState", "LaneState", "SafetyState", "TrackedObject", "WorldModel"]
