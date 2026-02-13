"""Backward-compatible re-exports from canonical src.types package."""
from __future__ import annotations

# TrackedObject kept here for backward compat (maps to fusion-specific tracked object)
from dataclasses import dataclass

from src.types.ego import EgoState
from src.types.lanes import LaneGeometry
from src.types.safety import SafetyState
from src.types.world_model import DrivableArea


@dataclass
class TrackedObject:
    track_id: int
    class_name: str
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    distance_m: float | None = None
    relative_velocity_mps: float | None = None


__all__ = [
    "DrivableArea",
    "EgoState",
    "LaneGeometry",
    "SafetyState",
    "TrackedObject",
]
