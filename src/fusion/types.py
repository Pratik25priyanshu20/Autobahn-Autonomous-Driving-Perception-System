"""Backward-compatible re-exports from canonical src.types package."""
from __future__ import annotations

from src.types.ego import EgoState
from src.types.lanes import LaneGeometry
from src.types.safety import SafetyState
from src.types.world_model import DrivableArea

# TrackedObject kept here for backward compat (maps to fusion-specific tracked object)
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class TrackedObject:
    track_id: int
    class_name: str
    bbox_xyxy: Tuple[int, int, int, int]
    confidence: float
    distance_m: Optional[float] = None
    relative_velocity_mps: Optional[float] = None


__all__ = [
    "DrivableArea",
    "EgoState",
    "LaneGeometry",
    "SafetyState",
    "TrackedObject",
]
