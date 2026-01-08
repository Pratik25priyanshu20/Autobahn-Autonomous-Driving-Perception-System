"""Canonical lane types for APS++."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class LaneGeometry:
    """Geometric lane state from any lane detector backend."""

    left_lane: Optional[np.ndarray] = None
    right_lane: Optional[np.ndarray] = None
    ego_offset_m: Optional[float] = None
    ego_offset_px: float = 0.0
    confidence: float = 0.0
    stable: bool = False


@dataclass
class LaneState:
    """High-level lane detection state."""

    left_detected: bool = False
    right_detected: bool = False
    center_offset_px: float = 0.0
    jitter_px: float = 0.0
    stable: bool = False
    confidence: float = 0.0
