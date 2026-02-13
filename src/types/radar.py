"""Radar detection types for APS++."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RadarDetection:
    """Single radar return."""

    range_m: float = 0.0
    azimuth_deg: float = 0.0
    velocity_mps: float = 0.0
    rcs_dbsm: float = 0.0
    x_m: float = 0.0
    y_m: float = 0.0


@dataclass
class RadarFrame:
    """Collection of radar detections for a single scan."""

    detections: list[RadarDetection] = field(default_factory=list)
    timestamp: float = 0.0
