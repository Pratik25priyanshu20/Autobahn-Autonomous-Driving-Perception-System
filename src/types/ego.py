"""Canonical EgoState type for APS++."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EgoState:
    """Vehicle ego state in world coordinates."""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    speed: float = 0.0
    speed_mps: float = 0.0
    yaw_rate: float = 0.0
    acceleration: float = 0.0
