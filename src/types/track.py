"""Canonical Track type for APS++."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Track:
    """Unified tracked-object representation.

    Produced by any tracker (DeepSORT, ByteTrack, etc.) and consumed
    by fusion, safety, and visualization subsystems.
    """

    track_id: int
    bbox_xyxy: tuple[int, int, int, int]
    class_name: str
    conf: float
    age: int = 0
    is_confirmed: bool = True
    velocity_px_per_frame: tuple[float, float] | None = None

    # World-frame attributes (populated by orchestrator / Kalman)
    x: float | None = None
    y: float | None = None
    vx: float | None = None
    vy: float | None = None
    ttc: float | None = None
    risk: str | None = None
