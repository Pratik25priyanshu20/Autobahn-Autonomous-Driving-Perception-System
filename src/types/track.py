"""Canonical Track type for APS++."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Track:
    """Unified tracked-object representation.

    Produced by any tracker (DeepSORT, ByteTrack, etc.) and consumed
    by fusion, safety, and visualization subsystems.
    """

    track_id: int
    bbox_xyxy: Tuple[int, int, int, int]
    class_name: str
    conf: float
    age: int = 0
    is_confirmed: bool = True
    velocity_px_per_frame: Optional[Tuple[float, float]] = None

    # World-frame attributes (populated by orchestrator / Kalman)
    x: Optional[float] = None
    y: Optional[float] = None
    vx: Optional[float] = None
    vy: Optional[float] = None
    ttc: Optional[float] = None
    risk: Optional[str] = None
