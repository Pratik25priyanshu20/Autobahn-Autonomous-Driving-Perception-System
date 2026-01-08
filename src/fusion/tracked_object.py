"""Deprecated — kept for backward compat.  Prefer src.types.track.Track."""
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TrackedObject:
    track_id: int
    cls: str
    bbox: Tuple[int, int, int, int]
    confidence: float

    x: Optional[float] = None
    y: Optional[float] = None
    vx: Optional[float] = None
    vy: Optional[float] = None

    age: int = 0
