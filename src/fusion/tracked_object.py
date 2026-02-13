"""Deprecated — kept for backward compat.  Prefer src.types.track.Track."""
from dataclasses import dataclass


@dataclass
class TrackedObject:
    track_id: int
    cls: str
    bbox: tuple[int, int, int, int]
    confidence: float

    x: float | None = None
    y: float | None = None
    vx: float | None = None
    vy: float | None = None

    age: int = 0
