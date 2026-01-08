from dataclasses import dataclass
from typing import Tuple


@dataclass
class TrackedObject:
    track_id: int
    cls: str
    bbox: Tuple[int, int, int, int]
    confidence: float

    x: float | None = None
    y: float | None = None
    vx: float | None = None
    vy: float | None = None

    age: int = 0
