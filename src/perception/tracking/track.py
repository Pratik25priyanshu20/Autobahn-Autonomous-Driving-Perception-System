from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Track:
    track_id: int
    bbox_xyxy: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    class_name: str
    conf: float
    age: int = 0
    is_confirmed: bool = True
    velocity_px_per_frame: Optional[Tuple[float, float]] = None  # (vx, vy)
