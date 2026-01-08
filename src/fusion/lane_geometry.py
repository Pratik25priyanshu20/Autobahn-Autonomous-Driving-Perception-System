from dataclasses import dataclass
from typing import Optional


@dataclass
class LaneGeometry:
    left_lane: Optional[object] = None
    right_lane: Optional[object] = None
    ego_offset_px: float = 0.0
    confidence: float = 0.0
