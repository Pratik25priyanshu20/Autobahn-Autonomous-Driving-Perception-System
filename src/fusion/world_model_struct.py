from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time

from src.fusion.ego_state import EgoState
from src.fusion.lane_geometry import LaneGeometry
from src.fusion.tracked_object import TrackedObject


@dataclass
class WorldModel:
    timestamp: float = field(default_factory=time.time)
    ego: EgoState = field(default_factory=EgoState)
    objects: List[TrackedObject] = field(default_factory=list)
    lanes: Optional[LaneGeometry] = None

    safety_state: Dict[str, bool] = field(default_factory=dict)
    confidence: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"t={self.timestamp:.2f}s | "
            f"objects={len(self.objects)} | "
            f"lanes={'yes' if self.lanes else 'no'}"
        )
