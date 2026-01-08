from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class FramePacket:
    frame: object
    timestamp: float
    sensor_id: str = "camera_front"
    pose: Optional[Dict[str, float]] = None


@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]
    score: float
    label: str


@dataclass
class Track:
    track_id: int
    detection: Detection
    age: int = 0


@dataclass
class PerceptionOutput:
    detections: List[Detection] = field(default_factory=list)
    tracks: List[Track] = field(default_factory=list)
    lanes: Optional[object] = None
    segmentation: Optional[object] = None
    depth: Optional[object] = None


@dataclass
class WorldModel:
    objects: List[Detection] = field(default_factory=list)
    tracks: List[Track] = field(default_factory=list)
    lane_graph: Optional[object] = None
    drivable_area: Optional[object] = None
    depth_map: Optional[object] = None
    ego_state: Optional[Dict[str, float]] = None


@dataclass
class SafetyStatus:
    ttc_s: Optional[float]
    risk_score: Optional[float]
    warnings: List[str] = field(default_factory=list)
    degraded_mode: bool = False
