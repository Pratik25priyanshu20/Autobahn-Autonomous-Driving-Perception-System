"""Backward-compatible re-exports from canonical src.types package."""
from src.types.detection import Detection
from src.types.perception import FramePacket, PerceptionOutput
from src.types.safety import SafetyStatus
from src.types.track import Track
from src.types.world_model import WorldModel

__all__ = [
    "Detection",
    "FramePacket",
    "PerceptionOutput",
    "SafetyStatus",
    "Track",
    "WorldModel",
]
