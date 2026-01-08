"""Backward-compatible re-exports — canonical WorldModel lives in src.types.world_model."""
from src.types.world_model import DrivableArea, RuntimeStats, WorldModel

__all__ = ["DrivableArea", "RuntimeStats", "WorldModel"]


def build_world_model(perception_output):
    """Simple factory used by fusion_engine (legacy compat)."""
    from src.types.world_model import WorldModel as WM

    wm = WM()
    wm.detections = list(getattr(perception_output, "detections", []))
    wm.tracks = list(getattr(perception_output, "tracks", []))
    return wm
