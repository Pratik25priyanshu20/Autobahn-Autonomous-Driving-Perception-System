"""Deprecated — re-exports from canonical types.  Use src.types instead."""
from src.types.ego import EgoState
from src.types.world_model import WorldModel
from src.fusion.tracked_object import TrackedObject
from src.fusion.lane_geometry import LaneGeometry

__all__ = ["EgoState", "LaneGeometry", "TrackedObject", "WorldModel"]
