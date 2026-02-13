"""3D detection types for APS++ (Phase 1.4)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Detection3D:
    """Pseudo-3D bounding box derived from 2D detection + depth."""

    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    class_id: int
    class_name: str

    # 3D position in camera frame (meters)
    x_m: float = 0.0
    y_m: float = 0.0
    z_m: float = 0.0

    # 3D extent estimates (meters)
    width_m: float | None = None
    height_m: float | None = None
    depth_m: float | None = None

    # Orientation (radians)
    yaw: float = 0.0

    @property
    def center_3d(self) -> tuple[float, float, float]:
        return (self.x_m, self.y_m, self.z_m)
