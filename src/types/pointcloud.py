"""Point cloud and LIDAR detection types for APS++."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PointCloud:
    """Raw point cloud data (Nx4: x, y, z, intensity)."""

    points: np.ndarray  # shape (N, 4)

    def __post_init__(self):
        if self.points.ndim != 2 or self.points.shape[1] < 4:
            raise ValueError(
                f"PointCloud expects (N, 4) array, got {self.points.shape}"
            )

    @property
    def xyz(self) -> np.ndarray:
        return self.points[:, :3]

    @property
    def intensity(self) -> np.ndarray:
        return self.points[:, 3]

    def __len__(self) -> int:
        return self.points.shape[0]


@dataclass
class LidarDetection3D:
    """Single 3D detection from LIDAR processing."""

    center: np.ndarray  # shape (3,) — x, y, z
    dimensions: np.ndarray  # shape (3,) — length, width, height
    yaw: float = 0.0
    class_name: str = "unknown"
    confidence: float = 0.0


@dataclass
class BEVGrid:
    """Bird's-eye-view pillar encoding."""

    grid: np.ndarray  # shape (C, H, W)
    resolution_m: float = 0.2
    origin: tuple[float, float] = (-40.0, -40.0)
