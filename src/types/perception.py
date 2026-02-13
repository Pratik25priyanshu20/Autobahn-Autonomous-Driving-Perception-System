"""Canonical perception I/O types for APS++."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from src.types.pointcloud import PointCloud
    from src.types.radar import RadarFrame


@dataclass
class FramePacket:
    """Single sensor frame with metadata."""

    frame: np.ndarray | None = None
    timestamp: float = 0.0
    sensor_id: str = "camera_front"
    pose: dict[str, float] | None = None
    point_cloud: PointCloud | None = None
    calibration: Any | None = None  # KITTICalibration (varies by dataset)
    labels: list[Any] | None = None  # ground truth labels
    radar_frame: RadarFrame | None = None


@dataclass
class PerceptionOutput:
    """Aggregated output from all perception modules."""

    detections: list[Any] = field(default_factory=list)
    tracks: list[Any] = field(default_factory=list)
    lanes: dict[str, Any] | None = None
    segmentation: np.ndarray | None = None
    depth: np.ndarray | None = None
