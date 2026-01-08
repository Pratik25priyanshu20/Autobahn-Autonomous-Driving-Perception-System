"""Canonical perception I/O types for APS++."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class FramePacket:
    """Single sensor frame with metadata."""

    frame: object  # np.ndarray (H, W, 3)
    timestamp: float = 0.0
    sensor_id: str = "camera_front"
    pose: Optional[Dict[str, float]] = None


@dataclass
class PerceptionOutput:
    """Aggregated output from all perception modules."""

    detections: List[object] = field(default_factory=list)
    tracks: List[object] = field(default_factory=list)
    lanes: Optional[object] = None
    segmentation: Optional[object] = None
    depth: Optional[object] = None
