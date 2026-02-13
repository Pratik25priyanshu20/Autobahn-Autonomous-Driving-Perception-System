"""Types for data recording and replay."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RecordedFrame:
    """Serialized snapshot of a single world model frame."""

    frame_id: int = 0
    timestamp: float = 0.0
    tracks: list[dict[str, Any]] = field(default_factory=list)
    lanes: dict[str, Any] = field(default_factory=dict)
    fcw_state: str = "NORMAL"
    safety_state: str = "NORMAL"
    detections_count: int = 0
    predictions: dict[str, Any] = field(default_factory=dict)
    sensor_health: dict[str, float] = field(default_factory=dict)
