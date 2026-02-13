"""Base class for lane detection backends (Phase 1.2)."""
from __future__ import annotations

import abc
from typing import Any

import numpy as np


class BaseLaneDetector(abc.ABC):
    @abc.abstractmethod
    def infer(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        """Return dict with: left_line, right_line, ego_offset_px, lane_confidence, lane_center_x, lane_stable, lane_center_jitter_px."""
        ...
