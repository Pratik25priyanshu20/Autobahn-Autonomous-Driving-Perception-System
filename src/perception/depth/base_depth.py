"""Base class for monocular depth estimation (Phase 1.3)."""
from __future__ import annotations

import abc
from typing import Dict

import numpy as np


class BaseDepthEstimator(abc.ABC):
    @abc.abstractmethod
    def infer(self, frame: np.ndarray) -> Dict[str, object]:
        """Return dict with keys: depth_map (H,W float32), confidence (float), latency_ms (float)."""
        ...
