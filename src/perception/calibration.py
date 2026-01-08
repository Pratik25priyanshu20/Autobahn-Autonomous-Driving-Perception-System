"""Confidence calibration via temperature scaling (Phase 0.5)."""
from __future__ import annotations

import math
from typing import List

from src.types.detection import Detection


class ConfidenceCalibrator:
    """Applies temperature scaling to detection confidence scores.

    Temperature > 1 softens probabilities (under-confident model).
    Temperature < 1 sharpens probabilities (over-confident model).
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = max(0.01, temperature)

    def calibrate(self, detections: List[Detection]) -> List[Detection]:
        if self.temperature == 1.0:
            return detections
        for det in detections:
            logit = math.log(max(det.conf, 1e-7) / max(1.0 - det.conf, 1e-7))
            scaled = logit / self.temperature
            det.conf = 1.0 / (1.0 + math.exp(-scaled))
        return detections
