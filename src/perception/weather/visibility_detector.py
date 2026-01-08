"""Visibility / weather condition detector (Phase 3.3).

Uses image statistics (contrast, brightness, saturation) to classify
conditions: clear | fog | dark | glare.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


@dataclass
class VisibilityResult:
    condition: str  # "clear" | "fog" | "dark" | "glare"
    confidence: float
    brightness: float
    contrast: float
    degraded: bool


class VisibilityDetector:
    """Detect weather/visibility conditions from image statistics."""

    def __init__(
        self,
        dark_threshold: float = 60.0,
        glare_threshold: float = 210.0,
        fog_contrast_threshold: float = 30.0,
        fog_brightness_min: float = 100.0,
    ):
        self.dark_threshold = dark_threshold
        self.glare_threshold = glare_threshold
        self.fog_contrast_threshold = fog_contrast_threshold
        self.fog_brightness_min = fog_brightness_min

    def detect(self, frame_bgr: np.ndarray) -> VisibilityResult:
        if cv2 is None:
            return VisibilityResult("clear", 0.0, 0.0, 0.0, False)

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))

        condition = "clear"
        degraded = False
        confidence = 0.8

        if brightness < self.dark_threshold:
            condition = "dark"
            degraded = True
            confidence = min(1.0, (self.dark_threshold - brightness) / self.dark_threshold)
        elif brightness > self.glare_threshold:
            condition = "glare"
            degraded = True
            confidence = min(1.0, (brightness - self.glare_threshold) / (255 - self.glare_threshold + 1e-6))
        elif contrast < self.fog_contrast_threshold and brightness > self.fog_brightness_min:
            condition = "fog"
            degraded = True
            confidence = min(1.0, (self.fog_contrast_threshold - contrast) / self.fog_contrast_threshold)
        else:
            confidence = min(1.0, contrast / 60.0)

        return VisibilityResult(
            condition=condition,
            confidence=confidence,
            brightness=brightness,
            contrast=contrast,
            degraded=degraded,
        )
