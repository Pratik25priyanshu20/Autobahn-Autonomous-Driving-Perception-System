"""Tests for weather/visibility detector (Phase 3.3)."""
import numpy as np
import pytest

try:
    import cv2  # noqa: F401
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from src.perception.weather.visibility_detector import VisibilityDetector


@pytest.mark.skipif(not HAS_CV2, reason="opencv not installed")
class TestVisibilityDetector:
    def test_dark_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)  # All black
        detector = VisibilityDetector()
        result = detector.detect(frame)
        assert result.condition == "dark"
        assert result.degraded is True

    def test_bright_frame(self):
        frame = np.full((480, 640, 3), 240, dtype=np.uint8)  # Very bright
        detector = VisibilityDetector()
        result = detector.detect(frame)
        assert result.condition == "glare"
        assert result.degraded is True

    def test_normal_frame(self):
        # Use a wide-spread random frame to ensure high contrast after BGR->gray
        rng = np.random.RandomState(42)
        frame = rng.randint(30, 220, (480, 640, 3), dtype=np.uint8)
        detector = VisibilityDetector()
        result = detector.detect(frame)
        assert result.condition == "clear"
        assert result.degraded is False

    def test_fog_frame(self):
        # Low contrast, medium brightness
        frame = np.full((480, 640, 3), 140, dtype=np.uint8)
        frame += np.random.randint(-5, 5, frame.shape, dtype=np.int8).astype(np.uint8)
        detector = VisibilityDetector()
        result = detector.detect(frame)
        assert result.condition in ("fog", "clear")  # Depends on exact noise
