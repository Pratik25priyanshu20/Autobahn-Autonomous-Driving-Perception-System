"""Tests for UFLDv2 lane detector (Phase 1.2)."""
import numpy as np
import pytest

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


@pytest.mark.skipif(not HAS_CV2, reason="opencv not installed")
def test_ufld_fallback_mode():
    from src.perception.lanes.ufld_detector import UFLDv2LaneDetector

    detector = UFLDv2LaneDetector()  # No model loaded — uses fallback
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Draw some lines to detect
    cv2.line(frame, (200, 720), (500, 400), (255, 255, 255), 3)
    cv2.line(frame, (1080, 720), (780, 400), (255, 255, 255), 3)
    result = detector.infer(frame)
    assert "lane_confidence" in result
    assert "ego_offset_px" in result
    assert "lane_stable" in result


def test_ufld_base_class():
    from src.perception.lanes.base_lane_detector import BaseLaneDetector
    assert hasattr(BaseLaneDetector, "infer")
