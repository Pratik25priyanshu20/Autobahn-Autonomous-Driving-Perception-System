"""End-to-end smoke tests for the full perception pipeline with synthetic data."""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml

try:
    import cv2
except ImportError:
    cv2 = None

_cv2_required = pytest.mark.skipif(cv2 is None, reason="cv2 required")


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

class _MockDetector:
    """Drop-in replacement for YOLODetector that returns no detections."""

    def infer(self, frame, conf_thres=0.25):
        return []


class _MockTracker:
    """Drop-in replacement for DeepSORTTracker that returns empty tracks."""

    def update(self, frame, detections):
        return [], {}


# ---------------------------------------------------------------------------
# Minimal config factory
# ---------------------------------------------------------------------------

def _minimal_cfg(**overrides) -> dict:
    """Return a minimal Orchestrator config with most features disabled."""
    cfg: dict = {
        "perception": {"runtime": "pytorch", "conf_thres": 0.25},
        "tracking": {"enabled": False, "interval": 1, "kalman": False},
        "lane": {"enabled": False},
        "segmentation": {"enabled": False},
        "depth": {"enabled": False},
        "weather": {"enabled": False},
        "ldw": {"enabled": False},
        "fcw": {"enabled": False},
        "bsd": {"enabled": False},
        "occupancy_grid": {"enabled": False},
        "safety": {"asil": {"enabled": False}},
        "performance": {"target_fps": 30, "fps_smoothing": 0.9},
        "video": {"resize": {"enabled": False}},
    }
    # Apply overrides (shallow merge of top-level keys)
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(val)
        else:
            cfg[key] = val
    return cfg


def _make_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """Create a synthetic BGR frame filled with mid-grey."""
    return np.full((h, w, 3), 128, dtype=np.uint8)


def _build_orchestrator(cfg: dict):
    """Instantiate Orchestrator with mocked heavy dependencies."""
    with (
        patch("src.perception.detection.yolo.YOLODetector", return_value=_MockDetector()),
        patch("src.perception.tracking.deepsort_tracker.DeepSORTTracker", return_value=_MockTracker()),
    ):
        from src.runtime.orchestrator import Orchestrator

        logger = logging.getLogger("test_e2e_smoke")
        orch = Orchestrator(cfg, logger)

    # Belt-and-suspenders: ensure mocked detector/tracker are active
    orch.detector = _MockDetector()
    orch.tracker = _MockTracker()
    return orch


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

@_cv2_required
class TestE2ESmoke:
    """Smoke tests that exercise the full Orchestrator pipeline on synthetic frames."""

    # 1 -----------------------------------------------------------------
    def test_single_frame_processing(self):
        """Process one synthetic frame and verify WorldModel structure."""
        cfg = _minimal_cfg()
        orch = _build_orchestrator(cfg)
        frame = _make_frame()

        wm = orch.process_frame(frame_id=0, frame=frame, packet=None)

        # WorldModel identity checks
        assert wm.frame_id == 0
        assert wm.frame is not None
        assert isinstance(wm.warnings, list)

        # Runtime stats populated
        assert wm.runtime is not None
        assert wm.runtime.fps >= 0
        assert isinstance(wm.runtime.stages_ms, dict)

    # 2 -----------------------------------------------------------------
    def test_multi_frame_pipeline(self):
        """Process 10 frames in sequence; FPS meter should produce non-zero fps."""
        cfg = _minimal_cfg()
        orch = _build_orchestrator(cfg)

        wm = None
        for i in range(10):
            frame = _make_frame()
            wm = orch.process_frame(frame_id=i, frame=frame, packet=None)
            assert wm is not None
            assert wm.frame_id == i

        # After 10 ticks the EMA FPS meter must report something positive
        assert wm is not None
        assert wm.runtime.fps > 0

    # 3 -----------------------------------------------------------------
    def test_lane_detection_integration(self):
        """Enable Canny-Hough lane detector on a frame with drawn lane lines."""
        cfg = _minimal_cfg(
            lane={"enabled": True, "backend": "canny_hough"},
        )
        orch = _build_orchestrator(cfg)

        # Draw two strong diagonal lines mimicking lane markings.
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Left lane line
        cv2.line(frame, (160, 479), (300, 264), (255, 255, 255), 3)
        # Right lane line
        cv2.line(frame, (480, 479), (340, 264), (255, 255, 255), 3)

        wm = orch.process_frame(frame_id=0, frame=frame, packet=None)

        # The lane detector should have populated the lanes dict
        assert isinstance(wm.lanes, dict)
        assert len(wm.lanes) > 0, "Expected lanes dict to be populated with detected lane info"

    # 4 -----------------------------------------------------------------
    def test_safety_output_structure(self):
        """Verify wm.safety contains the expected keys."""
        cfg = _minimal_cfg()
        orch = _build_orchestrator(cfg)
        frame = _make_frame()

        wm = orch.process_frame(frame_id=0, frame=frame, packet=None)

        assert isinstance(wm.safety, dict)
        assert "state" in wm.safety, f"Missing 'state' key in safety: {wm.safety}"
        assert "message" in wm.safety, f"Missing 'message' key in safety: {wm.safety}"
        assert "color" in wm.safety, f"Missing 'color' key in safety: {wm.safety}"

    # 5 -----------------------------------------------------------------
    def test_config_validator_integration(self):
        """Run validate_config against the real system.yaml and safety.yaml."""
        from src.utils.config_validator import validate_config

        base = Path(__file__).resolve().parent.parent
        system_path = base / "configs" / "system.yaml"
        safety_path = base / "configs" / "safety.yaml"

        with open(system_path) as f:
            system_cfg = yaml.safe_load(f)
        with open(safety_path) as f:
            safety_cfg = yaml.safe_load(f)

        # Should NOT raise ConfigValidationError
        warns = validate_config(system_cfg, safety_cfg=safety_cfg)
        assert isinstance(warns, list)
