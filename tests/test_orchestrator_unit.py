"""Unit tests for Orchestrator extracted stage methods."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np


def _minimal_cfg() -> dict:
    """Return a minimal config dict with all heavy features disabled."""
    return {
        "perception": {"runtime": "pytorch", "device": "cpu"},
        "tracking": {"enabled": False, "interval": 2, "kalman": False},
        "lane": {"enabled": False},
        "ldw": {"enabled": False},
        "fcw": {"enabled": False},
        "segmentation": {"enabled": False},
        "depth": {"enabled": False},
        "weather": {"enabled": False},
        "bsd": {"enabled": False},
        "occupancy_grid": {"enabled": False},
        "performance": {"fps_smoothing": 0.9},
        "video": {"resize": {"enabled": False}},
        "safety": {"sensor_health": {"enabled": False}},
        "explainability": {"enabled": False},
        "interaction": {"enabled": False},
        "radar": {"enabled": False},
        "lidar": {"enabled": False},
    }


def _build_orchestrator(cfg_overrides: dict | None = None):
    """Construct an Orchestrator with mocked heavy dependencies."""
    cfg = _minimal_cfg()
    if cfg_overrides:
        for key, value in cfg_overrides.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key].update(value)
            else:
                cfg[key] = value

    mock_logger = MagicMock()

    with (
        patch("src.runtime.orchestrator.YOLODetector") as mock_yolo,
        patch("src.runtime.orchestrator.DeepSORTTracker") as mock_tracker,
        patch("src.runtime.orchestrator.CannyHoughLaneDetector"),
        patch("src.runtime.orchestrator.DeepLabV3Segmenter"),
        patch("src.runtime.orchestrator.SafetyManager") as mock_safety,
    ):
        mock_yolo.return_value = MagicMock()
        mock_tracker.return_value = MagicMock()
        mock_safety.return_value = MagicMock()

        from src.runtime.orchestrator import Orchestrator

        orch = Orchestrator(cfg, mock_logger)
    return orch


# ── _preprocess ──────────────────────────────────────────────────────


class TestPreprocess:
    def test_resize_disabled_returns_same_frame(self):
        orch = _build_orchestrator()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out, ms = orch._preprocess(frame)
        assert out is frame
        assert ms >= 0.0

    def test_resize_enabled_returns_resized(self):
        orch = _build_orchestrator(
            {"video": {"resize": {"enabled": True, "width": 320, "height": 240}}}
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out, ms = orch._preprocess(frame)
        assert out.shape == (240, 320, 3)
        assert ms >= 0.0


# ── _assess_sensors ──────────────────────────────────────────────────


class TestAssessSensors:
    def test_returns_empty_dict_when_monitor_is_none(self):
        orch = _build_orchestrator()
        assert orch.sensor_health_monitor is None
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        health, ms = orch._assess_sensors(frame, None, [])
        assert health == {}
        assert ms == 0.0


# ── _run_tracking_update ─────────────────────────────────────────────


class TestRunTrackingUpdate:
    def test_returns_cached_tracks_when_not_on_interval(self):
        orch = _build_orchestrator({"tracking": {"enabled": True, "interval": 5, "kalman": False}})
        orch.tracking_enabled = True
        orch.tracking_interval = 5

        # Seed cached tracks
        cached_tracks = [MagicMock()]
        cached_trajectories = {1: [(10, 20)]}
        orch._last_tracks = cached_tracks
        orch._last_trajectories = cached_trajectories

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # frame_id=3 is NOT a multiple of 5 -> should return cached
        tracks, trajectories, ms = orch._run_tracking_update(frame, 3, [])
        assert tracks is cached_tracks
        assert trajectories is cached_trajectories
        assert ms == 0.0


# ── _analyze_lanes ───────────────────────────────────────────────────


class TestAnalyzeLanes:
    def test_empty_dict_returns_ldw_not_allowed(self):
        orch = _build_orchestrator()
        ldw_allowed, departure = orch._analyze_lanes({}, [])
        assert ldw_allowed is False
        assert departure is None

    def test_none_lanes_returns_ldw_not_allowed(self):
        orch = _build_orchestrator()
        ldw_allowed, departure = orch._analyze_lanes(None, [])
        assert ldw_allowed is False
        assert departure is None


# ── _evaluate_interactions ───────────────────────────────────────────


class TestEvaluateInteractions:
    def test_returns_empty_list_when_model_is_none(self):
        orch = _build_orchestrator()
        assert orch.interaction_model is None
        interactions, ms = orch._evaluate_interactions([], [])
        assert interactions == []
        assert ms == 0.0


# ── _evaluate_bsd ────────────────────────────────────────────────────


class TestEvaluateBSD:
    def test_returns_none_when_detector_is_none(self):
        orch = _build_orchestrator()
        assert orch.bsd_detector is None
        result = orch._evaluate_bsd([], [])
        assert result is None


# ── _process_lidar ───────────────────────────────────────────────────


class TestProcessLidar:
    def test_returns_default_dict_when_processor_is_none(self):
        orch = _build_orchestrator()
        assert orch.lidar_processor is None
        result = orch._process_lidar(None, [], [])
        assert result["detections"] == []
        assert result["fused"] == []
        assert result["point_cloud"] is None
        assert result["bev_grid"] is None
        assert result["lidar_ms"] == 0.0
        assert result["bev_ms"] == 0.0
        assert result["fusion_ms"] == 0.0


# ── _process_radar ───────────────────────────────────────────────────


class TestProcessRadar:
    def test_returns_default_dict_when_processor_is_none(self):
        orch = _build_orchestrator()
        assert orch.radar_processor is None
        result = orch._process_radar(None, [], [])
        assert result["detections"] == []
        assert result["fused"] is False
        assert result["radar_ms"] == 0.0
        assert result["fusion_ms"] == 0.0
