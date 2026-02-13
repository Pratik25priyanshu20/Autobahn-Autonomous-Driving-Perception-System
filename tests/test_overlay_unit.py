"""Unit tests for visualization overlay helpers."""
from __future__ import annotations

import numpy as np

from src.visualization.overlay import draw_saliency, draw_sensor_health


class TestDrawSensorHealth:
    def test_draws_bars_without_crash(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        health = {"camera": 0.95, "lidar": 0.6, "radar": 0.3}
        result = draw_sensor_health(frame, health)
        assert result is not None
        assert result.shape == (480, 640, 3)
        # The result should differ from the all-black input (bars drawn)
        assert result.sum() > 0

    def test_empty_health_returns_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = draw_sensor_health(frame, {})
        # Empty dict -> early return, should return original frame
        assert result is frame


class TestDrawSaliency:
    def test_blends_heatmap_onto_frame(self):
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        heatmap = np.random.rand(480, 640).astype(np.float32)
        result = draw_saliency(frame, heatmap)
        assert result is not None
        assert result.shape == (480, 640, 3)
        # Result should be different from input since heatmap is blended
        assert not np.array_equal(result, frame)

    def test_none_heatmap_returns_frame(self):
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        result = draw_saliency(frame, None)
        assert result is frame
