"""Tests for sensor degradation scoring (Task 4)."""
from __future__ import annotations

import numpy as np

from src.safety.sensor_health import (
    SensorHealthMonitor,
)


class _MockPointCloud:
    def __init__(self, num_points: int, intensity_std: float = 10.0):
        self.points = np.random.rand(num_points, 4).astype(np.float32)
        if intensity_std < 0.001:
            self.points[:, 3] = 0.5  # constant intensity


class _MockRadarFrame:
    def __init__(self, num_detections: int):
        self.detections = [None] * num_detections


class TestCameraHealth:
    def test_normal_frame_high_score(self):
        monitor = SensorHealthMonitor()
        frame = np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8)
        health = monitor.assess_camera(frame)
        assert health.score > 0.5
        assert 40 < health.brightness < 220

    def test_dark_frame_low_score(self):
        monitor = SensorHealthMonitor(brightness_range=(40.0, 220.0))
        frame = np.zeros((480, 640, 3), dtype=np.uint8) + 5
        health = monitor.assess_camera(frame)
        assert health.score < 0.5
        assert health.brightness < 10

    def test_blurred_frame_lower_score(self):
        monitor = SensorHealthMonitor(blur_threshold=100.0)
        # Uniform frame = zero blur
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        health = monitor.assess_camera(frame)
        assert health.blur < 5  # almost no variance

    def test_bright_frame_overexposed(self):
        monitor = SensorHealthMonitor(brightness_range=(40.0, 220.0))
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 250
        health = monitor.assess_camera(frame)
        assert health.score < 0.8  # brightness penalty

    def test_empty_frame(self):
        monitor = SensorHealthMonitor()
        health = monitor.assess_camera(np.array([]))
        assert health.score == 0.0


class TestLidarHealth:
    def test_normal_point_cloud(self):
        monitor = SensorHealthMonitor(expected_lidar_points=1000)
        pc = _MockPointCloud(1000)
        health = monitor.assess_lidar(pc)
        assert health.score > 0.7
        assert health.point_ratio >= 1.0

    def test_low_point_cloud(self):
        monitor = SensorHealthMonitor(expected_lidar_points=10000)
        pc = _MockPointCloud(100)
        health = monitor.assess_lidar(pc)
        assert health.point_ratio < 0.1
        assert health.score < 0.5

    def test_none_point_cloud(self):
        monitor = SensorHealthMonitor()
        health = monitor.assess_lidar(None)
        assert health.score == 0.0

    def test_constant_intensity_warning(self):
        monitor = SensorHealthMonitor(expected_lidar_points=100)
        pc = _MockPointCloud(200, intensity_std=0.0)
        health = monitor.assess_lidar(pc)
        assert not health.intensity_ok


class TestRadarHealth:
    def test_consistent_detections(self):
        monitor = SensorHealthMonitor()
        for _ in range(5):
            monitor.assess_radar(_MockRadarFrame(10))
        health = monitor.assess_radar(_MockRadarFrame(10))
        assert health.score > 0.7

    def test_empty_radar(self):
        monitor = SensorHealthMonitor()
        health = monitor.assess_radar(None)
        assert health.score == 0.5

    def test_inconsistent_detections(self):
        monitor = SensorHealthMonitor()
        for _ in range(5):
            monitor.assess_radar(_MockRadarFrame(10))
        health = monitor.assess_radar(_MockRadarFrame(0))
        assert health.score < 0.7


class TestOverallHealth:
    def test_all_healthy(self):
        monitor = SensorHealthMonitor(expected_lidar_points=100)
        frame = np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8)
        monitor.assess_camera(frame)
        monitor.assess_lidar(_MockPointCloud(200))
        for _ in range(3):
            monitor.assess_radar(_MockRadarFrame(5))
        assert monitor.overall_health() > 0.4
        assert not monitor.degraded()

    def test_degraded_flag(self):
        monitor = SensorHealthMonitor(health_threshold=0.9)
        monitor.assess_camera(np.zeros((100, 100, 3), dtype=np.uint8))
        assert monitor.degraded()
