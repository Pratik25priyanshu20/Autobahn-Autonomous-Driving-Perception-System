"""Sensor degradation scoring for camera, LIDAR, and radar (Task 4)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


@dataclass
class CameraHealth:
    score: float = 1.0
    brightness: float = 128.0
    blur: float = 500.0
    occlusion: float = 0.0


@dataclass
class LidarHealth:
    score: float = 1.0
    point_ratio: float = 1.0
    intensity_ok: bool = True


@dataclass
class RadarHealth:
    score: float = 1.0
    detection_consistency: float = 1.0


class SensorHealthMonitor:
    """Monitors health of camera, LIDAR, and radar sensors."""

    def __init__(
        self,
        brightness_range: tuple[float, float] = (40.0, 220.0),
        blur_threshold: float = 100.0,
        expected_lidar_points: int = 10000,
        health_threshold: float = 0.5,
    ):
        self.brightness_range = brightness_range
        self.blur_threshold = blur_threshold
        self.expected_lidar_points = expected_lidar_points
        self.health_threshold = health_threshold
        self._radar_history: deque[int] = deque(maxlen=10)
        self._latest: dict[str, float] = {}

    def assess_camera(self, frame: np.ndarray) -> CameraHealth:
        """Assess camera health via brightness, blur, and edge density."""
        if frame is None or frame.size == 0:
            return CameraHealth(score=0.0, brightness=0.0, blur=0.0, occlusion=1.0)

        # Brightness: mean pixel value of grayscale
        gray = np.mean(frame, axis=2) if len(frame.shape) == 3 else frame.astype(np.float64)
        brightness = float(np.mean(gray))

        # Brightness score
        lo, hi = self.brightness_range
        if brightness < lo:
            brightness_score = brightness / lo
        elif brightness > hi:
            brightness_score = max(0.0, 1.0 - (brightness - hi) / (255.0 - hi))
        else:
            brightness_score = 1.0

        # Blur: Laplacian variance (higher = sharper)
        if cv2 is not None:
            gray_u8 = gray.astype(np.uint8) if gray.dtype != np.uint8 else gray
            laplacian = cv2.Laplacian(gray_u8, cv2.CV_64F)
            blur_var = float(laplacian.var())
        else:
            # Approximate Laplacian variance without cv2
            dx = np.diff(gray, axis=1)
            dy = np.diff(gray, axis=0)
            blur_var = float(np.var(dx) + np.var(dy))

        blur_score = min(1.0, blur_var / self.blur_threshold)

        # Occlusion: edge density via simple gradient magnitude
        if cv2 is not None:
            edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
            edge_density = float(np.count_nonzero(edges)) / max(1, edges.size)
        else:
            edge_density = float(np.mean(np.abs(np.diff(gray, axis=1)))) / 255.0
        occlusion_score = min(1.0, edge_density / 0.05) if edge_density > 0 else 0.0

        score = 0.5 * brightness_score + 0.3 * blur_score + 0.2 * occlusion_score
        score = max(0.0, min(1.0, score))

        health = CameraHealth(score=score, brightness=brightness, blur=blur_var, occlusion=1.0 - occlusion_score)
        self._latest["camera"] = score
        return health

    def assess_lidar(self, point_cloud: Any) -> LidarHealth:
        """Assess LIDAR health via point count and intensity distribution."""
        if point_cloud is None:
            return LidarHealth(score=0.0, point_ratio=0.0, intensity_ok=False)

        points = getattr(point_cloud, "points", None)
        if points is None:
            return LidarHealth(score=0.0, point_ratio=0.0, intensity_ok=False)

        num_points = len(points)
        point_ratio = min(1.0, num_points / max(1, self.expected_lidar_points))

        # Intensity check (column 3 if available)
        intensity_ok = True
        if points.shape[1] >= 4:
            intensities = points[:, 3]
            if np.std(intensities) < 1e-3 and len(intensities) > 10:
                intensity_ok = False  # all same intensity = likely sensor error

        score = 0.7 * point_ratio + 0.3 * (1.0 if intensity_ok else 0.0)
        score = max(0.0, min(1.0, score))

        health = LidarHealth(score=score, point_ratio=point_ratio, intensity_ok=intensity_ok)
        self._latest["lidar"] = score
        return health

    def assess_radar(self, radar_frame: Any) -> RadarHealth:
        """Assess radar health via detection count consistency (moving average)."""
        if radar_frame is None:
            return RadarHealth(score=0.5, detection_consistency=0.5)

        dets = getattr(radar_frame, "detections", [])
        count = len(dets)
        self._radar_history.append(count)

        if len(self._radar_history) < 2:
            return RadarHealth(score=0.8, detection_consistency=0.8)

        avg = sum(self._radar_history) / len(self._radar_history)
        if avg < 1:
            consistency = 0.3
        else:
            deviation = abs(count - avg) / avg
            consistency = max(0.0, min(1.0, 1.0 - deviation))

        score = max(0.0, min(1.0, consistency))
        health = RadarHealth(score=score, detection_consistency=consistency)
        self._latest["radar"] = score
        return health

    def overall_health(self) -> float:
        """Weighted average of latest sensor scores."""
        if not self._latest:
            return 1.0
        weights = {"camera": 0.5, "lidar": 0.3, "radar": 0.2}
        total_w = 0.0
        total_s = 0.0
        for sensor, score in self._latest.items():
            w = weights.get(sensor, 0.1)
            total_w += w
            total_s += w * score
        return total_s / max(total_w, 1e-6)

    def degraded(self) -> bool:
        """True if overall health is below threshold."""
        return self.overall_health() < self.health_threshold
