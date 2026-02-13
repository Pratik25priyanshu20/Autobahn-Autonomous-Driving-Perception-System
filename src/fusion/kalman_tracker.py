"""Per-object Kalman filter for world-frame state estimation (Phase 2.2).

State vector: [x, y, vx, vy]  (position and velocity in meters).
"""
from __future__ import annotations

import numpy as np


class ObjectKalmanFilter:
    """Linear Kalman filter for a single tracked object."""

    def __init__(
        self,
        process_noise: float = 0.5,
        measurement_noise: float = 1.0,
        dt: float = 0.033,
    ):
        self.dt = dt
        # State: [x, y, vx, vy]
        self.x = np.zeros(4, dtype=np.float64)
        # State transition
        self.F = np.eye(4, dtype=np.float64)
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        # Measurement matrix (observe x, y)
        self.H = np.zeros((2, 4), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        # Covariance
        self.P = np.eye(4, dtype=np.float64) * 10.0
        # Process noise
        self.Q = np.eye(4, dtype=np.float64) * process_noise
        # Measurement noise
        self.R = np.eye(2, dtype=np.float64) * measurement_noise

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z: np.ndarray) -> np.ndarray:
        """z = [x_measured, y_measured]"""
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R  # noqa: N806
        K = self.P @ self.H.T @ np.linalg.inv(S)  # noqa: N806
        self.x = self.x + K @ y
        eye = np.eye(4)
        self.P = (eye - K @ self.H) @ self.P
        return self.x.copy()

    @property
    def position(self) -> tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.x[2]), float(self.x[3])


class KalmanTrackManager:
    """Manages per-ID Kalman filters."""

    def __init__(
        self,
        process_noise: float = 0.5,
        measurement_noise: float = 1.0,
        dt: float = 0.033,
    ):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.dt = dt
        self.filters: dict[int, ObjectKalmanFilter] = {}

    def update_track(self, track_id: int, x_m: float, y_m: float) -> tuple[float, float, float, float]:
        """Update (or create) the filter for a track. Returns (x, y, vx, vy)."""
        if track_id not in self.filters:
            kf = ObjectKalmanFilter(self.process_noise, self.measurement_noise, self.dt)
            kf.x[:2] = [x_m, y_m]
            self.filters[track_id] = kf

        kf = self.filters[track_id]
        kf.predict()
        state = kf.update(np.array([x_m, y_m]))
        return float(state[0]), float(state[1]), float(state[2]), float(state[3])

    def prune(self, alive_ids: set) -> None:
        """Remove filters for tracks that are no longer alive."""
        dead = [tid for tid in self.filters if tid not in alive_ids]
        for tid in dead:
            del self.filters[tid]

    def get_filter(self, track_id: int) -> ObjectKalmanFilter | None:
        return self.filters.get(track_id)
