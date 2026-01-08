"""Temporal object prediction using Kalman velocity (Phase 2.3).

Predicts future positions at configurable horizons (default 0.5s, 1.0s, 2.0s).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from src.fusion.kalman_tracker import KalmanTrackManager


@dataclass
class PredictionPoint:
    t_s: float
    x_m: float
    y_m: float


class TemporalPredictor:
    """Predict future positions from Kalman velocity estimates."""

    def __init__(self, horizons_s: Tuple[float, ...] = (0.5, 1.0, 2.0)):
        self.horizons_s = horizons_s

    def predict(self, kalman_manager: KalmanTrackManager, alive_ids: set) -> Dict[int, List[PredictionPoint]]:
        predictions: Dict[int, List[PredictionPoint]] = {}
        for tid in alive_ids:
            kf = kalman_manager.get_filter(tid)
            if kf is None:
                continue
            x, y = kf.position
            vx, vy = kf.velocity
            pts: List[PredictionPoint] = []
            for t in self.horizons_s:
                pts.append(PredictionPoint(t_s=t, x_m=x + vx * t, y_m=y + vy * t))
            predictions[tid] = pts
        return predictions
