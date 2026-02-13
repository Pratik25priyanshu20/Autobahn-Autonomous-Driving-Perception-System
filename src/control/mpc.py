"""Model Predictive Control (Phase 6.2).

Basic trajectory optimization with safety constraints.
"""
from __future__ import annotations

import math
from typing import Any

from src.control.base_controller import BaseController


class MPCController(BaseController):
    """Simple MPC controller with horizon-based optimization."""

    def __init__(self, horizon: int = 10, dt: float = 0.1, wheelbase_m: float = 2.8, max_steer_deg: float = 30.0):
        self.horizon = horizon
        self.dt = dt
        self.wheelbase_m = wheelbase_m
        self.max_steer_deg = max_steer_deg

    def plan(self, world_model: Any) -> dict[str, Any]:
        lanes = getattr(world_model, "lanes", {})
        safety = getattr(world_model, "safety", {})

        ego_offset_px = lanes.get("ego_offset_px", 0.0) if isinstance(lanes, dict) else 0.0
        lane_stable = lanes.get("lane_stable", False) if isinstance(lanes, dict) else False
        px_to_m = 0.02
        lateral_error_m = (ego_offset_px or 0.0) * px_to_m

        # Simple trajectory optimization: minimize lateral error over horizon
        trajectory: list[dict[str, float]] = []
        x, y, yaw = 0.0, 0.0, 0.0
        speed = 5.0  # m/s assumed

        safety_state = safety.get("state", "NORMAL") if isinstance(safety, dict) else "NORMAL"
        if safety_state == "CRITICAL":
            speed = 0.0
        elif safety_state == "WARNING":
            speed = 2.0

        best_steer = 0.0
        if lane_stable and abs(lateral_error_m) > 0.01:
            # P-controller for MPC first step
            best_steer = -0.5 * lateral_error_m
            best_steer = max(-math.radians(self.max_steer_deg), min(math.radians(self.max_steer_deg), best_steer))

        for _step in range(self.horizon):
            x += speed * math.cos(yaw) * self.dt
            y += speed * math.sin(yaw) * self.dt
            yaw += (speed / self.wheelbase_m) * math.tan(best_steer) * self.dt
            trajectory.append({"x": x, "y": y, "yaw": yaw, "speed": speed, "steer": best_steer})

        return {
            "steer_deg": math.degrees(best_steer),
            "steer_rad": best_steer,
            "throttle": min(0.5, speed / 10.0),
            "brake": 1.0 if safety_state == "CRITICAL" else 0.0,
            "trajectory": trajectory,
            "horizon": self.horizon,
            "safety_state": safety_state,
        }
