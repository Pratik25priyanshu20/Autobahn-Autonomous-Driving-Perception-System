"""Pure Pursuit controller (Phase 6.2).

Computes steering angle from lane ego_offset with safety gating.
"""
from __future__ import annotations

import math
from typing import Any

from src.control.base_controller import BaseController


def pure_pursuit(world_model: Any, lookahead_m: float, wheelbase_m: float, max_steer_deg: float = 30.0) -> dict[str, float]:
    """Compute steering from lane ego offset and safety state."""
    lanes = getattr(world_model, "lanes", {})
    safety = getattr(world_model, "safety", {})

    ego_offset_px = lanes.get("ego_offset_px", 0.0) if isinstance(lanes, dict) else 0.0
    lane_stable = lanes.get("lane_stable", False) if isinstance(lanes, dict) else False

    # Convert pixel offset to meters (approx)
    px_to_m = 0.02
    lateral_error_m = (ego_offset_px or 0.0) * px_to_m

    # Pure pursuit geometry
    if abs(lateral_error_m) < 0.01 or not lane_stable:
        steer_rad = 0.0
    else:
        # Simplified: atan(2 * L * sin(alpha) / ld)
        alpha = math.atan2(lateral_error_m, lookahead_m)
        steer_rad = math.atan2(2.0 * wheelbase_m * math.sin(alpha), lookahead_m)

    steer_deg = math.degrees(steer_rad)
    steer_deg = max(-max_steer_deg, min(max_steer_deg, steer_deg))

    # Safety-gated throttle
    safety_state = safety.get("state", "NORMAL") if isinstance(safety, dict) else "NORMAL"
    if safety_state == "CRITICAL":
        throttle = 0.0
        brake = 1.0
    elif safety_state == "WARNING":
        throttle = 0.1
        brake = 0.5
    elif safety_state == "CAUTION":
        throttle = 0.3
        brake = 0.0
    else:
        throttle = 0.5
        brake = 0.0

    return {
        "steer_deg": steer_deg,
        "steer_rad": steer_rad,
        "throttle": throttle,
        "brake": brake,
        "lateral_error_m": lateral_error_m,
        "lane_stable": lane_stable,
        "safety_state": safety_state,
    }


class PurePursuitController(BaseController):
    def __init__(self, lookahead_m: float = 6.0, wheelbase_m: float = 2.8, max_steer_deg: float = 30.0):
        self.lookahead_m = lookahead_m
        self.wheelbase_m = wheelbase_m
        self.max_steer_deg = max_steer_deg

    def plan(self, world_model: Any) -> dict[str, float]:
        return pure_pursuit(world_model, self.lookahead_m, self.wheelbase_m, self.max_steer_deg)
