from typing import Any, Dict

from src.control.base_controller import BaseController


class MPCController(BaseController):
    def __init__(self, horizon: int = 10, dt: float = 0.1):
        self.horizon = horizon
        self.dt = dt

    def plan(self, world_model: Any) -> Dict[str, float]:
        # Placeholder for MPC optimization result
        return {"steer": 0.0, "throttle": 0.0, "horizon": self.horizon, "dt": self.dt}
