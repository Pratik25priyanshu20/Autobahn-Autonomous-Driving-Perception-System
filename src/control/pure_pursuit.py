from typing import Any, Dict

from src.control.base_controller import BaseController


def pure_pursuit(world_model: Any, lookahead_m: float, wheelbase_m: float) -> Dict[str, float]:
    # Placeholder steering computation
    return {"steer": 0.0, "throttle": 0.0, "lookahead_m": lookahead_m, "wheelbase_m": wheelbase_m}


class PurePursuitController(BaseController):
    def __init__(self, lookahead_m: float = 6.0, wheelbase_m: float = 2.8):
        self.lookahead_m = lookahead_m
        self.wheelbase_m = wheelbase_m

    def plan(self, world_model: Any) -> Dict[str, float]:
        return pure_pursuit(world_model, self.lookahead_m, self.wheelbase_m)
