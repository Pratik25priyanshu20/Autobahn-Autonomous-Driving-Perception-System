from dataclasses import dataclass


@dataclass
class EgoState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    speed: float = 0.0
    acceleration: float = 0.0
