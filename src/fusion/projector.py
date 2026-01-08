from typing import Tuple


def pixel_to_camera(x: float, y: float, depth: float, fx: float, fy: float, cx: float, cy: float) -> Tuple[float, float, float]:
    X = (x - cx) * depth / fx
    Y = (y - cy) * depth / fy
    Z = depth
    return X, Y, Z
