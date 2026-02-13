

def pixel_to_camera(x: float, y: float, depth: float, fx: float, fy: float, cx: float, cy: float) -> tuple[float, float, float]:
    X = (x - cx) * depth / fx  # noqa: N806
    Y = (y - cy) * depth / fy  # noqa: N806
    Z = depth  # noqa: N806
    return X, Y, Z
