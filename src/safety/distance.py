def estimate_distance_px(bbox, frame_h):
    """
    Estimate distance proxy using bbox height.
    Larger bbox → closer object.
    """
    x1, y1, x2, y2 = bbox
    box_h = max(1, y2 - y1)
    return frame_h / box_h


def distance_proxy_px(bbox, frame_h):
    """
    Alias for estimate_distance_px with explicit naming.
    Returns a proxy distance in pixel-based units (not meters).
    """
    return estimate_distance_px(bbox, frame_h)
