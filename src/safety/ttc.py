from __future__ import annotations


def compute_ttc(distance_proxy: float, closing_rate: float) -> float | None:
    """
    TTC = distance / closing_rate
    closing_rate must be > 0 (approaching).
    Returns TTC in seconds-like units (proxy-based).
    """
    if closing_rate is None or closing_rate <= 1e-6:
        return None
    if distance_proxy is None or distance_proxy <= 1e-6:
        return None
    return distance_proxy / closing_rate
