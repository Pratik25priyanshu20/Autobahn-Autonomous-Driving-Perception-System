def risk_score(distance_m: float, velocity_mps: float, weights=None) -> float:
    defaults = {"velocity_weight": 0.7, "distance_weight": 0.3}
    if weights is not None:
        defaults.update(weights)
    distance_term = 1.0 / max(distance_m, 1e-3)
    velocity_term = velocity_mps
    return defaults["velocity_weight"] * velocity_term + defaults["distance_weight"] * distance_term
