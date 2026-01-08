def risk_score(distance_m: float, velocity_mps: float, weights=None) -> float:
    weights = weights or {"velocity_weight": 0.7, "distance_weight": 0.3}
    distance_term = 1.0 / max(distance_m, 1e-3)
    velocity_term = velocity_mps
    return weights["velocity_weight"] * velocity_term + weights["distance_weight"] * distance_term
