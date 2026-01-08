from typing import Dict, List

from src.safety.risk import risk_score
from src.safety.ttc import compute_ttc
from src.utils.types import SafetyStatus


def evaluate(distance_m: float, relative_speed_mps: float, config: Dict) -> SafetyStatus:
    warnings: List[str] = []
    ttc_value = compute_ttc(distance_m, relative_speed_mps)
    ttc_cfg = config.get("ttc", {})
    if ttc_value is not None:
        if ttc_value <= ttc_cfg.get("critical_s", 1.5):
            warnings.append("critical_ttc")
        elif ttc_value <= ttc_cfg.get("warning_s", 3.0):
            warnings.append("ttc_warning")

    score = risk_score(distance_m, relative_speed_mps, config.get("risk"))
    if score >= config.get("risk", {}).get("escalation_threshold", 0.6):
        warnings.append("high_risk")

    degraded = bool(config.get("degraded_mode", {}).get("enable", False) and warnings)
    return SafetyStatus(ttc_s=ttc_value, risk_score=score, warnings=warnings, degraded_mode=degraded)
