from src.safety.rules import evaluate


def test_safety_warning_triggers():
    status = evaluate(distance_m=5.0, relative_speed_mps=10.0, config={"ttc": {"warning_s": 3.0}, "risk": {"escalation_threshold": 0.5}})
    assert status.warnings
    assert status.risk_score is not None
