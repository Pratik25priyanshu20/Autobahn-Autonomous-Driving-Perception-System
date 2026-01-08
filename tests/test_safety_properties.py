"""Property-based safety tests (Phase 5.2)."""
import pytest

try:
    from hypothesis import given, strategies as st, assume
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False
    # Provide stubs so class bodies parse without error
    def given(**kw):  # type: ignore
        return lambda f: f
    class _St:
        def __getattr__(self, _):
            return lambda *a, **kw: None
    st = _St()  # type: ignore

from src.safety.ttc import compute_ttc
from src.safety.risk import risk_score
from src.safety.fcw import fcw_state
from src.safety.safety_manager import SafetyManager, SafetyStateEnum


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestTTCProperties:
    @given(distance=st.floats(min_value=0.01, max_value=1000), speed=st.floats(min_value=0.01, max_value=100))
    def test_ttc_non_negative(self, distance, speed):
        ttc = compute_ttc(distance, speed)
        if ttc is not None:
            assert ttc >= 0, f"TTC must be non-negative, got {ttc}"

    @given(distance=st.floats(min_value=0.01, max_value=1000), speed=st.floats(min_value=-100, max_value=-0.01))
    def test_ttc_none_when_diverging(self, distance, speed):
        ttc = compute_ttc(distance, speed)
        # When speed is negative (diverging), TTC should be None or very large
        assert ttc is None or ttc < 0 or ttc > 100


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestRiskProperties:
    @given(
        distance=st.floats(min_value=0.1, max_value=1000),
        speed=st.floats(min_value=0.0, max_value=100),
    )
    def test_risk_score_finite(self, distance, speed):
        score = risk_score(distance, speed)
        assert score is not None
        import math
        assert math.isfinite(score), f"Risk score must be finite, got {score}"


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestFCWStateProperties:
    @given(ttc=st.one_of(st.none(), st.floats(min_value=-10, max_value=100)))
    def test_fcw_state_valid(self, ttc):
        state = fcw_state(ttc)
        assert state in ("NORMAL", "CAUTION", "WARNING", "CRITICAL", None)


class TestSafetyManagerNeverCrashes:
    def test_evaluate_with_none_inputs(self):
        sm = SafetyManager()
        out = sm.evaluate()
        assert out.state in SafetyStateEnum

    def test_evaluate_with_all_inputs(self):
        sm = SafetyManager()
        out = sm.evaluate(
            ldw_departure="LEFT",
            fcw_state="WARNING",
            fcw_ttc_s=1.2,
            fcw_pre_active=True,
            lane_ok=True,
            bsd_warnings=[{"side": "left"}],
        )
        assert out.state in SafetyStateEnum
        assert isinstance(out.message, str)
