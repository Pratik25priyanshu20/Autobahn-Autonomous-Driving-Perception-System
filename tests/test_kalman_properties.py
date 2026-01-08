"""Property-based Kalman filter tests (Phase 5.2)."""
import numpy as np
import pytest

try:
    from hypothesis import given, strategies as st
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

from src.fusion.kalman_tracker import ObjectKalmanFilter, KalmanTrackManager


class TestKalmanFilter:
    def test_basic_predict_update(self):
        kf = ObjectKalmanFilter()
        kf.x[:2] = [5.0, 10.0]
        kf.predict()
        state = kf.update(np.array([5.1, 10.2]))
        assert state[0] != 0 or state[1] != 0

    def test_covariance_positive_semidefinite(self):
        kf = ObjectKalmanFilter()
        kf.x[:2] = [1.0, 2.0]
        for _ in range(10):
            kf.predict()
            kf.update(np.array([1.0 + np.random.randn() * 0.1, 2.0 + np.random.randn() * 0.1]))
            eigenvalues = np.linalg.eigvalsh(kf.P)
            assert np.all(eigenvalues >= -1e-10), f"Covariance not PSD: eigenvalues={eigenvalues}"


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestKalmanProperties:
    @given(
        x=st.floats(min_value=-100, max_value=100),
        y=st.floats(min_value=-100, max_value=100),
    )
    def test_update_returns_finite(self, x, y):
        kf = ObjectKalmanFilter()
        kf.predict()
        state = kf.update(np.array([x, y]))
        assert np.all(np.isfinite(state))


class TestKalmanTrackManager:
    def test_update_and_prune(self):
        mgr = KalmanTrackManager()
        x, y, vx, vy = mgr.update_track(1, 5.0, 10.0)
        assert x is not None
        mgr.update_track(2, 3.0, 7.0)
        mgr.prune({1})
        assert 2 not in mgr.filters
        assert 1 in mgr.filters
