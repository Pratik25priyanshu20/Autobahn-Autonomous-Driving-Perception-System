"""Tests for rule-based interaction model (Task 7)."""
from __future__ import annotations

from src.prediction.interaction_model import InteractionModel


class _MockTrack:
    def __init__(self, track_id, x, y, vx=0.0, vy=0.0):
        self.track_id = track_id
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy


class TestGapAcceptance:
    def test_safe_gap(self):
        model = InteractionModel(min_gap_s=3.0)
        ego = {"x": 0.0, "y": 0.0, "vx": 0.0, "vy": 10.0}
        adjacent = [_MockTrack(1, 3.5, 50.0, 0.0, 10.0)]  # far ahead
        result = model.gap_acceptance(ego, adjacent)
        assert result is None

    def test_unsafe_gap(self):
        model = InteractionModel(min_gap_s=3.0)
        ego = {"x": 0.0, "y": 0.0, "vx": 0.0, "vy": 10.0}
        # Adjacent vehicle close ahead with different speed → small time gap
        adjacent = [_MockTrack(1, 3.5, 5.0, 0.0, 5.0)]  # 5m ahead, slower → closing
        result = model.gap_acceptance(ego, adjacent)
        assert result is not None
        assert result.type == "gap_acceptance"
        assert result.risk_level in ("medium", "high")


class TestYieldHeuristic:
    def test_vehicle_from_right(self):
        model = InteractionModel(lane_width_m=3.5)
        ego = {"x": 0.0, "y": 0.0}
        tracks = [_MockTrack(1, 5.0, 0.5, vx=-2.0, vy=0.0)]  # from right, approaching
        result = model.yield_heuristic(ego, tracks)
        assert result is not None
        assert result.type == "yield"
        assert "Rechts-vor-Links" in result.description

    def test_no_yield_needed(self):
        model = InteractionModel(lane_width_m=3.5)
        ego = {"x": 0.0, "y": 0.0}
        tracks = [_MockTrack(1, 5.0, 0.5, vx=2.0, vy=0.0)]  # moving away
        result = model.yield_heuristic(ego, tracks)
        assert result is None


class TestFollowingDistance:
    def test_following_too_close(self):
        model = InteractionModel(safe_following_s=2.0)
        ego = {"x": 0.0, "y": 0.0, "vy": 15.0}  # 15 m/s = 54 km/h
        lead = _MockTrack(1, 0.0, 10.0)  # 10m ahead
        result = model.following_distance(ego, lead)
        # 10m / 15 m/s = 0.67s < 2.0s
        assert result is not None
        assert result.type == "following_distance"
        assert result.risk_level == "high"

    def test_safe_following(self):
        model = InteractionModel(safe_following_s=2.0)
        ego = {"x": 0.0, "y": 0.0, "vy": 10.0}
        lead = _MockTrack(1, 0.0, 30.0)  # 30m ahead
        result = model.following_distance(ego, lead)
        # 30m / 10 m/s = 3.0s > 2.0s
        assert result is None


class TestCutInPrediction:
    def test_cut_in_from_left(self):
        model = InteractionModel(lane_width_m=3.5, cut_in_lateral_threshold=0.5)
        ego = {"x": 0.0, "y": 0.0}
        # Vehicle in left lane moving right (positive vx)
        tracks = [_MockTrack(1, -4.0, 10.0, vx=1.5, vy=0.0)]
        result = model.cut_in_prediction(ego, tracks)
        assert result is not None
        assert result.type == "cut_in"

    def test_cut_in_from_right(self):
        model = InteractionModel(lane_width_m=3.5, cut_in_lateral_threshold=0.5)
        ego = {"x": 0.0, "y": 0.0}
        # Vehicle in right lane moving left (negative vx)
        tracks = [_MockTrack(1, 4.0, 10.0, vx=-1.5, vy=0.0)]
        result = model.cut_in_prediction(ego, tracks)
        assert result is not None
        assert result.type == "cut_in"

    def test_no_cut_in(self):
        model = InteractionModel(lane_width_m=3.5, cut_in_lateral_threshold=0.5)
        ego = {"x": 0.0, "y": 0.0}
        # Vehicle in adjacent lane moving away
        tracks = [_MockTrack(1, 4.0, 10.0, vx=1.0, vy=0.0)]
        result = model.cut_in_prediction(ego, tracks)
        assert result is None


class TestEvaluateAll:
    def test_multiple_events(self):
        model = InteractionModel(safe_following_s=2.0, lane_width_m=3.5)
        ego = {"x": 0.0, "y": 0.0, "vx": 0.0, "vy": 15.0}
        tracks = [
            _MockTrack(1, 0.0, 10.0, 0.0, 15.0),   # too close lead
            _MockTrack(2, 4.0, 10.0, -1.5, 0.0),    # cutting in from right
        ]
        events = model.evaluate(ego, tracks)
        assert len(events) >= 1
        types = [e.type for e in events]
        # Should detect at least following distance or cut-in
        assert "following_distance" in types or "cut_in" in types

    def test_empty_tracks(self):
        model = InteractionModel()
        ego = {"x": 0.0, "y": 0.0, "vx": 0.0, "vy": 10.0}
        events = model.evaluate(ego, [])
        assert events == []
