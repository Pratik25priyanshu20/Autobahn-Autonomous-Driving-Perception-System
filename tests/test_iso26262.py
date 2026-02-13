"""Tests for ISO 26262 Safety Architecture (Task 4).

Covers ASIL classification, plausibility checks, redundant detection,
and DTC logging.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from src.safety.asil_classifier import COMPONENT_ASIL_MAP, ASILClassifier, ASILLevel
from src.safety.dtc_logger import DTCLogger
from src.safety.plausibility_checker import PlausibilityChecker
from src.safety.redundant_detector import RedundantDetector

# ---------------------------------------------------------------------------
# Helpers / mock objects
# ---------------------------------------------------------------------------

@dataclass
class MockTrack:
    track_id: int
    x: float | None = None
    y: float | None = None
    vx: float | None = None
    vy: float | None = None


@dataclass
class MockDetection:
    bbox_xyxy: tuple = (0, 0, 0, 0)
    confidence: float = 0.9
    class_name: str = "car"


class MockDetector:
    """A fake detector that returns pre-configured detections."""

    def __init__(self, detections: list[MockDetection]):
        self._detections = detections

    def infer(self, frame, **kwargs) -> list[MockDetection]:
        return list(self._detections)


# ---------------------------------------------------------------------------
# ASIL Classifier
# ---------------------------------------------------------------------------

class TestASILClassifier:

    def test_detection_is_asil_b(self):
        c = ASILClassifier()
        assert c.get_level("detection") == ASILLevel.B

    def test_fcw_is_asil_c(self):
        c = ASILClassifier()
        assert c.get_level("fcw") == ASILLevel.C

    def test_controller_is_asil_d(self):
        c = ASILClassifier()
        assert c.get_level("controller") == ASILLevel.D

    def test_unknown_component_defaults_to_qm(self):
        c = ASILClassifier()
        assert c.get_level("nonexistent_module") == ASILLevel.QM

    def test_requires_redundancy_true_for_c_and_d(self):
        c = ASILClassifier()
        assert c.requires_redundancy("fcw") is True        # C
        assert c.requires_redundancy("controller") is True  # D
        assert c.requires_redundancy("fusion") is True      # C

    def test_requires_redundancy_false_for_a_and_b(self):
        c = ASILClassifier()
        assert c.requires_redundancy("detection") is False         # B
        assert c.requires_redundancy("lane_detection") is False    # A
        assert c.requires_redundancy("nonexistent") is False       # QM

    def test_escalation_levels(self):
        c = ASILClassifier()
        assert c.escalation_level("lane_detection") == "monitoring"   # A
        assert c.escalation_level("detection") == "monitoring"        # B
        assert c.escalation_level("fcw") == "redundant"               # C
        assert c.escalation_level("controller") == "fail_safe"        # D
        assert c.escalation_level("nonexistent") == "none"            # QM

    def test_get_all_assignments(self):
        c = ASILClassifier()
        assignments = c.get_all_assignments()
        assert len(assignments) == len(COMPONENT_ASIL_MAP)
        assert "detection" in assignments

    def test_overrides(self):
        c = ASILClassifier(overrides={"detection": "D"})
        assert c.get_level("detection") == ASILLevel.D


# ---------------------------------------------------------------------------
# Plausibility Checker
# ---------------------------------------------------------------------------

class TestPlausibilityChecker:

    def test_velocity_violation(self):
        checker = PlausibilityChecker(max_velocity_kmh=100.0)
        # 50 m/s = 180 km/h which exceeds 100 km/h limit
        fast_track = MockTrack(track_id=1, x=0.0, y=0.0, vx=30.0, vy=40.0)
        violations = checker.check(tracks=[fast_track], detections=[])
        names = [v.check_name for v in violations]
        assert "velocity" in names

    def test_no_velocity_violation_when_under_limit(self):
        checker = PlausibilityChecker(max_velocity_kmh=200.0)
        slow_track = MockTrack(track_id=1, x=0.0, y=0.0, vx=5.0, vy=5.0)
        violations = checker.check(tracks=[slow_track], detections=[])
        names = [v.check_name for v in violations]
        assert "velocity" not in names

    def test_position_jump_detected(self):
        checker = PlausibilityChecker(max_position_jump_m=5.0)
        prev = [MockTrack(track_id=1, x=0.0, y=0.0)]
        curr = [MockTrack(track_id=1, x=100.0, y=0.0)]
        violations = checker.check(tracks=curr, detections=[], prev_tracks=prev)
        names = [v.check_name for v in violations]
        assert "position_jump" in names

    def test_no_position_jump_when_small(self):
        checker = PlausibilityChecker(max_position_jump_m=10.0)
        prev = [MockTrack(track_id=1, x=0.0, y=0.0)]
        curr = [MockTrack(track_id=1, x=1.0, y=1.0)]
        violations = checker.check(tracks=curr, detections=[], prev_tracks=prev)
        names = [v.check_name for v in violations]
        assert "position_jump" not in names

    def test_detection_count_anomaly(self):
        checker = PlausibilityChecker(max_detection_count=5)
        dets = [MockDetection() for _ in range(10)]
        violations = checker.check(tracks=[], detections=dets)
        names = [v.check_name for v in violations]
        assert "detection_count" in names

    def test_no_detection_count_anomaly(self):
        checker = PlausibilityChecker(max_detection_count=100)
        dets = [MockDetection() for _ in range(10)]
        violations = checker.check(tracks=[], detections=dets)
        names = [v.check_name for v in violations]
        assert "detection_count" not in names

    def test_bbox_overlap_check(self):
        checker = PlausibilityChecker(max_bbox_overlap=0.5)
        # Two nearly identical bboxes -> high IoU
        d1 = MockDetection(bbox_xyxy=(0, 0, 100, 100))
        d2 = MockDetection(bbox_xyxy=(5, 5, 105, 105))
        violations = checker.check(tracks=[], detections=[d1, d2])
        names = [v.check_name for v in violations]
        assert "bbox_overlap" in names


# ---------------------------------------------------------------------------
# Redundant Detector
# ---------------------------------------------------------------------------

class TestRedundantDetector:

    def test_matching_detectors(self):
        dets = [
            MockDetection(bbox_xyxy=(10, 10, 50, 50)),
            MockDetection(bbox_xyxy=(100, 100, 200, 200)),
        ]
        primary = MockDetector(dets)
        secondary = MockDetector(dets)
        rd = RedundantDetector(primary, secondary, iou_threshold=0.5)
        result = rd.detect(frame=None)
        assert len(result.agreed) == 2
        assert len(result.primary_only) == 0
        assert len(result.secondary_only) == 0
        assert result.agreement_ratio == 1.0

    def test_mismatching_detectors(self):
        dets_a = [MockDetection(bbox_xyxy=(10, 10, 50, 50))]
        dets_b = [MockDetection(bbox_xyxy=(500, 500, 600, 600))]
        primary = MockDetector(dets_a)
        secondary = MockDetector(dets_b)
        rd = RedundantDetector(primary, secondary, iou_threshold=0.5)
        result = rd.detect(frame=None)
        assert len(result.agreed) == 0
        assert len(result.primary_only) == 1
        assert len(result.secondary_only) == 1
        assert result.agreement_ratio == 0.0

    def test_partial_agreement(self):
        shared = MockDetection(bbox_xyxy=(10, 10, 50, 50))
        extra = MockDetection(bbox_xyxy=(500, 500, 600, 600))
        primary = MockDetector([shared, extra])
        secondary = MockDetector([shared])
        rd = RedundantDetector(primary, secondary, iou_threshold=0.5)
        result = rd.detect(frame=None)
        assert len(result.agreed) == 1
        assert len(result.primary_only) == 1
        assert result.agreement_ratio == pytest.approx(0.5)

    def test_empty_detectors(self):
        primary = MockDetector([])
        secondary = MockDetector([])
        rd = RedundantDetector(primary, secondary, iou_threshold=0.5)
        result = rd.detect(frame=None)
        assert len(result.agreed) == 0
        assert result.agreement_ratio == 0.0


# ---------------------------------------------------------------------------
# DTC Logger
# ---------------------------------------------------------------------------

class TestDTCLogger:

    def test_log_and_retrieve(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dtc = DTCLogger(output_dir=Path(tmpdir))
            dtc.log("DTC_DET_001", details={"module": "yolo"}, frame_id=42)
            active = dtc.get_active()
            assert len(active) == 1
            assert active[0]["code"] == "DTC_DET_001"
            assert active[0]["frame_id"] == 42

    def test_has_critical_false_when_no_critical(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dtc = DTCLogger(output_dir=Path(tmpdir))
            dtc.log("DTC_DET_001")  # warning
            assert dtc.has_critical() is False

    def test_has_critical_true_when_critical(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dtc = DTCLogger(output_dir=Path(tmpdir))
            dtc.log("DTC_FCW_002")  # critical
            assert dtc.has_critical() is True

    def test_clear_code(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dtc = DTCLogger(output_dir=Path(tmpdir))
            dtc.log("DTC_DET_001")
            dtc.clear("DTC_DET_001")
            assert len(dtc.get_active()) == 0

    def test_summary_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dtc = DTCLogger(output_dir=Path(tmpdir))
            dtc.log("DTC_DET_001")   # warning
            dtc.log("DTC_TRK_001")   # info
            dtc.log("DTC_FCW_002")   # critical
            s = dtc.summary()
            assert s["warning"] == 1
            assert s["info"] == 1
            assert s["critical"] == 1

    def test_jsonl_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dtc = DTCLogger(output_dir=Path(tmpdir))
            dtc.log("DTC_DET_001", frame_id=1)
            dtc.log("DTC_FCW_002", frame_id=2)
            log_file = Path(tmpdir) / "dtc_log.jsonl"
            assert log_file.exists()
            lines = log_file.read_text().strip().split("\n")
            assert len(lines) == 2
