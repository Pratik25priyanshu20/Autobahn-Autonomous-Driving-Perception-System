"""Tests for MOT metrics: IDF1, HOTA, format parsers (Task 5)."""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from src.evaluation.mot_formatter import format_aps_predictions, parse_mot17
from src.evaluation.mot_metrics import CLEARMOTEvaluator


class TestIDF1:
    def test_perfect_tracking_idf1(self):
        """Perfect tracking should give IDF1 = 1.0."""
        gt = [
            [(1, (10, 10, 50, 50)), (2, (100, 100, 150, 150))],
            [(1, (12, 12, 52, 52)), (2, (102, 102, 152, 152))],
            [(1, (14, 14, 54, 54)), (2, (104, 104, 154, 154))],
        ]
        pred = gt.copy()
        evaluator = CLEARMOTEvaluator(iou_threshold=0.3)
        idf1 = evaluator.compute_idf1(gt, pred)
        assert idf1 > 0.99

    def test_imperfect_tracking_idf1(self):
        """Swapped IDs should reduce IDF1."""
        gt = [
            [(1, (10, 10, 50, 50)), (2, (100, 100, 150, 150))],
            [(1, (12, 12, 52, 52)), (2, (102, 102, 152, 152))],
        ]
        # Swap pred IDs in frame 2
        pred = [
            [(1, (10, 10, 50, 50)), (2, (100, 100, 150, 150))],
            [(2, (12, 12, 52, 52)), (1, (102, 102, 152, 152))],
        ]
        evaluator = CLEARMOTEvaluator(iou_threshold=0.3)
        idf1 = evaluator.compute_idf1(gt, pred)
        assert 0.0 < idf1 < 1.0


class TestHOTA:
    def test_perfect_hota(self):
        gt = [
            [(1, (10, 10, 50, 50))],
            [(1, (12, 12, 52, 52))],
        ]
        pred = gt.copy()
        evaluator = CLEARMOTEvaluator(iou_threshold=0.3)
        hota = evaluator.compute_hota(gt, pred)
        assert 0.5 < hota <= 1.0

    def test_hota_range(self):
        """HOTA should be between 0 and 1."""
        gt = [
            [(1, (10, 10, 50, 50)), (2, (100, 100, 150, 150))],
            [(1, (12, 12, 52, 52))],
        ]
        pred = [
            [(1, (10, 10, 50, 50))],
            [(1, (12, 12, 52, 52)), (3, (200, 200, 250, 250))],
        ]
        evaluator = CLEARMOTEvaluator(iou_threshold=0.3)
        hota = evaluator.compute_hota(gt, pred)
        assert 0.0 <= hota <= 1.0


class TestEvaluateFull:
    def test_returns_all_metrics(self):
        gt = [[(1, (10, 10, 50, 50))]]
        pred = [[(1, (10, 10, 50, 50))]]
        evaluator = CLEARMOTEvaluator()
        result = evaluator.evaluate_full(gt, pred)
        assert "MOTA" in result
        assert "MOTP" in result
        assert "IDF1" in result
        assert "HOTA" in result
        assert "FP" in result
        assert "FN" in result
        assert "IDSW" in result


class TestMOT17Parser:
    def test_parse_mot17(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow([1, 1, 10, 10, 40, 40, 1.0, 1, 1.0])
            writer.writerow([1, 2, 100, 100, 50, 50, 1.0, 1, 1.0])
            writer.writerow([2, 1, 12, 12, 40, 40, 1.0, 1, 1.0])
            path = f.name
        frames = parse_mot17(path)
        assert len(frames) == 2
        assert len(frames[0]) == 2  # frame 1 has 2 objects
        assert len(frames[1]) == 1  # frame 2 has 1 object
        Path(path).unlink()


class TestAPSFormatter:
    def test_format_aps_predictions(self):
        class _Trk:
            def __init__(self, tid, bbox):
                self.track_id = tid
                self.bbox_xyxy = bbox

        tracks_by_frame = [
            [_Trk(1, (10, 10, 50, 50)), _Trk(2, (100, 100, 150, 150))],
            [_Trk(1, (12, 12, 52, 52))],
        ]
        result = format_aps_predictions(tracks_by_frame)
        assert len(result) == 2
        assert result[0][0][0] == 1  # track_id
        assert result[0][0][1] == (10, 10, 50, 50)  # bbox
