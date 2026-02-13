"""Tests for radar processing, fusion, and input (Task 1)."""
from __future__ import annotations

import csv
import math
import tempfile
from pathlib import Path

import numpy as np

from src.fusion.radar_camera_fusion import RadarCameraFusion
from src.perception.radar.radar_processor import RadarProcessor
from src.types.radar import RadarDetection, RadarFrame

# ── Radar Processor Tests ──────────────────────────────────────────

class TestRadarProcessorGhostFilter:
    def test_removes_low_rcs(self):
        proc = RadarProcessor(min_rcs_dbsm=-5.0)
        dets = [
            RadarDetection(range_m=10, azimuth_deg=0, velocity_mps=5, rcs_dbsm=-3),
            RadarDetection(range_m=15, azimuth_deg=5, velocity_mps=3, rcs_dbsm=-10),
        ]
        result = proc.ghost_filter(dets)
        assert len(result) == 1
        assert result[0].rcs_dbsm == -3

    def test_removes_2x_ghost(self):
        proc = RadarProcessor(min_rcs_dbsm=-20.0, ghost_azimuth_tolerance_deg=5.0)
        dets = [
            RadarDetection(range_m=10, azimuth_deg=0, velocity_mps=5, rcs_dbsm=5),
            RadarDetection(range_m=20, azimuth_deg=1, velocity_mps=5, rcs_dbsm=0),  # 2x ghost
        ]
        result = proc.ghost_filter(dets)
        assert len(result) == 1
        assert result[0].range_m == 10

    def test_removes_3x_ghost(self):
        proc = RadarProcessor(min_rcs_dbsm=-20.0)
        dets = [
            RadarDetection(range_m=10, azimuth_deg=0, velocity_mps=5, rcs_dbsm=5),
            RadarDetection(range_m=30, azimuth_deg=0, velocity_mps=5, rcs_dbsm=0),  # 3x ghost
        ]
        result = proc.ghost_filter(dets)
        assert len(result) == 1

    def test_keeps_non_ghosts(self):
        proc = RadarProcessor(min_rcs_dbsm=-20.0)
        dets = [
            RadarDetection(range_m=10, azimuth_deg=0, velocity_mps=5, rcs_dbsm=5),
            RadarDetection(range_m=25, azimuth_deg=45, velocity_mps=3, rcs_dbsm=5),  # different azimuth
        ]
        result = proc.ghost_filter(dets)
        assert len(result) == 2


class TestCartesianConversion:
    def test_zero_azimuth(self):
        det = RadarDetection(range_m=10, azimuth_deg=0)
        result = RadarProcessor.to_cartesian(det)
        assert abs(result.x_m) < 1e-6
        assert abs(result.y_m - 10.0) < 1e-6

    def test_90_deg_azimuth(self):
        det = RadarDetection(range_m=10, azimuth_deg=90)
        result = RadarProcessor.to_cartesian(det)
        assert abs(result.x_m - 10.0) < 1e-6
        assert abs(result.y_m) < 1e-6

    def test_45_deg_azimuth(self):
        det = RadarDetection(range_m=10, azimuth_deg=45)
        result = RadarProcessor.to_cartesian(det)
        expected = 10 * math.sin(math.radians(45))
        assert abs(result.x_m - expected) < 1e-6
        assert abs(result.y_m - expected) < 1e-6


class TestClustering:
    def test_merges_nearby(self):
        proc = RadarProcessor(cluster_distance_m=3.0)
        dets = [
            RadarDetection(x_m=1, y_m=1),
            RadarDetection(x_m=1.5, y_m=1.5),
            RadarDetection(x_m=20, y_m=20),
        ]
        result = proc.cluster_detections(dets)
        assert len(result) == 2

    def test_empty_input(self):
        proc = RadarProcessor()
        assert proc.cluster_detections([]) == []

    def test_single_detection(self):
        proc = RadarProcessor()
        dets = [RadarDetection(x_m=5, y_m=5, velocity_mps=10, rcs_dbsm=2)]
        result = proc.cluster_detections(dets)
        assert len(result) == 1
        assert abs(result[0].x_m - 5) < 1e-6


class TestRadarProcessorIntegration:
    def test_full_pipeline(self):
        proc = RadarProcessor(min_rcs_dbsm=-10.0, cluster_distance_m=2.0)
        frame = RadarFrame(
            detections=[
                RadarDetection(range_m=10, azimuth_deg=5, velocity_mps=5, rcs_dbsm=5),
                RadarDetection(range_m=11, azimuth_deg=6, velocity_mps=4, rcs_dbsm=3),
                RadarDetection(range_m=50, azimuth_deg=-30, velocity_mps=10, rcs_dbsm=10),
            ],
            timestamp=1.0,
        )
        result = proc.process(frame)
        assert len(result) >= 1
        for det in result:
            assert det.x_m is not None
            assert det.y_m is not None


# ── Radar-Camera Fusion Tests ──────────────────────────────────────

class _MockTrack:
    def __init__(self, track_id, bbox_xyxy):
        self.track_id = track_id
        self.bbox_xyxy = bbox_xyxy
        self.radar_velocity_mps = None
        self.radar_range_m = None


class TestRadarCameraFusion:
    def test_project_radar_to_image(self):
        cam = np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1]], dtype=np.float64)
        det = RadarDetection(x_m=0, y_m=10)
        u, v = RadarCameraFusion.project_radar_to_image(det, cam)
        assert abs(u - 640.0) < 1e-3  # centered
        assert abs(v - 360.0) < 1e-3  # centered (y_cam=0)

    def test_project_lateral_offset(self):
        cam = np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1]], dtype=np.float64)
        det = RadarDetection(x_m=5, y_m=10)
        u, v = RadarCameraFusion.project_radar_to_image(det, cam)
        assert u > 640.0  # shifted right

    def test_fusion_matching(self):
        fusion = RadarCameraFusion(match_threshold_px=200.0)
        tracks = [
            _MockTrack(1, (600, 320, 680, 400)),  # center ~(640, 360)
            _MockTrack(2, (100, 100, 200, 200)),
        ]
        radar_dets = [
            RadarDetection(x_m=0, y_m=10, velocity_mps=15, range_m=10),
        ]
        cam = np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1]], dtype=np.float64)
        matches = fusion.match_radar_to_tracks(radar_dets, tracks, cam)
        assert 1 in matches
        assert matches[1].velocity_mps == 15

    def test_enrichment(self):
        tracks = [_MockTrack(1, (0, 0, 100, 100))]
        matches = {1: RadarDetection(velocity_mps=20, range_m=15)}
        RadarCameraFusion.enrich_tracks(tracks, matches)
        assert tracks[0].radar_velocity_mps == 20
        assert tracks[0].radar_range_m == 15

    def test_fuse_returns_tracks(self):
        fusion = RadarCameraFusion()
        tracks = [_MockTrack(1, (600, 320, 680, 400))]
        radar_dets = [RadarDetection(x_m=0, y_m=10, velocity_mps=15, range_m=10)]
        result = fusion.fuse(tracks, radar_dets)
        assert len(result) == 1


# ── Radar Input Tests ──────────────────────────────────────────────

class TestRadarInput:
    def test_csv_parsing(self):
        from src.inputs.radar_input import RadarInput

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["frame_id", "range_m", "azimuth_deg", "velocity_mps", "rcs_dbsm"])
            writer.writerow([0, 10.0, 5.0, 3.0, 2.0])
            writer.writerow([0, 15.0, -10.0, 5.0, 4.0])
            writer.writerow([1, 20.0, 0.0, 8.0, 6.0])
            csv_path = f.name

        ri = RadarInput(csv_path)
        ri.start()
        packets = list(ri.frames())
        assert len(packets) == 2
        assert packets[0].radar_frame is not None
        assert len(packets[0].radar_frame.detections) == 2
        assert len(packets[1].radar_frame.detections) == 1
        ri.stop()
        Path(csv_path).unlink()
