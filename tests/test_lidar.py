"""Tests for LIDAR pipeline, BEV encoding, fusion, and KITTI calibration."""
from __future__ import annotations

import numpy as np
import pytest

from src.types.pointcloud import BEVGrid, LidarDetection3D, PointCloud

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_ground_plane(n: int = 2000, z: float = -1.5, noise: float = 0.05) -> np.ndarray:
    """Generate a flat ground plane at height `z` with slight noise."""
    rng = np.random.default_rng(0)
    xy = rng.uniform(-30, 30, size=(n, 2))
    zz = z + rng.normal(0, noise, size=(n, 1))
    intensity = rng.uniform(0, 1, size=(n, 1)).astype(np.float32)
    return np.hstack([xy, zz, intensity]).astype(np.float32)


def _make_cluster(center: tuple, n: int = 50, spread: float = 0.3) -> np.ndarray:
    """Generate a tight Gaussian cluster around `center`."""
    rng = np.random.default_rng(42)
    xyz = rng.normal(loc=center, scale=spread, size=(n, 3)).astype(np.float32)
    intensity = rng.uniform(0.1, 1.0, size=(n, 1)).astype(np.float32)
    return np.hstack([xyz, intensity])


# ---------------------------------------------------------------------------
# RANSAC ground removal
# ---------------------------------------------------------------------------

class TestRANSACGroundRemoval:
    def test_removes_flat_plane(self):
        from src.perception.lidar.point_cloud_processor import PointCloudProcessor

        proc = PointCloudProcessor()
        ground = _make_ground_plane(n=1500, z=-1.5, noise=0.03)
        # Add some above-ground points
        above = _make_cluster(center=(5.0, 5.0, 0.5), n=100, spread=0.2)
        pts = np.vstack([ground, above])

        result = proc.ground_removal_ransac(pts, max_iterations=200, distance_threshold=0.2)

        # Most ground points should be removed; above-ground points should remain
        assert len(result) < len(pts), "RANSAC should remove ground points"
        # At least some above-ground points survive
        above_z = result[:, 2]
        assert np.any(above_z > -1.0), "Above-ground points should survive"

    def test_empty_input(self):
        from src.perception.lidar.point_cloud_processor import PointCloudProcessor

        proc = PointCloudProcessor()
        pts = np.empty((0, 4), dtype=np.float32)
        result = proc.ground_removal_ransac(pts)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# DBSCAN clustering
# ---------------------------------------------------------------------------

class TestDBSCANClustering:
    def test_two_clusters(self):
        from src.perception.lidar.point_cloud_processor import PointCloudProcessor

        proc = PointCloudProcessor()
        c1 = _make_cluster(center=(10.0, 0.0, 0.0), n=60, spread=0.3)
        c2 = _make_cluster(center=(-10.0, 0.0, 0.0), n=60, spread=0.3)
        pts = np.vstack([c1, c2])

        clusters = proc.cluster_dbscan(pts, eps=1.5, min_samples=10)
        assert len(clusters) == 2, f"Expected 2 clusters, got {len(clusters)}"

    def test_no_clusters_in_noise(self):
        from src.perception.lidar.point_cloud_processor import PointCloudProcessor

        proc = PointCloudProcessor()
        rng = np.random.default_rng(99)
        noise = rng.uniform(-50, 50, size=(20, 4)).astype(np.float32)
        clusters = proc.cluster_dbscan(noise, eps=0.5, min_samples=10)
        assert len(clusters) == 0, "Sparse noise should yield no clusters"

    def test_empty_input(self):
        from src.perception.lidar.point_cloud_processor import PointCloudProcessor

        proc = PointCloudProcessor()
        pts = np.empty((0, 4), dtype=np.float32)
        clusters = proc.cluster_dbscan(pts)
        assert clusters == []


# ---------------------------------------------------------------------------
# BEV encoding
# ---------------------------------------------------------------------------

class TestBEVEncoding:
    def test_output_shape(self):
        from src.perception.lidar.bev_encoder import BEVEncoder

        enc = BEVEncoder(
            x_range=(-20, 20), y_range=(-20, 20), z_range=(-3, 3), resolution=0.5
        )
        rng = np.random.default_rng(7)
        pts = rng.uniform(-15, 15, size=(5000, 4)).astype(np.float32)
        pc = PointCloud(points=pts)

        bev = enc.encode(pc)
        assert isinstance(bev, BEVGrid)
        expected_h = int(40 / 0.5)  # 80
        expected_w = int(40 / 0.5)  # 80
        assert bev.grid.shape == (7, expected_h, expected_w), (
            f"Expected (7, {expected_h}, {expected_w}), got {bev.grid.shape}"
        )

    def test_count_channel_nonzero(self):
        from src.perception.lidar.bev_encoder import BEVEncoder

        enc = BEVEncoder(resolution=0.5)
        rng = np.random.default_rng(3)
        pts = rng.uniform(-10, 10, size=(1000, 4)).astype(np.float32)
        pc = PointCloud(points=pts)

        bev = enc.encode(pc)
        # Count channel should have nonzero cells
        assert bev.grid[0].sum() > 0, "Count channel should have nonzero values"

    def test_empty_cloud(self):
        from src.perception.lidar.bev_encoder import BEVEncoder

        enc = BEVEncoder(resolution=1.0, x_range=(-10, 10), y_range=(-10, 10))
        pts = np.zeros((1, 4), dtype=np.float32)  # single point at origin
        pts[0, :3] = [0, 0, 0]
        pc = PointCloud(points=pts)
        bev = enc.encode(pc)
        assert bev.grid[0].sum() == 1.0, "Single point should occupy one cell"


# ---------------------------------------------------------------------------
# Fusion IoU
# ---------------------------------------------------------------------------

class TestFusionIoU:
    def test_identical_boxes(self):
        from src.fusion.lidar_camera_fusion import LidarCameraFusion

        det = LidarDetection3D(
            center=np.array([5.0, 5.0, 0.0]),
            dimensions=np.array([4.0, 2.0, 1.5]),
        )
        iou = LidarCameraFusion.compute_bev_iou(det, det)
        assert abs(iou - 1.0) < 1e-5, f"Identical boxes should have IoU=1.0, got {iou}"

    def test_non_overlapping(self):
        from src.fusion.lidar_camera_fusion import LidarCameraFusion

        det_a = LidarDetection3D(
            center=np.array([0.0, 0.0, 0.0]),
            dimensions=np.array([2.0, 2.0, 1.5]),
        )
        det_b = LidarDetection3D(
            center=np.array([100.0, 100.0, 0.0]),
            dimensions=np.array([2.0, 2.0, 1.5]),
        )
        iou = LidarCameraFusion.compute_bev_iou(det_a, det_b)
        assert iou == 0.0, f"Non-overlapping boxes should have IoU=0, got {iou}"

    def test_partial_overlap(self):
        from src.fusion.lidar_camera_fusion import LidarCameraFusion

        det_a = LidarDetection3D(
            center=np.array([0.0, 0.0, 0.0]),
            dimensions=np.array([4.0, 4.0, 1.5]),
        )
        det_b = LidarDetection3D(
            center=np.array([2.0, 0.0, 0.0]),
            dimensions=np.array([4.0, 4.0, 1.5]),
        )
        iou = LidarCameraFusion.compute_bev_iou(det_a, det_b)
        assert 0.0 < iou < 1.0, f"Partial overlap should have 0 < IoU < 1, got {iou}"


# ---------------------------------------------------------------------------
# KITTI calibration parsing
# ---------------------------------------------------------------------------

class TestKITTICalibration:
    def test_parse_calib(self, tmp_path):
        from src.inputs.kitti_input import KITTICalibration

        # Create a mock calibration file
        calib_text = (
            "P0: 7.070493e+02 0.000000e+00 6.040814e+02 0.000000e+00 "
            "0.000000e+00 7.070493e+02 1.805066e+02 0.000000e+00 "
            "0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00\n"
            "P1: 7.070493e+02 0.000000e+00 6.040814e+02 -3.797842e+02 "
            "0.000000e+00 7.070493e+02 1.805066e+02 0.000000e+00 "
            "0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00\n"
            "P2: 7.070493e+02 0.000000e+00 6.040814e+02 4.575831e+01 "
            "0.000000e+00 7.070493e+02 1.805066e+02 -3.454157e-01 "
            "0.000000e+00 0.000000e+00 1.000000e+00 4.981016e-03\n"
            "P3: 7.070493e+02 0.000000e+00 6.040814e+02 -3.341081e+02 "
            "0.000000e+00 7.070493e+02 1.805066e+02 2.330660e+00 "
            "0.000000e+00 0.000000e+00 1.000000e+00 3.201153e-03\n"
            "R0_rect: 9.999128e-01 1.009263e-02 -8.511932e-03 "
            "-1.012729e-02 9.999406e-01 -4.037671e-03 "
            "8.470675e-03 4.123522e-03 9.999556e-01\n"
            "Tr_velo_to_cam: 6.927964e-03 -9.999722e-01 -2.757829e-03 -2.457729e-02 "
            "-1.162982e-03 2.749836e-03 -9.999955e-01 -6.127237e-02 "
            "9.999753e-01 6.931141e-03 -1.143899e-03 -3.321029e-01\n"
            "Tr_imu_to_velo: 9.999976e-01 7.553071e-04 -2.035826e-03 -8.086759e-01 "
            "-7.854027e-04 9.998898e-01 -1.482298e-02 3.195559e-01 "
            "2.024406e-03 1.482454e-02 9.998881e-01 -7.997231e-01\n"
        )
        calib_file = tmp_path / "test_calib.txt"
        calib_file.write_text(calib_text)

        calib = KITTICalibration.from_file(calib_file)

        # Check shapes
        assert calib.P2.shape == (3, 4), f"P2 shape {calib.P2.shape}"
        assert calib.R0_rect.shape == (3, 3), f"R0_rect shape {calib.R0_rect.shape}"
        assert calib.Tr_velo_to_cam.shape == (3, 4), f"Tr_velo_to_cam shape {calib.Tr_velo_to_cam.shape}"

        # Check velo_to_cam is 4x4
        v2c = calib.velo_to_cam
        assert v2c.shape == (4, 4), f"velo_to_cam should be 4x4, got {v2c.shape}"

        # Check velo_to_image is 3x4
        v2i = calib.velo_to_image
        assert v2i.shape == (3, 4), f"velo_to_image should be 3x4, got {v2i.shape}"

    def test_project_points(self, tmp_path):
        from src.inputs.kitti_input import KITTICalibration

        # Use identity-like calibration for simple test
        calib = KITTICalibration()
        calib.P2 = np.array([
            [500.0, 0.0, 300.0, 0.0],
            [0.0, 500.0, 200.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ])
        calib.R0_rect = np.eye(3)
        calib.Tr_velo_to_cam = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ])

        pts = np.array([[0.0, 0.0, 10.0]])  # 10m in front
        uv = calib.project_velo_to_image(pts)
        assert uv.shape == (1, 2), f"Expected (1,2), got {uv.shape}"
        # Point at (0, 0, 10) should project near (300, 200) with these params
        assert uv[0, 0] == pytest.approx(300.0, abs=1.0)
        assert uv[0, 1] == pytest.approx(200.0, abs=1.0)


# ---------------------------------------------------------------------------
# Full pipeline smoke test
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_process_returns_detections(self):
        from src.perception.lidar.point_cloud_processor import PointCloudProcessor

        proc = PointCloudProcessor(
            max_range=50.0,
            min_range=0.5,
            voxel_size=0.2,
            cluster_eps=1.5,
            cluster_min_samples=10,
        )

        # Create ground + two objects
        ground = _make_ground_plane(n=1000, z=-1.5, noise=0.03)
        car = _make_cluster(center=(10.0, 2.0, 0.0), n=80, spread=0.4)
        person = _make_cluster(center=(5.0, -3.0, 0.5), n=40, spread=0.2)
        pts = np.vstack([ground, car, person])
        pc = PointCloud(points=pts)

        dets = proc.process(pc)
        assert isinstance(dets, list)
        # Should detect at least one object (the car cluster)
        assert len(dets) >= 1, f"Expected at least 1 detection, got {len(dets)}"
        for d in dets:
            assert isinstance(d, LidarDetection3D)
            assert d.center.shape == (3,)
            assert d.dimensions.shape == (3,)


# ---------------------------------------------------------------------------
# PointCloud validation
# ---------------------------------------------------------------------------

class TestPointCloudDataclass:
    def test_valid_construction(self):
        pts = np.random.rand(100, 4).astype(np.float32)
        pc = PointCloud(points=pts)
        assert len(pc) == 100
        assert pc.xyz.shape == (100, 3)
        assert pc.intensity.shape == (100,)

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            PointCloud(points=np.random.rand(100, 3).astype(np.float32))
