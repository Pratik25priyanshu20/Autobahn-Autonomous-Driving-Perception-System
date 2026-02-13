"""LIDAR point cloud processing pipeline.

Stages: range filter -> ground removal (RANSAC) -> voxel downsample
        -> DBSCAN clustering -> oriented bbox fitting -> classification.

Uses only numpy (no scipy dependency).
"""
from __future__ import annotations

import numpy as np

from src.types.pointcloud import LidarDetection3D, PointCloud
from src.utils.logger import get_logger


class PointCloudProcessor:
    """Full LIDAR processing pipeline producing 3D detections."""

    def __init__(
        self,
        max_range: float = 80.0,
        min_range: float = 1.0,
        voxel_size: float = 0.1,
        ground_method: str = "ransac",
        max_iterations: int = 100,
        distance_threshold: float = 0.2,
        cluster_eps: float = 0.8,
        cluster_min_samples: int = 10,
    ):
        self.max_range = max_range
        self.min_range = min_range
        self.voxel_size = voxel_size
        self.ground_method = ground_method
        self.max_iterations = max_iterations
        self.distance_threshold = distance_threshold
        self.cluster_eps = cluster_eps
        self.cluster_min_samples = cluster_min_samples
        self.logger = get_logger(__name__)

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def range_filter(
        self, points: np.ndarray, max_range: float = 80.0, min_range: float = 1.0
    ) -> np.ndarray:
        """Filter points by euclidean distance from origin."""
        dist = np.linalg.norm(points[:, :3], axis=1)
        mask = (dist >= min_range) & (dist <= max_range)
        return points[mask]

    def ground_removal_ransac(
        self,
        points: np.ndarray,
        max_iterations: int = 100,
        distance_threshold: float = 0.2,
    ) -> np.ndarray:
        """Remove ground plane via RANSAC (pure numpy).

        Fits a plane ax + by + cz + d = 0, removes inliers as ground.
        """
        if len(points) < 3:
            return points

        best_inlier_mask = np.zeros(len(points), dtype=bool)
        best_inlier_count = 0
        xyz = points[:, :3]
        n_pts = len(xyz)
        rng = np.random.default_rng(42)

        for _ in range(max_iterations):
            # Sample 3 random points
            idxs = rng.choice(n_pts, size=3, replace=False)
            p1, p2, p3 = xyz[idxs[0]], xyz[idxs[1]], xyz[idxs[2]]

            # Compute plane normal
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-10:
                continue
            normal = normal / norm_len
            d = -np.dot(normal, p1)

            # Distance of all points to the plane
            distances = np.abs(xyz @ normal + d)
            inlier_mask = distances < distance_threshold
            inlier_count = int(np.sum(inlier_mask))

            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inlier_mask = inlier_mask

        # Return non-ground points
        return points[~best_inlier_mask]

    def voxel_downsample(self, points: np.ndarray, voxel_size: float = 0.1) -> np.ndarray:
        """Voxel grid downsampling — keep one point per voxel cell."""
        if len(points) == 0:
            return points

        # Quantize coordinates to voxel grid indices
        voxel_indices = np.floor(points[:, :3] / voxel_size).astype(np.int32)

        # Encode voxel indices as unique keys
        # Shift to positive range for hashing
        mins = voxel_indices.min(axis=0)
        shifted = voxel_indices - mins
        dims = shifted.max(axis=0) + 1

        keys = (
            shifted[:, 0].astype(np.int64) * int(dims[1]) * int(dims[2])
            + shifted[:, 1].astype(np.int64) * int(dims[2])
            + shifted[:, 2].astype(np.int64)
        )

        # Keep first occurrence per voxel
        _, unique_idx = np.unique(keys, return_index=True)
        return points[unique_idx]

    def cluster_dbscan(
        self, points: np.ndarray, eps: float = 0.8, min_samples: int = 10
    ) -> list[np.ndarray]:
        """DBSCAN clustering (pure numpy implementation).

        Returns a list of point arrays, one per cluster.
        """
        if len(points) == 0:
            return []

        n = len(points)
        xyz = points[:, :3]
        labels = np.full(n, -1, dtype=np.int32)
        cluster_id = 0
        visited = np.zeros(n, dtype=bool)

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True

            # Find neighbors
            dists = np.linalg.norm(xyz - xyz[i], axis=1)
            neighbors = np.where(dists <= eps)[0]

            if len(neighbors) < min_samples:
                # Noise point
                continue

            labels[i] = cluster_id
            seed_queue = list(neighbors)
            seed_set = set(neighbors)
            q_idx = 0

            while q_idx < len(seed_queue):
                j = seed_queue[q_idx]
                q_idx += 1

                if not visited[j]:
                    visited[j] = True
                    dists_j = np.linalg.norm(xyz - xyz[j], axis=1)
                    neighbors_j = np.where(dists_j <= eps)[0]
                    if len(neighbors_j) >= min_samples:
                        for nj in neighbors_j:
                            if nj not in seed_set:
                                seed_set.add(nj)
                                seed_queue.append(nj)

                if labels[j] == -1:
                    labels[j] = cluster_id

            cluster_id += 1

        clusters: list[np.ndarray] = []
        for cid in range(cluster_id):
            mask = labels == cid
            if np.sum(mask) >= min_samples:
                clusters.append(points[mask])

        return clusters

    def fit_oriented_bbox(
        self, cluster_points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Compute oriented bounding box for a cluster.

        Returns (center_xyz, dimensions_lwh, yaw_rad).
        Uses PCA on the XY plane to determine orientation.
        """
        xyz = cluster_points[:, :3]
        center = xyz.mean(axis=0)

        # PCA on XY
        xy = xyz[:, :2] - center[:2]
        cov = np.cov(xy, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Principal direction = eigenvector with largest eigenvalue
        principal = eigenvectors[:, np.argmax(eigenvalues)]
        yaw = float(np.arctan2(principal[1], principal[0]))

        # Rotate points to aligned frame
        cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)
        rot = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
        aligned_xy = (xyz[:, :2] - center[:2]) @ rot.T

        # Compute bounding box in aligned frame
        xy_min = aligned_xy.min(axis=0)
        xy_max = aligned_xy.max(axis=0)
        z_min = xyz[:, 2].min()
        z_max = xyz[:, 2].max()

        length = float(xy_max[0] - xy_min[0])
        width = float(xy_max[1] - xy_min[1])
        height = float(z_max - z_min)

        # Adjust center z to midpoint
        center[2] = (z_min + z_max) / 2.0

        return center, np.array([length, width, height]), yaw

    def classify_by_dimensions(self, dimensions: np.ndarray) -> tuple[str, float]:
        """Heuristic classification based on bounding box dimensions.

        Returns (class_name, confidence).
        """
        length, width, height = float(dimensions[0]), float(dimensions[1]), float(dimensions[2])

        # Ensure length >= width for consistent matching
        if width > length:
            length, width = width, length

        # Person: tall, narrow
        if height > 1.2 and width < 1.0 and length < 1.2:
            conf = 1.0 - min(1.0, abs(height - 1.7) / 1.0)
            return "person", max(0.3, conf)

        # Cyclist: similar to person but slightly wider
        if height > 1.2 and width < 1.2 and 1.0 <= length <= 2.2:
            return "cyclist", 0.5

        # Truck / bus: long vehicles
        if length > 6.0:
            return "truck", 0.7

        # Car: typical dimensions around 4.5 x 1.8 x 1.5
        if 2.5 <= length <= 6.0 and 1.2 <= width <= 2.5 and 1.0 <= height <= 2.5:
            # Score by proximity to typical car dims
            l_err = abs(length - 4.5) / 4.5
            w_err = abs(width - 1.8) / 1.8
            h_err = abs(height - 1.5) / 1.5
            conf = 1.0 - min(1.0, (l_err + w_err + h_err) / 3.0)
            return "car", max(0.3, conf)

        return "unknown", 0.2

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def process(self, point_cloud: PointCloud) -> list[LidarDetection3D]:
        """Run full LIDAR processing pipeline.

        Pipeline: range filter -> ground removal -> voxel downsample
                  -> DBSCAN cluster -> fit bbox -> classify.
        """
        pts = point_cloud.points.copy()
        self.logger.debug("LIDAR pipeline: input %d points", len(pts))

        # 1. Range filter
        pts = self.range_filter(pts, max_range=self.max_range, min_range=self.min_range)
        self.logger.debug("After range filter: %d points", len(pts))

        # 2. Ground removal
        if self.ground_method == "ransac":
            pts = self.ground_removal_ransac(
                pts,
                max_iterations=self.max_iterations,
                distance_threshold=self.distance_threshold,
            )
        self.logger.debug("After ground removal: %d points", len(pts))

        # 3. Voxel downsampling
        pts = self.voxel_downsample(pts, voxel_size=self.voxel_size)
        self.logger.debug("After voxel downsample: %d points", len(pts))

        # 4. Clustering
        clusters = self.cluster_dbscan(pts, eps=self.cluster_eps, min_samples=self.cluster_min_samples)
        self.logger.debug("DBSCAN found %d clusters", len(clusters))

        # 5. Fit bounding boxes and classify
        detections: list[LidarDetection3D] = []
        for cluster_pts in clusters:
            center, dims, yaw = self.fit_oriented_bbox(cluster_pts)
            class_name, confidence = self.classify_by_dimensions(dims)
            detections.append(
                LidarDetection3D(
                    center=center,
                    dimensions=dims,
                    yaw=yaw,
                    class_name=class_name,
                    confidence=confidence,
                )
            )

        self.logger.debug("LIDAR pipeline: %d detections", len(detections))
        return detections
