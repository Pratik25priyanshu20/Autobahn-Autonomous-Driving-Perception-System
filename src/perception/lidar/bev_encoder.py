"""Bird's-eye-view pillar encoder for LIDAR point clouds.

Produces a (7, H, W) feature grid:
  Channel 0: point count per pillar
  Channel 1: mean z
  Channel 2: max z
  Channel 3: mean intensity
  Channel 4: std z
  Channel 5: z range (max - min)
  Channel 6: density (count / max_count)
"""
from __future__ import annotations

import numpy as np

from src.types.pointcloud import BEVGrid, PointCloud
from src.utils.logger import get_logger


class BEVEncoder:
    """Pillar-based BEV encoder (pure numpy)."""

    NUM_CHANNELS = 7

    def __init__(
        self,
        x_range: tuple[float, float] = (-40.0, 40.0),
        y_range: tuple[float, float] = (-40.0, 40.0),
        z_range: tuple[float, float] = (-3.0, 3.0),
        resolution: float = 0.2,
    ):
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.resolution = resolution

        self.grid_w = int((x_range[1] - x_range[0]) / resolution)
        self.grid_h = int((y_range[1] - y_range[0]) / resolution)
        self.logger = get_logger(__name__)

    def encode(self, point_cloud: PointCloud) -> BEVGrid:
        """Encode a point cloud into a BEV pillar grid.

        Returns a BEVGrid with shape (7, grid_h, grid_w).
        """
        pts = point_cloud.points
        x, y, z, intensity = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]

        # Filter to BEV range
        mask = (
            (x >= self.x_range[0])
            & (x < self.x_range[1])
            & (y >= self.y_range[0])
            & (y < self.y_range[1])
            & (z >= self.z_range[0])
            & (z < self.z_range[1])
        )
        x, y, z, intensity = x[mask], y[mask], z[mask], intensity[mask]

        # Compute grid indices
        col = ((x - self.x_range[0]) / self.resolution).astype(np.int32)
        row = ((y - self.y_range[0]) / self.resolution).astype(np.int32)

        # Clamp to valid range
        col = np.clip(col, 0, self.grid_w - 1)
        row = np.clip(row, 0, self.grid_h - 1)

        # Initialize output grid
        grid = np.zeros((self.NUM_CHANNELS, self.grid_h, self.grid_w), dtype=np.float32)

        # Compute per-pillar statistics using bincount-based aggregation
        flat_idx = row * self.grid_w + col
        n_cells = self.grid_h * self.grid_w

        # Channel 0: count
        counts = np.bincount(flat_idx, minlength=n_cells).astype(np.float32)
        grid[0] = counts.reshape(self.grid_h, self.grid_w)

        valid = counts > 0

        # Channel 1: mean z
        sum_z = np.bincount(flat_idx, weights=z, minlength=n_cells).astype(np.float32)
        mean_z = np.zeros(n_cells, dtype=np.float32)
        mean_z[valid] = sum_z[valid] / counts[valid]
        grid[1] = mean_z.reshape(self.grid_h, self.grid_w)

        # Channel 2: max z — iterate is the simplest numpy-only way
        max_z = np.full(n_cells, -np.inf, dtype=np.float32)
        np.maximum.at(max_z, flat_idx, z.astype(np.float32))
        max_z[~valid] = 0.0
        grid[2] = max_z.reshape(self.grid_h, self.grid_w)

        # Channel 3: mean intensity
        sum_int = np.bincount(flat_idx, weights=intensity, minlength=n_cells).astype(np.float32)
        mean_int = np.zeros(n_cells, dtype=np.float32)
        mean_int[valid] = sum_int[valid] / counts[valid]
        grid[3] = mean_int.reshape(self.grid_h, self.grid_w)

        # Channel 4: std z
        sum_z2 = np.bincount(flat_idx, weights=(z ** 2), minlength=n_cells).astype(np.float32)
        var_z = np.zeros(n_cells, dtype=np.float32)
        var_z[valid] = sum_z2[valid] / counts[valid] - mean_z[valid] ** 2
        var_z = np.maximum(var_z, 0.0)  # numerical safety
        grid[4] = np.sqrt(var_z).reshape(self.grid_h, self.grid_w)

        # Channel 5: z range (max - min)
        min_z = np.full(n_cells, np.inf, dtype=np.float32)
        np.minimum.at(min_z, flat_idx, z.astype(np.float32))
        min_z[~valid] = 0.0
        z_range = np.zeros(n_cells, dtype=np.float32)
        z_range[valid] = max_z[valid] - min_z[valid]
        grid[5] = z_range.reshape(self.grid_h, self.grid_w)

        # Channel 6: density (count normalized by global max count)
        max_count = counts.max() if counts.max() > 0 else 1.0
        grid[6] = (counts / max_count).reshape(self.grid_h, self.grid_w)

        self.logger.debug(
            "BEV encoded: grid (%d, %d, %d), %d points in range",
            self.NUM_CHANNELS,
            self.grid_h,
            self.grid_w,
            int(mask.sum()),
        )

        return BEVGrid(
            grid=grid,
            resolution_m=self.resolution,
            origin=(self.x_range[0], self.y_range[0]),
        )
