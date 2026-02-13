"""Occupancy grid from depth + segmentation (Phase 3.1)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OccupancyGrid:
    """BEV occupancy grid."""

    grid: np.ndarray  # (rows, cols) float32 in [0,1], 1=occupied
    resolution_m: float = 0.2
    max_range_m: float = 40.0
    origin: tuple = (0.0, 0.0)  # ego position in grid


class OccupancyGridBuilder:
    """Projects depth + segmentation into a BEV occupancy grid."""

    def __init__(
        self,
        resolution_m: float = 0.2,
        max_range_m: float = 40.0,
        fx: float = 700.0,
        fy: float = 700.0,
        cx: float = 640.0,
        cy: float = 360.0,
    ):
        self.resolution_m = resolution_m
        self.max_range_m = max_range_m
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.grid_size = int(2 * max_range_m / resolution_m)

    def build(
        self,
        depth_map: np.ndarray | None = None,
        drivable_mask: np.ndarray | None = None,
    ) -> OccupancyGrid:
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        if depth_map is None:
            return OccupancyGrid(grid=grid, resolution_m=self.resolution_m, max_range_m=self.max_range_m)

        h, w = depth_map.shape[:2]
        # Subsample for speed
        step = 4
        for v in range(0, h, step):
            for u in range(0, w, step):
                z = float(depth_map[v, u])
                if z <= 0 or z > self.max_range_m:
                    continue
                # Check if pixel is on non-drivable surface
                if drivable_mask is not None and drivable_mask[v, u] == 1:
                    continue  # drivable — skip

                # Project to BEV
                x_m = (u - self.cx) * z / self.fx
                y_m = z  # forward distance

                gx = int((x_m + self.max_range_m) / self.resolution_m)
                gy = int((self.max_range_m - y_m) / self.resolution_m)

                if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                    grid[gy, gx] = min(1.0, grid[gy, gx] + 0.3)

        return OccupancyGrid(
            grid=grid,
            resolution_m=self.resolution_m,
            max_range_m=self.max_range_m,
            origin=(self.grid_size // 2, self.grid_size - 1),
        )
