"""Tests for occupancy grid (Phase 3.1)."""
import numpy as np

from src.safety.occupancy_grid import OccupancyGridBuilder, OccupancyGrid


def test_empty_depth_returns_empty_grid():
    builder = OccupancyGridBuilder(resolution_m=0.5, max_range_m=20.0)
    grid = builder.build(depth_map=None)
    assert isinstance(grid, OccupancyGrid)
    assert grid.grid.sum() == 0


def test_depth_populates_grid():
    builder = OccupancyGridBuilder(resolution_m=0.5, max_range_m=20.0)
    depth = np.ones((100, 100), dtype=np.float32) * 10.0
    grid = builder.build(depth_map=depth, drivable_mask=None)
    # Some cells should be occupied since all pixels have depth = 10m
    assert grid.grid.sum() > 0


def test_drivable_mask_suppresses():
    builder = OccupancyGridBuilder(resolution_m=0.5, max_range_m=20.0)
    depth = np.ones((100, 100), dtype=np.float32) * 10.0
    drivable = np.ones((100, 100), dtype=np.uint8)  # All drivable
    grid = builder.build(depth_map=depth, drivable_mask=drivable)
    assert grid.grid.sum() == 0  # All suppressed
