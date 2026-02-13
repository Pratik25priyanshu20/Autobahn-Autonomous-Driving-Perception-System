"""Unit tests for BEVRenderer."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from src.bev.bev_renderer import BEVRenderer


class TestBEVRendererInit:
    def test_can_be_instantiated(self):
        renderer = BEVRenderer()
        assert renderer.size == 500
        assert renderer.ppm == 10.0
        assert renderer.origin == (250, 450)

    def test_custom_parameters(self):
        renderer = BEVRenderer(size=300, pixels_per_meter=5.0)
        assert renderer.size == 300
        assert renderer.ppm == 5.0


class TestDrawInteractions:
    def test_empty_interactions_does_not_crash(self):
        renderer = BEVRenderer(size=200)
        canvas = np.zeros((200, 200, 3), dtype=np.uint8)
        # world mock with empty interactions and tracks
        world = MagicMock()
        world.interactions = []
        world.tracks = []
        # Should return without error
        renderer._draw_interactions(canvas, world)
        # Canvas should remain all black (nothing drawn)
        assert canvas.sum() == 0

    def test_no_interactions_attribute(self):
        renderer = BEVRenderer(size=200)
        canvas = np.zeros((200, 200, 3), dtype=np.uint8)
        world = MagicMock(spec=[])  # no attributes
        # getattr(..., "interactions", []) should return []
        renderer._draw_interactions(canvas, world)
        assert canvas.sum() == 0
