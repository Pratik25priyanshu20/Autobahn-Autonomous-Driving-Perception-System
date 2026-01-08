"""Tests for depth estimation module (Phase 1.3)."""
import numpy as np

from src.perception.depth.base_depth import BaseDepthEstimator


class DummyDepth(BaseDepthEstimator):
    def infer(self, frame):
        h, w = frame.shape[:2]
        return {
            "depth_map": np.ones((h, w), dtype=np.float32) * 5.0,
            "confidence": 1.0,
            "latency_ms": 0.1,
        }


def test_dummy_depth_returns_correct_shape():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    d = DummyDepth()
    out = d.infer(frame)
    assert out["depth_map"].shape == (480, 640)
    assert out["confidence"] == 1.0
