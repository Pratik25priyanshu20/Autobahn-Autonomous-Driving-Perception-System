"""Tests for saliency maps / Grad-CAM explainability (Task 6)."""
from __future__ import annotations

import numpy as np

from src.perception.explainability.attention_overlay import overlay_saliency
from src.perception.explainability.grad_cam import GradCAMExplainer


class _MockDetection:
    def __init__(self, x1, y1, x2, y2, conf=0.8):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.conf = conf


class TestGradCAMExplainer:
    def test_output_shape(self):
        explainer = GradCAMExplainer(model=None, num_detections=3)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dets = [_MockDetection(100, 100, 200, 200, 0.9)]
        heatmap = explainer.explain(frame, dets)
        assert heatmap.shape == (480, 640)

    def test_normalized_output(self):
        explainer = GradCAMExplainer(model=None, num_detections=5)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dets = [
            _MockDetection(100, 100, 200, 200, 0.9),
            _MockDetection(300, 300, 400, 400, 0.7),
        ]
        heatmap = explainer.explain(frame, dets)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0 + 1e-6

    def test_no_detections(self):
        explainer = GradCAMExplainer(model=None)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        heatmap = explainer.explain(frame, [])
        assert heatmap.shape == (480, 640)
        assert heatmap.max() == 0.0

    def test_heatmap_peak_at_detection(self):
        explainer = GradCAMExplainer(model=None, num_detections=1)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dets = [_MockDetection(300, 200, 340, 280, 0.9)]
        heatmap = explainer.explain(frame, dets)
        # Peak should be near detection center
        cy, cx = 240, 320
        region = heatmap[cy - 20:cy + 20, cx - 20:cx + 20]
        assert region.mean() > heatmap.mean()


class TestOverlaySaliency:
    def test_overlay_shape(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        heatmap = np.random.rand(480, 640).astype(np.float32)
        result = overlay_saliency(frame, heatmap)
        assert result.shape == frame.shape

    def test_overlay_zero_heatmap(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        heatmap = np.zeros((480, 640), dtype=np.float32)
        result = overlay_saliency(frame, heatmap)
        assert result.shape == frame.shape

    def test_overlay_none_heatmap(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = overlay_saliency(frame, None)
        assert result.shape == frame.shape

    def test_alpha_blending(self):
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        heatmap = np.ones((100, 100), dtype=np.float32)
        result_low = overlay_saliency(frame, heatmap, alpha=0.1)
        result_high = overlay_saliency(frame, heatmap, alpha=0.9)
        # High alpha should differ more from original
        diff_low = np.abs(result_low.astype(float) - frame.astype(float)).mean()
        diff_high = np.abs(result_high.astype(float) - frame.astype(float)).mean()
        assert diff_high > diff_low
