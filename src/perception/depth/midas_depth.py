"""MiDAS monocular depth estimator (Phase 1.3)."""
from __future__ import annotations

import time
from typing import Dict

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from src.perception.depth.base_depth import BaseDepthEstimator


class MiDASDepth(BaseDepthEstimator):
    """Monocular depth via MiDAS (torch.hub)."""

    def __init__(self, model_type: str = "MiDaS_small", device: str = "cpu"):
        if torch is None:
            raise ImportError("torch is required for MiDASDepth")
        self.device = device
        self.model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        self.model.to(self.device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        if model_type == "MiDaS_small":
            self.transform = transforms.small_transform
        else:
            self.transform = transforms.dpt_transform

    @torch.no_grad()
    def infer(self, frame: np.ndarray) -> Dict[str, object]:
        start = time.perf_counter()
        input_batch = self.transform(frame).to(self.device)
        prediction = self.model(input_batch)
        depth_map = prediction.squeeze().cpu().numpy()
        h, w = frame.shape[:2]
        if depth_map.shape != (h, w):
            import cv2
            depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return {
            "depth_map": depth_map.astype(np.float32),
            "confidence": float(np.mean(depth_map > 0)),
            "latency_ms": latency_ms,
        }
