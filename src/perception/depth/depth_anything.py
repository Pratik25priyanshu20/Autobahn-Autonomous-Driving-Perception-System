"""Depth Anything V2 monocular depth estimator (Phase 1.3)."""
from __future__ import annotations

import time

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from src.perception.depth.base_depth import BaseDepthEstimator


class DepthAnythingV2(BaseDepthEstimator):
    """Monocular depth via Depth Anything V2 (torch.hub)."""

    def __init__(self, device: str = "cpu"):
        if torch is None:
            raise ImportError("torch is required for DepthAnythingV2")
        self.device = device
        self.model = torch.hub.load("huggingface/pytorch-transformers", "model", "Intel/dpt-hybrid-midas", trust_repo=True)
        # Fallback: use MiDAS-style loading if Depth Anything not available
        try:
            self.model = torch.hub.load("LiheYoung/Depth-Anything", "depth_anything_vits14", trust_repo=True)
        except Exception:
            # Fall back to MiDAS
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        self.model.to(self.device).eval()
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform

    @torch.no_grad()
    def infer(self, frame: np.ndarray) -> dict[str, object]:
        start = time.perf_counter()
        input_batch = self.transform(frame).to(self.device)
        prediction = self.model(input_batch)
        depth_map = prediction.squeeze().cpu().numpy()
        # Resize to original frame size
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
