"""Lightweight activation-based saliency (Grad-CAM style) for YOLO/ONNX models (Task 6).

Uses activation magnitudes weighted by detection confidence rather than backprop,
making it compatible with both PyTorch and ONNX inference paths.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


class GradCAMExplainer:
    """Generates saliency heatmaps based on detector activations and detection locations."""

    def __init__(
        self,
        model: Any = None,
        target_layer: str = "model.model.9",
        num_detections: int = 5,
    ):
        self.model = model
        self.target_layer = target_layer
        self.num_detections = num_detections
        self._activations: np.ndarray | None = None
        self._hook = None

        if model is not None:
            self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward hook on target layer to capture activations."""
        try:
            import torch
            parts = self.target_layer.split(".")
            layer = self.model
            for part in parts:
                layer = layer[int(part)] if part.isdigit() else getattr(layer, part)

            def hook_fn(module, input, output):
                if isinstance(output, torch.Tensor):
                    self._activations = output.detach().cpu().numpy()

            self._hook = layer.register_forward_hook(hook_fn)
            logger.debug("Registered activation hook on %s", self.target_layer)
        except Exception as e:
            logger.debug("Could not register hook: %s", e)

    def explain(self, frame: np.ndarray, detections: list[Any]) -> np.ndarray:
        """Generate a (H, W) heatmap in [0, 1].

        Lightweight approach: create Gaussian blobs at detection centers,
        weighted by confidence. If activations are available from hook,
        use activation magnitude as additional weighting.
        """
        h, w = frame.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)

        if not detections:
            return heatmap

        # Use top-N detections by confidence
        sorted_dets = sorted(
            detections,
            key=lambda d: getattr(d, "conf", getattr(d, "confidence", 0.0)),
            reverse=True,
        )[:self.num_detections]

        for det in sorted_dets:
            conf = float(getattr(det, "conf", getattr(det, "confidence", 0.5)))
            x1 = int(getattr(det, "x1", 0))
            y1 = int(getattr(det, "y1", 0))
            x2 = int(getattr(det, "x2", w))
            y2 = int(getattr(det, "y2", h))

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            sigma_x = max(10, (x2 - x1) // 2)
            sigma_y = max(10, (y2 - y1) // 2)

            # Generate Gaussian blob
            yy, xx = np.ogrid[:h, :w]
            gauss = np.exp(
                -((xx - cx) ** 2) / (2 * sigma_x ** 2)
                - ((yy - cy) ** 2) / (2 * sigma_y ** 2)
            )
            heatmap += gauss.astype(np.float32) * conf

        # Incorporate activation magnitudes if available
        if self._activations is not None:
            try:
                act = self._activations
                if act.ndim == 4:
                    act = act[0]  # batch dim
                act_map = np.mean(np.abs(act), axis=0)  # channel mean
                if cv2 is not None:
                    act_resized = cv2.resize(act_map, (w, h), interpolation=cv2.INTER_LINEAR)
                else:
                    # Simple nearest-neighbor resize
                    act_resized = np.repeat(
                        np.repeat(act_map, h // act_map.shape[0] + 1, axis=0),
                        w // act_map.shape[1] + 1,
                        axis=1,
                    )[:h, :w]
                if act_resized.max() > 0:
                    act_resized = act_resized / act_resized.max()
                heatmap = heatmap * (0.5 + 0.5 * act_resized)
            except Exception:
                pass

        # Normalize to [0, 1]
        max_val = heatmap.max()
        if max_val > 0:
            heatmap = heatmap / max_val

        return heatmap
