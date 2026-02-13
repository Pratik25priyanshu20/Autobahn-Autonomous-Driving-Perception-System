"""Saliency overlay rendering for explainability (Task 6)."""
from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def overlay_saliency(
    frame: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Blend a saliency heatmap onto a frame using a colormap.

    Args:
        frame: BGR image (H, W, 3)
        heatmap: float32 (H, W) in [0, 1]
        alpha: blending factor (0 = frame only, 1 = heatmap only)

    Returns:
        Blended BGR image.
    """
    if heatmap is None or heatmap.max() == 0:
        return frame.copy()

    h, w = frame.shape[:2]
    hmap = heatmap.copy()

    # Resize if needed
    if hmap.shape[:2] != (h, w):
        hmap = cv2.resize(hmap, (w, h), interpolation=cv2.INTER_LINEAR) if cv2 is not None else np.resize(hmap, (h, w))

    # Convert to uint8 for colormap
    hmap_u8 = (hmap * 255).clip(0, 255).astype(np.uint8)

    if cv2 is not None:
        colored = cv2.applyColorMap(hmap_u8, cv2.COLORMAP_JET)
    else:
        # Simple fallback: red channel proportional to heatmap
        colored = np.zeros_like(frame)
        colored[:, :, 2] = hmap_u8  # red channel

    blended = cv2.addWeighted(colored, alpha, frame, 1.0 - alpha, 0) if cv2 is not None else (
        (colored.astype(np.float32) * alpha + frame.astype(np.float32) * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
    )
    return blended
