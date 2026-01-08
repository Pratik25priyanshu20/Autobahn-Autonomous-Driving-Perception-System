"""Mono3D detection stub (Phase 1.4).

Converts 2D detections + depth map into pseudo-3D boxes.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from src.types.detection import Detection
from src.types.detection3d import Detection3D


class Mono3DDetector:
    """Convert 2D detections + monocular depth into pseudo-3D boxes."""

    def __init__(self, fx: float = 700.0, fy: float = 700.0, cx: float = 640.0, cy: float = 360.0):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def infer(self, detections: List[Detection], depth_map: Optional[np.ndarray]) -> List[Detection3D]:
        if depth_map is None:
            return []

        results: List[Detection3D] = []
        for det in detections:
            cx_px = (det.x1 + det.x2) / 2.0
            cy_px = (det.y1 + det.y2) / 2.0
            cx_i = int(min(max(cx_px, 0), depth_map.shape[1] - 1))
            cy_i = int(min(max(cy_px, 0), depth_map.shape[0] - 1))
            z = float(depth_map[cy_i, cx_i])
            if z <= 0:
                continue
            x_m = (cx_px - self.cx) * z / self.fx
            y_m = (cy_px - self.cy) * z / self.fy
            width_m = (det.x2 - det.x1) * z / self.fx
            height_m = (det.y2 - det.y1) * z / self.fy
            results.append(Detection3D(
                x1=det.x1, y1=det.y1, x2=det.x2, y2=det.y2,
                conf=det.conf, class_id=det.class_id, class_name=det.class_name,
                x_m=x_m, y_m=y_m, z_m=z,
                width_m=width_m, height_m=height_m,
            ))
        return results
