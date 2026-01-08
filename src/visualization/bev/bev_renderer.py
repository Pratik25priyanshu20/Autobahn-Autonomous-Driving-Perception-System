import cv2
import numpy as np
from typing import List, Dict


class BEVRenderer:
    def __init__(self, size_px: int = 600, range_m: float = 50.0, lane_width_m: float = 3.5):
        self.size_px = size_px
        self.range_m = range_m
        self.lane_width_m = lane_width_m
        self.scale = size_px / (2 * range_m)

    def _world_to_bev(self, x_m: float, y_m: float):
        cx = self.size_px // 2
        cy = self.size_px // 2
        px = int(cx + x_m * self.scale)
        py = int(cy - y_m * self.scale)
        return px, py

    def render(self, tracks: List[Dict], lanes: Dict, image_shape: tuple, fcw_state: str = "NORMAL"):
        canvas = np.zeros((self.size_px, self.size_px, 3), dtype=np.uint8)

        # Grid
        for m in range(0, int(self.range_m), 5):
            y = int(self.size_px / 2 - m * self.scale)
            cv2.line(canvas, (0, y), (self.size_px, y), (40, 40, 40), 1)

        # Ego
        cv2.rectangle(
            canvas,
            (self.size_px // 2 - 10, self.size_px // 2 - 20),
            (self.size_px // 2 + 10, self.size_px // 2 + 20),
            (0, 255, 0),
            -1,
        )

        img_h, img_w = image_shape[:2]
        img_cx = img_w / 2

        # Objects
        for t in tracks:
            bbox = getattr(t, "bbox_xyxy", None) or getattr(t, "bbox", None)
            if not bbox:
                continue

            x1, y1, x2, y2 = bbox
            bbox_h = max(1, y2 - y1)
            bbox_cx = (x1 + x2) / 2

            y_m = min(self.range_m, 1400.0 / bbox_h)
            x_m = (bbox_cx - img_cx) / img_w * self.lane_width_m * 2

            px, py = self._world_to_bev(x_m, y_m)

            cv2.circle(canvas, (px, py), 6, (255, 0, 0), -1)
            cv2.putText(
                canvas,
                f"ID {getattr(t, 'track_id', '')}",
                (px + 6, py - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

        # FCW zone
        if fcw_state in ("PRE", "WARNING", "CRITICAL"):
            radius = {"PRE": 15, "WARNING": 25, "CRITICAL": 35}[fcw_state]
            color = {"PRE": (0, 255, 255), "WARNING": (0, 165, 255), "CRITICAL": (0, 0, 255)}[fcw_state]
            cv2.circle(canvas, (self.size_px // 2, self.size_px // 2), radius, color, 2)

        return canvas
