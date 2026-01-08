from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class LaneDetectorConfig:
    roi_top_ratio: float = 0.55
    roi_bottom_ratio: float = 1.0
    canny_low: int = 50
    canny_high: int = 150
    hough_rho: int = 2
    hough_theta: float = np.pi / 180
    hough_threshold: int = 50
    hough_min_line_len: int = 40
    hough_max_line_gap: int = 150
    min_abs_slope: float = 0.4
    smooth_alpha: float = 0.85


class CannyHoughLaneDetector:
    """
    Fast lane detector baseline. Swappable with UFLD later behind same interface.
    """

    def __init__(self, cfg: LaneDetectorConfig | None = None):
        self.cfg = cfg or LaneDetectorConfig()
        self._prev_left: Optional[Tuple[float, float]] = None
        self._prev_right: Optional[Tuple[float, float]] = None
        self._lane_center_hist: deque = deque(maxlen=15)

    def infer(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        h, w = frame_bgr.shape[:2]
        roi_top = int(h * self.cfg.roi_top_ratio)
        roi_bottom = int(h * self.cfg.roi_bottom_ratio)

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self.cfg.canny_low, self.cfg.canny_high)

        mask = np.zeros_like(edges)
        polygon = np.array(
            [
                [
                    (int(0.08 * w), roi_bottom),
                    (int(0.45 * w), roi_top),
                    (int(0.55 * w), roi_top),
                    (int(0.92 * w), roi_bottom),
                ]
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, polygon, 255)
        masked = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(
            masked,
            rho=self.cfg.hough_rho,
            theta=self.cfg.hough_theta,
            threshold=self.cfg.hough_threshold,
            minLineLength=self.cfg.hough_min_line_len,
            maxLineGap=self.cfg.hough_max_line_gap,
        )

        left_params, right_params = self._separate_and_fit(lines)
        left_params = self._ema(self._prev_left, left_params)
        right_params = self._ema(self._prev_right, right_params)
        self._prev_left = left_params
        self._prev_right = right_params

        left_line = self._params_to_points(left_params, roi_top, roi_bottom, w) if left_params else None
        right_line = self._params_to_points(right_params, roi_top, roi_bottom, w) if right_params else None
        lane_confidence = 0.0
        if left_line and right_line:
            lane_confidence = 1.0
        elif left_line or right_line:
            lane_confidence = 0.5

        ego_offset_px = self._estimate_ego_offset(left_line, right_line, w, roi_bottom)
        lane_stable = False
        lane_center_x = None
        lane_center_jitter_px = None
        if left_line and right_line:
            lx = left_line[0][0]
            rx = right_line[0][0]
            lane_center_x = (lx + rx) / 2.0
            self._lane_center_hist.append(lane_center_x)
            if len(self._lane_center_hist) >= 5:
                recent = list(self._lane_center_hist)[-5:]
                jitter = max(recent) - min(recent)
                lane_center_jitter_px = float(jitter)
                lane_stable = True

        return {
            "roi": {"top": roi_top, "bottom": roi_bottom},
            "left_line": left_line,
            "right_line": right_line,
            "ego_offset_px": ego_offset_px,
            "lane_confidence": lane_confidence,
            "lane_center_x": lane_center_x,
            "lane_stable": lane_stable,
            "lane_center_jitter_px": lane_center_jitter_px,
        }

    def _separate_and_fit(self, lines):
        if lines is None:
            return None, None

        left_pts: List[Tuple[int, int]] = []
        right_pts: List[Tuple[int, int]] = []

        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < self.cfg.min_abs_slope:
                continue
            if slope < 0:
                left_pts.extend([(x1, y1), (x2, y2)])
            else:
                right_pts.extend([(x1, y1), (x2, y2)])

        left = self._fit_line(left_pts) if left_pts else None
        right = self._fit_line(right_pts) if right_pts else None
        return left, right

    def _fit_line(self, pts: List[Tuple[int, int]]):
        xs = np.array([p[0] for p in pts], dtype=np.float32)
        ys = np.array([p[1] for p in pts], dtype=np.float32)
        m, b = np.polyfit(xs, ys, 1)
        return float(m), float(b)

    def _params_to_points(self, params, y_top: int, y_bottom: int, w: int):
        m, b = params
        if abs(m) < 1e-6:
            return None
        x1 = int((y_bottom - b) / m)
        x2 = int((y_top - b) / m)
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        return [(x1, y_bottom), (x2, y_top)]

    def _estimate_ego_offset(self, left_line, right_line, w: int, y_ref: int):
        if not left_line or not right_line:
            return None
        lx = left_line[0][0]
        rx = right_line[0][0]
        lane_center = (lx + rx) / 2.0
        ego_center = w / 2.0
        return float(ego_center - lane_center)

    def _ema(self, prev, cur):
        if cur is None:
            return prev
        if prev is None:
            return cur
        a = self.cfg.smooth_alpha
        return (a * prev[0] + (1 - a) * cur[0], a * prev[1] + (1 - a) * cur[1])
