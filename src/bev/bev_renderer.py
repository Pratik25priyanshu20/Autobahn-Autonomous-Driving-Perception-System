from __future__ import annotations

import cv2
import numpy as np
from typing import Any


class BEVRenderer:
    """
    Lightweight BEV renderer for debugging/fusion demos.

    Coordinate frame:
      - +Y is forward (up on the canvas)
      - +X is right
      - Ego sits near the bottom center of the canvas
      - Distances are approximate/proxy (no calibration)
    """

    def __init__(self, size: int = 500, pixels_per_meter: float = 10.0):
        self.size = size
        self.ppm = pixels_per_meter
        # ego placed slightly above the bottom edge
        self.origin = (size // 2, size - 50)
        self.lane_width_m = 3.5
        self.lane_length_m = 40.0

        # Scale/grid
        self.grid_step_m = 5
        self.max_forward_m = self.lane_length_m
        # Safety zones (forward ranges in meters)
        self.safety_zones = [
            ("GREEN", 0, 10, (0, 60, 0)),
            ("YELLOW", 10, 25, (0, 200, 200)),
            ("RED", 25, 40, (0, 0, 120)),
        ]

    def world_to_bev(self, x_m: float, y_m: float) -> tuple[int, int]:
        """Map (x, y) meters into pixel coordinates on the canvas."""
        px = int(self.origin[0] + x_m * self.ppm)
        py = int(self.origin[1] - y_m * self.ppm)
        return px, py

    def render(self, world: Any, fcw: dict | None = None) -> np.ndarray:
        """
        Render a BEV image from the current world model.
        Expects world.objects with (x, y) in meters if available.
        """
        canvas = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        self._draw_grid(canvas)
        self._draw_corridor_shading(canvas)
        self._draw_safety_zones(canvas)
        self._draw_ego(canvas)
        self._draw_corridor(canvas)
        self._draw_lanes(canvas, world)
        self._draw_objects(canvas, world, fcw or {})
        self._draw_fcw_cone(canvas, fcw_state=(fcw or {}).get("state"))
        return canvas

    # ------------------------------------------------------------------ #
    # Draw helpers
    # ------------------------------------------------------------------ #
    def _draw_grid(self, canvas: np.ndarray) -> None:
        """Simple forward grid every 5 m for depth perception with labels."""
        for m in range(0, int(self.max_forward_m) + 1, self.grid_step_m):
            _, y = self.world_to_bev(0.0, m)
            cv2.line(canvas, (0, y), (self.size, y), (50, 50, 50), 1)
            if m > 0:
                cv2.putText(
                    canvas,
                    f"{m}m",
                    (5, max(12, y - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (200, 200, 200),
                    1,
                )

    def _draw_safety_zones(self, canvas: np.ndarray) -> None:
        """Forward safety envelopes (comfort/caution/emergency)."""
        for _, start_m, end_m, color in self.safety_zones:
            p0 = self.world_to_bev(-self.lane_width_m * 2, start_m)
            p1 = self.world_to_bev(self.lane_width_m * 2, end_m)
            x1, y1 = p0
            x2, y2 = p1
            overlay = canvas.copy()
            cv2.rectangle(overlay, (min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2)), color, -1)
            canvas[:] = cv2.addWeighted(overlay, 0.08, canvas, 0.92, 0)

    def _draw_ego(self, canvas: np.ndarray) -> None:
        x, y = self.origin
        cv2.rectangle(canvas, (x - 10, y - 20), (x + 10, y), (255, 255, 255), -1)

    def _draw_corridor(self, canvas: np.ndarray) -> None:
        """Visualize the ego lane corridor used for FCW reasoning."""
        half = self.lane_width_m / 2
        x1, y1 = self.world_to_bev(-half, 0)
        x2, y2 = self.world_to_bev(half, self.lane_length_m)
        cv2.rectangle(canvas, (min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2)), (80, 80, 80), 1)

    def _draw_corridor_shading(self, canvas: np.ndarray) -> None:
        """Forward corridor shading by zones (comfort/caution/danger)."""
        lane_half_width_px = int((self.lane_width_m / 2) * self.ppm)
        cx = self.size // 2
        y0 = self.size // 2
        zones = [
            (0, 15, (0, 80, 0)),    # green
            (15, 25, (0, 80, 80)),  # yellow
            (25, 40, (0, 0, 80)),   # red
        ]
        for start_m, end_m, color in zones:
            y_start = int(y0 - start_m * self.ppm)
            y_end = int(y0 - end_m * self.ppm)
            overlay = canvas.copy()
            cv2.rectangle(
                overlay,
                (cx - lane_half_width_px, y_end),
                (cx + lane_half_width_px, y_start),
                color,
                -1,
            )
            canvas[:] = cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0)

    def _draw_lanes(self, canvas: np.ndarray, world: Any) -> None:
        """
        Approximate lane lines. Uses a fixed lane width and optional ego offset
        (if provided by lane detector) to shift the center.
        """
        lanes = getattr(world, "lanes", None)
        if lanes is None:
            return

        ego_offset_px = lanes.get("ego_offset_px") if isinstance(lanes, dict) else getattr(lanes, "ego_offset_px", None)
        # documented assumption: 1 px ~ 0.02 m for BEV viz only
        px_to_m = 0.02
        center_offset_m = float(ego_offset_px) * px_to_m if ego_offset_px else 0.0

        lane_center_x = -center_offset_m
        half_lane = self.lane_width_m / 2
        left_x = lane_center_x - half_lane
        right_x = lane_center_x + half_lane

        for side_x in (left_x, right_x, lane_center_x):
            p0 = self.world_to_bev(side_x, 0.0)
            p1 = self.world_to_bev(side_x, self.lane_length_m)
            cv2.line(canvas, p0, p1, (0, 255, 0), 2 if side_x == lane_center_x else 1)

    def _draw_objects(self, canvas: np.ndarray, world: Any, fcw: dict) -> None:
        lead_id = fcw.get("lead_id")
        fcw_state = (fcw.get("state") or "NORMAL").upper()

        def color_for(ttc: float | None, is_lead: bool, risk: str | None) -> tuple[int, int, int]:
            # Risk first if provided
            risk = (risk or "").upper()
            risk_map = {
                "CRITICAL": (0, 0, 255),
                "WARNING": (0, 165, 255),
                "PRE": (0, 255, 255),
                "NORMAL": (0, 255, 0),
            }
            if risk in risk_map:
                return risk_map[risk]

            if ttc is not None:
                if ttc < 2.0:
                    return (0, 0, 255)
                if ttc < 4.0:
                    return (0, 255, 255)
            if is_lead:
                if fcw_state == "CRITICAL":
                    return (0, 0, 255)
                if fcw_state == "WARNING":
                    return (0, 165, 255)
                if fcw_state == "PRE":
                    return (0, 255, 255)
            # grey if no info, else green safe
            return (200, 200, 200) if ttc is None else (0, 255, 0)

        objs = getattr(world, "objects", []) or getattr(world, "tracks", [])
        for obj in objs:
            x_m = getattr(obj, "x", None)
            y_m = getattr(obj, "y", None)
            if x_m is None or y_m is None:
                continue

            px, py = self.world_to_bev(x_m, y_m)
            ttc = getattr(obj, "ttc", None)
            risk = getattr(obj, "risk", None)
            is_lead = getattr(obj, "track_id", None) == lead_id
            color = color_for(ttc, is_lead, risk)
            in_corr = abs(x_m) <= (self.lane_width_m / 2.0)
            if in_corr:
                cv2.circle(canvas, (px, py), 7, color, -1)
            else:
                cv2.circle(canvas, (px, py), 7, (100, 100, 100), 1)

            # Velocity vector (intent)
            vx = getattr(obj, "vx", 0.0) or 0.0
            vy = getattr(obj, "vy", 0.0) or 0.0
            # scale velocity into meters for viz; small factor to keep arrows reasonable
            arrow_scale = 1.5
            end_px, end_py = self.world_to_bev(x_m + vx * arrow_scale, y_m + vy * arrow_scale)
            cv2.arrowedLine(canvas, (px, py), (end_px, end_py), color, 2, tipLength=0.3)

            # Prediction point at ~1s ahead
            pred_px, pred_py = self.world_to_bev(x_m + vx * 1.0, y_m + vy * 1.0)
            cv2.circle(canvas, (pred_px, pred_py), 4, color, 1)

            cv2.putText(
                canvas,
                f"ID {getattr(obj, 'track_id', '')}" + (f" | TTC {ttc:.1f}s" if ttc is not None else ""),
                (px + 5, py - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 255, 255),
                1,
            )

    def _draw_fcw_cone(self, canvas: np.ndarray, fcw_state: str | None) -> None:
        fcw_state = (fcw_state or "NORMAL").upper()
        if fcw_state not in ("PRE", "WARNING", "CRITICAL"):
            return
        cx = self.size // 2
        y0 = self.size // 2
        length_m = {"PRE": 20, "WARNING": 30, "CRITICAL": 40}[fcw_state]
        width_m = 3.5
        length_px = int(length_m * self.ppm)
        width_px = int(width_m * self.ppm)
        pts = np.array(
            [
                (cx - width_px, y0),
                (cx + width_px, y0),
                (cx + width_px // 2, y0 - length_px),
                (cx - width_px // 2, y0 - length_px),
            ]
        )
        color = {"PRE": (0, 255, 255), "WARNING": (0, 165, 255), "CRITICAL": (0, 0, 255)}[fcw_state]
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [pts], color)
        canvas[:] = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)
