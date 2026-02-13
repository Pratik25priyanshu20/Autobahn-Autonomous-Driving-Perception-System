"""Runtime plausibility checks on perception output (ISO 26262).

Flags physically impossible or suspicious detections/tracks so that
downstream safety modules can react accordingly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.utils.logger import get_logger

logger = get_logger("plausibility_checker")


@dataclass
class PlausibilityViolation:
    check_name: str
    description: str
    severity: str  # "info", "warning", "critical"
    track_id: int | None = None


class PlausibilityChecker:
    """Runtime plausibility checks on perception output."""

    def __init__(
        self,
        max_velocity_kmh: float = 200.0,
        max_position_jump_m: float = 10.0,
        max_bbox_overlap: float = 0.8,
        max_detection_count: int = 100,
    ):
        self.max_velocity_kmh = max_velocity_kmh
        self.max_position_jump_m = max_position_jump_m
        self.max_bbox_overlap = max_bbox_overlap
        self.max_detection_count = max_detection_count

    def check(
        self,
        tracks: list[Any],
        detections: list[Any],
        prev_tracks: list[Any] | None = None,
    ) -> list[PlausibilityViolation]:
        """Run all plausibility checks and return violations."""
        violations: list[PlausibilityViolation] = []
        violations.extend(self._velocity_check(tracks))
        violations.extend(self._position_jump_check(tracks, prev_tracks))
        violations.extend(self._bbox_overlap_check(detections))
        violations.extend(self._detection_count_check(detections))
        return violations

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _velocity_check(self, tracks: list[Any]) -> list[PlausibilityViolation]:
        """Flag any track whose velocity exceeds *max_velocity_kmh*."""
        violations: list[PlausibilityViolation] = []
        for trk in tracks:
            vx = getattr(trk, "vx", None)
            vy = getattr(trk, "vy", None)
            if vx is None or vy is None:
                continue
            speed_ms = (vx ** 2 + vy ** 2) ** 0.5
            speed_kmh = speed_ms * 3.6
            if speed_kmh > self.max_velocity_kmh:
                tid = getattr(trk, "track_id", None)
                violations.append(
                    PlausibilityViolation(
                        check_name="velocity",
                        description=f"Track {tid} velocity {speed_kmh:.1f} km/h exceeds limit {self.max_velocity_kmh} km/h",
                        severity="warning",
                        track_id=tid,
                    )
                )
        return violations

    def _position_jump_check(
        self, tracks: list[Any], prev_tracks: list[Any] | None
    ) -> list[PlausibilityViolation]:
        """Flag any track that teleports more than *max_position_jump_m*."""
        violations: list[PlausibilityViolation] = []
        if prev_tracks is None:
            return violations

        prev_map: dict[int, Any] = {}
        for pt in prev_tracks:
            tid = getattr(pt, "track_id", None)
            if tid is not None:
                prev_map[tid] = pt

        for trk in tracks:
            tid = getattr(trk, "track_id", None)
            if tid is None or tid not in prev_map:
                continue
            x = getattr(trk, "x", None)
            y = getattr(trk, "y", None)
            px = getattr(prev_map[tid], "x", None)
            py = getattr(prev_map[tid], "y", None)
            if x is None or y is None or px is None or py is None:
                continue
            jump = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
            if jump > self.max_position_jump_m:
                violations.append(
                    PlausibilityViolation(
                        check_name="position_jump",
                        description=f"Track {tid} jumped {jump:.2f} m (limit {self.max_position_jump_m} m)",
                        severity="warning",
                        track_id=tid,
                    )
                )
        return violations

    def _bbox_overlap_check(self, detections: list[Any]) -> list[PlausibilityViolation]:
        """Flag pairs of detections whose IoU exceeds *max_bbox_overlap*."""
        violations: list[PlausibilityViolation] = []
        n = len(detections)
        for i in range(n):
            for j in range(i + 1, n):
                iou = self._iou(detections[i], detections[j])
                if iou > self.max_bbox_overlap:
                    violations.append(
                        PlausibilityViolation(
                            check_name="bbox_overlap",
                            description=f"Detections {i} and {j} overlap IoU={iou:.2f} > {self.max_bbox_overlap}",
                            severity="info",
                        )
                    )
        return violations

    def _detection_count_check(self, detections: list[Any]) -> list[PlausibilityViolation]:
        """Flag if total detection count exceeds *max_detection_count*."""
        violations: list[PlausibilityViolation] = []
        if len(detections) > self.max_detection_count:
            violations.append(
                PlausibilityViolation(
                    check_name="detection_count",
                    description=f"{len(detections)} detections exceeds limit {self.max_detection_count}",
                    severity="warning",
                )
            )
        return violations

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _iou(a: Any, b: Any) -> float:
        a_box = getattr(a, "bbox_xyxy", None) or getattr(a, "bbox", None)
        b_box = getattr(b, "bbox_xyxy", None) or getattr(b, "bbox", None)
        if a_box is None or b_box is None:
            return 0.0

        x1 = max(a_box[0], b_box[0])
        y1 = max(a_box[1], b_box[1])
        x2 = min(a_box[2], b_box[2])
        y2 = min(a_box[3], b_box[3])

        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, a_box[2] - a_box[0]) * max(0.0, a_box[3] - a_box[1])
        area_b = max(0.0, b_box[2] - b_box[0]) * max(0.0, b_box[3] - b_box[1])
        union = area_a + area_b - inter

        if union <= 0:
            return 0.0
        return inter / union
