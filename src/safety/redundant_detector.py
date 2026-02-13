"""Redundant detection with IoU-based majority voting (ISO 26262).

Runs two detectors in parallel and cross-validates their outputs via
Intersection-over-Union matching.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from src.utils.logger import get_logger

logger = get_logger("redundant_detector")


@dataclass
class RedundantDetectionResult:
    """Outcome of running two detectors and matching their outputs."""

    agreed: list[Any] = field(default_factory=list)        # detections both agree on
    primary_only: list[Any] = field(default_factory=list)  # only primary detected
    secondary_only: list[Any] = field(default_factory=list)  # only secondary detected
    agreement_ratio: float = 0.0  # len(agreed) / max(len(primary), len(secondary))


def _bbox_iou(a: Any, b: Any) -> float:
    """Compute IoU between two detection objects with bbox_xyxy or bbox attrs."""
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


class RedundantDetector:
    """Run two detectors in parallel, cross-validate via IoU-based majority voting."""

    def __init__(
        self,
        primary_detector: Any,
        secondary_detector: Any,
        iou_threshold: float = 0.5,
        min_agreement: float = 0.6,
    ):
        self.primary = primary_detector
        self.secondary = secondary_detector
        self.iou_threshold = iou_threshold
        self.min_agreement = min_agreement
        self._executor = ThreadPoolExecutor(max_workers=2)

    def detect(self, frame: Any, **kwargs) -> RedundantDetectionResult:
        """Run both detectors in parallel and match outputs."""
        future_primary = self._executor.submit(self.primary.infer, frame, **kwargs)
        future_secondary = self._executor.submit(self.secondary.infer, frame, **kwargs)

        primary_dets = future_primary.result()
        secondary_dets = future_secondary.result()

        return self._match(primary_dets, secondary_dets)

    def _match(
        self, primary_dets: list[Any], secondary_dets: list[Any]
    ) -> RedundantDetectionResult:
        """Match detections by IoU, split into agreed / primary-only / secondary-only."""
        matched_secondary: set = set()
        agreed: list[Any] = []
        primary_only: list[Any] = []

        for p_det in primary_dets:
            best_iou = 0.0
            best_idx: int | None = None
            for idx, s_det in enumerate(secondary_dets):
                if idx in matched_secondary:
                    continue
                iou = _bbox_iou(p_det, s_det)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx is not None and best_iou >= self.iou_threshold:
                agreed.append(p_det)
                matched_secondary.add(best_idx)
            else:
                primary_only.append(p_det)

        secondary_only = [
            s for idx, s in enumerate(secondary_dets) if idx not in matched_secondary
        ]

        total = max(len(primary_dets), len(secondary_dets), 1)
        ratio = len(agreed) / total

        result = RedundantDetectionResult(
            agreed=agreed,
            primary_only=primary_only,
            secondary_only=secondary_only,
            agreement_ratio=ratio,
        )

        if ratio < self.min_agreement:
            logger.warning(
                "Redundant detection agreement %.2f < threshold %.2f",
                ratio,
                self.min_agreement,
            )

        return result
