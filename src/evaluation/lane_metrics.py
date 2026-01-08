"""Lane detection evaluation metrics (Phase 6.4)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class LaneMetricsResult:
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    mean_iou: float = 0.0


class LaneIoUEvaluator:
    """Computes precision/recall/F1 and IoU for lane detection."""

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

    def evaluate(
        self,
        gt_masks: List[np.ndarray],
        pred_masks: List[np.ndarray],
    ) -> LaneMetricsResult:
        tp = 0
        fp = 0
        fn = 0
        ious: List[float] = []

        for gt, pred in zip(gt_masks, pred_masks):
            gt_bin = (gt > 0).astype(np.uint8)
            pred_bin = (pred > 0).astype(np.uint8)

            intersection = np.sum(gt_bin & pred_bin)
            union = np.sum(gt_bin | pred_bin)
            iou = intersection / max(union, 1e-6)
            ious.append(iou)

            if iou >= self.iou_threshold:
                tp += 1
            else:
                if np.sum(pred_bin) > 0:
                    fp += 1
                if np.sum(gt_bin) > 0:
                    fn += 1

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)

        return LaneMetricsResult(
            precision=precision,
            recall=recall,
            f1=f1,
            mean_iou=float(np.mean(ious)) if ious else 0.0,
        )
