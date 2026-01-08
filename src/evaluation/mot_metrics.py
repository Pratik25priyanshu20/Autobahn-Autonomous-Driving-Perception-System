"""CLEAR MOT evaluation metrics (Phase 6.4).

Computes MOTA, MOTP, ID switches for multi-object tracking evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class MOTResult:
    mota: float = 0.0
    motp: float = 0.0
    id_switches: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    total_gt: int = 0


class CLEARMOTEvaluator:
    """Computes CLEAR MOT metrics over a sequence of frames."""

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self._prev_matches: Dict[int, int] = {}  # gt_id -> pred_id

    def evaluate_sequence(
        self,
        gt_frames: List[List[Tuple[int, Tuple[int, int, int, int]]]],
        pred_frames: List[List[Tuple[int, Tuple[int, int, int, int]]]],
    ) -> MOTResult:
        total_gt = 0
        total_fp = 0
        total_fn = 0
        total_idsw = 0
        total_dist = 0.0
        total_matches = 0

        for gt_objs, pred_objs in zip(gt_frames, pred_frames):
            gt_boxes = {gid: bbox for gid, bbox in gt_objs}
            pred_boxes = {pid: bbox for pid, bbox in pred_objs}

            total_gt += len(gt_boxes)

            # Compute IoU matrix
            matched_gt = set()
            matched_pred = set()
            idsw = 0
            frame_matches: Dict[int, int] = {}

            for gid, gbox in gt_boxes.items():
                best_iou = 0.0
                best_pid = None
                for pid, pbox in pred_boxes.items():
                    if pid in matched_pred:
                        continue
                    iou = self._iou(gbox, pbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_pid = pid

                if best_pid is not None and best_iou >= self.iou_threshold:
                    matched_gt.add(gid)
                    matched_pred.add(best_pid)
                    frame_matches[gid] = best_pid
                    total_dist += (1.0 - best_iou)
                    total_matches += 1

                    if gid in self._prev_matches and self._prev_matches[gid] != best_pid:
                        idsw += 1

            total_fp += len(pred_boxes) - len(matched_pred)
            total_fn += len(gt_boxes) - len(matched_gt)
            total_idsw += idsw
            self._prev_matches = frame_matches

        mota = 1.0 - (total_fp + total_fn + total_idsw) / max(1, total_gt)
        motp = total_dist / max(1, total_matches)

        return MOTResult(
            mota=mota,
            motp=motp,
            id_switches=total_idsw,
            false_positives=total_fp,
            false_negatives=total_fn,
            total_gt=total_gt,
        )

    @staticmethod
    def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / max(union, 1e-6)
