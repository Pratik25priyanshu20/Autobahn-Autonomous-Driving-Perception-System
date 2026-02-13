"""CLEAR MOT evaluation metrics (Phase 6.4) + IDF1/HOTA (Task 5).

Computes MOTA, MOTP, IDF1, HOTA, ID switches for multi-object tracking evaluation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class MOTResult:
    mota: float = 0.0
    motp: float = 0.0
    id_switches: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    total_gt: int = 0
    idf1: float = 0.0
    hota: float = 0.0


class CLEARMOTEvaluator:
    """Computes CLEAR MOT metrics over a sequence of frames."""

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self._prev_matches: dict[int, int] = {}  # gt_id -> pred_id

    def evaluate_sequence(
        self,
        gt_frames: list[list[tuple[int, tuple[int, int, int, int]]]],
        pred_frames: list[list[tuple[int, tuple[int, int, int, int]]]],
    ) -> MOTResult:
        total_gt = 0
        total_fp = 0
        total_fn = 0
        total_idsw = 0
        total_dist = 0.0
        total_matches = 0

        for gt_objs, pred_objs in zip(gt_frames, pred_frames, strict=False):
            gt_boxes = {gid: bbox for gid, bbox in gt_objs}
            pred_boxes = {pid: bbox for pid, bbox in pred_objs}

            total_gt += len(gt_boxes)

            # Compute IoU matrix
            matched_gt = set()
            matched_pred = set()
            idsw = 0
            frame_matches: dict[int, int] = {}

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

    def compute_idf1(
        self,
        gt_frames: list[list[tuple[int, tuple[int, int, int, int]]]],
        pred_frames: list[list[tuple[int, tuple[int, int, int, int]]]],
    ) -> float:
        """Compute IDF1: ID precision/recall harmonic mean.

        Builds global ID assignment (gt_id → pred_id) based on longest shared association.
        """
        # Build per-ID association counts: (gt_id, pred_id) → count of matched frames
        assoc_counts: dict[tuple[int, int], int] = {}
        gt_id_counts: dict[int, int] = {}
        pred_id_counts: dict[int, int] = {}

        for gt_objs, pred_objs in zip(gt_frames, pred_frames, strict=False):
            gt_boxes = {gid: bbox for gid, bbox in gt_objs}
            pred_boxes = {pid: bbox for pid, bbox in pred_objs}

            for gid in gt_boxes:
                gt_id_counts[gid] = gt_id_counts.get(gid, 0) + 1
            for pid in pred_boxes:
                pred_id_counts[pid] = pred_id_counts.get(pid, 0) + 1

            matched_pred: set[int] = set()
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
                    matched_pred.add(best_pid)
                    key = (gid, best_pid)
                    assoc_counts[key] = assoc_counts.get(key, 0) + 1

        # Greedy global assignment: pick (gt, pred) pairs with highest association count
        used_gt: set[int] = set()
        used_pred: set[int] = set()
        idtp = 0

        sorted_assocs = sorted(assoc_counts.items(), key=lambda x: x[1], reverse=True)
        for (gid, pid), count in sorted_assocs:
            if gid in used_gt or pid in used_pred:
                continue
            used_gt.add(gid)
            used_pred.add(pid)
            idtp += count

        total_gt_count = sum(gt_id_counts.values())
        total_pred_count = sum(pred_id_counts.values())

        idfn = total_gt_count - idtp
        idfp = total_pred_count - idtp

        idp = idtp / max(1, idtp + idfp)
        idr = idtp / max(1, idtp + idfn)
        idf1 = 2.0 * idp * idr / max(1e-6, idp + idr)

        return idf1

    def compute_hota(
        self,
        gt_frames: list[list[tuple[int, tuple[int, int, int, int]]]],
        pred_frames: list[list[tuple[int, tuple[int, int, int, int]]]],
    ) -> float:
        """Compute HOTA: Higher Order Tracking Accuracy = sqrt(DetA × AssA).

        Simplified single-threshold implementation.
        """
        # Detection accuracy: TP / (TP + FP + FN)
        tp = 0
        fp_total = 0
        fn_total = 0
        assoc_counts: dict[tuple[int, int], int] = {}

        for gt_objs, pred_objs in zip(gt_frames, pred_frames, strict=False):
            gt_boxes = {gid: bbox for gid, bbox in gt_objs}
            pred_boxes = {pid: bbox for pid, bbox in pred_objs}

            matched_gt: set[int] = set()
            matched_pred: set[int] = set()

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
                    tp += 1
                    key = (gid, best_pid)
                    assoc_counts[key] = assoc_counts.get(key, 0) + 1

            fp_total += len(pred_boxes) - len(matched_pred)
            fn_total += len(gt_boxes) - len(matched_gt)

        det_a = tp / max(1, tp + fp_total + fn_total)

        # Association accuracy
        if not assoc_counts:
            return 0.0

        # For each matched (gt, pred) pair, compute association score
        # AssA = average over all TPs of |TPA(c)| / |FPA(c) + FNA(c) + TPA(c)|
        gt_total: dict[int, int] = {}
        pred_total: dict[int, int] = {}
        for (gid, pid), count in assoc_counts.items():
            gt_total[gid] = gt_total.get(gid, 0) + count
            pred_total[pid] = pred_total.get(pid, 0) + count

        ass_scores: list[float] = []
        for (gid, pid), tpa in assoc_counts.items():
            fpa = pred_total.get(pid, 0) - tpa
            fna = gt_total.get(gid, 0) - tpa
            ass_score = tpa / max(1, tpa + fpa + fna)
            # Weight by number of TPs in this association
            for _ in range(tpa):
                ass_scores.append(ass_score)

        ass_a = sum(ass_scores) / max(1, len(ass_scores))

        hota = math.sqrt(det_a * ass_a)
        return hota

    def evaluate_full(
        self,
        gt_frames: list[list[tuple[int, tuple[int, int, int, int]]]],
        pred_frames: list[list[tuple[int, tuple[int, int, int, int]]]],
    ) -> dict[str, float | int]:
        """Compute all metrics: MOTA, MOTP, IDF1, HOTA, FP, FN, IDSW."""
        self._prev_matches = {}
        basic = self.evaluate_sequence(gt_frames, pred_frames)
        idf1 = self.compute_idf1(gt_frames, pred_frames)
        hota = self.compute_hota(gt_frames, pred_frames)

        return {
            "MOTA": basic.mota,
            "MOTP": basic.motp,
            "IDF1": idf1,
            "HOTA": hota,
            "FP": basic.false_positives,
            "FN": basic.false_negatives,
            "IDSW": basic.id_switches,
            "total_gt": basic.total_gt,
        }

    @staticmethod
    def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / max(union, 1e-6)
