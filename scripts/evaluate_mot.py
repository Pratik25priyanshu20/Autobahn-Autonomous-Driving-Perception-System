#!/usr/bin/env python3
"""MOT Benchmark Evaluation CLI (Task 5).

Usage:
  python scripts/evaluate_mot.py --gt data/gt.txt --pred data/pred.txt --format mot17
  python scripts/evaluate_mot.py --gt data/kitti/labels/ --pred data/pred.txt --format kitti
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.evaluation.mot_formatter import parse_kitti_tracking, parse_mot17
from src.evaluation.mot_metrics import CLEARMOTEvaluator


def main():
    parser = argparse.ArgumentParser(description="MOT Benchmark Evaluation")
    parser.add_argument("--gt", required=True, help="Path to ground truth (file or dir)")
    parser.add_argument("--pred", required=True, help="Path to predictions file")
    parser.add_argument("--format", choices=["mot17", "kitti"], default="mot17", help="GT format")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    console = Console()
    console.print("[bold]APS++ MOT Evaluation[/bold]")

    # Parse ground truth
    gt_frames = parse_kitti_tracking(args.gt) if args.format == "kitti" else parse_mot17(args.gt)

    # Parse predictions (always MOT17 format)
    pred_frames = parse_mot17(args.pred)

    # Align frame counts
    min_frames = min(len(gt_frames), len(pred_frames))
    if min_frames == 0:
        console.print("[red]No frames to evaluate[/red]")
        sys.exit(1)
    gt_frames = gt_frames[:min_frames]
    pred_frames = pred_frames[:min_frames]

    # Evaluate
    evaluator = CLEARMOTEvaluator(iou_threshold=0.5)
    result = evaluator.evaluate_full(gt_frames, pred_frames)

    # Display
    table = Table(title="MOT Evaluation Results")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    for key, val in result.items():
        if isinstance(val, float):
            table.add_row(key, f"{val:.4f}")
        else:
            table.add_row(key, str(val))
    console.print(table)

    # Output
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
        console.print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
