"""CLI runner for APS++ evaluation framework (Phase 6.4)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="APS++ Evaluation Framework")
    parser.add_argument("--metrics", required=True, help="Path to metrics.json from a run")
    parser.add_argument("--gt", default=None, help="Path to ground truth annotations (optional)")
    parser.add_argument("--output", default="evaluation_report.json", help="Output report path")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        print(f"Metrics file not found: {metrics_path}")
        return

    with open(metrics_path) as f:
        metrics = json.load(f)

    report = {
        "source": str(metrics_path),
        "total_frames": len(metrics.get("frames", [])),
        "summary": {},
    }

    frames = metrics.get("frames", [])
    if frames:
        fps_values = [f["fps"] for f in frames if f.get("fps")]
        report["summary"]["mean_fps"] = sum(fps_values) / len(fps_values) if fps_values else 0
        report["summary"]["min_fps"] = min(fps_values) if fps_values else 0

        det_counts = [f.get("detection_count", 0) for f in frames]
        report["summary"]["mean_detections"] = sum(det_counts) / len(det_counts)
        report["summary"]["max_detections"] = max(det_counts)

        track_counts = [f.get("track_count", 0) for f in frames]
        report["summary"]["mean_tracks"] = sum(track_counts) / len(track_counts)

        # Aggregate stage timings
        stage_totals: dict = {}
        stage_counts: dict = {}
        for f in frames:
            for stage, ms in f.get("stages_ms", {}).items():
                stage_totals[stage] = stage_totals.get(stage, 0) + ms
                stage_counts[stage] = stage_counts.get(stage, 0) + 1
        report["summary"]["mean_stage_ms"] = {
            s: stage_totals[s] / stage_counts[s] for s in stage_totals
        }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(report, indent=2))
    print(f"Evaluation report saved: {output_path}")


if __name__ == "__main__":
    main()
