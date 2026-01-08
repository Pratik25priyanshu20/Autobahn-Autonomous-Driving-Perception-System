#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from statistics import mean, median


def safe_mean(xs):
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else None


def pct(n, d):
    return (100.0 * n / d) if d else 0.0


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/summarize_run.py results/run_YYYYMMDD_HHMMSS")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing: {metrics_path}")

    m = json.loads(metrics_path.read_text())
    frames = m.get("frames", [])
    n = len(frames)
    if n == 0:
        print("No frames found in metrics.json")
        return

    fps_vals = [f.get("fps") for f in frames if f.get("fps") is not None]
    det_ms = [f.get("stages_ms", {}).get("detection") for f in frames]
    lane_ms = [f.get("stages_ms", {}).get("lane") for f in frames]
    seg_ms = [f.get("stages_ms", {}).get("segmentation") for f in frames]
    bev_ms = [f.get("stages_ms", {}).get("bev") for f in frames]

    lane_ok = 0
    for f in frames:
        ego_offset = (f.get("lane") or {}).get("ego_offset_px", None)
        if ego_offset is not None:
            lane_ok += 1

    fcw_counts = {"NORMAL": 0, "PRE": 0, "WARNING": 0, "CRITICAL": 0, "NONE": 0}
    ttc_vals = []
    for f in frames:
        fcw = f.get("fcw", {}) or {}
        st = fcw.get("state", "NONE")
        fcw_counts[st] = fcw_counts.get(st, 0) + 1
        ttc = fcw.get("ttc_s")
        if ttc is not None:
            ttc_vals.append(ttc)

    ldw_right = 0
    ldw_left = 0
    for f in frames:
        details = (f.get("safety") or {}).get("details", {}) or {}
        dep = details.get("ldw_departure")
        if dep == "RIGHT":
            ldw_right += 1
        elif dep == "LEFT":
            ldw_left += 1

    print("\n================ APS++ RUN SUMMARY ================")
    print(f"Run dir: {run_dir}")
    print(f"Frames: {n}")
    if fps_vals:
        print(f"FPS  avg={mean(fps_vals):.2f}  med={median(fps_vals):.2f}  min={min(fps_vals):.2f}  max={max(fps_vals):.2f}")
    else:
        print("FPS: (missing)")

    print("\nLatency (ms) (avg):")
    sm = safe_mean(det_ms)
    print(f"  detection:     {sm:.2f}" if sm is not None else "  detection:     (missing)")
    sm = safe_mean(lane_ms)
    print(f"  lane:          {sm:.2f}" if sm is not None else "  lane:          (missing)")
    sm = safe_mean(seg_ms)
    print(f"  segmentation:  {sm:.2f}" if sm is not None else "  segmentation:  (missing)")
    sm = safe_mean(bev_ms)
    print(f"  bev:           {sm:.2f}" if sm is not None else "  bev:           (missing)")

    print("\nLane availability:")
    print(f"  lane frames: {lane_ok}/{n} ({pct(lane_ok, n):.1f}%)")

    print("\nFCW state distribution:")
    for k in ["NORMAL", "PRE", "WARNING", "CRITICAL", "NONE"]:
        c = fcw_counts.get(k, 0)
        print(f"  {k:8s}: {c:4d} ({pct(c, n):.1f}%)")

    if ttc_vals:
        print(f"\nTTC stats (s): avg={mean(ttc_vals):.2f}  min={min(ttc_vals):.2f}  max={max(ttc_vals):.2f}")
    else:
        print("\nTTC stats: (no TTC values logged)")

    print("\nLDW departures (frame-level flags):")
    print(f"  LEFT:  {ldw_left}")
    print(f"  RIGHT: {ldw_right}")
    print("===================================================\n")


if __name__ == "__main__":
    main()
