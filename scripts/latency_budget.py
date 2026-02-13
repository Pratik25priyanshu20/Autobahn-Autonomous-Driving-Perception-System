#!/usr/bin/env python3
"""Profile per-stage latency of the Autobahn Perception Stack pipeline.

Runs 100 synthetic frames through the Orchestrator, collects ``stages_ms``
from each WorldModel, and produces a latency breakdown table printed via
Rich and saved to ``demo/latency_budget.md``.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

# ---------------------------------------------------------------------------
# Path setup so we can import project modules regardless of working directory
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Rich console (best-effort import)
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.table import Table
except ImportError:
    Console = None  # type: ignore[assignment,misc]
    Table = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "demo" / "latency_budget.md"
NUM_FRAMES = 100
FRAME_H, FRAME_W = 480, 640
BUDGET_MS = 50.0  # target total pipeline budget in milliseconds


# ---------------------------------------------------------------------------
# Mock helpers (same pattern as tests/test_e2e_smoke.py)
# ---------------------------------------------------------------------------

class _MockDetector:
    """Drop-in replacement for YOLODetector that returns no detections."""

    def infer(self, frame, conf_thres=0.25):  # noqa: ARG002
        return []


class _MockTracker:
    """Drop-in replacement for DeepSORTTracker that returns empty tracks."""

    def update(self, frame, detections):  # noqa: ARG002
        return [], {}


# ---------------------------------------------------------------------------
# Orchestrator factory
# ---------------------------------------------------------------------------

def _minimal_cfg() -> dict:
    """Return a minimal Orchestrator config with lane detection enabled."""
    return {
        "perception": {"runtime": "pytorch", "conf_thres": 0.25},
        "tracking": {"enabled": False, "interval": 1, "kalman": False},
        "lane": {"enabled": True, "backend": "canny_hough"},
        "segmentation": {"enabled": False},
        "depth": {"enabled": False},
        "weather": {"enabled": False},
        "ldw": {"enabled": False},
        "fcw": {"enabled": False},
        "bsd": {"enabled": False},
        "occupancy_grid": {"enabled": False},
        "safety": {"asil": {"enabled": False}},
        "performance": {"target_fps": 30, "fps_smoothing": 0.9},
        "video": {"resize": {"enabled": False}},
    }


def _build_orchestrator():
    """Instantiate Orchestrator with mocked heavy dependencies."""
    cfg = _minimal_cfg()

    with (
        patch(
            "src.perception.detection.yolo.YOLODetector",
            return_value=_MockDetector(),
        ),
        patch(
            "src.perception.tracking.deepsort_tracker.DeepSORTTracker",
            return_value=_MockTracker(),
        ),
    ):
        from src.runtime.orchestrator import Orchestrator

        logger = logging.getLogger("latency_budget")
        logger.setLevel(logging.WARNING)
        orch = Orchestrator(cfg, logger)

    # Belt-and-suspenders: ensure mocked detector/tracker are active
    orch.detector = _MockDetector()
    orch.tracker = _MockTracker()
    return orch


def _make_frame() -> np.ndarray:
    """Create a synthetic BGR frame filled with mid-grey."""
    return np.full((FRAME_H, FRAME_W, 3), 128, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _percentile(data: list[float], p: float) -> float:
    """Compute the p-th percentile of *data* (0-100 scale)."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    floor_k = int(k)
    ceil_k = min(floor_k + 1, len(sorted_data) - 1)
    frac = k - floor_k
    return sorted_data[floor_k] + frac * (sorted_data[ceil_k] - sorted_data[floor_k])


def _mean(data: list[float]) -> float:
    return sum(data) / max(len(data), 1)


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------

def _generate_markdown(stage_stats: dict[str, dict[str, float]], total_stats: dict[str, float]) -> str:
    lines: list[str] = []
    lines.append("# Latency Budget Breakdown")
    lines.append("")
    lines.append(f"> Profiled over **{NUM_FRAMES}** synthetic frames ({FRAME_H}x{FRAME_W} BGR)")
    lines.append(">")
    lines.append("> Auto-generated by `scripts/latency_budget.py`")
    lines.append("")

    # --- Per-stage table ---
    lines.append("## Per-Stage Latency")
    lines.append("")
    lines.append("| Stage | Mean (ms) | P95 (ms) | P99 (ms) | Max (ms) | % of Total |")
    lines.append("|-------|----------:|----------:|----------:|----------:|-----------:|")
    for stage, stats in stage_stats.items():
        pct = (stats["mean"] / max(total_stats["mean"], 1e-6)) * 100.0
        lines.append(
            f"| {stage} | {stats['mean']:.3f} | {stats['p95']:.3f} "
            f"| {stats['p99']:.3f} | {stats['max']:.3f} | {pct:.1f}% |"
        )
    lines.append(
        f"| **TOTAL** | **{total_stats['mean']:.3f}** | **{total_stats['p95']:.3f}** "
        f"| **{total_stats['p99']:.3f}** | **{total_stats['max']:.3f}** | **100.0%** |"
    )
    lines.append("")

    # --- Budget analysis ---
    lines.append("## Budget Analysis")
    lines.append("")
    headroom = BUDGET_MS - total_stats["mean"]
    headroom_pct = (headroom / BUDGET_MS) * 100.0
    status = "WITHIN BUDGET" if headroom >= 0 else "OVER BUDGET"
    lines.append("| Metric | Value |")
    lines.append("|--------|------:|")
    lines.append(f"| Total Budget | {BUDGET_MS:.1f} ms |")
    lines.append(f"| Mean Used | {total_stats['mean']:.3f} ms |")
    lines.append(f"| P95 Used | {total_stats['p95']:.3f} ms |")
    lines.append(f"| Headroom (mean) | {headroom:.3f} ms ({headroom_pct:.1f}%) |")
    lines.append(f"| Status | **{status}** |")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*Report generated by the Autobahn Perception Stack latency profiler.*")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rich stdout
# ---------------------------------------------------------------------------

def _print_rich_table(stage_stats: dict[str, dict[str, float]], total_stats: dict[str, float]) -> None:
    if Console is None or Table is None:
        print("[INFO] Rich not installed -- skipping console table.")
        return

    console = Console()
    console.print()
    console.rule(f"[bold blue]Latency Budget -- {NUM_FRAMES} frames @ {FRAME_H}x{FRAME_W}[/bold blue]")
    console.print()

    table = Table(title="Per-Stage Latency", show_lines=True)
    table.add_column("Stage", style="cyan", no_wrap=True)
    table.add_column("Mean (ms)", justify="right", style="green")
    table.add_column("P95 (ms)", justify="right", style="yellow")
    table.add_column("P99 (ms)", justify="right", style="yellow")
    table.add_column("Max (ms)", justify="right", style="red")
    table.add_column("% of Total", justify="right", style="magenta")

    for stage, stats in stage_stats.items():
        pct = (stats["mean"] / max(total_stats["mean"], 1e-6)) * 100.0
        table.add_row(
            stage,
            f"{stats['mean']:.3f}",
            f"{stats['p95']:.3f}",
            f"{stats['p99']:.3f}",
            f"{stats['max']:.3f}",
            f"{pct:.1f}%",
        )

    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{total_stats['mean']:.3f}[/bold]",
        f"[bold]{total_stats['p95']:.3f}[/bold]",
        f"[bold]{total_stats['p99']:.3f}[/bold]",
        f"[bold]{total_stats['max']:.3f}[/bold]",
        "[bold]100.0%[/bold]",
    )
    console.print(table)
    console.print()

    # --- Budget analysis ---
    headroom = BUDGET_MS - total_stats["mean"]
    headroom_pct = (headroom / BUDGET_MS) * 100.0
    status = "[green]WITHIN BUDGET[/green]" if headroom >= 0 else "[red]OVER BUDGET[/red]"

    budget_table = Table(title="Budget Analysis", show_lines=True)
    budget_table.add_column("Metric", style="cyan")
    budget_table.add_column("Value", justify="right", style="green")
    budget_table.add_row("Total Budget", f"{BUDGET_MS:.1f} ms")
    budget_table.add_row("Mean Used", f"{total_stats['mean']:.3f} ms")
    budget_table.add_row("P95 Used", f"{total_stats['p95']:.3f} ms")
    budget_table.add_row("Headroom (mean)", f"{headroom:.3f} ms ({headroom_pct:.1f}%)")
    budget_table.add_row("Status", status)
    console.print(budget_table)
    console.print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Building orchestrator (mocked detector + tracker) ...")
    orch = _build_orchestrator()

    # Collect stages_ms from each frame
    all_stages: dict[str, list[float]] = {}
    all_totals: list[float] = []

    print(f"Running {NUM_FRAMES} frames through the pipeline ...")
    for i in range(NUM_FRAMES):
        frame = _make_frame()
        wm = orch.process_frame(frame_id=i, frame=frame, packet=None)

        stages = wm.runtime.stages_ms
        frame_total = 0.0
        for stage_name, ms in stages.items():
            all_stages.setdefault(stage_name, []).append(ms)
            frame_total += ms
        all_totals.append(frame_total)

    # Compute statistics per stage
    stage_stats: dict[str, dict[str, float]] = {}
    for stage_name, values in all_stages.items():
        stage_stats[stage_name] = {
            "mean": _mean(values),
            "p95": _percentile(values, 95),
            "p99": _percentile(values, 99),
            "max": max(values),
        }

    total_stats = {
        "mean": _mean(all_totals),
        "p95": _percentile(all_totals, 95),
        "p99": _percentile(all_totals, 99),
        "max": max(all_totals),
    }

    # Generate markdown report
    md = _generate_markdown(stage_stats, total_stats)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(md)
    print(f"[OK] Latency budget report saved to {OUTPUT_PATH}")

    # Print Rich table to stdout
    _print_rich_table(stage_stats, total_stats)


if __name__ == "__main__":
    main()
