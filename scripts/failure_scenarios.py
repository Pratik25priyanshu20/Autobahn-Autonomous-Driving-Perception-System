"""Controlled sensor failure scenario demonstrations for APS++.

Showcases how the perception stack detects and responds to degraded
sensor inputs (camera blackout, LIDAR dropout, radar inconsistency).
Ideal for interview presentations and system-resilience demos.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.safety.sensor_health import SensorHealthMonitor
from src.types.radar import RadarDetection, RadarFrame

console = Console()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"


@dataclass
class ScenarioResult:
    name: str
    normal_summary: str
    degraded_summary: str
    system_response: str
    details: list[str]


def _header(title: str) -> None:
    console.rule(f"[bold cyan]{title}[/bold cyan]")


# ---------------------------------------------------------------------------
# Scenario 1 -- Camera Blackout
# ---------------------------------------------------------------------------

def scenario_camera_blackout() -> ScenarioResult:
    _header("Scenario 1: Camera Blackout")

    monitor = SensorHealthMonitor(brightness_range=(40, 220))

    # Normal frame
    normal_frame = np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8)
    cam_normal = monitor.assess_camera(normal_frame)

    # Dark / blackout frame
    dark_frame = np.zeros((480, 640, 3), dtype=np.uint8) + 5
    cam_dark = monitor.assess_camera(dark_frame)

    table = Table(title="Camera Health Assessment", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Normal", style="green")
    table.add_column("Blackout", style="red")

    table.add_row("Score", f"{cam_normal.score:.3f}", f"{cam_dark.score:.3f}")
    table.add_row("Brightness", f"{cam_normal.brightness:.1f}", f"{cam_dark.brightness:.1f}")
    table.add_row("Blur (Laplacian var)", f"{cam_normal.blur:.1f}", f"{cam_dark.blur:.1f}")
    table.add_row("Occlusion", f"{cam_normal.occlusion:.3f}", f"{cam_dark.occlusion:.3f}")
    table.add_row("Degraded?", "No", "Yes" if cam_dark.score < 0.5 else "No")
    console.print(table)

    score_drop = cam_normal.score - cam_dark.score
    normal_summary = (
        f"Score={cam_normal.score:.3f}  |  brightness={cam_normal.brightness:.1f}  |  "
        f"blur={cam_normal.blur:.1f}  |  status=HEALTHY"
    )
    degraded_summary = (
        f"Score={cam_dark.score:.3f}  |  brightness={cam_dark.brightness:.1f}  |  "
        f"blur={cam_dark.blur:.1f}  |  status=DEGRADED"
    )
    system_response = (
        "Orchestrator detects brightness below threshold (40). "
        "Switches to degraded mode: increases tracking interval, "
        "skips non-critical stages (segmentation, depth), and "
        "issues WARNING: Sensor degradation detected."
    )

    console.print(Panel(
        f"[bold]Score drop:[/bold] {score_drop:.3f}\n"
        f"[bold]Brightness:[/bold] {cam_normal.brightness:.1f} -> {cam_dark.brightness:.1f}\n"
        f"[bold]System response:[/bold] {system_response}",
        title="[bold yellow]Camera Blackout Analysis[/bold yellow]",
        border_style="yellow",
    ))

    return ScenarioResult(
        name="Camera Blackout",
        normal_summary=normal_summary,
        degraded_summary=degraded_summary,
        system_response=system_response,
        details=[
            f"Score drop: {score_drop:.3f}",
            f"Brightness: {cam_normal.brightness:.1f} -> {cam_dark.brightness:.1f}",
            f"Normal blur variance: {cam_normal.blur:.1f}",
            f"Blackout blur variance: {cam_dark.blur:.1f}",
        ],
    )


# ---------------------------------------------------------------------------
# Scenario 2 -- LIDAR Point Cloud Dropout
# ---------------------------------------------------------------------------

@dataclass
class _MockPointCloud:
    """Minimal mock matching the points-attribute contract used by SensorHealthMonitor."""
    points: np.ndarray | None


def scenario_lidar_dropout() -> ScenarioResult:
    _header("Scenario 2: LIDAR Point Cloud Dropout")

    monitor = SensorHealthMonitor(expected_lidar_points=10_000)

    # Healthy cloud -- 10 000 points, 4 channels (x, y, z, intensity)
    healthy_pts = np.random.randn(10_000, 4).astype(np.float32)
    healthy_pts[:, 3] = np.random.uniform(0.1, 1.0, 10_000)  # varied intensity
    lidar_healthy = monitor.assess_lidar(_MockPointCloud(points=healthy_pts))

    # Sparse cloud -- 50 points
    sparse_pts = np.random.randn(50, 4).astype(np.float32)
    sparse_pts[:, 3] = np.random.uniform(0.1, 1.0, 50)
    lidar_sparse = monitor.assess_lidar(_MockPointCloud(points=sparse_pts))

    # Total dropout -- None
    lidar_none = monitor.assess_lidar(None)

    table = Table(title="LIDAR Health Assessment", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Healthy (10k)", style="green")
    table.add_column("Sparse (50)", style="yellow")
    table.add_column("Dropout (None)", style="red")

    table.add_row("Score", f"{lidar_healthy.score:.3f}", f"{lidar_sparse.score:.3f}", f"{lidar_none.score:.3f}")
    table.add_row("Point Ratio", f"{lidar_healthy.point_ratio:.3f}", f"{lidar_sparse.point_ratio:.3f}", f"{lidar_none.point_ratio:.3f}")
    table.add_row("Intensity OK", str(lidar_healthy.intensity_ok), str(lidar_sparse.intensity_ok), str(lidar_none.intensity_ok))
    console.print(table)

    normal_summary = (
        f"Score={lidar_healthy.score:.3f}  |  point_ratio={lidar_healthy.point_ratio:.3f}  |  "
        f"intensity_ok={lidar_healthy.intensity_ok}  |  status=HEALTHY"
    )
    degraded_summary = (
        f"Sparse: score={lidar_sparse.score:.3f}  |  point_ratio={lidar_sparse.point_ratio:.3f}\n"
        f"Dropout: score={lidar_none.score:.3f}  |  point_ratio={lidar_none.point_ratio:.3f}"
    )
    system_response = (
        "On sparse cloud: sensor health score drops below 0.5 threshold. "
        "Orchestrator falls back to camera-only perception. "
        "On total dropout (None): score=0.0, LIDAR pipeline skipped, "
        "fusion disabled, DTC_SH_001 diagnostic code logged."
    )

    console.print(Panel(
        f"[bold]Healthy -> Sparse drop:[/bold] {lidar_healthy.score - lidar_sparse.score:.3f}\n"
        f"[bold]Healthy -> Dropout drop:[/bold] {lidar_healthy.score - lidar_none.score:.3f}\n"
        f"[bold]System response:[/bold] {system_response}",
        title="[bold yellow]LIDAR Dropout Analysis[/bold yellow]",
        border_style="yellow",
    ))

    return ScenarioResult(
        name="LIDAR Point Cloud Dropout",
        normal_summary=normal_summary,
        degraded_summary=degraded_summary,
        system_response=system_response,
        details=[
            f"Healthy score: {lidar_healthy.score:.3f}",
            f"Sparse (50pts) score: {lidar_sparse.score:.3f}",
            f"Dropout score: {lidar_none.score:.3f}",
            f"Point ratio (sparse): {lidar_sparse.point_ratio:.4f}",
        ],
    )


# ---------------------------------------------------------------------------
# Scenario 3 -- Radar Inconsistency
# ---------------------------------------------------------------------------

def scenario_radar_inconsistency() -> ScenarioResult:
    _header("Scenario 3: Radar Detection Inconsistency")

    monitor = SensorHealthMonitor()

    # Feed 5 consistent frames with ~10 detections each
    consistent_scores = []
    for _ in range(5):
        dets = [RadarDetection(range_m=float(i), azimuth_deg=0.0, velocity_mps=5.0) for i in range(10)]
        frame = RadarFrame(detections=dets, timestamp=0.0)
        health = monitor.assess_radar(frame)
        consistent_scores.append(health)

    last_consistent = consistent_scores[-1]

    # Feed one frame with 0 detections (sudden dropout)
    empty_frame = RadarFrame(detections=[], timestamp=0.0)
    radar_inconsistent = monitor.assess_radar(empty_frame)

    table = Table(title="Radar Health Assessment", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Consistent (10 dets)", style="green")
    table.add_column("Inconsistent (0 dets)", style="red")

    table.add_row("Score", f"{last_consistent.score:.3f}", f"{radar_inconsistent.score:.3f}")
    table.add_row("Detection Consistency", f"{last_consistent.detection_consistency:.3f}", f"{radar_inconsistent.detection_consistency:.3f}")
    table.add_row("Detection Count", "10", "0")
    console.print(table)

    score_drop = last_consistent.score - radar_inconsistent.score
    normal_summary = (
        f"Score={last_consistent.score:.3f}  |  "
        f"detection_consistency={last_consistent.detection_consistency:.3f}  |  "
        f"status=STABLE"
    )
    degraded_summary = (
        f"Score={radar_inconsistent.score:.3f}  |  "
        f"detection_consistency={radar_inconsistent.detection_consistency:.3f}  |  "
        f"status=INCONSISTENT"
    )
    system_response = (
        "Sudden detection count change (10 -> 0) triggers consistency drop. "
        "Moving average detects anomaly. Radar health score falls, "
        "reducing radar weight in fusion. If sustained, orchestrator "
        "disables radar-camera fusion and logs DTC_RADAR_001."
    )

    console.print(Panel(
        f"[bold]Score drop:[/bold] {score_drop:.3f}\n"
        f"[bold]Consistency:[/bold] {last_consistent.detection_consistency:.3f} -> "
        f"{radar_inconsistent.detection_consistency:.3f}\n"
        f"[bold]System response:[/bold] {system_response}",
        title="[bold yellow]Radar Inconsistency Analysis[/bold yellow]",
        border_style="yellow",
    ))

    return ScenarioResult(
        name="Radar Detection Inconsistency",
        normal_summary=normal_summary,
        degraded_summary=degraded_summary,
        system_response=system_response,
        details=[
            f"Score drop: {score_drop:.3f}",
            f"Consistency: {last_consistent.detection_consistency:.3f} -> {radar_inconsistent.detection_consistency:.3f}",
            "Frames fed before anomaly: 5",
            "Detections per consistent frame: 10",
        ],
    )


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------

def _write_markdown(results: list[ScenarioResult]) -> Path:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    md_path = DEMO_DIR / "failure_scenarios.md"

    lines: list[str] = [
        "# APS++ Controlled Failure Scenarios",
        "",
        "Demonstrates how the perception stack detects and responds to degraded sensor inputs.",
        "",
    ]

    for i, r in enumerate(results, 1):
        lines.append(f"## Scenario {i}: {r.name}")
        lines.append("")
        lines.append("### Normal State")
        lines.append("```")
        lines.append(r.normal_summary)
        lines.append("```")
        lines.append("")
        lines.append("### Degraded State")
        lines.append("```")
        lines.append(r.degraded_summary)
        lines.append("```")
        lines.append("")
        lines.append("### Details")
        for d in r.details:
            lines.append(f"- {d}")
        lines.append("")
        lines.append("### System Response")
        lines.append(f"> {r.system_response}")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("*Generated by `scripts/failure_scenarios.py`*")
    lines.append("")

    md_path.write_text("\n".join(lines))
    return md_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    console.print(Panel(
        "[bold white]APS++ Controlled Failure Scenarios[/bold white]\n"
        "Demonstrates sensor degradation detection and system response",
        title="[bold blue]Failure Scenario Demo[/bold blue]",
        border_style="blue",
    ))

    results: list[ScenarioResult] = []

    results.append(scenario_camera_blackout())
    console.print()
    results.append(scenario_lidar_dropout())
    console.print()
    results.append(scenario_radar_inconsistency())
    console.print()

    # Overall health after all scenarios
    _header("Overall Summary")
    summary_table = Table(title="Failure Scenario Summary", show_lines=True)
    summary_table.add_column("Scenario", style="bold")
    summary_table.add_column("Normal Score", style="green")
    summary_table.add_column("Degraded Score", style="red")
    summary_table.add_column("Response")

    for r in results:
        # Extract first score from summary
        normal_score = r.normal_summary.split("Score=")[1].split(" ")[0].rstrip()
        degraded_score = r.degraded_summary.split("score=")[1].split(" ")[0].rstrip() if "score=" in r.degraded_summary else r.degraded_summary.split("Score=")[1].split(" ")[0].rstrip()
        summary_table.add_row(r.name, normal_score, degraded_score, r.system_response[:80] + "...")

    console.print(summary_table)

    # Write markdown
    md_path = _write_markdown(results)
    console.print(f"\n[bold green]Markdown report written to:[/bold green] {md_path}")


if __name__ == "__main__":
    main()
