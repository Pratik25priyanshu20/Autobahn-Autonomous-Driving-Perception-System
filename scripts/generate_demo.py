"""Generate professional demo screenshots from synthetic data for APS++.

Creates camera overlay, BEV top-down view, and sensor health visualization
images suitable for portfolio presentations and interviews.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src.bev.bev_renderer import BEVRenderer
from src.visualization.overlay import draw_sensor_health

DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

# -- Color palette (BGR) ---------------------------------------------------
WHITE = (255, 255, 255)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (60, 60, 60)
ROAD_GRAY = (80, 80, 80)
LANE_WHITE = (230, 230, 230)
GREEN = (0, 200, 0)
BRIGHT_GREEN = (0, 255, 0)
YELLOW = (0, 220, 255)
RED = (0, 0, 255)
ORANGE = (0, 165, 255)
CYAN = (255, 255, 0)
BLUE = (255, 140, 0)
SKY_BLUE = (210, 180, 140)
DARK_BG = (30, 30, 30)
PANEL_BG = (40, 40, 40)

# -- Dimensions -------------------------------------------------------------
WIDTH, HEIGHT = 1280, 720


# ---------------------------------------------------------------------------
# 1. Overlay Screenshot (camera view with perception annotations)
# ---------------------------------------------------------------------------

def _draw_road_scene(canvas: np.ndarray) -> np.ndarray:
    """Draw a synthetic road scene with sky, road, and horizon."""
    h, w = canvas.shape[:2]
    horizon_y = int(h * 0.38)

    # Sky gradient
    for y in range(horizon_y):
        ratio = y / max(1, horizon_y)
        b = int(210 - 80 * ratio)
        g = int(180 - 60 * ratio)
        r = int(140 - 40 * ratio)
        canvas[y, :] = (b, g, r)

    # Road surface
    canvas[horizon_y:, :] = ROAD_GRAY

    # Road edges (converging perspective lines)
    vanish_x = w // 2
    vanish_y = horizon_y
    left_bottom = (int(w * 0.1), h)
    right_bottom = (int(w * 0.9), h)
    cv2.line(canvas, (vanish_x, vanish_y), left_bottom, DARK_GRAY, 2)
    cv2.line(canvas, (vanish_x, vanish_y), right_bottom, DARK_GRAY, 2)

    return canvas


def _draw_lane_lines(canvas: np.ndarray) -> np.ndarray:
    """Draw dashed center and solid edge lane lines with perspective."""
    h, w = canvas.shape[:2]
    horizon_y = int(h * 0.38)
    vanish_x = w // 2

    # Lane positions at bottom
    left_lane_bottom = int(w * 0.35)
    center_bottom = w // 2
    right_lane_bottom = int(w * 0.65)

    # Draw solid edge lines
    cv2.line(canvas, (vanish_x, horizon_y), (left_lane_bottom - 80, h), LANE_WHITE, 3)
    cv2.line(canvas, (vanish_x, horizon_y), (right_lane_bottom + 80, h), LANE_WHITE, 3)

    # Draw dashed center line
    num_dashes = 12
    for i in range(num_dashes):
        t_start = i / num_dashes
        t_end = (i + 0.5) / num_dashes
        if i % 2 == 0:
            y_s = int(horizon_y + (h - horizon_y) * t_start)
            y_e = int(horizon_y + (h - horizon_y) * t_end)
            x_s = int(vanish_x + (center_bottom - vanish_x) * t_start)
            x_e = int(vanish_x + (center_bottom - vanish_x) * t_end)
            thickness = max(1, int(1 + 2 * t_start))
            cv2.line(canvas, (x_s, y_s), (x_e, y_e), LANE_WHITE, thickness)

    # Lane departure indicator (left side -- green = no departure)
    cv2.rectangle(canvas, (20, h - 60), (35, h - 20), GREEN, -1)
    cv2.putText(canvas, "LDW: CLEAR", (42, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1, cv2.LINE_AA)

    return canvas


def _draw_vehicle_bbox(
    canvas: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    label: str,
    color: tuple[int, int, int],
    conf: float,
    track_id: int | None = None,
) -> np.ndarray:
    """Draw a single detection bounding box with label and optional track ID."""
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

    # Label background
    text = f"{label} {conf:.2f}"
    if track_id is not None:
        text = f"ID {track_id} | {text}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(canvas, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(canvas, text, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)

    return canvas


def _draw_detections_on_scene(canvas: np.ndarray) -> np.ndarray:
    """Draw synthetic vehicle bounding boxes on the road scene."""
    # Vehicle 1 -- car ahead-left (medium distance)
    canvas = _draw_vehicle_bbox(canvas, 280, 310, 420, 430, "car", GREEN, 0.92, track_id=3)

    # Vehicle 2 -- truck ahead-right (closer)
    canvas = _draw_vehicle_bbox(canvas, 700, 280, 920, 500, "truck", YELLOW, 0.87, track_id=7)

    # Vehicle 3 -- car far ahead (small)
    canvas = _draw_vehicle_bbox(canvas, 580, 285, 640, 330, "car", GREEN, 0.78, track_id=12)

    # Vehicle 4 -- car in adjacent lane
    canvas = _draw_vehicle_bbox(canvas, 1000, 350, 1140, 480, "car", ORANGE, 0.81, track_id=5)

    # Draw simple vehicle rectangles inside bboxes for visual realism
    cv2.rectangle(canvas, (300, 340), (400, 420), (100, 100, 120), -1)
    cv2.rectangle(canvas, (720, 310), (900, 490), (90, 90, 110), -1)
    cv2.rectangle(canvas, (590, 295), (630, 325), (110, 110, 130), -1)
    cv2.rectangle(canvas, (1020, 370), (1120, 470), (105, 105, 125), -1)

    return canvas


def _draw_fcw_status(canvas: np.ndarray) -> np.ndarray:
    """Draw FCW status indicator in the top-right corner."""
    h, w = canvas.shape[:2]
    # FCW panel
    panel_x = w - 240
    panel_y = 12
    panel_w = 225
    panel_h = 36
    overlay = canvas.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 40, 0), -1)
    cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)
    cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), GREEN, 1)
    cv2.putText(canvas, "FCW: NORMAL", (panel_x + 12, panel_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2, cv2.LINE_AA)
    return canvas


def _draw_hud(canvas: np.ndarray) -> np.ndarray:
    """Draw HUD overlay with FPS, stage timings, and system status."""
    # Semi-transparent background for HUD
    overlay = canvas.copy()
    cv2.rectangle(overlay, (8, 8), (260, 195), PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)
    cv2.rectangle(canvas, (8, 8), (260, 195), LIGHT_GRAY, 1)

    y = 30
    cv2.putText(canvas, "APS++ | FPS: 28.5", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)
    y += 26

    stages = [
        ("detection", 12.3),
        ("tracking", 4.1),
        ("lane", 2.8),
        ("fcw", 0.4),
        ("fusion", 1.2),
        ("total", 20.8),
    ]
    for name, ms in stages:
        color = WHITE if name != "total" else CYAN
        cv2.putText(canvas, f"  {name}: {ms:5.1f} ms", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        y += 20

    return canvas


def _draw_sensor_health_bars(canvas: np.ndarray) -> np.ndarray:
    """Draw sensor health bars at the bottom-right of the frame."""
    health_dict = {"camera": 0.95, "lidar": 0.82, "radar": 0.91}
    return draw_sensor_health(canvas, health_dict)


def generate_overlay_screenshot() -> Path:
    """Generate the main camera overlay screenshot."""
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    canvas = _draw_road_scene(canvas)
    canvas = _draw_lane_lines(canvas)
    canvas = _draw_detections_on_scene(canvas)
    canvas = _draw_fcw_status(canvas)
    canvas = _draw_hud(canvas)
    canvas = _draw_sensor_health_bars(canvas)

    # Info warnings at bottom
    cv2.putText(canvas, "INFO: 4 detections | 4 tracks | ego_offset=+3.2px",
                (18, HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, LIGHT_GRAY, 1, cv2.LINE_AA)

    path = DEMO_DIR / "overlay_screenshot.png"
    cv2.imwrite(str(path), canvas)
    return path


# ---------------------------------------------------------------------------
# 2. BEV Screenshot (top-down Bird's Eye View)
# ---------------------------------------------------------------------------

class _MockWorldObj:
    """Minimal mock object for BEV rendering."""
    def __init__(self, track_id: int, x: float, y: float, vx: float = 0.0, vy: float = 0.0,
                 ttc: float | None = None, risk: str | None = None, class_name: str = "car"):
        self.track_id = track_id
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.ttc = ttc
        self.risk = risk
        self.class_name = class_name


class _MockWorld:
    """Minimal world model mock for BEV renderer."""
    def __init__(self):
        self.tracks = [
            _MockWorldObj(3, x=-0.8, y=15.0, vx=0.0, vy=-2.0, ttc=7.5, risk="NORMAL"),
            _MockWorldObj(7, x=1.2, y=8.0, vx=0.0, vy=-5.0, ttc=3.2, risk="CAUTION"),
            _MockWorldObj(12, x=0.2, y=25.0, vx=0.1, vy=-1.5, ttc=12.0, risk="NORMAL"),
            _MockWorldObj(5, x=4.5, y=10.0, vx=-1.0, vy=-3.0, ttc=5.0, risk="NORMAL"),
        ]
        self.objects = self.tracks
        self.lanes = {"ego_offset_px": 3.2}
        self.predictions = {}
        self.predictions_topk = {}
        self.interactions = []


def generate_bev_screenshot() -> Path:
    """Generate a BEV top-down view screenshot."""
    renderer = BEVRenderer(size=600, pixels_per_meter=10.0)
    world = _MockWorld()
    fcw = {"state": "NORMAL", "lead_id": 7}

    canvas = renderer.render(world, fcw)

    # Add title bar
    title_bar = np.zeros((40, canvas.shape[1], 3), dtype=np.uint8)
    title_bar[:] = PANEL_BG
    cv2.putText(title_bar, "APS++ Bird's Eye View (BEV)", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
    canvas = np.vstack([title_bar, canvas])

    # Add legend at bottom
    legend_h = 50
    legend = np.zeros((legend_h, canvas.shape[1], 3), dtype=np.uint8)
    legend[:] = PANEL_BG

    items = [
        ("Ego", WHITE, 15),
        ("Safe", BRIGHT_GREEN, 115),
        ("Caution", CYAN, 200),
        ("Warning", ORANGE, 310),
        ("Critical", RED, 430),
    ]
    for label, color, x_off in items:
        cv2.circle(legend, (x_off, 25), 6, color, -1)
        cv2.putText(legend, label, (x_off + 12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, LIGHT_GRAY, 1, cv2.LINE_AA)

    canvas = np.vstack([canvas, legend])

    path = DEMO_DIR / "bev_screenshot.png"
    cv2.imwrite(str(path), canvas)
    return path


# ---------------------------------------------------------------------------
# 3. Sensor Health Screenshot
# ---------------------------------------------------------------------------

def generate_sensor_health_screenshot() -> Path:
    """Generate a sensor health monitoring visualization."""
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    # Background -- dark with subtle grid
    canvas[:] = DARK_BG
    for y_line in range(0, HEIGHT, 40):
        cv2.line(canvas, (0, y_line), (WIDTH, y_line), (40, 40, 40), 1)
    for x_line in range(0, WIDTH, 40):
        cv2.line(canvas, (x_line, 0), (x_line, HEIGHT), (40, 40, 40), 1)

    # Title
    cv2.putText(canvas, "APS++ Sensor Health Monitor", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 2, cv2.LINE_AA)
    cv2.line(canvas, (30, 65), (530, 65), LIGHT_GRAY, 1)

    # Use the overlay function for the standard health bars
    health_dict = {"camera": 0.95, "lidar": 0.82, "radar": 0.91}
    canvas = draw_sensor_health(canvas, health_dict)

    # Draw large detailed gauges in center
    sensors = [
        ("Camera", 0.95, GREEN, "Brightness: OK | Blur: Sharp | Occlusion: 0.05"),
        ("LIDAR", 0.82, GREEN, "Points: 8,200/10,000 | Intensity: Varied"),
        ("Radar", 0.91, GREEN, "Consistency: 0.91 | Detections: 12 avg"),
    ]

    y_start = 120
    for i, (name, score, _color, detail) in enumerate(sensors):
        y = y_start + i * 180

        # Sensor name
        cv2.putText(canvas, name.upper(), (60, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2, cv2.LINE_AA)

        # Score text
        score_color = GREEN if score > 0.7 else ORANGE if score > 0.4 else RED
        cv2.putText(canvas, f"{score:.0%}", (300, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, score_color, 2, cv2.LINE_AA)

        # Progress bar
        bar_x = 60
        bar_y = y + 30
        bar_w = 500
        bar_h = 30
        # Background
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        # Filled portion
        fill_w = int(bar_w * score)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), score_color, -1)
        # Border
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), LIGHT_GRAY, 1)

        # Detail text
        cv2.putText(canvas, detail, (60, y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, LIGHT_GRAY, 1, cv2.LINE_AA)

        # Status badge
        status = "HEALTHY" if score > 0.7 else "DEGRADED" if score > 0.4 else "CRITICAL"
        badge_color = GREEN if status == "HEALTHY" else ORANGE if status == "DEGRADED" else RED
        badge_x = 600
        cv2.rectangle(canvas, (badge_x, y - 5), (badge_x + 120, y + 25), badge_color, -1)
        cv2.putText(canvas, status, (badge_x + 8, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2, cv2.LINE_AA)

    # Overall health
    overall = 0.5 * 0.95 + 0.3 * 0.82 + 0.2 * 0.91
    overall_y = y_start + 3 * 180 + 20
    cv2.line(canvas, (60, overall_y - 20), (720, overall_y - 20), LIGHT_GRAY, 1)
    cv2.putText(canvas, f"Overall Health: {overall:.1%}", (60, overall_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)

    # Timestamp
    cv2.putText(canvas, "System uptime: 00:12:34 | Frame: #18720",
                (60, HEIGHT - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, LIGHT_GRAY, 1, cv2.LINE_AA)

    path = DEMO_DIR / "sensor_health_screenshot.png"
    cv2.imwrite(str(path), canvas)
    return path


# ---------------------------------------------------------------------------
# 4. README generation
# ---------------------------------------------------------------------------

def generate_readme() -> Path:
    readme_path = DEMO_DIR / "README.md"
    content = """\
# APS++ Demo Artifacts

## Screenshots
- `overlay_screenshot.png` -- Camera feed with perception overlay
- `bev_screenshot.png` -- Bird's Eye View (BEV) representation
- `sensor_health_screenshot.png` -- Sensor health monitoring visualization

## Reports
- `METRICS.md` -- Full metrics summary
- `latency_budget.md` -- Per-stage latency breakdown
- `failure_scenarios.md` -- Controlled failure demonstrations

## How to Regenerate
```bash
python3 scripts/generate_metrics_report.py
python3 scripts/latency_budget.py
python3 scripts/failure_scenarios.py
python3 scripts/generate_demo.py
```
"""
    readme_path.write_text(content)
    return readme_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  APS++ Demo Screenshot Generator")
    print("=" * 60)

    overlay_path = generate_overlay_screenshot()
    print(f"  [1/4] Overlay screenshot  -> {overlay_path}")

    bev_path = generate_bev_screenshot()
    print(f"  [2/4] BEV screenshot      -> {bev_path}")

    health_path = generate_sensor_health_screenshot()
    print(f"  [3/4] Sensor health       -> {health_path}")

    readme_path = generate_readme()
    print(f"  [4/4] README.md           -> {readme_path}")

    print()
    print("  All demo artifacts generated successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
