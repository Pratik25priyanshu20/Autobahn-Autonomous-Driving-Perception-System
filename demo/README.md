# APS++ Demo Artifacts

Pre-generated screenshots and reports demonstrating the full perception pipeline capabilities.

## Screenshots

### Perception Overlay (`overlay_screenshot.png`)
Camera-view visualization showing:
- Tracked vehicles with bounding boxes, track IDs, class labels, and confidence scores
- Perspective-correct lane lines with dashed center markings
- FCW status indicator (NORMAL / PRE / WARNING / CRITICAL)
- Real-time latency HUD: per-stage timing breakdown (detection, tracking, lane, FCW, fusion)
- Sensor health bars (camera, LIDAR, radar) with color-coded status
- Lane departure warning indicator

### Bird's Eye View (`bev_screenshot.png`)
Top-down representation showing:
- Ego vehicle at bottom center
- Tracked objects with TTC-based coloring (green=safe, yellow=caution, orange=warning, red=critical)
- Velocity arrows showing heading and speed
- TTC labels per object (e.g., "ID 7 | TTC 3.2s")
- Lane corridor boundaries
- Distance grid (5m intervals)

### Sensor Health Dashboard (`sensor_health_screenshot.png`)
Detailed sensor monitoring showing:
- Camera health: 95% — brightness OK, blur sharp, occlusion 0.05
- LIDAR health: 82% — 8,200/10,000 points, intensity varied
- Radar health: 91% — consistency 0.91, 12 avg detections
- Overall health: 90.3%
- Color-coded progress bars and status badges (HEALTHY / DEGRADED / FAILED)

## Reports

### Metrics Summary (`METRICS.md`)
Quantitative evaluation covering:
- **Detection**: mAP = 0.65 (baseline)
- **Tracking**: MOTA = 1.0 (perfect) / 0.667 (imperfect), IDF1, HOTA, ID switches
- **System**: FPS, latency, inference backend
- **Safety**: FCW/LDW/BSD module status, sensor health, ASIL coverage
- **Fusion**: LIDAR + radar enrichment capabilities

### Latency Budget (`latency_budget.md`)
Per-stage timing breakdown with:
- Mean, P95, P99, and max latency per pipeline stage
- Percentage of total budget consumed by each stage
- Budget analysis: total used vs 50 ms target
- Headroom calculation

### Failure Scenarios (`failure_scenarios.md`)
Controlled degradation demonstrations:
- **Camera blackout**: Score drops from 1.00 to 0.06 — system triggers degradation warning
- **LIDAR dropout**: 50 points vs 10,000 expected — score drops to 0.30, camera-only fallback
- **Radar inconsistency**: 0 detections after consistent 10 — consistency drops to 0.00, DTC logged

## Regeneration

All artifacts can be regenerated from source:

```bash
# From the project root (autonomous-perception-system/)
python scripts/generate_metrics_report.py    # -> demo/METRICS.md
python scripts/latency_budget.py             # -> demo/latency_budget.md
python scripts/failure_scenarios.py          # -> demo/failure_scenarios.md
python scripts/generate_demo.py              # -> demo/*.png + demo/README.md
```

No external data or GPU required — all scripts use synthetic data and mocked models.
