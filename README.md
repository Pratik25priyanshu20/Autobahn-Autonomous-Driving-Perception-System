# APS++ — Autonomous Perception Stack

APS++ is a production-grade autonomous driving perception stack engineered for German Autobahn scenarios. It processes multi-sensor inputs (camera, LIDAR, radar) and outputs real-time detections, multi-object tracks, lane geometry, depth maps, safety warnings, BEV visualization, sensor health assessments, and runtime metrics — all within a 50 ms latency budget.

**179 tests | 0 ruff errors | 100+ source modules | ISO 26262 safety layer | Config-validated startup**

---

## Key Features

### Perception
- **Object Detection** — YOLOv8/v11 with ONNX Runtime and TensorRT inference paths; INT8 quantization support
- **Multi-Object Tracking** — DeepSORT or ByteTrack (config-selectable) with per-object Kalman filtering in world-frame coordinates
- **Lane Detection** — Canny+Hough baseline or UFLDv2 neural backend with stability gating and EMA smoothing
- **Monocular Depth** — MiDAS or Depth Anything V2 backends for dense per-frame depth maps
- **Semantic Segmentation** — DeepLabV3+ (MobileNetV3) for drivable area extraction
- **Weather/Visibility** — Image-statistics classifier (clear / fog / dark / glare) with automatic degraded-mode triggering

### Sensor Fusion
- **LIDAR Processing** — Point cloud voxelization, RANSAC ground removal, 3D clustering, BEV encoding
- **LIDAR-Camera Fusion** — Projects 3D LIDAR clusters onto 2D tracks for depth enrichment
- **Radar Processing** — Ghost filtering, polar-to-cartesian conversion, spatial clustering
- **Radar-Camera Fusion** — Projects radar detections to image space, enriches tracks with velocity and range
- **Temporal Prediction** — Kalman-based trajectory prediction at 0.5 s, 1.0 s, and 2.0 s horizons with top-K hypotheses

### Safety (ISO 26262)
- **Forward Collision Warning (FCW)** — TTC-based 4-state machine (NORMAL → PRE → WARNING → CRITICAL)
- **Lane Departure Warning (LDW)** — Stability-gated offset monitoring
- **Blind Spot Detection (BSD)** — Lateral track monitoring with configurable zones
- **Rear Cross-Traffic Alert (RCTA)** — Cross-traffic TTC evaluation
- **Sensor Health Monitor** — Camera (brightness/blur/occlusion), LIDAR (point density), radar (detection consistency)
- **ASIL Classification** — Risk classifier per hazard scenario (ASIL-A through ASIL-D)
- **Plausibility Checker** — Cross-module sanity validation (detection/track/TTC coherence)
- **Redundant Detector** — Lightweight fallback detector for cross-validation
- **DTC Logger** — ISO 14229-style diagnostic trouble code logging
- **Occupancy Grid** — BEV grid projected from depth + semantic segmentation

### Prediction & Planning
- **Interaction Model** — Rule-based gap acceptance, yield heuristic, following distance, cut-in prediction
- **Trajectory Prediction** — Multi-horizon Kalman projection with top-K hypotheses
- **Control** — Pure Pursuit steering and MPC trajectory optimization (safety-gated)

### Explainability
- **Saliency Maps** — Activation-based heatmaps showing what the detector is attending to
- **Attention Overlay** — Blended heatmap visualization on camera frames

### Infrastructure
- **Data Recording** — Gzipped msgpack streams (`.apsrec`) with configurable interval and size limits
- **Replay Input** — Frame-level seeking and variable playback speed from recorded sessions
- **KITTI Integration** — Native KITTI dataset loader with calibration matrix support
- **CARLA Integration** — CARLA simulator bridge with scenario runner and metrics dashboard
- **Streaming** — FastAPI + WebSocket server for real-time frame and world-model JSON streaming
- **Config Validation** — Startup validation of all config values (enums, ranges, dependencies)
- **Evaluation** — CLEAR MOT (MOTA/MOTP), IDF1, HOTA, lane IoU, safety response latency

### Performance
- **Parallel Pipeline** — ThreadPoolExecutor for concurrent perception stages
- **Adaptive Frame Skip** — Health-monitor watchdog with graceful degradation
- **ONNX/TensorRT** — Export pipeline with FP16/INT8 quantization and accuracy guardrails
- **BEV Visualization** — Top-down ego corridor with safety zones, velocity arrows, and prediction circles

---

## System Pipeline

```
Input Sources
  Video / Webcam / CARLA / KITTI / Replay (.apsrec)
    |
    v
  Orchestrator (config-driven, 15 extracted stage methods)
    |
    +--[Parallel Block]--+------------------+------------------+
    |                    |                  |                  |
    v                    v                  v                  v
  Detection          Lanes             Depth           Segmentation
  YOLOv8/v11       CH | UFLDv2       MiDAS | DA       DeepLabV3+
  ONNX | TRT
    |
    v
  Sensor Health Assessment (camera brightness/blur, LIDAR points, radar consistency)
    |
    v
  Saliency / Grad-CAM (activation-based heatmap)
    |
    v
  Tracking (DeepSORT | ByteTrack) + Confidence Calibration
    |
    v
  Kalman Filter (per-track world-frame state: x, y, vx, vy)
    |
    v
  Temporal Prediction (0.5s, 1.0s, 2.0s) + Top-K Hypotheses
    |
    v
  Interaction Model (gap acceptance, yield, following, cut-in)
    |
    v
  LIDAR Processing (voxelize -> ground removal -> clustering -> BEV)
    |
    v
  LIDAR-Camera Fusion (3D cluster -> 2D track depth enrichment)
    |
    v
  Radar Processing (ghost filter -> cartesian -> cluster)
    |
    v
  Radar-Camera Fusion (project -> match -> velocity enrichment)
    |
    v
  Safety (FCW / LDW / BSD / RCTA / Occupancy / Plausibility)
    |
    v
  Control (Pure Pursuit / MPC, safety-gated)
    |
    v
  World Model (canonical per-frame state + EMA smoothing)
    |
    +----------+-----------+-----------+
    |          |           |           |
    v          v           v           v
  Overlay    BEV       Streaming   Recording
  (HUD)   (top-down)  (WebSocket) (.apsrec)
```

---

## Quick Start

### Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                  # core dependencies via pyproject.toml
pip install -e '.[dev]'           # + pytest, hypothesis, ruff, mypy
pip install -e '.[onnx]'          # + onnxruntime
pip install -e '.[streaming]'     # + fastapi, uvicorn, websockets
pip install -e '.[carla]'         # + carla client
```

Or using requirements.txt:

```bash
pip install -r requirements.txt
```

### Run

```bash
# Video replay (default)
python src/app.py --config configs/system.yaml --input data/samples/test_drive.mp4

# Live webcam
python src/app.py --config configs/system.yaml --live

# CARLA simulator
python src/app.py --config configs/system.yaml --carla

# KITTI dataset
python src/app.py --config configs/system.yaml --kitti data/kitti/

# Replay a recorded session
python src/app.py --config configs/system.yaml --replay recordings/session_001.apsrec

# Record while running
python src/app.py --config configs/system.yaml --input data/samples/test_drive.mp4 --record

# With real-time streaming dashboard
python src/app.py --config configs/system.yaml --input data/samples/test_drive.mp4 --stream --stream-port 8765
```

### Test

```bash
pytest tests/ -v                          # 179 tests, 27 modules
pytest tests/test_safety*.py -v           # Safety-focused tests
pytest tests/test_e2e_smoke.py -v         # End-to-end smoke tests
pytest tests/ --cov=src --cov-report=html # Coverage report
```

### Docker

```bash
docker compose up --build
```

The container exposes port `8765` for WebSocket streaming when enabled.

---

## Configuration

All features are controlled through YAML config files under `configs/`. New features default to `enabled: false` to preserve backward compatibility. Config is validated at startup — invalid values raise `ConfigValidationError` with clear messages.

| File | Scope |
|------|-------|
| `system.yaml` | Detection, runtime backend, tracking, lanes, depth, weather, parallel execution, adaptive skip, CARLA, control, radar, radar fusion, recording, explainability, interaction model |
| `safety.yaml` | FCW/LDW thresholds, occupancy grid, BSD/RCTA, sensor health |
| `fusion.yaml` | Kalman process/measurement noise, smoothing |
| `perception.yaml` | Resolution, detection confidence thresholds |
| `control.yaml` | Pure Pursuit / MPC parameters |

### Feature toggles

```yaml
# Radar fusion
radar:
  enabled: true
  min_rcs_dbsm: -10.0
  cluster_distance_m: 2.0
radar_fusion:
  enabled: true
  match_threshold_px: 50

# Sensor health monitoring
sensor_health:
  enabled: true
  brightness_range: [40, 220]
  blur_threshold: 100.0
  health_threshold: 0.5

# Saliency / explainability
explainability:
  enabled: true
  target_layer: "model.model[-2]"

# Interaction model
interaction:
  enabled: true
  min_gap_s: 3.0
  safe_following_distance_s: 2.0

# Data recording
recording:
  enabled: true
  record_interval: 1
  max_file_size_mb: 100
```

---

## Outputs

Each run produces a timestamped directory under `results/`:

| File | Contents |
|------|----------|
| `output.mp4` | Annotated video with detections, tracks, lanes, FCW/LDW, BSD, sensor health indicators |
| `bev.mp4` | Bird's-eye view with ego corridor, safety zones, prediction circles, interaction arrows |
| `metrics.json` | FPS, per-stage timing, git commit hash |
| `safety_events.jsonl` | Safety state transitions with timestamps |
| `dtc_log.jsonl` | ISO 14229-style diagnostic trouble codes |

---

## Demo Artifacts

Pre-generated screenshots and reports are in `demo/`:

| File | Description |
|------|-------------|
| `overlay_screenshot.png` | Camera view with perception overlay — bboxes, lane lines, FCW status, latency HUD, sensor health bars |
| `bev_screenshot.png` | Bird's-eye view — tracked objects with TTC coloring, velocity arrows, lane corridor |
| `sensor_health_screenshot.png` | Sensor health dashboard — camera/LIDAR/radar scores with progress bars |
| `METRICS.md` | Detection (mAP), tracking (MOTA, IDF1, HOTA), safety module summary |
| `latency_budget.md` | Per-stage mean/P95/P99 latency breakdown with budget analysis |
| `failure_scenarios.md` | Controlled degradation demos — camera blackout, LIDAR dropout, radar inconsistency |

Regenerate all artifacts:

```bash
python scripts/generate_metrics_report.py
python scripts/latency_budget.py
python scripts/failure_scenarios.py
python scripts/generate_demo.py
```

---

## Project Structure

```
src/
  app.py                                  # CLI entrypoint
  types/                                  # Canonical type definitions
    detection.py                          #   Detection, bbox/score/label properties
    track.py                              #   Track with world-frame attrs (x, y, vx, vy, ttc, risk)
    ego.py                                #   EgoState
    lanes.py                              #   LaneGeometry, LaneState
    safety.py                             #   SafetyStateEnum, SafetyStatus, SafetyOutput
    world_model.py                        #   WorldModel (canonical per-frame state)
    perception.py                         #   FramePacket, PerceptionOutput
    detection3d.py                        #   Detection3D (pseudo-3D boxes)
    pointcloud.py                         #   PointCloud, BEVGrid
    radar.py                              #   RadarDetection, RadarFrame
  runtime/
    orchestrator.py                       # Central pipeline (15 extracted stage methods)
    health_monitor.py                     # Latency watchdog with degraded mode
    frame_sync.py                         # Multi-sensor frame synchronization
    parallel_executor.py                  # ThreadPoolExecutor for concurrent stages
  inputs/
    base_input.py                         # Input ABC
    video_input.py                        # File-based video input
    webcam_input.py                       # Live camera input
    carla_input.py                        # CARLA simulator integration
    kitti_input.py                        # KITTI dataset loader with calibration
    radar_input.py                        # Radar CSV input
  perception/
    detection/
      yolo.py                             # YOLOv8/v11 (auto device: CUDA/MPS/CPU)
      onnx_detector.py                    # ONNX Runtime with CPU/CUDA/TensorRT providers
      mono3d_stub.py                      # 2D + depth -> pseudo-3D projection
    tracking/
      deepsort_tracker.py                 # DeepSORT multi-object tracker
      bytetrack_tracker.py                # ByteTrack via supervision library
    lanes/
      base_lane_detector.py               # Lane detector ABC
      lane_detector.py                    # Canny + Hough baseline
      ufld_detector.py                    # UFLDv2 neural lane detector
    depth/
      base_depth.py                       # Depth estimator ABC
      midas_depth.py                      # MiDAS via torch.hub
      depth_anything.py                   # Depth Anything V2 via torch.hub
    segmentation/
      base_segmenter.py                   # Segmenter ABC
      deeplabv3_segmenter.py              # DeepLabV3+ (MobileNet)
      postprocess.py                      # Drivable area extraction
    weather/
      visibility_detector.py              # Clear/fog/dark/glare classifier
    lidar/
      point_cloud_processor.py            # Voxelization, RANSAC ground removal, clustering
      bev_encoder.py                      # Point cloud -> BEV grid encoding
    radar/
      radar_processor.py                  # Ghost filter, cartesian, clustering
    explainability/
      grad_cam.py                         # Activation-based saliency heatmaps
      attention_overlay.py                # Colormap + blend overlay
    calibration.py                        # Confidence temperature scaling
  fusion/
    kalman_tracker.py                     # Per-object Kalman filter (state=[x,y,vx,vy])
    temporal_predictor.py                 # Trajectory prediction at 0.5s/1.0s/2.0s
    lidar_camera_fusion.py                # LIDAR 3D cluster -> 2D track fusion
    radar_camera_fusion.py                # Radar -> image projection + track enrichment
    fusion_engine.py                      # Perception -> WorldModel fusion
    projector.py                          # Pixel -> world-frame projection
  safety/
    safety_manager.py                     # Unified safety state manager (FCW+LDW+BSD)
    fcw.py                                # FCW state machine
    ttc.py                                # Time-to-collision computation
    risk.py                               # Risk score calculation
    rules.py                              # Rule-based safety evaluation
    occupancy_grid.py                     # BEV occupancy from depth + segmentation
    bsd_rcta.py                           # Blind Spot Detection + Rear Cross-Traffic Alert
    sensor_health.py                      # Camera/LIDAR/radar health scoring
    asil_classifier.py                    # ISO 26262 ASIL risk classification
    plausibility_checker.py               # Cross-module sanity validation
    redundant_detector.py                 # Lightweight fallback detector
    dtc_logger.py                         # Diagnostic trouble code logging
    safety_logger.py                      # Safety event logging
    distance.py                           # Distance estimation
  prediction/
    interaction_model.py                  # Gap acceptance, yield, following, cut-in
  recording/
    data_recorder.py                      # Gzipped msgpack stream writer
    replay_input.py                       # Frame-level replay with seeking
    recording_types.py                    # RecordedFrame dataclass
  control/
    base_controller.py                    # Controller ABC
    pure_pursuit.py                       # Steering from lane offset + safety-gated throttle
    mpc.py                                # Model predictive control
  bev/
    bev_renderer.py                       # Top-down BEV with safety zones and predictions
  visualization/
    overlay.py                            # HUD, detections, tracks, FCW, LDW, BSD, sensor health
    dashboard.py                          # Metrics dashboard
  streaming/
    server.py                             # FastAPI + WebSocket JPEG/JSON streaming
  simulation/
    scenario_runner.py                    # CARLA scenario execution
    test_harness.py                       # Simulation test framework
    metrics_dashboard.py                  # Live metrics visualization
  evaluation/
    mot_metrics.py                        # CLEAR MOT (MOTA/MOTP/IDF1/HOTA/ID-switches)
    mot_formatter.py                      # MOT17/KITTI format parsers
    lane_metrics.py                       # Lane IoU precision/recall/F1
    safety_metrics.py                     # Safety response latency evaluation
  adas/
    ttc_filter.py                         # TTC persistence smoothing
  utils/
    config.py                             # YAML config loader
    config_validator.py                   # Startup config validation (enums, ranges, deps)
    logger.py                             # Structured logging
    timing.py                             # Stage timing utilities

configs/
  system.yaml                             # Main system configuration
  safety.yaml                             # Safety thresholds, sensor health
  fusion.yaml                             # Kalman noise parameters
  perception.yaml                         # Perception tuning
  control.yaml                            # Controller parameters

scripts/
  evaluate.py                             # CLI evaluation runner
  evaluate_mot.py                         # MOT benchmark (IDF1, HOTA) CLI
  export_onnx.py                          # ONNX model export
  quantize_int8.py                        # INT8 quantization with accuracy checks
  build_tensorrt.py                       # TensorRT engine builder
  benchmark.py                            # Basic performance benchmark
  benchmark_onnx.py                       # ONNX vs PyTorch comparison
  benchmark_all.py                        # Comprehensive all-backends benchmark
  ci_latency_gate.py                      # CI latency budget enforcement
  run_scenarios.py                        # CARLA scenario runner
  generate_metrics_report.py              # Generates demo/METRICS.md
  latency_budget.py                       # Generates demo/latency_budget.md
  failure_scenarios.py                    # Generates demo/failure_scenarios.md
  generate_demo.py                        # Generates demo/ screenshots
  generate_safety_report.py               # Safety report generator
  summarize_run.py                        # Run metrics summary

tests/                                    # 27 test modules, 179 tests
  test_e2e_smoke.py                       # End-to-end pipeline smoke tests
  test_orchestrator_unit.py               # Orchestrator extracted method tests
  test_overlay_unit.py                    # Overlay rendering tests
  test_bev_unit.py                        # BEV renderer tests
  test_config_validator.py                # Config validation (16 tests)
  test_types_consolidation.py             # Canonical type verification
  test_kalman_properties.py               # Kalman filter property-based tests
  test_safety_properties.py               # Safety property-based tests (hypothesis)
  test_iso26262.py                        # ISO 26262 ASIL/plausibility/DTC tests
  test_occupancy.py                       # Occupancy grid tests
  test_visibility.py                      # Weather/visibility tests
  test_parallel.py                        # Parallel executor tests
  test_lane_ufld.py                       # UFLDv2 lane detector tests
  test_depth.py                           # Depth estimation tests
  test_bytetrack.py                       # ByteTrack tracker tests
  test_lidar.py                           # LIDAR processing + fusion tests
  test_radar.py                           # Radar processing + fusion tests
  test_recording.py                       # Data recording + replay tests
  test_saliency.py                        # Grad-CAM saliency tests
  test_sensor_health.py                   # Sensor health monitor tests
  test_interaction.py                     # Interaction model tests
  test_mot_metrics.py                     # MOT benchmark (IDF1/HOTA) tests
  test_safety.py                          # TTC/risk rule evaluation
  test_fusion.py                          # Perception-to-WorldModel fusion
  test_end_to_end.py                      # Import chain verification
  test_inputs.py                          # Input source tests
  test_perception_contracts.py            # Perception output schema validation

demo/                                     # Pre-generated presentation artifacts
  overlay_screenshot.png                  # Camera perception overlay
  bev_screenshot.png                      # Bird's-eye view
  sensor_health_screenshot.png            # Sensor health dashboard
  METRICS.md                              # Full metrics report
  latency_budget.md                       # Per-stage latency breakdown
  failure_scenarios.md                    # Controlled failure demonstrations

baselines/
  benchmark_baseline.json                 # Reference performance numbers

.github/workflows/
  ci.yml                                  # Lint + typecheck + test + coverage + latency gate
  model-regression.yml                    # Model performance regression checks

docker-compose.yml                        # Docker with GPU runtime + streaming port
pyproject.toml                            # Project metadata and dependencies
```

---

## Evaluation

### MOT Benchmark

```bash
# Full metrics (MOTA, MOTP, IDF1, HOTA, ID switches)
python scripts/evaluate_mot.py \
  --gt data/gt/mot_labels.csv \
  --pred results/tracks.csv \
  --format mot17

# KITTI tracking format
python scripts/evaluate_mot.py \
  --gt data/kitti/label_02/ \
  --pred results/kitti_tracks/ \
  --format kitti
```

### Lane and Safety Evaluation

```bash
python scripts/evaluate.py \
  --predictions results/run_*/tracks.json \
  --ground-truth data/gt/mot_labels.json \
  --mode mot    # or: lane, safety
```

### Latency Budget Verification

```bash
python scripts/latency_budget.py          # Generates per-stage breakdown
python scripts/ci_latency_gate.py         # CI-friendly pass/fail check
```

---

## CI Pipeline

`.github/workflows/ci.yml` runs on every push and PR:

1. **Lint** — `ruff check src/ tests/`
2. **Type check** — `mypy src/ --ignore-missing-imports`
3. **Unit tests** — `pytest tests/ -v` (179 tests)
4. **Coverage** — `pytest --cov=src --cov-report=xml`
5. **Latency gate** — `python scripts/ci_latency_gate.py` (50 ms budget)
6. **Safety tests** — `pytest tests/test_safety*.py tests/test_iso26262.py -v`

`.github/workflows/model-regression.yml` runs on model path changes:

1. **Benchmark** — Runs full detection + tracking benchmark
2. **Regression check** — Fails if mAP drops >5% or latency increases >20% vs baseline

---

## Limitations

- BEV uses proxy geometry (no camera calibration matrix required).
- Mono3D projection is approximate without stereo or LiDAR validation.
- CARLA integration requires a running CARLA server (0.9.x).
- Property-based tests require `hypothesis` (optional dev dependency).
- ByteTrack requires `supervision>=0.18` (optional dependency).
- Radar CSV input expects pre-formatted columns (frame_id, range_m, azimuth_deg, velocity_mps, rcs_dbsm).
- KITTI input requires the standard KITTI object detection directory layout.
