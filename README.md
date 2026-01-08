# APS++ — Autonomous Perception Stack

APS++ is a real-time autonomous driving perception stack built for highway scenarios. It processes forward-facing video (or live camera / CARLA simulator feeds) and outputs detections, tracks, lane cues, depth maps, safety warnings, BEV visualization, and runtime metrics.

## Key Features

- **Object detection** — YOLOv8/v11 with config-driven model switching and ONNX/TensorRT inference paths.
- **Multi-object tracking** — DeepSORT or ByteTrack (config-selectable), with per-object Kalman filtering in world-frame coordinates.
- **Lane detection** — Canny+Hough baseline or UFLDv2 neural backend, with stability gating and EMA smoothing.
- **Monocular depth estimation** — MiDAS or Depth Anything V2 backends for per-frame dense depth maps.
- **Temporal prediction** — Kalman-based trajectory prediction at 0.5 s, 1.0 s, and 2.0 s horizons.
- **Occupancy grid** — BEV grid projected from depth + semantic segmentation.
- **Safety system** — Forward Collision Warning (FCW), Lane Departure Warning (LDW), Blind Spot Detection (BSD), Rear Cross-Traffic Alert (RCTA).
- **Weather/visibility detection** — Image-statistics classifier (clear / fog / dark / glare) with automatic degraded-mode triggering.
- **Adaptive performance** — Parallel pipeline stages, health-monitor watchdog, adaptive frame skipping under load.
- **Control modules** — Pure Pursuit steering and MPC trajectory optimization (safety-gated).
- **BEV visualization** — Top-down ego corridor, safety zones, tracked objects with velocity arrows and prediction circles.
- **Streaming** — FastAPI + WebSocket server for real-time frame and world-model JSON streaming.
- **Evaluation framework** — CLEAR MOT (MOTA/MOTP/ID-switches), lane IoU, and safety response latency metrics.
- **CI** — GitHub Actions with linting (ruff), type checking (mypy), and unit tests (pytest).

## System Pipeline

```
Input (Video / Webcam / CARLA)
  -> Orchestrator
     -> [Parallel where enabled]
        -> Detection (YOLOv8/v11 | ONNX | TensorRT)
        -> Lane Detection (Canny+Hough | UFLDv2)
        -> Segmentation (DeepLabV3+)
        -> Depth Estimation (MiDAS | Depth Anything V2)
     -> Tracking (DeepSORT | ByteTrack)
     -> Confidence Calibration (temperature scaling)
     -> Kalman Filter (world-frame x/y/vx/vy)
     -> Temporal Prediction (0.5s, 1.0s, 2.0s)
     -> Mono3D Projection (2D + depth -> pseudo-3D)
     -> Weather / Visibility Detection
     -> Occupancy Grid (depth + segmentation -> BEV)
     -> Safety (FCW / LDW / BSD / RCTA)
     -> Control (Pure Pursuit / MPC)
     -> World Model + Metrics
     -> Visualization (Overlay + BEV)
     -> Streaming (WebSocket)
```

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

Or using requirements.txt (legacy):

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

# With real-time streaming dashboard
python src/app.py --config configs/system.yaml --input data/samples/test_drive.mp4 --stream --stream-port 8765
```

### Test

```bash
pytest tests/ -v
```

### Docker

```bash
docker compose up --build
```

The container exposes port `8765` for WebSocket streaming when enabled.

## Configuration

All features are controlled through YAML config files under `configs/`. New features default to `enabled: false` to preserve backward compatibility.

| File | Scope |
|------|-------|
| `system.yaml` | Detection model, runtime backend, tracking type, lane backend, depth, weather, parallel execution, adaptive skip, CARLA, control |
| `safety.yaml` | FCW/LDW thresholds, occupancy grid, BSD/RCTA |
| `fusion.yaml` | Kalman process/measurement noise, smoothing |
| `perception.yaml` | Resolution, detection confidence thresholds |
| `control.yaml` | Pure Pursuit / MPC parameters |

### Enabling new features

Edit `configs/system.yaml`:

```yaml
# Switch to ByteTrack
tracking:
  type: "bytetrack"

# Enable depth estimation
depth:
  enabled: true
  backend: "midas"        # or "depth_anything"

# Enable parallel pipeline
performance:
  parallel:
    enabled: true
    max_workers: 4

# Enable weather detection
weather:
  enabled: true
```

## Outputs

Each run produces a timestamped directory under `results/`:

| File | Contents |
|------|----------|
| `output.mp4` | Annotated video with detections, tracks, lanes, FCW/LDW, BSD indicators |
| `bev.mp4` | Bird's-eye view with ego corridor, safety zones, prediction circles |
| `metrics.json` | FPS, per-stage timing, git commit hash |
| `safety_events.jsonl` | Safety state transitions with timestamps |

## Project Structure

```
src/
  app.py                          # CLI entrypoint (--input, --live, --carla, --stream)
  types/                          # Canonical type definitions
    detection.py                  #   Detection, bbox/score/label properties
    track.py                      #   Track with world-frame attrs (x, y, vx, vy, ttc, risk)
    ego.py                        #   EgoState
    lanes.py                      #   LaneGeometry, LaneState
    safety.py                     #   SafetyStateEnum, SafetyStatus, SafetyOutput
    world_model.py                #   WorldModel (canonical per-frame state)
    perception.py                 #   FramePacket, PerceptionOutput
    detection3d.py                #   Detection3D (pseudo-3D boxes)
  runtime/
    orchestrator.py               # Central pipeline with config-driven module gating
    health_monitor.py             # Latency watchdog with consecutive-miss degraded mode
    frame_sync.py                 # Multi-sensor frame synchronization
    parallel_executor.py          # ThreadPoolExecutor for concurrent pipeline stages
  inputs/
    base_input.py                 # Input ABC
    video_input.py                # File-based video input
    webcam_input.py               # Live camera input
    carla_input.py                # CARLA simulator integration
  perception/
    detection/
      yolo.py                     # YOLOv8/v11 (auto device: CUDA/MPS/CPU)
      onnx_detector.py            # ONNX Runtime with CPU/CUDA/TensorRT providers
      mono3d_stub.py              # 2D + depth -> pseudo-3D projection
    tracking/
      deepsort_tracker.py         # DeepSORT multi-object tracker
      bytetrack_tracker.py        # ByteTrack via supervision library
      track.py                    # Track type re-export
    lanes/
      base_lane_detector.py       # Lane detector ABC
      lane_detector.py            # Canny + Hough baseline
      ufld_detector.py            # UFLDv2 neural lane detector
    depth/
      base_depth.py               # Depth estimator ABC
      midas_depth.py              # MiDAS via torch.hub
      depth_anything.py           # Depth Anything V2 via torch.hub
    segmentation/
      base_segmenter.py           # Segmenter ABC
      deeplabv3_segmenter.py      # DeepLabV3+ (MobileNet)
      postprocess.py              # Drivable area extraction
    weather/
      visibility_detector.py      # Clear/fog/dark/glare classifier
    calibration.py                # Confidence temperature scaling
  fusion/
    world_model.py                # WorldModel re-exports + build_world_model factory
    kalman_tracker.py             # Per-object Kalman filter (state=[x,y,vx,vy])
    temporal_predictor.py         # Trajectory prediction at 0.5s/1.0s/2.0s
    fusion_engine.py              # Perception -> WorldModel fusion
    projector.py                  # Pixel -> world-frame projection
  safety/
    safety_manager.py             # Unified safety state manager (FCW+LDW+BSD)
    rules.py                      # Rule-based safety evaluation
    fcw.py                        # FCW state machine
    ttc.py                        # Time-to-collision computation
    risk.py                       # Risk score calculation
    occupancy_grid.py             # BEV occupancy from depth + segmentation
    bsd_rcta.py                   # Blind Spot Detection + Rear Cross-Traffic Alert
    safety_logger.py              # Safety event logging
    distance.py                   # Distance estimation
  control/
    base_controller.py            # Controller ABC
    pure_pursuit.py               # Steering from lane offset + safety-gated throttle
    mpc.py                        # Model predictive control
  bev/
    bev_renderer.py               # Top-down BEV with safety zones and predictions
  visualization/
    overlay.py                    # HUD, detections, tracks, FCW, LDW, BSD indicators
  streaming/
    server.py                     # FastAPI + WebSocket JPEG/JSON streaming
  evaluation/
    mot_metrics.py                # CLEAR MOT (MOTA/MOTP/ID-switches)
    lane_metrics.py               # Lane IoU precision/recall/F1
    safety_metrics.py             # Safety response latency evaluation
  adas/
    ttc_filter.py                 # TTC persistence smoothing
  utils/
    config.py                     # YAML config loader
    types.py                      # Backward-compatible re-exports
    logger.py, timing.py          # Utilities

configs/
  system.yaml                     # Main system configuration
  safety.yaml                     # Safety thresholds and features
  fusion.yaml                     # Kalman noise parameters
  perception.yaml                 # Perception tuning
  control.yaml                    # Controller parameters

scripts/
  evaluate.py                     # CLI evaluation runner
  build_tensorrt.py               # TensorRT engine builder
  export_onnx.py                  # ONNX model export
  benchmark.py, benchmark_onnx.py # Performance benchmarking
  summarize_run.py                # Run metrics summary
  test_bev.py                     # BEV smoke test

tests/                            # 14 test modules, 40 tests
  test_types_consolidation.py     # Canonical type verification
  test_kalman_properties.py       # Kalman filter + property-based tests
  test_safety_properties.py       # Safety property-based tests (hypothesis)
  test_occupancy.py               # Occupancy grid tests
  test_visibility.py              # Weather/visibility tests
  test_parallel.py                # Parallel executor tests
  test_lane_ufld.py               # UFLDv2 lane detector tests
  test_depth.py                   # Depth estimation tests
  test_bytetrack.py               # ByteTrack tracker tests
  ...

.github/workflows/ci.yml         # Lint + typecheck + test + safety CI
pyproject.toml                    # Project metadata and dependencies
docker-compose.yml                # Docker with GPU runtime + streaming port
```

## Evaluation

Run offline evaluation against ground truth annotations:

```bash
python scripts/evaluate.py \
  --predictions results/run_*/tracks.json \
  --ground-truth data/gt/mot_labels.json \
  --mode mot
```

Available modes: `mot` (MOTA/MOTP), `lane` (IoU/F1), `safety` (response latency).

## Limitations

- BEV uses proxy geometry (no camera calibration).
- Mono3D projection is approximate without stereo or LiDAR validation.
- CARLA integration requires a running CARLA server (0.9.x).
- Property-based tests require `hypothesis` (optional dev dependency).
- ByteTrack requires `supervision>=0.18` (optional dependency).
