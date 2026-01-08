# APS++ Architecture

APS++ is a modular perception pipeline with a single runtime orchestrator. All modules are config-gated, allowing features to be toggled without code changes. The system supports video replay, live camera, and CARLA simulator inputs.

## Design Goals

- Low-latency, frame-by-frame perception with clear stage boundaries.
- Config-driven module selection — swap detectors, trackers, lane backends, and depth estimators via YAML.
- Unified type system (`src/types/`) shared across all modules.
- Safety reasoning that is inspectable, loggable, and separable from perception.
- Deployment path from PyTorch through ONNX to TensorRT without architecture changes.
- Parallel execution of independent stages when hardware allows.

## High-Level System Diagram

```
                         +-------------------+
                         |   Input Source     |
                         | Video/Webcam/CARLA |
                         +---------+---------+
                                   |
                                   v
                         +---------+---------+
                         |    Orchestrator    |
                         |  (config-driven)   |
                         +---------+---------+
                                   |
               +-------------------+-------------------+
               |           Parallel Stage Block         |
               |  (ThreadPoolExecutor, when enabled)    |
               +---+--------+--------+--------+--------+
                   |        |        |        |
                   v        v        v        v
              Detection   Lanes    Depth   Segmentation
            YOLOv8/v11   CH|UFLD  MiDAS|DA  DeepLabV3+
            ONNX|TRT
               +---+--------+--------+--------+
                   |
                   v
          +--------+--------+
          |    Tracking      |
          | DeepSORT|ByteTrack|
          +--------+---------+
                   |
          +--------v---------+
          | Confidence Calib  |
          | (temperature)     |
          +--------+---------+
                   |
          +--------v---------+
          |  Kalman Filter    |
          |  (per-track KF)   |
          | state=[x,y,vx,vy] |
          +--------+---------+
                   |
          +--------v---------+
          | Temporal Predict  |
          | 0.5s, 1.0s, 2.0s |
          +--------+---------+
                   |
          +--------v---------+
          | Mono3D Projection |
          | (2D + depth)      |
          +--------+---------+
                   |
     +-------------+-------------+
     |             |             |
     v             v             v
  Weather     Occupancy      BSD/RCTA
  Detector    Grid           (lateral)
     |             |             |
     +------+------+------+-----+
            |
     +------v------+
     |   Safety     |
     | FCW/LDW/BSD  |
     +------+------+
            |
     +------v------+
     |  Control     |
     | PurePursuit  |
     |    / MPC     |
     +------+------+
            |
     +------v------+
     | World Model  |
     +------+------+
            |
     +------v------+------+
     |             |       |
     v             v       v
  Overlay       BEV     Streaming
  (annotated)  (top-down) (WebSocket)
```

## Type System

All canonical types live in `src/types/`:

| Module | Types |
|--------|-------|
| `detection.py` | `Detection` — x1/y1/x2/y2, conf, class_id, class_name; properties: bbox, score, label, center |
| `track.py` | `Track` — bbox_xyxy, class_name, conf, age, velocity_px_per_frame; world-frame: x, y, vx, vy, ttc, risk |
| `ego.py` | `EgoState` — x, y, yaw, speed, speed_mps, yaw_rate, acceleration |
| `lanes.py` | `LaneGeometry`, `LaneState` |
| `safety.py` | `SafetyStateEnum`, `SafetyStatus`, `SafetyState`, `SafetyOutput` |
| `world_model.py` | `WorldModel` — canonical per-frame state; `RuntimeStats`, `DrivableArea` |
| `perception.py` | `FramePacket`, `PerceptionOutput` |
| `detection3d.py` | `Detection3D` — 3D position and extent |

Legacy import paths (`src/utils/types`, `src/fusion/types`, `src/fusion/ego_state`, etc.) are preserved as re-exports for backward compatibility.

## Runtime Flow

1. **Input** delivers `FramePacket` instances to the orchestrator.
2. **Parallel block** (optional) runs detection, lanes, depth, and segmentation concurrently via `ParallelStageExecutor`.
3. **Tracking** runs after detection completes (dependency).
4. **Confidence calibration** applies temperature scaling to detection scores.
5. **Kalman filter** (`KalmanTrackManager`) updates world-frame state per track.
6. **Temporal predictor** projects trajectories at 0.5 s, 1.0 s, 2.0 s horizons.
7. **Mono3D** combines 2D boxes + depth map into pseudo-3D positions.
8. **Weather detector** classifies visibility conditions and triggers degraded mode.
9. **Occupancy grid** projects depth + segmentation into a BEV grid.
10. **BSD/RCTA** checks lateral tracks for blind-spot hazards.
11. **Safety manager** merges FCW, LDW, and BSD into a unified safety state.
12. **Control** (Pure Pursuit or MPC) computes steering and throttle, gated by safety state.
13. **World model** snapshot is emitted — all downstream consumers (overlay, BEV, streaming, metrics) read from this single object.

## Perception Modules

| Module | File | Backends |
|--------|------|----------|
| Detection | `perception/detection/yolo.py` | YOLOv8n/s/m, YOLOv11n/s |
| Detection (ONNX) | `perception/detection/onnx_detector.py` | CPU, CUDA, TensorRT EP |
| Tracking | `perception/tracking/deepsort_tracker.py` | DeepSORT |
| Tracking | `perception/tracking/bytetrack_tracker.py` | ByteTrack (supervision) |
| Lanes | `perception/lanes/lane_detector.py` | Canny + Hough |
| Lanes | `perception/lanes/ufld_detector.py` | UFLDv2 (with Hough fallback) |
| Depth | `perception/depth/midas_depth.py` | MiDAS v2.1 small (torch.hub) |
| Depth | `perception/depth/depth_anything.py` | Depth Anything V2 (torch.hub) |
| Segmentation | `perception/segmentation/deeplabv3_segmenter.py` | DeepLabV3+ MobileNetV3 |
| Weather | `perception/weather/visibility_detector.py` | Image statistics classifier |
| Calibration | `perception/calibration.py` | Temperature scaling |
| Mono3D | `perception/detection/mono3d_stub.py` | 2D + depth projection |

## Safety Stack

```
SafetyManager.evaluate()
  ├── FCW: TTC-based states (NORMAL -> PRE -> WARNING -> CRITICAL)
  │   └── TTC smoothing via TTCFilter (src/adas/ttc_filter.py)
  ├── LDW: lane stability + offset persistence gating
  ├── BSD: lateral track monitoring for blind-spot hazards
  │   └── BlindSpotDetector (src/safety/bsd_rcta.py)
  └── Occupancy: BEV grid from depth + segmentation
      └── OccupancyGridBuilder (src/safety/occupancy_grid.py)
```

The safety manager escalates the unified state and produces a `SafetyOutput` with state, message, color, and details. Safety events are logged to `safety_events.jsonl`.

## Fusion

- **Kalman filtering** (`src/fusion/kalman_tracker.py`): Linear KF with state `[x, y, vx, vy]` per track. Configurable process/measurement noise via `configs/fusion.yaml`.
- **Temporal prediction** (`src/fusion/temporal_predictor.py`): Uses Kalman velocity estimates to project positions at multiple time horizons.
- **World model** (`src/types/world_model.py`): Single dataclass holding all per-frame state — detections, tracks, trajectories, lanes, depth_map, predictions, occupancy, FCW, safety, control, warnings, runtime stats.

## Control

Two controller implementations are available:

- **Pure Pursuit** (`src/control/pure_pursuit.py`): Computes steering angle from lane ego offset with a configurable lookahead distance. Throttle is safety-gated — reduces speed under WARNING, brakes under CRITICAL.
- **MPC** (`src/control/mpc.py`): Optimizes a trajectory over a finite horizon minimizing lateral error, heading error, and control effort. Uses a kinematic bicycle model.

Both controllers are disabled by default (`control.enabled: false` in `system.yaml`).

## Performance Architecture

- **Parallel execution**: `ParallelStageExecutor` runs independent perception stages (detection, lanes, depth, segmentation) concurrently using `ThreadPoolExecutor`.
- **Health monitor**: Tracks per-frame latency against a configurable watchdog threshold. After N consecutive misses, sets `degraded=True`, which triggers adaptive frame skipping.
- **Adaptive skip**: When degraded, the orchestrator doubles the tracking interval and optionally disables depth/segmentation to maintain core detection + tracking throughput.
- **ONNX/TensorRT**: `ONNXDetector` uses `onnxruntime` with CPU, CUDA, or TensorRT execution providers. `scripts/build_tensorrt.py` builds optimized TRT engines from ONNX models.

## Configuration

All tunables live under `configs/*.yaml` and are loaded by `src/utils/config.py`. Each module reads only its namespace. New features default to `enabled: false`.

Key configuration namespaces:

| Namespace | Controls |
|-----------|----------|
| `perception.detector_model` | YOLOv8n/s/m, YOLOv11n/s |
| `perception.runtime` | `pytorch`, `onnx`, `tensorrt` |
| `tracking.type` | `deepsort`, `bytetrack` |
| `lane.backend` | `canny_hough`, `ufldv2` |
| `depth.backend` | `midas`, `depth_anything` |
| `weather.enabled` | Visibility detection |
| `performance.parallel.enabled` | Concurrent stage execution |
| `performance.adaptive_skip` | Load-based frame skipping |
| `control.enabled` / `control.type` | Pure Pursuit or MPC |

## Streaming

`src/streaming/server.py` provides a FastAPI application with:

- **WebSocket endpoint** (`/ws`): Sends JPEG-encoded frames and world-model JSON on each frame.
- **HTTP endpoint** (`/health`): Health check returning current FPS and frame count.

Launched in a daemon thread when `--stream` is passed to `app.py`.

## CI Pipeline

`.github/workflows/ci.yml` runs on push/PR:

1. **Lint** — `ruff check src/ tests/`
2. **Type check** — `mypy src/ --ignore-missing-imports`
3. **Unit tests** — `pytest tests/ -v`
4. **Safety tests** — `pytest tests/test_safety*.py -v`
