# APS++ Architecture

APS++ is a modular, multi-sensor perception pipeline with a single runtime orchestrator. All modules are config-gated, allowing features to be toggled without code changes. The system supports video replay, live camera, CARLA simulator, KITTI dataset, and recorded session replay inputs.

## Design Goals

- Low-latency, frame-by-frame perception with clear stage boundaries.
- Config-driven module selection — swap detectors, trackers, lane backends, depth estimators, and sensor fusion modes via YAML.
- Unified type system (`src/types/`) shared across all modules with concrete types via `TYPE_CHECKING`.
- Safety reasoning that is inspectable, loggable, and separable from perception (ISO 26262 alignment).
- Multi-sensor fusion path: camera + LIDAR + radar with independent processing and joint enrichment.
- Deployment path from PyTorch through ONNX to TensorRT without architecture changes.
- Parallel execution of independent stages when hardware allows.
- Startup config validation to catch misconfigurations before the pipeline runs.

## High-Level System Diagram

```
                    +-------------------------------+
                    |        Input Sources           |
                    | Video / Webcam / CARLA / KITTI |
                    |        / Replay (.apsrec)      |
                    +---------------+---------------+
                                    |
                         +----------v----------+
                         |     Orchestrator      |
                         |  (15 stage methods)   |
                         +----------+----------+
                                    |
                    +---------------+---------------+
                    |     Parallel Stage Block       |
                    | (ThreadPoolExecutor, optional) |
                    +--+-------+-------+-------+---+
                       |       |       |       |
                       v       v       v       v
                   Detection Lanes   Depth  Segmentation
                   YOLO v8   CH|    MiDAS|  DeepLabV3+
                   /v11     UFLD    DA
                   ONNX|TRT
                    +--+-------+-------+-------+---+
                                    |
                    +---------------v---------------+
                    |     Sensor Health Assessment    |
                    | Camera | LIDAR | Radar scoring  |
                    +---------------+---------------+
                                    |
                    +---------------v---------------+
                    |    Saliency / Grad-CAM         |
                    |  Activation-based heatmaps      |
                    +---------------+---------------+
                                    |
                    +---------------v---------------+
                    |          Tracking               |
                    |    DeepSORT | ByteTrack          |
                    |  + Confidence Calibration        |
                    +---------------+---------------+
                                    |
                    +---------------v---------------+
                    |       Kalman Filter             |
                    |  Per-track KF: [x, y, vx, vy]   |
                    +---------------+---------------+
                                    |
                    +---------------v---------------+
                    |     Temporal Prediction          |
                    |  0.5s, 1.0s, 2.0s + Top-K       |
                    +---------------+---------------+
                                    |
                    +---------------v---------------+
                    |     Interaction Model            |
                    | Gap / Yield / Following / Cut-in |
                    +---------------+---------------+
                                    |
               +--------------------+--------------------+
               |                                         |
    +----------v----------+                   +----------v----------+
    |   LIDAR Pipeline     |                   |   Radar Pipeline     |
    | Voxelize -> RANSAC   |                   | Ghost filter ->      |
    | -> Cluster -> BEV    |                   | Cartesian -> Cluster |
    +----------+----------+                   +----------+----------+
               |                                         |
    +----------v----------+                   +----------v----------+
    | LIDAR-Camera Fusion  |                   | Radar-Camera Fusion  |
    | 3D cluster -> depth  |                   | Project -> velocity  |
    +----------+----------+                   +----------+----------+
               +--------------------+--------------------+
                                    |
                    +---------------v---------------+
                    |          Safety Stack           |
                    | FCW / LDW / BSD / RCTA          |
                    | Occupancy / Plausibility         |
                    | ASIL / DTC / Redundant Detector  |
                    +---------------+---------------+
                                    |
                    +---------------v---------------+
                    |           Control               |
                    |    Pure Pursuit / MPC            |
                    |      (safety-gated)              |
                    +---------------+---------------+
                                    |
                    +---------------v---------------+
                    |         World Model              |
                    |   Canonical per-frame state      |
                    |     + EMA smoothing              |
                    +--+--------+--------+--------+---+
                       |        |        |        |
                       v        v        v        v
                    Overlay    BEV    Streaming  Recording
                  (annotated) (top-   (WebSocket) (.apsrec)
                              down)
```

## Type System

All canonical types live in `src/types/`. Types use `TYPE_CHECKING` imports for concrete type annotations without circular dependency issues.

| Module | Types |
|--------|-------|
| `detection.py` | `Detection` — x1/y1/x2/y2, conf, class_id, class_name; properties: bbox, score, label, center |
| `track.py` | `Track` — bbox_xyxy, class_name, conf, age, velocity_px_per_frame; world-frame: x, y, vx, vy, ttc, risk |
| `ego.py` | `EgoState` — x, y, yaw, speed, speed_mps, yaw_rate, acceleration |
| `lanes.py` | `LaneGeometry`, `LaneState` |
| `safety.py` | `SafetyStateEnum`, `SafetyStatus`, `SafetyState`, `SafetyOutput` |
| `world_model.py` | `WorldModel` — canonical per-frame state with concrete typed fields; `RuntimeStats`, `DrivableArea` |
| `perception.py` | `FramePacket` (with `PointCloud`, `RadarFrame`), `PerceptionOutput` |
| `detection3d.py` | `Detection3D` — 3D position and extent |
| `pointcloud.py` | `PointCloud` (N x 4 array), `BEVGrid` (2D occupancy) |
| `radar.py` | `RadarDetection` (range, azimuth, velocity, RCS, x, y), `RadarFrame` |

### WorldModel Fields

The `WorldModel` dataclass is the central per-frame state object. Key typed fields:

| Field | Type | Source |
|-------|------|--------|
| `detections` | `list[Any]` | Detection stage |
| `tracks` | `list[Any]` | Tracking stage |
| `predictions` | `dict[int, list[PredictionPoint]]` | Temporal predictor |
| `predictions_topk` | `dict[int, list[PredictionPoint]]` | Top-K hypotheses |
| `occupancy` | `OccupancyGrid \| None` | Safety occupancy |
| `fused_detections` | `list[FusedDetection]` | LIDAR-camera fusion |
| `point_cloud` | `PointCloud \| None` | LIDAR input |
| `bev_grid` | `BEVGrid \| None` | BEV encoder |
| `radar_detections` | `list[RadarDetection]` | Radar processor |
| `interactions` | `list[InteractionEvent]` | Interaction model |
| `saliency_map` | `np.ndarray \| None` | Grad-CAM |
| `sensor_health` | `dict[str, float]` | Health monitor |
| `depth_map` | `np.ndarray \| None` | Depth estimator |

## Runtime Flow

1. **Config validation** at startup — enums, ranges, and dependencies checked before pipeline runs.
2. **Input** delivers `FramePacket` instances (camera frame + optional LIDAR point cloud + optional radar frame).
3. **Parallel block** (optional) runs detection, lanes, depth, and segmentation concurrently via `ParallelStageExecutor`.
4. **Sensor health** assesses camera brightness/blur, LIDAR point density, and radar detection consistency.
5. **Saliency** computes activation-based heatmaps for explainability (config-gated).
6. **Tracking** runs after detection completes (dependency). Confidence calibration applies temperature scaling.
7. **Kalman filter** (`KalmanTrackManager`) updates world-frame state per track.
8. **Temporal predictor** projects trajectories at 0.5 s, 1.0 s, 2.0 s horizons with top-K hypotheses.
9. **Interaction model** evaluates gap acceptance, yield heuristics, following distance, and cut-in prediction.
10. **LIDAR pipeline** (if available) — voxelization, RANSAC ground removal, 3D clustering, BEV encoding.
11. **LIDAR-camera fusion** projects 3D clusters onto 2D tracks for depth enrichment.
12. **Radar pipeline** (if available) — ghost filtering, polar-to-cartesian conversion, spatial clustering.
13. **Radar-camera fusion** projects radar detections to image space, enriches tracks with velocity and range.
14. **Safety manager** merges FCW, LDW, BSD, occupancy, and plausibility into a unified safety state.
15. **Post-safety** plausibility check validates detection/track/TTC coherence.
16. **Control** (Pure Pursuit or MPC) computes steering and throttle, gated by safety state.
17. **World model** snapshot is emitted — all downstream consumers read from this single object.
18. **Recording** (optional) serializes the world model to a gzipped msgpack stream.

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
| LIDAR | `perception/lidar/point_cloud_processor.py` | Voxelization, RANSAC, clustering |
| LIDAR BEV | `perception/lidar/bev_encoder.py` | Point cloud -> BEV grid |
| Radar | `perception/radar/radar_processor.py` | Ghost filter, cartesian, clustering |
| Saliency | `perception/explainability/grad_cam.py` | Activation-based heatmaps |
| Calibration | `perception/calibration.py` | Temperature scaling |
| Mono3D | `perception/detection/mono3d_stub.py` | 2D + depth projection |

## Safety Stack

```
SafetyManager.evaluate()
  |
  +-- FCW: TTC-based states (NORMAL -> PRE -> WARNING -> CRITICAL)
  |     +-- TTC smoothing via TTCFilter (src/adas/ttc_filter.py)
  |
  +-- LDW: lane stability + offset persistence gating
  |
  +-- BSD: lateral track monitoring for blind-spot hazards
  |     +-- BlindSpotDetector (src/safety/bsd_rcta.py)
  |
  +-- Occupancy: BEV grid from depth + segmentation
  |     +-- OccupancyGridBuilder (src/safety/occupancy_grid.py)
  |
  +-- Plausibility: cross-module sanity checks
  |     +-- PlausibilityChecker (src/safety/plausibility_checker.py)
  |     +-- Detection count vs track count coherence
  |     +-- TTC bounds validation
  |
  +-- ASIL Classification: per-hazard risk level
  |     +-- ASILClassifier (src/safety/asil_classifier.py)
  |     +-- ASIL-A through ASIL-D assignment
  |
  +-- Redundant Detector: lightweight fallback
  |     +-- RedundantDetector (src/safety/redundant_detector.py)
  |     +-- Cross-validation with primary detector
  |
  +-- DTC Logger: diagnostic trouble codes
  |     +-- DTCLogger (src/safety/dtc_logger.py)
  |     +-- ISO 14229-style logging
  |
  +-- Sensor Health: per-sensor degradation scoring
        +-- SensorHealthMonitor (src/safety/sensor_health.py)
        +-- Camera: brightness, blur (Laplacian variance), occlusion
        +-- LIDAR: point count ratio vs expected
        +-- Radar: detection consistency (moving average)
```

The safety manager escalates the unified state and produces a `SafetyOutput` with state, message, color, and details. Safety events are logged to `safety_events.jsonl`, DTCs to `dtc_log.jsonl`.

## Sensor Fusion

### Camera (Primary)
- **Detection**: YOLOv8/v11 produces 2D bounding boxes with class and confidence.
- **Tracking**: DeepSORT or ByteTrack maintains temporal identity across frames.
- **Depth**: MiDAS or Depth Anything V2 provides per-pixel depth estimates.

### LIDAR
- **Processing** (`src/perception/lidar/point_cloud_processor.py`): Voxelization, RANSAC ground plane removal, DBSCAN-style clustering.
- **BEV Encoding** (`src/perception/lidar/bev_encoder.py`): Projects clusters into a 2D BEV grid.
- **Fusion** (`src/fusion/lidar_camera_fusion.py`): Projects 3D LIDAR clusters onto 2D camera tracks, enriching them with metric depth.

### Radar
- **Processing** (`src/perception/radar/radar_processor.py`): Ghost filtering (low-RCS, multi-bounce), polar-to-cartesian conversion, spatial clustering.
- **Fusion** (`src/fusion/radar_camera_fusion.py`): Projects radar detections to image coordinates via camera matrix, matches to tracks, enriches with Doppler velocity and range.

### Temporal Fusion
- **Kalman filtering** (`src/fusion/kalman_tracker.py`): Linear KF with state `[x, y, vx, vy]` per track. Configurable process/measurement noise.
- **Temporal prediction** (`src/fusion/temporal_predictor.py`): Uses Kalman velocity estimates to project positions at multiple time horizons. Supports top-K trajectory hypotheses.
- **World model smoothing** (`src/types/world_model.py`): EMA smoothing on lane geometry and track confidence scores across frames.

## Prediction

- **Interaction Model** (`src/prediction/interaction_model.py`): Rule-based behavioral prediction with four heuristics:
  - **Gap acceptance**: Checks if adjacent lane gap exceeds minimum threshold.
  - **Yield heuristic**: German right-before-left rule for intersections.
  - **Following distance**: 2-second rule validation against lead track.
  - **Cut-in prediction**: Detects lateral velocity toward ego lane, computes time-to-cut-in.
  - Each heuristic produces an `InteractionEvent` with risk level and time-to-event.

## Control

Two controller implementations are available:

- **Pure Pursuit** (`src/control/pure_pursuit.py`): Computes steering angle from lane ego offset with a configurable lookahead distance. Throttle is safety-gated.
- **MPC** (`src/control/mpc.py`): Optimizes a trajectory over a finite horizon minimizing lateral error, heading error, and control effort. Uses a kinematic bicycle model.

Both controllers are disabled by default and safety-gated at runtime.

## Recording & Replay

- **Data Recorder** (`src/recording/data_recorder.py`): Serializes world model state to gzipped msgpack streams (`.apsrec`). Configurable recording interval and max file size.
- **Replay Input** (`src/recording/replay_input.py`): Reads `.apsrec` files with frame-level seeking and variable playback speed.
- Round-trip verified: record N frames -> replay -> verify frame count matches.

## Performance Architecture

- **Parallel execution**: `ParallelStageExecutor` runs detection, lanes, depth, and segmentation concurrently using `ThreadPoolExecutor`.
- **Health monitor**: Tracks per-frame latency against a configurable watchdog threshold. After N consecutive misses, sets `degraded=True`.
- **Adaptive skip**: When degraded, doubles tracking interval and optionally disables depth/segmentation.
- **ONNX/TensorRT**: `ONNXDetector` uses `onnxruntime` with CPU, CUDA, or TensorRT execution providers. INT8 quantization with accuracy guardrails.

## Configuration

All tunables live under `configs/*.yaml` and are loaded by `src/utils/config.py`. Config is validated at startup by `src/utils/config_validator.py`. Each module reads only its namespace. New features default to `enabled: false`.

| Namespace | Controls |
|-----------|----------|
| `perception.detector_model` | YOLOv8n/s/m, YOLOv11n/s |
| `perception.runtime` | `pytorch`, `onnx`, `tensorrt` |
| `tracking.type` | `deepsort`, `bytetrack` |
| `lane.backend` | `canny_hough`, `ufldv2` |
| `depth.backend` | `midas`, `depth_anything` |
| `weather.enabled` | Visibility detection |
| `radar.enabled` | Radar processing |
| `radar_fusion.enabled` | Radar-camera fusion |
| `sensor_health.enabled` | Camera/LIDAR/radar health scoring |
| `explainability.enabled` | Saliency / Grad-CAM |
| `interaction.enabled` | Behavioral interaction model |
| `recording.enabled` | Data recording to `.apsrec` |
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
3. **Unit tests** — `pytest tests/ -v` (179 tests across 27 modules)
4. **Coverage** — `pytest --cov=src --cov-report=xml`
5. **Latency gate** — `python scripts/ci_latency_gate.py` (50 ms budget)
6. **Safety tests** — `pytest tests/test_safety*.py tests/test_iso26262.py -v`

`.github/workflows/model-regression.yml` runs on model path changes:

1. **Benchmark** — Runs detection + tracking on synthetic data
2. **Regression check** — Fails if mAP drops >5% or latency increases >20% vs `baselines/benchmark_baseline.json`
