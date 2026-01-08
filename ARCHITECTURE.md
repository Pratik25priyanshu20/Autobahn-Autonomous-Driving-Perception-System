# APS++ Architecture

APS++ is organized as a modular perception pipeline with a single runtime orchestrator. The current implementation is optimized for video replay and uses proxy geometry for BEV and TTC estimation.

## Design Goals
- Low-latency, frame-by-frame perception with clear stage boundaries.
- Modular components that can be swapped or upgraded independently.
- Safety reasoning that is easy to inspect and log.
- Deployment path toward ONNX/TensorRT without rewrites.

## High-Level System Diagram
```
Input (Video)
  -> Orchestrator
     -> Detection (YOLOv8)
     -> Tracking (DeepSORT)
     -> Lane Detection
     -> Optional Segmentation
     -> Safety (FCW/LDW + TTC)
     -> World Model + Metrics
     -> Visualization (Overlay + BEV)
```

## Runtime Flow
1. Inputs deliver `FramePacket` instances to the orchestrator.
2. Perception modules run in sequence per frame (detection, tracking, lanes, segmentation).
3. Safety logic computes TTC proxies and safety states.
4. A `WorldModel` snapshot is emitted and rendered to overlays/BEV.
5. Metrics and safety events are logged for offline analysis.

## World Model
`src/fusion/world_model.py` defines the runtime world snapshot used by overlays and BEV:
- `frame_id`, `frame`, detections, tracks, trajectories.
- Lane state, drivable area, and FCW signals.
- Safety banner data (state/message/color).
- Per-stage timings and FPS metrics.

Separate type contracts exist in `src/utils/types.py` for unit tests and future fusion work. These should be consolidated over time.

## Perception Modules
- Detection: `src/perception/detection/yolo.py` (YOLOv8, CPU/MPS).
- Tracking: `src/perception/tracking/deepsort_tracker.py` (DeepSORT).
- Lanes: `src/perception/lanes/lane_detector.py` (Canny + Hough).
- Segmentation (optional): `src/perception/segmentation/deeplabv3_segmenter.py`.

## Safety Stack (LDW / FCW)
Safety is computed inside `src/runtime/orchestrator.py` and merged via `src/safety/safety_manager.py`.
- LDW: lane stability + offset persistence gating.
- FCW: TTC estimates from track history and proxy distances.
- TTC smoothing: `src/adas/ttc_filter.py` reduces flicker.
- Events: `results/run_*/safety_events.jsonl`.

## BEV Projection
`src/bev/bev_renderer.py` renders a top-down view with:
- Ego corridor and safety zones.
- Tracked objects projected using proxy x/y derived from image geometry.
This is a visualization tool and not a calibrated metric reconstruction.

## Runtime & Performance
- Orchestrator reports per-stage timings and FPS in `metrics.json`.
- `src/runtime/health_monitor.py` and `src/runtime/frame_sync.py` are ready for multi-sensor expansion.
- Performance targets and optimizations are tracked in `PERFORMANCE.md`.

## Deployment Strategy (ONNX / TensorRT-ready)
- `scripts/export_onnx.py`: export YOLOv8 to ONNX.
- `scripts/benchmark_onnx.py`: compare PyTorch vs ONNX on CPU.
- `scripts/build_tensorrt.py`: placeholder for TensorRT engine builds.

## Configuration
All tunables live under `configs/*.yaml` and are loaded by `src/utils/config.py`. Each module reads only its namespace, enabling mode switches and profile tuning.
