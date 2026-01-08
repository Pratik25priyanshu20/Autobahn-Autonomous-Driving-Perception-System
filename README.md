# APS++ — Autonomous Perception Stack

APS++ is a real-time autonomous perception stack aimed at highway-style scenarios. It processes forward-facing video, performs detection, tracking, lane estimation, BEV projection, and safety reasoning, and produces annotated video plus run metrics.

## Overview
APS++ is built around a modular pipeline that can be extended with additional perception heads or simulated inputs. The current implementation focuses on video replay with YOLOv8 detection, DeepSORT tracking, lane cues, optional segmentation, and FCW/LDW safety logic.

## Key Features
- Real-time object detection (YOLOv8) with stable tracking IDs (DeepSORT).
- Lane detection with lane-departure warnings (LDW) and stability gating.
- Forward collision warning (FCW) with TTC-based states (NORMAL, PRE, WARNING, CRITICAL).
- Bird's-Eye View (BEV) visualization with ego corridor and safety zones.
- Per-frame metrics, latency breakdowns, and safety event logging.
- ONNX export utilities and CPU benchmarking scripts.

## System Pipeline
```
Video Input
  -> Detection (YOLOv8)
  -> Tracking (DeepSORT)
  -> Lane Detection
  -> Optional Segmentation
  -> Safety (FCW/LDW + TTC)
  -> World Model + Metrics
  -> BEV + Annotated Video
```

## Bird's-Eye View (BEV)
`src/bev/bev_renderer.py` renders a top-down visualization of tracked objects and the ego corridor. It uses a proxy metric mapping from image geometry and is intended for debugging and demonstrations rather than calibrated perception.

## Safety Intelligence
Safety is computed per frame in the runtime orchestrator and merged in `src/safety/safety_manager.py`:
- FCW uses TTC estimates with persistence smoothing (`src/adas/ttc_filter.py`).
- LDW flags lane departure with confidence and stability gating.
- Safety events are logged to `results/run_*/safety_events.jsonl`.

## Performance
Per-stage timings and FPS are recorded in `results/run_*/metrics.json`. Use `scripts/summarize_run.py` for a quick report. Targets and tuning notes live in `PERFORMANCE.md`.

## ONNX Deployment
Export YOLOv8 to ONNX and run a simple CPU benchmark:
```bash
python scripts/export_onnx.py --weights yolov8n.pt --out models/yolo_v8.onnx
python scripts/benchmark_onnx.py
```
TensorRT build is stubbed in `scripts/build_tensorrt.py` for future acceleration.

## How to Run
1. Install dependencies:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Run the perception pipeline on the sample clip:
   ```bash
   python src/app.py --config configs/system.yaml --input data/samples/test_drive.mp4
   ```
3. Summarize a run:
   ```bash
   python scripts/summarize_run.py results/run_YYYYMMDD_HHMMSS
   ```
4. Optional BEV smoke test (writes `results/test_bev.png`):
   ```bash
   python scripts/test_bev.py
   ```
5. Docker (GPU runtime configured in `docker-compose.yml`):
   ```bash
   docker compose up --build
   ```

## Project Structure
- `src/app.py`: main entrypoint for video replay.
- `src/runtime/`: orchestrator, frame sync, health monitor.
- `src/inputs/`: video and CARLA input adapters.
- `src/perception/`: detection, tracking, lanes, segmentation stubs.
- `src/safety/`: FCW/LDW logic, TTC utilities, safety manager.
- `src/fusion/`: world model helpers (in-progress).
- `src/bev/`: BEV renderer.
- `src/visualization/`: overlays and HUD.
- `configs/`: runtime and module configuration.
- `scripts/`: benchmarking, export, and testing utilities.

## Limitations & Future Work
- CARLA input is a stub; live sensor integration is not implemented.
- Depth and full fusion modules are placeholders.
- BEV uses proxy geometry (no camera calibration).
- ONNX/TensorRT path currently covers detection only.
- World model types are duplicated across modules and should be consolidated.
