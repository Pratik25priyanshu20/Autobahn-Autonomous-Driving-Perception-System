# Performance

Targets focus on low latency, stable throughput, and bounded jitter to support real-time perception and control across camera, LIDAR, and radar sensor inputs.

## Latency Budgets

| Stage | Target | Notes |
|-------|--------|-------|
| End-to-end | ≤ 50 ms | Camera to fused world model (stretch goal 30 ms on Orin) |
| Detection | ≤ 30 ms | YOLOv8n on GPU; ~60 ms on CPU |
| Tracking | ≤ 10 ms | ByteTrack preferred for speed |
| Lane detection | ≤ 15 ms | Canny+Hough; UFLDv2 adds ~10 ms |
| Depth estimation | ≤ 25 ms | MiDAS small on GPU |
| Segmentation | ≤ 20 ms | DeepLabV3+ MobileNet |
| LIDAR processing | ≤ 8 ms | Voxelize + RANSAC + clustering |
| LIDAR-camera fusion | ≤ 3 ms | 3D cluster projection |
| Radar processing | ≤ 3 ms | Ghost filter + cartesian + cluster |
| Radar-camera fusion | ≤ 2 ms | Image projection + matching |
| Sensor health | ≤ 2 ms | Brightness/blur/point-count checks |
| Saliency (Grad-CAM) | ≤ 5 ms | Activation extraction (no backprop) |
| Kalman + prediction | ≤ 2 ms | Pure NumPy, no GPU |
| Interaction model | ≤ 1 ms | Rule-based, no neural inference |
| Safety evaluation | ≤ 5 ms | FCW + LDW + BSD + plausibility |
| Fusion + world model | ≤ 5 ms | Dataclass assembly + EMA smoothing |
| Recording overhead | ≤ 2 ms | Gzipped msgpack serialization |
| Frame sync buffering | ≤ 2 frames | Jitter < 10 ms |

### Measured Latency (Synthetic Benchmark)

Run `python scripts/latency_budget.py` to regenerate. The output is saved to `demo/latency_budget.md`.

Typical results on CPU (mocked detector/tracker, lane detection enabled):

| Stage | Mean | P95 | P99 |
|-------|------|-----|-----|
| Preprocess | < 0.1 ms | < 0.1 ms | < 0.1 ms |
| Lane detection | ~0.4 ms | ~0.5 ms | ~0.6 ms |
| Tracking | < 0.1 ms | < 0.1 ms | < 0.1 ms |
| Safety | < 0.1 ms | < 0.1 ms | < 0.1 ms |
| **Total** | **~0.5 ms** | **~0.6 ms** | **~0.7 ms** |

The synthetic benchmark uses mocked ML models. Real-world latency depends on GPU and model size.

## Parallel Pipeline

When `performance.parallel.enabled: true`, the orchestrator dispatches independent perception stages concurrently via `ParallelStageExecutor` (`src/runtime/parallel_executor.py`):

```
Detection ─┐
Lanes     ─┤ ThreadPoolExecutor (max_workers from config)
Depth     ─┤
Segment   ─┘
              -> Tracking (serial, depends on detection)
              -> LIDAR pipeline (serial, depends on point cloud)
              -> Radar pipeline (serial, depends on radar frame)
```

This reduces wall-clock time for the perception block from the sum of all stages to approximately the slowest single stage.

Configure in `configs/system.yaml`:

```yaml
performance:
  parallel:
    enabled: true
    max_workers: 4
```

## Health Monitor & Degraded Mode

`src/runtime/health_monitor.py` tracks per-frame latency against a configurable watchdog threshold:

- **watchdog_ms**: Maximum allowed frame processing time (default: 100 ms).
- **degraded_after_misses**: Number of consecutive watchdog misses before entering degraded mode (default: 3).

When degraded mode activates:

1. Tracking interval doubles (process every other frame).
2. Depth estimation and segmentation are temporarily disabled.
3. Core detection + tracking + safety continue at reduced load.
4. The system recovers automatically when latency drops below threshold.

```yaml
performance:
  watchdog_ms: 100
  degraded_after_misses: 3
  adaptive_skip: true
```

## Sensor Health Monitoring

`src/safety/sensor_health.py` provides real-time sensor quality assessment:

| Sensor | Metrics | Healthy Score |
|--------|---------|---------------|
| Camera | Brightness (mean pixel), blur (Laplacian variance), occlusion (edge density) | > 0.7 |
| LIDAR | Point count ratio vs expected, intensity distribution | > 0.7 |
| Radar | Detection consistency (moving average of detection count) | > 0.7 |

When overall health drops below threshold, the system logs a degradation warning and can trigger degraded mode.

Run `python scripts/failure_scenarios.py` to see controlled degradation demos:
- Camera blackout: score drops from 1.0 to 0.06
- LIDAR dropout: score drops from 1.0 to 0.30
- Radar inconsistency: consistency drops from 1.0 to 0.0

## Inference Backends

Three inference paths are available for the detector:

| Backend | Provider | Config Value |
|---------|----------|--------------|
| PyTorch | `torch` + `ultralytics` | `pytorch` (default) |
| ONNX Runtime | `onnxruntime` (CPU/CUDA) | `onnx` |
| TensorRT | `onnxruntime` TRT EP or native TRT | `tensorrt` |

Switch via `configs/system.yaml`:

```yaml
perception:
  runtime: "onnx"
  onnx_path: "models/yolov8n.onnx"
```

### Building TensorRT Engines

```bash
# Export to ONNX first
python scripts/export_onnx.py --weights yolov8n.pt --out models/yolov8n.onnx

# Quantize to INT8 (with accuracy guardrails)
python scripts/quantize_int8.py --all

# Build TRT engine (requires TensorRT SDK)
python scripts/build_tensorrt.py --onnx models/yolov8n.onnx --output models/yolov8n.engine

# Benchmark ONNX vs PyTorch
python scripts/benchmark_onnx.py
```

## Model Selection

Detector models are config-switchable:

```yaml
perception:
  detector_model: "yolov8n"   # Options: yolov8n, yolov8s, yolov8m, yolov11n, yolov11s
```

Smaller models (yolov8n) favor speed; larger models (yolov8m) favor accuracy. The system auto-selects the best available device (CUDA > MPS > CPU).

## Confidence Calibration

Temperature scaling calibration can reduce overconfident detections, improving downstream tracking and safety:

```yaml
perception:
  confidence_calibration:
    enabled: true
    temperature: 1.5
```

## Optimization Levers

- **ONNX/TensorRT** export for detection and depth models.
- **Model quantization** (INT8/FP16) with accuracy guardrails via TRT builder.
- **Parallel pipeline** for independent perception stages.
- **Adaptive frame skipping** under sustained overload.
- **Resolution/downsampling** controls in `configs/perception.yaml`.
- **Tracker selection**: ByteTrack is faster than DeepSORT for high object counts.
- **Depth backend**: MiDAS small is faster than Depth Anything V2 but less accurate.
- **Sensor fusion gating**: Disable LIDAR/radar fusion when sensors unavailable to save overhead.
- **Saliency gating**: Disable explainability in production for lower latency.
- **Recording interval**: Reduce recording frequency to lower serialization overhead.

## Benchmarking

```bash
# Full pipeline benchmark (all backends)
python scripts/benchmark_all.py --json results/bench.json

# Basic pipeline benchmark
python scripts/benchmark.py --input data/samples/test_drive.mp4 --config configs/system.yaml

# ONNX-specific benchmark
python scripts/benchmark_onnx.py

# Latency budget breakdown
python scripts/latency_budget.py

# CI latency gate (pass/fail)
python scripts/ci_latency_gate.py

# Summarize a run's metrics
python scripts/summarize_run.py results/run_YYYYMMDD_HHMMSS
```

Metrics recorded per run:

- FPS (mean, p50, p95)
- Per-stage latency breakdown (detection, tracking, lanes, depth, segmentation, LIDAR, radar, safety, fusion)
- Frame drop rate
- Sensor health scores
- Git commit hash for reproducibility

## Regression Checks

### CI Pipeline

`.github/workflows/ci.yml` includes:

- **Latency gate**: `python scripts/ci_latency_gate.py` enforces 50 ms budget on synthetic data.
- **Coverage**: `pytest --cov=src --cov-report=xml` tracks test coverage.

`.github/workflows/model-regression.yml` runs on model path changes:

- Compares against `baselines/benchmark_baseline.json` (mAP=0.65, latency=35ms, fps=28.5).
- Fails if mAP drops >5% or latency increases >20%.

### Performance-Sensitive Tests

- `test_parallel.py`: Verifies concurrent execution achieves expected speedup (4 stages in < 0.3 s).
- `test_orchestrator_unit.py`: Tests 10 extracted stage methods for correctness and return types.
- `test_e2e_smoke.py`: End-to-end pipeline smoke test (single frame, multi-frame, lane integration).
- Health monitor tests: Validates degraded-mode triggering and recovery.
- Per-stage timing is logged in `metrics.json` for offline comparison across commits.
