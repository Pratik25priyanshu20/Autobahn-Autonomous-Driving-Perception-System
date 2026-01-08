# Performance

Targets focus on low latency, stable throughput, and bounded jitter to support real-time perception and control.

## Latency Budgets

| Stage | Target | Notes |
|-------|--------|-------|
| End-to-end | ≤ 120 ms | Camera to fused world model; stretch goal 80 ms |
| Detection | ≤ 30 ms | YOLOv8n on GPU; ~60 ms on CPU |
| Tracking | ≤ 10 ms | ByteTrack preferred for speed |
| Lane detection | ≤ 15 ms | Canny+Hough; UFLDv2 adds ~10 ms |
| Depth estimation | ≤ 25 ms | MiDAS small on GPU |
| Segmentation | ≤ 20 ms | DeepLabV3+ MobileNet |
| Kalman + prediction | ≤ 2 ms | Pure NumPy, no GPU |
| Safety evaluation | ≤ 5 ms | Rule-based, no neural inference |
| Fusion + world model | ≤ 5 ms | Dataclass assembly |
| Frame sync buffering | ≤ 2 frames | Jitter < 10 ms |

## Parallel Pipeline

When `performance.parallel.enabled: true`, the orchestrator dispatches independent perception stages concurrently via `ParallelStageExecutor` (`src/runtime/parallel_executor.py`):

```
Detection ─┐
Lanes     ─┤ ThreadPoolExecutor (max_workers from config)
Depth     ─┤
Segment   ─┘
              -> Tracking (serial, depends on detection)
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

## Benchmarking

```bash
# Full pipeline benchmark
python scripts/benchmark.py --input data/samples/test_drive.mp4 --config configs/system.yaml

# ONNX-specific benchmark
python scripts/benchmark_onnx.py

# Summarize a run's metrics
python scripts/summarize_run.py results/run_YYYYMMDD_HHMMSS
```

Metrics recorded per run:

- FPS (mean, p50, p95)
- Per-stage latency breakdown (detection, tracking, lanes, depth, segmentation, safety, fusion)
- Frame drop rate
- Git commit hash for reproducibility

## Regression Checks

CI includes performance-sensitive tests:

- `test_parallel.py`: Verifies concurrent execution achieves expected speedup (4 stages in < 0.3 s).
- Health monitor tests: Validates degraded-mode triggering and recovery.
- Per-stage timing is logged in `metrics.json` for offline comparison across commits.
