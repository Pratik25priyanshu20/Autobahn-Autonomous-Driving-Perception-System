# Performance Plan

Targets focus on low latency, stable throughput, and bounded jitter to support real-time control.

## Budgets (initial)
- End-to-end latency: ≤ 120 ms (camera to fused world model), stretch 80 ms.
- Detector + tracker: ≤ 60 ms combined on target hardware (TensorRT where possible).
- Frame sync buffering: ≤ 2 frames; jitter < 10 ms.
- Fusion + safety: ≤ 20 ms.

## Optimization levers
- ONNX/TensorRT export (`scripts/export_onnx.py`, `scripts/build_tensorrt.py`).
- Model quantization (INT8/FP16) with accuracy guardrails.
- Asynchronous pipelines with bounded queues in `runtime/orchestrator.py`.
- Resolution/downsampling controls in `configs/perception.yaml` and `configs/system.yaml`.
- Optional batch inference for trackers to amortize cost.

## Benchmarking
- `scripts/benchmark.py` should replay sample clips, logging FPS, p50/p95 latency, and drop rate.
- Store metrics under `results/metrics/` and reports in `results/reports/`.
- Include GPU/CPU utilization and memory footprint where available.

## Regression checks
- Performance CI thresholding on FPS and tail latency.
- Compare fused object counts and stability before/after model changes.
