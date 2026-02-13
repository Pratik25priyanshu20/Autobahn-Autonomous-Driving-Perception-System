#!/usr/bin/env python3
"""
Benchmark PyTorch vs ONNX FP32 vs INT8 for all APS++ perception models.

Runs N forward passes per variant, computes mean latency and throughput,
then displays results in a Rich table and optionally exports to JSON.

Usage:
  python scripts/benchmark_all.py
  python scripts/benchmark_all.py --iters 100 --warmup 10
  python scripts/benchmark_all.py --json results/benchmark.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

try:
    from rich.console import Console
    from rich.table import Table
except ImportError:
    print("Rich is required: pip install rich", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    model: str
    variant: str          # "PyTorch" | "ONNX FP32" | "ONNX INT8"
    mean_ms: float = 0.0
    std_ms: float = 0.0
    throughput_fps: float = 0.0
    file_size_mb: float = 0.0
    error: str = ""


# ---------------------------------------------------------------------------
# ONNX benchmark helper
# ---------------------------------------------------------------------------

def _bench_onnx(
    onnx_path: Path,
    input_name: str,
    input_shape: tuple[int, ...],
    n: int,
    warmup: int,
) -> tuple[float, float]:
    """Return (mean_ms, std_ms) for an ONNX model."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    dummy = np.random.randn(*input_shape).astype(np.float32)

    for _ in range(warmup):
        sess.run(None, {input_name: dummy})

    times: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy})
        times.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(times)), float(np.std(times))


# ---------------------------------------------------------------------------
# Per-model benchmarks
# ---------------------------------------------------------------------------

def _bench_yolo_pytorch(n: int, warmup: int) -> tuple[float, float]:
    import torch
    from ultralytics import YOLO

    weights = Path("yolov8n.pt")
    if not weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights}")

    model = YOLO(str(weights)).model
    model.eval().cpu()
    dummy = torch.randn(1, 3, 640, 640)

    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
        times: list[float] = []
        for _ in range(n):
            t0 = time.perf_counter()
            model(dummy)
            times.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(times)), float(np.std(times))


def _bench_midas_pytorch(n: int, warmup: int) -> tuple[float, float]:
    import torch

    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    model.eval().cpu()
    dummy = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
        times: list[float] = []
        for _ in range(n):
            t0 = time.perf_counter()
            model(dummy)
            times.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(times)), float(np.std(times))


def _bench_deeplabv3_pytorch(n: int, warmup: int) -> tuple[float, float]:
    import torch
    import torchvision

    model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights="DEFAULT")
    model.eval().cpu()
    dummy = torch.randn(1, 3, 513, 513)

    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
        times: list[float] = []
        for _ in range(n):
            t0 = time.perf_counter()
            model(dummy)
            times.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(times)), float(np.std(times))


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    name: str
    onnx_fp32: Path
    onnx_int8: Path
    input_name: str
    input_shape: tuple[int, ...]
    pytorch_fn: Callable[[int, int], tuple[float, float]]


MODELS: list[ModelSpec] = [
    ModelSpec(
        name="YOLOv8n",
        onnx_fp32=Path("models/yolo_v8.onnx"),
        onnx_int8=Path("models/yolo_v8_int8.onnx"),
        input_name="images",
        input_shape=(1, 3, 640, 640),
        pytorch_fn=_bench_yolo_pytorch,
    ),
    ModelSpec(
        name="MiDAS small",
        onnx_fp32=Path("models/midas_small.onnx"),
        onnx_int8=Path("models/midas_small_int8.onnx"),
        input_name="input",
        input_shape=(1, 3, 256, 256),
        pytorch_fn=_bench_midas_pytorch,
    ),
    ModelSpec(
        name="DeepLabV3",
        onnx_fp32=Path("models/deeplabv3_mobilenet.onnx"),
        onnx_int8=Path("models/deeplabv3_mobilenet_int8.onnx"),
        input_name="input",
        input_shape=(1, 3, 513, 513),
        pytorch_fn=_bench_deeplabv3_pytorch,
    ),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_benchmarks(iters: int, warmup: int) -> list[BenchResult]:
    """Run all benchmarks and return a flat list of results."""
    results: list[BenchResult] = []

    for spec in MODELS:
        # --- PyTorch ---
        try:
            mean, std = spec.pytorch_fn(iters, warmup)
            results.append(BenchResult(
                model=spec.name,
                variant="PyTorch",
                mean_ms=mean,
                std_ms=std,
                throughput_fps=1000.0 / mean if mean > 0 else 0.0,
            ))
        except Exception as exc:
            results.append(BenchResult(model=spec.name, variant="PyTorch", error=str(exc)))

        # --- ONNX FP32 ---
        if spec.onnx_fp32.exists():
            try:
                mean, std = _bench_onnx(spec.onnx_fp32, spec.input_name, spec.input_shape, iters, warmup)
                results.append(BenchResult(
                    model=spec.name,
                    variant="ONNX FP32",
                    mean_ms=mean,
                    std_ms=std,
                    throughput_fps=1000.0 / mean if mean > 0 else 0.0,
                    file_size_mb=spec.onnx_fp32.stat().st_size / (1024 * 1024),
                ))
            except Exception as exc:
                results.append(BenchResult(model=spec.name, variant="ONNX FP32", error=str(exc)))
        else:
            results.append(BenchResult(
                model=spec.name, variant="ONNX FP32", error=f"Not found: {spec.onnx_fp32}",
            ))

        # --- ONNX INT8 ---
        if spec.onnx_int8.exists():
            try:
                mean, std = _bench_onnx(spec.onnx_int8, spec.input_name, spec.input_shape, iters, warmup)
                results.append(BenchResult(
                    model=spec.name,
                    variant="ONNX INT8",
                    mean_ms=mean,
                    std_ms=std,
                    throughput_fps=1000.0 / mean if mean > 0 else 0.0,
                    file_size_mb=spec.onnx_int8.stat().st_size / (1024 * 1024),
                ))
            except Exception as exc:
                results.append(BenchResult(model=spec.name, variant="ONNX INT8", error=str(exc)))
        else:
            results.append(BenchResult(
                model=spec.name, variant="ONNX INT8", error=f"Not found: {spec.onnx_int8}",
            ))

    return results


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display_results(results: list[BenchResult], console: Console) -> None:
    """Render results as a Rich table."""
    table = Table(title="APS++ Model Benchmark (CPU)", show_lines=True)

    table.add_column("Model", style="bold cyan", no_wrap=True)
    table.add_column("Variant", style="bold")
    table.add_column("Mean (ms)", justify="right")
    table.add_column("Std (ms)", justify="right")
    table.add_column("FPS", justify="right")
    table.add_column("Size (MB)", justify="right")
    table.add_column("Status", justify="center")

    for r in results:
        if r.error:
            table.add_row(
                r.model, r.variant, "-", "-", "-", "-",
                f"[red]{r.error[:40]}[/red]",
            )
        else:
            size_str = f"{r.file_size_mb:.1f}" if r.file_size_mb > 0 else "-"
            table.add_row(
                r.model,
                r.variant,
                f"{r.mean_ms:.1f}",
                f"{r.std_ms:.1f}",
                f"{r.throughput_fps:.1f}",
                size_str,
                "[green]OK[/green]",
            )

    console.print()
    console.print(table)
    console.print()


def export_json(results: list[BenchResult], out_path: Path) -> None:
    """Write results to a JSON file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": "aps_model_benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": [asdict(r) for r in results],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Results saved to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark APS++ models: PyTorch vs ONNX FP32 vs INT8",
    )
    parser.add_argument(
        "--iters", "-n",
        type=int,
        default=50,
        help="Number of inference iterations per variant (default: 50)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup iterations before timing (default: 5)",
    )
    parser.add_argument(
        "--json",
        default=None,
        metavar="PATH",
        help="Export results to a JSON file",
    )
    args = parser.parse_args()

    console = Console()
    console.print("[bold]APS++ Benchmark[/bold] - PyTorch / ONNX FP32 / ONNX INT8")
    console.print(f"Iterations: {args.iters}  |  Warmup: {args.warmup}\n")

    results = run_benchmarks(args.iters, args.warmup)
    display_results(results, console)

    if args.json:
        export_json(results, Path(args.json))


if __name__ == "__main__":
    main()
