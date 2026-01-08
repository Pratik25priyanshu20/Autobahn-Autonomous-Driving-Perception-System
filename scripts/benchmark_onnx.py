#!/usr/bin/env python3
"""
Benchmark ONNX vs PyTorch (CPU) for YOLOv8n.
Mac-safe: uses CPUExecutionProvider.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO


N = 50
INPUT_SHAPE = (1, 3, 640, 640)


def bench_pytorch(weights: Path) -> float:
    model = YOLO(str(weights)).model
    model.eval().cpu()
    dummy = torch.randn(*INPUT_SHAPE)
    with torch.no_grad():
        # warmup
        for _ in range(5):
            _ = model(dummy)
        t0 = time.time()
        for _ in range(N):
            _ = model(dummy)
        t1 = time.time()
    return (t1 - t0) / N * 1000.0


def bench_onnx(onnx_path: Path) -> float:
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    dummy = np.random.randn(*INPUT_SHAPE).astype(np.float32)
    # warmup
    for _ in range(5):
        _ = sess.run(None, {"images": dummy})
    t0 = time.time()
    for _ in range(N):
        _ = sess.run(None, {"images": dummy})
    t1 = time.time()
    return (t1 - t0) / N * 1000.0


def main():
    weights = Path("yolov8n.pt")
    onnx_path = Path("models/yolo_v8.onnx")
    if not onnx_path.exists():
        raise FileNotFoundError(f"Export ONNX first: {onnx_path} not found")

    pt_ms = bench_pytorch(weights)
    onnx_ms = bench_onnx(onnx_path)

    print("\n=== APS++ ONNX Benchmark (CPU) ===")
    print(f"PyTorch avg latency: {pt_ms:.2f} ms")
    print(f"ONNX    avg latency: {onnx_ms:.2f} ms")
    print(f"Speedup: {pt_ms / onnx_ms:.2f}x" if onnx_ms > 0 else "Speedup: N/A")


if __name__ == "__main__":
    main()
