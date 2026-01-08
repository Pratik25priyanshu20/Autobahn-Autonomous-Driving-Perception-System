#!/usr/bin/env python3
"""
Export YOLOv8n to ONNX (CPU-safe).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def export(model_path: Path, out_path: Path) -> None:
    model = YOLO(str(model_path)).model
    model.eval()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 3, 640, 640)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        opset_version=12,
        input_names=["images"],
        output_names=["outputs"],
        dynamic_axes={"images": {0: "batch"}, "outputs": {0: "batch"}},
    )

    print(f"✅ YOLO exported to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8n to ONNX")
    parser.add_argument("--weights", default="yolov8n.pt", help="Path to YOLO weights")
    parser.add_argument("--out", default="models/yolo_v8.onnx", help="Output ONNX path")
    args = parser.parse_args()

    export(Path(args.weights), Path(args.out))


if __name__ == "__main__":
    main()
