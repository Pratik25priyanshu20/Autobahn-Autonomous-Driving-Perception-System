#!/usr/bin/env python3
"""
Dynamic INT8 quantization for ONNX models via onnxruntime.

Applies ``quantize_dynamic`` (weight-only INT8) to exported FP32 ONNX models.
This is calibration-free and works on any CPU without representative data.

Supported models:
  - yolo      : YOLOv8n object detector
  - midas     : MiDAS v2.1 small depth estimator
  - deeplabv3 : DeepLabV3+ MobileNetV3 semantic segmenter

Usage:
  python scripts/quantize_int8.py --model yolo
  python scripts/quantize_int8.py --all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, dict[str, Path]] = {
    "yolo": {
        "fp32": Path("models/yolo_v8.onnx"),
        "int8": Path("models/yolo_v8_int8.onnx"),
    },
    "midas": {
        "fp32": Path("models/midas_small.onnx"),
        "int8": Path("models/midas_small_int8.onnx"),
    },
    "deeplabv3": {
        "fp32": Path("models/deeplabv3_mobilenet.onnx"),
        "int8": Path("models/deeplabv3_mobilenet_int8.onnx"),
    },
}


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def quantize_model(name: str, fp32_path: Path, int8_path: Path) -> bool:
    """Run dynamic INT8 quantization on a single ONNX model.

    Returns True on success, False on failure.
    """
    if not fp32_path.exists():
        print(f"[SKIP] {name}: FP32 model not found at {fp32_path}", file=sys.stderr)
        print(f"       Run 'python scripts/export_onnx.py --model {name}' first.")
        return False

    int8_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[...] Quantizing {name}: {fp32_path} -> {int8_path}")

    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )

    fp32_size = fp32_path.stat().st_size
    int8_size = int8_path.stat().st_size
    reduction_pct = (1.0 - int8_size / fp32_size) * 100.0 if fp32_size > 0 else 0.0

    print(f"[OK]  {name}")
    print(f"      FP32 size : {fp32_size / (1024 * 1024):.2f} MB")
    print(f"      INT8 size : {int8_size / (1024 * 1024):.2f} MB")
    print(f"      Reduction : {reduction_pct:.1f}%")

    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dynamic INT8 quantization for APS++ ONNX models",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()),
        help="Quantize a specific model",
    )
    group.add_argument(
        "--all",
        action="store_true",
        dest="quantize_all",
        help="Quantize all available models",
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Override base directory for model files (default: models/)",
    )
    args = parser.parse_args()

    targets = list(MODEL_REGISTRY.keys()) if args.quantize_all else [args.model]

    success_count = 0
    total = len(targets)

    for name in targets:
        entry = MODEL_REGISTRY[name]
        fp32_path = entry["fp32"]
        int8_path = entry["int8"]

        # Allow overriding the models directory
        if args.models_dir:
            base = Path(args.models_dir)
            fp32_path = base / fp32_path.name
            int8_path = base / int8_path.name

        if quantize_model(name, fp32_path, int8_path):
            success_count += 1
        print()

    # Summary
    print("=" * 50)
    print(f"Quantization complete: {success_count}/{total} model(s) converted.")

    if success_count == 0:
        print("No models were quantized. Export FP32 models first:")
        print("  python scripts/export_onnx.py --model all")
        sys.exit(1)
    elif success_count < total:
        print("Some models were skipped. Check warnings above.")


if __name__ == "__main__":
    main()
