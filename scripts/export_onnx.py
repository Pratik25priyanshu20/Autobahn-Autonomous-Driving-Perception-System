#!/usr/bin/env python3
"""
Export perception models to ONNX format.

Supported models:
  - yolo      : YOLOv8n object detector
  - midas     : MiDAS v2.1 small depth estimator
  - deeplabv3 : DeepLabV3+ MobileNetV3 semantic segmenter
  - all       : Export all models

Usage:
  python scripts/export_onnx.py --model yolo
  python scripts/export_onnx.py --model all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# YOLO export
# ---------------------------------------------------------------------------

def export_yolo(weights: Path, out_path: Path) -> Path:
    """Export YOLOv8n to ONNX with dynamic batch axis."""
    from ultralytics import YOLO

    model = YOLO(str(weights)).model
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

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[OK] YOLO exported to {out_path}  ({size_mb:.1f} MB)")
    return out_path


# ---------------------------------------------------------------------------
# MiDAS export
# ---------------------------------------------------------------------------

def export_midas(out_path: Path, model_type: str = "MiDaS_small") -> Path:
    """Export MiDAS depth model to ONNX.

    Uses the small variant by default (256x256 input).
    """
    model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
    model.eval()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # MiDaS small expects 256x256 input
    dummy = torch.randn(1, 3, 256, 256)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        opset_version=12,
        input_names=["input"],
        output_names=["depth"],
        dynamic_axes={"input": {0: "batch"}, "depth": {0: "batch"}},
    )

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[OK] MiDAS exported to {out_path}  ({size_mb:.1f} MB)")
    return out_path


# ---------------------------------------------------------------------------
# DeepLabV3 export
# ---------------------------------------------------------------------------

def export_deeplabv3(out_path: Path) -> Path:
    """Export DeepLabV3+ MobileNetV3-Large to ONNX.

    Uses 513x513 input which is the standard Cityscapes resolution for this
    architecture.  Dynamic batch axis is enabled.
    """
    import torchvision

    model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights="DEFAULT")
    model.eval()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Standard segmentation input size
    dummy = torch.randn(1, 3, 513, 513)

    # DeepLabV3 returns an OrderedDict; we need to wrap it so ONNX sees a
    # single tensor output (the "out" key).
    class _Wrapper(torch.nn.Module):
        def __init__(self, backbone: torch.nn.Module):
            super().__init__()
            self.backbone = backbone

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.backbone(x)["out"]

    wrapped = _Wrapper(model)
    wrapped.eval()

    torch.onnx.export(
        wrapped,
        dummy,
        str(out_path),
        opset_version=12,
        input_names=["input"],
        output_names=["segmentation"],
        dynamic_axes={"input": {0: "batch"}, "segmentation": {0: "batch"}},
    )

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[OK] DeepLabV3 exported to {out_path}  ({size_mb:.1f} MB)")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MODEL_CHOICES = ("yolo", "midas", "deeplabv3", "all")

DEFAULT_PATHS = {
    "yolo": Path("models/yolo_v8.onnx"),
    "midas": Path("models/midas_small.onnx"),
    "deeplabv3": Path("models/deeplabv3_mobilenet.onnx"),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export APS++ perception models to ONNX",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="all",
        help="Which model to export (default: all)",
    )
    parser.add_argument(
        "--weights",
        default="yolov8n.pt",
        help="Path to YOLO weights (only used when model=yolo|all)",
    )
    parser.add_argument(
        "--out-dir",
        default="models",
        help="Output directory for ONNX files (default: models/)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Explicit output path (overrides --out-dir; single-model export only)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    targets: list[str] = list(DEFAULT_PATHS.keys()) if args.model == "all" else [args.model]

    exported: list[Path] = []

    for target in targets:
        out_path = Path(args.out) if args.out and len(targets) == 1 else out_dir / DEFAULT_PATHS[target].name

        try:
            if target == "yolo":
                exported.append(export_yolo(Path(args.weights), out_path))
            elif target == "midas":
                exported.append(export_midas(out_path))
            elif target == "deeplabv3":
                exported.append(export_deeplabv3(out_path))
        except Exception as exc:
            print(f"[FAIL] {target}: {exc}", file=sys.stderr)

    if exported:
        print(f"\nExported {len(exported)}/{len(targets)} model(s) to ONNX.")
    else:
        print("No models exported.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
