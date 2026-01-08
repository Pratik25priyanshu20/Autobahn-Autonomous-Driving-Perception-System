from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from src.types.detection import Detection

# Supported model variants for config-driven switching (Phase 1.1)
SUPPORTED_MODELS: Dict[str, str] = {
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
    "yolov11n": "yolo11n.pt",
    "yolov11s": "yolo11s.pt",
}


def _auto_device() -> str:
    """Select the best available inference device."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class YOLODetector:
    """YOLOv8/v11 wrapper for APS++ with automatic device selection."""

    def __init__(self, model_name: str = "yolov8n.pt", device: Optional[str] = None):
        from ultralytics import YOLO
        # Resolve shorthand model names
        resolved = SUPPORTED_MODELS.get(model_name, model_name)
        self.device = device or _auto_device()
        self.model = YOLO(resolved)
        self.model.to(self.device)

        # Automotive-relevant classes (COCO)
        self.allowed_classes = {
            0: "person",
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck",
        }

    def infer(self, frame: np.ndarray, conf_thres: float = 0.25) -> List[Detection]:
        results = self.model(
            frame,
            device=self.device,
            conf=conf_thres,
            verbose=False,
        )[0]

        detections: List[Detection] = []

        if results.boxes is None:
            return detections

        for box in results.boxes:
            cls_id = int(box.cls.item())
            if cls_id not in self.allowed_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf.item())

            detections.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    conf=conf,
                    class_id=cls_id,
                    class_name=self.allowed_classes[cls_id],
                )
            )

        return detections
