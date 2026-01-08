from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    class_id: int
    class_name: str


class YOLODetector:
    """
    YOLOv8 wrapper for APS++ with MPS acceleration where available.
    Output format is stable and contract-based.
    """

    def __init__(self, model_name: str = "yolov8n.pt", device: str | None = None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = YOLO(model_name)
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
        """
        Run YOLO inference on a single frame.
        """
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
