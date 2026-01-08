"""ONNX/TensorRT inference path for object detection (Phase 4.2).

Uses onnxruntime with CPU, CUDA, or TensorRT execution providers.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from src.types.detection import Detection


_COCO_AUTOMOTIVE = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


class ONNXDetector:
    """ONNX-based object detector with optional TensorRT backend."""

    def __init__(
        self,
        onnx_path: str = "yolov8n.onnx",
        provider: str = "cpu",  # "cpu" | "cuda" | "tensorrt"
        conf_thres: float = 0.25,
        input_size: tuple = (640, 640),
    ):
        if ort is None:
            raise ImportError("onnxruntime is required for ONNXDetector")
        providers = self._resolve_providers(provider)
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.conf_thres = conf_thres
        self.input_size = input_size
        self.input_name = self.session.get_inputs()[0].name

    def _resolve_providers(self, provider: str) -> list:
        if provider == "tensorrt":
            return ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        elif provider == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def infer(self, frame: np.ndarray, conf_thres: Optional[float] = None) -> List[Detection]:
        if cv2 is None:
            return []

        threshold = conf_thres if conf_thres is not None else self.conf_thres
        h_orig, w_orig = frame.shape[:2]

        # Preprocess
        img = cv2.resize(frame, self.input_size)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        # Inference
        outputs = self.session.run(None, {self.input_name: img})
        raw = outputs[0]  # shape varies by model export

        detections: List[Detection] = []
        if raw is None:
            return detections

        # Parse YOLO ONNX output format
        # Typical shape: (1, num_boxes, 85) for YOLOv8
        if len(raw.shape) == 3:
            raw = raw[0]  # (num_boxes, 85)

        # Handle transposed format (1, 84, num_boxes)
        if raw.shape[0] < raw.shape[-1]:
            raw = raw.T

        for row in raw:
            if len(row) < 6:
                continue
            # Format: x_center, y_center, w, h, obj_conf, class_scores...
            if len(row) > 6:
                class_scores = row[4:]
                class_id = int(np.argmax(class_scores))
                conf = float(class_scores[class_id])
            else:
                conf = float(row[4])
                class_id = int(row[5])

            if conf < threshold:
                continue
            if class_id not in _COCO_AUTOMOTIVE:
                continue

            cx, cy, bw, bh = row[:4]
            scale_x = w_orig / self.input_size[0]
            scale_y = h_orig / self.input_size[1]
            x1 = int((cx - bw / 2) * scale_x)
            y1 = int((cy - bh / 2) * scale_y)
            x2 = int((cx + bw / 2) * scale_x)
            y2 = int((cy + bh / 2) * scale_y)

            detections.append(Detection(
                x1=x1, y1=y1, x2=x2, y2=y2,
                conf=conf,
                class_id=class_id,
                class_name=_COCO_AUTOMOTIVE[class_id],
            ))

        return detections
