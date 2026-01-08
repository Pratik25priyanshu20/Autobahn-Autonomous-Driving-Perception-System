import time
from typing import Dict

import numpy as np
import torch
import torchvision

from src.perception.segmentation.base_segmenter import BaseSegmenter


class DeepLabV3Segmenter(BaseSegmenter):
    """
    DeepLabV3+ (MobileNet backbone) semantic segmentation.
    Pretrained on Cityscapes-style classes.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights="DEFAULT")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def infer(self, frame: np.ndarray) -> dict:
        """
        Args:
            frame: RGB image (H, W, 3), uint8

        Returns:
            dict with keys:
              - mask: (H, W) int class ids
              - confidence: float
              - latency_ms: float
        """
        start = time.perf_counter()
        img = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0).to(self.device)

        output = self.model(img)["out"]
        probs = torch.softmax(output, dim=1)
        mask = probs.argmax(dim=1).squeeze(0).cpu().numpy()
        confidence = probs.max().item()

        latency_ms = (time.perf_counter() - start) * 1000.0
        return {
            "mask": mask,
            "confidence": float(confidence),
            "latency_ms": latency_ms,
        }
