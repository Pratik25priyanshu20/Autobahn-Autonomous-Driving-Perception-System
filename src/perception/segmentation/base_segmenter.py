from __future__ import annotations

import abc
import numpy as np


class BaseSegmenter(abc.ABC):
    @abc.abstractmethod
    def infer(self, frame: np.ndarray) -> dict:
        """
        Input:
            frame: BGR or RGB image (H, W, 3)
        Output:
            {
              "mask": np.ndarray (H, W)   # class IDs
              "confidence": float
              "latency_ms": float
            }
        """
        raise NotImplementedError
