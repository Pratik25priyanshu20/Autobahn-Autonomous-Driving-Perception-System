"""Live webcam input source for APS++ (Phase 0.3)."""
from __future__ import annotations

import time
from typing import Generator, Tuple

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from src.inputs.base_input import BaseInput
from src.types.perception import FramePacket
from src.utils.logger import get_logger


class WebcamInput(BaseInput):
    """Captures frames from a local webcam via OpenCV."""

    def __init__(self, device_id: int = 0, fps: float = 30.0):
        if cv2 is None:
            raise ImportError("opencv-python is required for WebcamInput")
        self.device_id = device_id
        self.fps = fps
        self.logger = get_logger(__name__)
        self.cap = None

    def start(self) -> None:
        self.cap = cv2.VideoCapture(self.device_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam device {self.device_id}")
        self.logger.info("Webcam opened: device=%d", self.device_id)

    def frames(self) -> Generator[Tuple[int, FramePacket], None, None]:
        if self.cap is None:
            self.start()
        idx = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            idx += 1
            yield idx, FramePacket(frame=frame, timestamp=time.time(), sensor_id="webcam")

    def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.logger.info("Webcam closed: device=%d", self.device_id)
