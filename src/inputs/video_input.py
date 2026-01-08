from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Tuple

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from src.inputs.base_input import BaseInput
from src.utils.logger import get_logger
from src.utils.types import FramePacket


@dataclass
class VideoMeta:
    fps: float
    width: int
    height: int
    frame_count: int


class VideoInput(BaseInput):
    def __init__(self, path: str | Path, allow_missing: bool = False, frame_rate: Optional[int] = None):
        self.path = Path(path)
        self.logger = get_logger(__name__)
        self.frame_rate = frame_rate
        self.cap = None
        self.meta: Optional[VideoMeta] = None
        self.allow_missing = allow_missing

        if cv2 is None:
            if allow_missing:
                self.logger.warning("OpenCV not available; VideoInput will stay inert.")
                return
            raise ImportError("opencv-python is required for VideoInput")

        if not self.path.exists():
            if allow_missing:
                self.logger.warning("Video %s not found; proceeding inert for testing.", self.path)
                return
            raise FileNotFoundError(f"Video not found: {self.path}")

        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            if allow_missing:
                self.logger.warning("Could not open video %s; proceeding inert for testing.", self.path)
                self.cap = None
                return
            raise RuntimeError(f"Could not open video: {self.path}")

        self.meta = VideoMeta(
            fps=float(self.cap.get(cv2.CAP_PROP_FPS) or (frame_rate or 30.0)),
            width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        )
        self.logger.info(
            "Video opened: %s fps=%.2f size=%dx%d frames=%d",
            self.path,
            self.meta.fps,
            self.meta.width,
            self.meta.height,
            self.meta.frame_count,
        )

    def start(self) -> None:
        # Initialization handled in __init__
        return

    def frames(self) -> Generator[Tuple[int, FramePacket], None, None]:
        if self.cap is None:
            return
        idx = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            idx += 1
            yield idx, FramePacket(frame=frame, timestamp=idx / (self.meta.fps if self.meta else (self.frame_rate or 30)))

    def stop(self) -> None:
        if self.cap:
            self.cap.release()
            self.logger.info("Closed video %s", self.path)
