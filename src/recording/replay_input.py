"""Replay input: reads .apsrec files and yields FramePackets."""
from __future__ import annotations

import gzip
import pickle
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from src.inputs.base_input import BaseInput
from src.types.perception import FramePacket
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import msgpack
    _HAS_MSGPACK = True
except ImportError:
    msgpack = None  # type: ignore[assignment]
    _HAS_MSGPACK = False


class ReplayInput(BaseInput):
    """Replays recorded .apsrec files as a FramePacket stream."""

    def __init__(self, path: str | Path, playback_speed: float = 1.0):
        self.path = Path(path)
        self.playback_speed = max(0.1, playback_speed)
        self._data: dict[str, Any] = {}
        self._frames: list[dict[str, Any]] = []

    def start(self) -> None:
        logger.info("ReplayInput: loading %s", self.path)
        raw_compressed = self.path.read_bytes()
        raw = gzip.decompress(raw_compressed)

        if _HAS_MSGPACK:
            try:
                self._data = msgpack.unpackb(raw, raw=False)
            except Exception:
                self._data = pickle.loads(raw)
        else:
            self._data = pickle.loads(raw)

        self._frames = self._data.get("frames", [])
        logger.info("ReplayInput: loaded %d frames", len(self._frames))

    def stop(self) -> None:
        self._data = {}
        self._frames = []

    def get_frame(self, index: int) -> dict[str, Any] | None:
        """Random access to a specific frame by index."""
        if 0 <= index < len(self._frames):
            return self._frames[index]
        return None

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def frames(self) -> Iterator[FramePacket]:
        blank = np.zeros((720, 1280, 3), dtype=np.uint8)
        prev_ts = None

        for recorded in self._frames:
            ts = recorded.get("timestamp", 0.0)

            if prev_ts is not None and self.playback_speed < 100.0:
                delay = (ts - prev_ts) / self.playback_speed
                if delay > 0:
                    time.sleep(delay)
            prev_ts = ts

            packet = FramePacket(
                frame=blank.copy(),
                timestamp=ts,
                sensor_id="replay",
            )
            # Attach recorded metadata for downstream consumers
            packet._recorded = recorded  # type: ignore[attr-defined]

            yield packet
