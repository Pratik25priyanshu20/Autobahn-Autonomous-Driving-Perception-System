"""Data recorder: serializes WorldModel frames to compressed streams."""
from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from typing import Any

from src.recording.recording_types import RecordedFrame
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import msgpack
    _HAS_MSGPACK = True
except ImportError:
    msgpack = None  # type: ignore[assignment]
    _HAS_MSGPACK = False


class DataRecorder:
    """Records WorldModel snapshots to a compressed .apsrec file."""

    def __init__(
        self,
        output_dir: str | Path,
        record_interval: int = 1,
        max_size_mb: float = 100.0,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.record_interval = max(1, record_interval)
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self._path = self.output_dir / "recording.apsrec"
        self._buffer: list[dict[str, Any]] = []
        self._frame_count = 0
        self._total_bytes = 0
        self._closed = False
        logger.info("DataRecorder: output=%s interval=%d max_mb=%.1f", self._path, self.record_interval, max_size_mb)

    @staticmethod
    def _serialize_track(trk: Any) -> dict[str, Any]:
        """Extract serializable fields from a track object."""
        return {
            "track_id": getattr(trk, "track_id", None),
            "x": getattr(trk, "x", None),
            "y": getattr(trk, "y", None),
            "vx": getattr(trk, "vx", None),
            "vy": getattr(trk, "vy", None),
            "class_name": getattr(trk, "class_name", None),
            "conf": getattr(trk, "conf", None),
        }

    @staticmethod
    def _serialize_wm(wm: Any) -> RecordedFrame:
        """Convert a WorldModel into a RecordedFrame."""
        tracks = [DataRecorder._serialize_track(t) for t in getattr(wm, "tracks", [])]
        return RecordedFrame(
            frame_id=getattr(wm, "frame_id", 0),
            timestamp=getattr(wm, "frame_id", 0) / 20.0,
            tracks=tracks,
            lanes=dict(getattr(wm, "lanes", {}) or {}),
            fcw_state=(getattr(wm, "fcw", {}) or {}).get("state", "NORMAL"),
            safety_state=(getattr(wm, "safety", {}) or {}).get("state", "NORMAL"),
            detections_count=len(getattr(wm, "detections", [])),
            predictions={},
            sensor_health=dict(getattr(wm, "sensor_health", {}) or {}),
        )

    def record(self, wm: Any) -> None:
        """Serialize and buffer a world model frame."""
        if self._closed:
            return
        self._frame_count += 1
        if self._frame_count % self.record_interval != 0:
            return
        if self._total_bytes >= self.max_size_bytes:
            return

        rf = self._serialize_wm(wm)
        frame_dict = {
            "frame_id": rf.frame_id,
            "timestamp": rf.timestamp,
            "tracks": rf.tracks,
            "lanes": rf.lanes,
            "fcw_state": rf.fcw_state,
            "safety_state": rf.safety_state,
            "detections_count": rf.detections_count,
            "predictions": rf.predictions,
            "sensor_health": rf.sensor_health,
        }
        self._buffer.append(frame_dict)

    def close(self) -> None:
        """Flush buffer to compressed file."""
        if self._closed:
            return
        self._closed = True

        data = {"version": 1, "frames": self._buffer}

        raw = msgpack.packb(data, use_bin_type=True) if _HAS_MSGPACK else pickle.dumps(data)

        compressed = gzip.compress(raw)
        self._total_bytes = len(compressed)
        self._path.write_bytes(compressed)
        logger.info("DataRecorder: wrote %d frames (%.1f KB) to %s",
                     len(self._buffer), len(compressed) / 1024.0, self._path)
