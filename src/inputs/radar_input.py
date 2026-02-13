"""Radar CSV input source for APS++."""
from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from src.inputs.base_input import BaseInput
from src.types.perception import FramePacket
from src.types.radar import RadarDetection, RadarFrame
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RadarInput(BaseInput):
    """Reads radar detections from CSV (frame_id,range_m,azimuth_deg,velocity_mps,rcs_dbsm).

    Yields FramePackets with radar_frame populated and a blank camera frame.
    """

    def __init__(self, csv_path: str | Path, frame_width: int = 1280, frame_height: int = 720):
        self.csv_path = Path(csv_path)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self._frames_by_id: dict[int, list[RadarDetection]] = defaultdict(list)

    def start(self) -> None:
        logger.info("Loading radar CSV: %s", self.csv_path)
        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fid = int(row["frame_id"])
                det = RadarDetection(
                    range_m=float(row["range_m"]),
                    azimuth_deg=float(row["azimuth_deg"]),
                    velocity_mps=float(row["velocity_mps"]),
                    rcs_dbsm=float(row["rcs_dbsm"]),
                )
                self._frames_by_id[fid].append(det)
        logger.info("Loaded %d radar frames", len(self._frames_by_id))

    def stop(self) -> None:
        self._frames_by_id.clear()

    def frames(self) -> Iterator[FramePacket]:
        blank = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        for fid in sorted(self._frames_by_id.keys()):
            dets = self._frames_by_id[fid]
            rf = RadarFrame(detections=dets, timestamp=float(fid) / 20.0)
            yield FramePacket(
                frame=blank.copy(),
                timestamp=rf.timestamp,
                sensor_id="radar",
                radar_frame=rf,
            )
