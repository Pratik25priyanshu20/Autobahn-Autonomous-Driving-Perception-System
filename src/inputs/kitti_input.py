"""KITTI dataset input loader for APS++.

Loads from the KITTI raw/object detection format:
  - image_2/*.png     — left camera images
  - velodyne/*.bin    — LIDAR point clouds (N x 4 float32)
  - calib/*.txt       — calibration matrices
  - label_2/*.txt     — ground truth labels (optional)
"""
from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from src.inputs.base_input import BaseInput
from src.types.pointcloud import PointCloud
from src.utils.logger import get_logger
from src.utils.types import FramePacket

# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #

@dataclass
class KITTICalibration:
    """Parsed KITTI calibration data with projection utilities."""

    P0: np.ndarray = field(default_factory=lambda: np.eye(3, 4))
    P1: np.ndarray = field(default_factory=lambda: np.eye(3, 4))
    P2: np.ndarray = field(default_factory=lambda: np.eye(3, 4))
    P3: np.ndarray = field(default_factory=lambda: np.eye(3, 4))
    R0_rect: np.ndarray = field(default_factory=lambda: np.eye(3))
    Tr_velo_to_cam: np.ndarray = field(default_factory=lambda: np.eye(3, 4))
    Tr_imu_to_velo: np.ndarray = field(default_factory=lambda: np.eye(3, 4))

    @classmethod
    def from_file(cls, path: str | Path) -> KITTICalibration:
        """Parse a KITTI calib.txt file."""
        path = Path(path)
        data: dict[str, np.ndarray] = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                key, values = line.split(":", 1)
                key = key.strip()
                vals = np.array([float(v) for v in values.strip().split()])
                data[key] = vals

        calib = cls()
        if "P0" in data:
            calib.P0 = data["P0"].reshape(3, 4)
        if "P1" in data:
            calib.P1 = data["P1"].reshape(3, 4)
        if "P2" in data:
            calib.P2 = data["P2"].reshape(3, 4)
        if "P3" in data:
            calib.P3 = data["P3"].reshape(3, 4)
        if "R0_rect" in data:
            calib.R0_rect = data["R0_rect"].reshape(3, 3)
        if "Tr_velo_to_cam" in data:
            calib.Tr_velo_to_cam = data["Tr_velo_to_cam"].reshape(3, 4)
        if "Tr_imu_to_velo" in data:
            calib.Tr_imu_to_velo = data["Tr_imu_to_velo"].reshape(3, 4)

        return calib

    @property
    def velo_to_cam(self) -> np.ndarray:
        """4x4 Velodyne-to-camera transformation matrix."""
        T = np.eye(4)  # noqa: N806
        T[:3, :] = self.Tr_velo_to_cam
        R = np.eye(4)  # noqa: N806
        R[:3, :3] = self.R0_rect
        return R @ T

    @property
    def velo_to_image(self) -> np.ndarray:
        """3x4 projection from Velodyne coordinates to image pixels (camera 2)."""
        return self.P2 @ self.velo_to_cam

    def project_velo_to_image(self, points_3d: np.ndarray) -> np.ndarray:
        """Project Nx3 velodyne points to Nx2 image coordinates.

        Returns (N, 2) array of (u, v) pixel coordinates.
        Points behind the camera are set to (-1, -1).
        """
        n = points_3d.shape[0]
        pts_hom = np.hstack([points_3d[:, :3], np.ones((n, 1))])  # (N, 4)
        proj = (self.velo_to_image @ pts_hom.T).T  # (N, 3)

        # Mask points behind camera
        behind = proj[:, 2] <= 0
        proj[:, 2] = np.where(behind, 1.0, proj[:, 2])  # avoid div-by-zero

        uv = proj[:, :2] / proj[:, 2:3]
        uv[behind] = -1.0
        return uv


# --------------------------------------------------------------------------- #
# Label parsing
# --------------------------------------------------------------------------- #

@dataclass
class KITTILabel:
    """Single KITTI ground truth label."""

    class_name: str = "DontCare"
    truncated: float = 0.0
    occluded: int = 0
    alpha: float = 0.0
    bbox_2d: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # x1, y1, x2, y2
    dimensions: tuple[float, float, float] = (0.0, 0.0, 0.0)  # h, w, l
    location: tuple[float, float, float] = (0.0, 0.0, 0.0)  # x, y, z (camera frame)
    rotation_y: float = 0.0
    score: float = 1.0


def parse_kitti_labels(path: str | Path) -> list[KITTILabel]:
    """Parse a KITTI label_2 text file."""
    path = Path(path)
    if not path.exists():
        return []

    labels: list[KITTILabel] = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            label = KITTILabel(
                class_name=parts[0],
                truncated=float(parts[1]),
                occluded=int(parts[2]),
                alpha=float(parts[3]),
                bbox_2d=(float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])),
                dimensions=(float(parts[8]), float(parts[9]), float(parts[10])),
                location=(float(parts[11]), float(parts[12]), float(parts[13])),
                rotation_y=float(parts[14]),
                score=float(parts[15]) if len(parts) > 15 else 1.0,
            )
            labels.append(label)
    return labels


# --------------------------------------------------------------------------- #
# KITTI Input
# --------------------------------------------------------------------------- #

class KITTIInput(BaseInput):
    """Load frames from a KITTI dataset directory.

    Expected directory layout (object detection format):
        base_path/
            image_2/    — 000000.png, 000001.png, ...
            velodyne/   — 000000.bin, 000001.bin, ...
            calib/      — 000000.txt, 000001.txt, ...
            label_2/    — 000000.txt, ... (optional)

    For sequence-based datasets, set sequence to the sequence folder name
    and the loader will look under base_path/<sequence>/.
    """

    def __init__(self, base_path: str | Path, sequence: str | None = None):
        self.logger = get_logger(__name__)
        self.base_path = Path(base_path)
        if sequence is not None:
            self.base_path = self.base_path / sequence

        self.image_dir = self.base_path / "image_2"
        self.velodyne_dir = self.base_path / "velodyne"
        self.calib_dir = self.base_path / "calib"
        self.label_dir = self.base_path / "label_2"

        self._frame_ids: list[str] = []
        self.meta = None

    def start(self) -> None:
        """Discover available frames."""
        if not self.image_dir.exists():
            self.logger.warning("KITTI image dir not found: %s", self.image_dir)
            return

        # Collect frame IDs from image_2 directory
        image_files = sorted(self.image_dir.glob("*.png"))
        if not image_files:
            image_files = sorted(self.image_dir.glob("*.jpg"))
        self._frame_ids = [f.stem for f in image_files]
        self.logger.info(
            "KITTI dataset: %d frames at %s", len(self._frame_ids), self.base_path
        )

        # Create a simple meta object for compatibility
        @dataclass
        class _KITTIMeta:
            fps: float = 10.0
            width: int = 1242
            height: int = 375
            frame_count: int = len(self._frame_ids)

        self.meta = _KITTIMeta()

    def stop(self) -> None:
        self.logger.info("KITTI input stopped.")

    def frames(self) -> Generator[tuple[int, FramePacket], None, None]:
        """Yield (index, FramePacket) for each KITTI frame."""
        if cv2 is None:
            self.logger.error("OpenCV required for KITTI image loading")
            return

        for idx, fid in enumerate(self._frame_ids):
            # Load image
            img_path = self.image_dir / f"{fid}.png"
            if not img_path.exists():
                img_path = self.image_dir / f"{fid}.jpg"
            if not img_path.exists():
                self.logger.warning("Missing image: %s", img_path)
                continue
            image = cv2.imread(str(img_path))
            if image is None:
                self.logger.warning("Could not read image: %s", img_path)
                continue

            # Load point cloud
            pc = None
            velo_path = self.velodyne_dir / f"{fid}.bin"
            if velo_path.exists():
                raw = np.fromfile(str(velo_path), dtype=np.float32).reshape(-1, 4)
                pc = PointCloud(points=raw)
            else:
                self.logger.debug("No velodyne data for frame %s", fid)

            # Load calibration
            calib = None
            calib_path = self.calib_dir / f"{fid}.txt"
            if calib_path.exists():
                calib = KITTICalibration.from_file(calib_path)
            else:
                self.logger.debug("No calibration for frame %s", fid)

            # Load labels (optional)
            labels = None
            label_path = self.label_dir / f"{fid}.txt"
            if label_path.exists():
                labels = parse_kitti_labels(label_path)

            packet = FramePacket(
                frame=image,
                timestamp=idx / 10.0,  # KITTI ~10 Hz
                sensor_id="kitti_cam2",
                point_cloud=pc,
                calibration=calib,
                labels=labels,
            )
            yield idx, packet
