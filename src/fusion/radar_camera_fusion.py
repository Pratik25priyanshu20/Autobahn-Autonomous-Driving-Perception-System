"""Radar-camera fusion: project radar detections to image, match to tracks."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.types.radar import RadarDetection
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RadarCameraFusion:
    """Fuses radar detections with camera-based tracks."""

    def __init__(
        self,
        match_threshold_px: float = 100.0,
        camera_matrix: np.ndarray | None = None,
    ):
        self.match_threshold_px = match_threshold_px
        if camera_matrix is not None:
            self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        else:
            # Default pinhole camera intrinsics (fx, fy, cx, cy)
            self.camera_matrix = np.array([
                [800.0, 0.0, 640.0],
                [0.0, 800.0, 360.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)

    @staticmethod
    def project_radar_to_image(
        det: RadarDetection,
        cam_matrix: np.ndarray,
    ) -> tuple[float, float]:
        """Project a radar detection (world x, y) into image (u, v).

        Assumes radar x_m is lateral, y_m is forward depth.
        Camera convention: Z = forward (y_m), X = right (x_m), Y = down (0).
        """
        x_cam = det.x_m
        z_cam = max(det.y_m, 0.01)  # forward distance as depth
        y_cam = 0.0  # assume same height

        fx = cam_matrix[0, 0]
        fy = cam_matrix[1, 1]
        cx = cam_matrix[0, 2]
        cy = cam_matrix[1, 2]

        u = fx * (x_cam / z_cam) + cx
        v = fy * (y_cam / z_cam) + cy
        return float(u), float(v)

    def match_radar_to_tracks(
        self,
        radar_dets: list[RadarDetection],
        tracks: list[Any],
        cam_matrix: np.ndarray,
    ) -> dict[int, RadarDetection]:
        """Greedy nearest-neighbour matching in image space.

        Returns dict mapping track_id → matched RadarDetection.
        """
        if not radar_dets or not tracks:
            return {}

        # Project all radar detections to image
        radar_uv = []
        for det in radar_dets:
            u, v = self.project_radar_to_image(det, cam_matrix)
            radar_uv.append((u, v, det))

        matches: dict[int, RadarDetection] = {}
        used_radar: set[int] = set()

        for trk in tracks:
            bbox = getattr(trk, "bbox_xyxy", None)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            trk_cx = (x1 + x2) / 2.0
            trk_cy = (y1 + y2) / 2.0
            tid = getattr(trk, "track_id", None)
            if tid is None:
                continue

            best_dist = float("inf")
            best_idx = -1
            for idx, (u, v, _det) in enumerate(radar_uv):
                if idx in used_radar:
                    continue
                dist = math.sqrt((u - trk_cx) ** 2 + (v - trk_cy) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            if best_idx >= 0 and best_dist <= self.match_threshold_px:
                used_radar.add(best_idx)
                matches[tid] = radar_uv[best_idx][2]

        return matches

    @staticmethod
    def enrich_tracks(
        tracks: list[Any],
        matches: dict[int, RadarDetection],
    ) -> None:
        """Set radar_velocity_mps and radar_range_m on matched tracks."""
        for trk in tracks:
            tid = getattr(trk, "track_id", None)
            if tid is not None and tid in matches:
                det = matches[tid]
                trk.radar_velocity_mps = det.velocity_mps
                trk.radar_range_m = det.range_m

    def fuse(
        self,
        tracks: list[Any],
        radar_dets: list[RadarDetection],
        cam_matrix: np.ndarray | None = None,
    ) -> list[Any]:
        """Main entry: match radar to tracks and enrich with radar data."""
        mtx = cam_matrix if cam_matrix is not None else self.camera_matrix
        matches = self.match_radar_to_tracks(radar_dets, tracks, mtx)
        self.enrich_tracks(tracks, matches)
        logger.debug("Radar-camera fusion: %d/%d tracks matched", len(matches), len(tracks))
        return tracks
