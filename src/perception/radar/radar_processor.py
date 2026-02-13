"""Radar signal processing: ghost filtering, cartesian conversion, clustering."""
from __future__ import annotations

import math
from dataclasses import replace

from src.types.radar import RadarDetection, RadarFrame
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RadarProcessor:
    """Processes raw radar detections into filtered, clustered detections."""

    def __init__(
        self,
        min_rcs_dbsm: float = -10.0,
        cluster_distance_m: float = 2.0,
        ghost_azimuth_tolerance_deg: float = 5.0,
    ):
        self.min_rcs_dbsm = min_rcs_dbsm
        self.cluster_distance_m = cluster_distance_m
        self.ghost_azimuth_tolerance_deg = ghost_azimuth_tolerance_deg

    def ghost_filter(self, dets: list[RadarDetection]) -> list[RadarDetection]:
        """Remove low-RCS and multi-bounce ghost detections.

        Multi-bounce ghosts appear at 2x/3x the true range at similar azimuth.
        """
        # Remove low-RCS
        filtered = [d for d in dets if d.rcs_dbsm >= self.min_rcs_dbsm]

        # Remove multi-bounce ghosts (2x, 3x range duplicates)
        kept: list[RadarDetection] = []
        for det in filtered:
            is_ghost = False
            for ref in filtered:
                if ref is det:
                    continue
                az_diff = abs(det.azimuth_deg - ref.azimuth_deg)
                if az_diff > self.ghost_azimuth_tolerance_deg:
                    continue
                ratio = det.range_m / max(ref.range_m, 1e-6)
                if abs(ratio - 2.0) < 0.15 or abs(ratio - 3.0) < 0.15:
                    is_ghost = True
                    break
            if not is_ghost:
                kept.append(det)
        return kept

    @staticmethod
    def to_cartesian(det: RadarDetection) -> RadarDetection:
        """Convert polar (range, azimuth) to cartesian (x, y)."""
        az_rad = math.radians(det.azimuth_deg)
        x = det.range_m * math.sin(az_rad)
        y = det.range_m * math.cos(az_rad)
        return replace(det, x_m=x, y_m=y)

    def cluster_detections(self, dets: list[RadarDetection]) -> list[RadarDetection]:
        """Merge nearby detections into centroids using simple greedy clustering."""
        if not dets:
            return []

        used = [False] * len(dets)
        clusters: list[RadarDetection] = []

        for i, det_i in enumerate(dets):
            if used[i]:
                continue
            group = [det_i]
            used[i] = True
            for j in range(i + 1, len(dets)):
                if used[j]:
                    continue
                dx = det_i.x_m - dets[j].x_m
                dy = det_i.y_m - dets[j].y_m
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < self.cluster_distance_m:
                    group.append(dets[j])
                    used[j] = True

            n = len(group)
            cx = sum(d.x_m for d in group) / n
            cy = sum(d.y_m for d in group) / n
            avg_vel = sum(d.velocity_mps for d in group) / n
            avg_rcs = sum(d.rcs_dbsm for d in group) / n
            avg_range = math.sqrt(cx * cx + cy * cy)
            avg_az = math.degrees(math.atan2(cx, cy)) if avg_range > 0 else 0.0

            clusters.append(RadarDetection(
                range_m=avg_range,
                azimuth_deg=avg_az,
                velocity_mps=avg_vel,
                rcs_dbsm=avg_rcs,
                x_m=cx,
                y_m=cy,
            ))
        return clusters

    def process(self, radar_frame: RadarFrame) -> list[RadarDetection]:
        """Full processing pipeline: ghost filter → cartesian → cluster."""
        dets = self.ghost_filter(radar_frame.detections)
        dets = [self.to_cartesian(d) for d in dets]
        dets = self.cluster_detections(dets)
        logger.debug("Radar: %d raw → %d processed", len(radar_frame.detections), len(dets))
        return dets
