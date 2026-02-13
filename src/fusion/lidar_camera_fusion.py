"""Late-fusion module for LIDAR and camera detections.

Strategy:
  1. Project LIDAR 3D centers to 2D image coordinates.
  2. Compute BEV IoU between pairs of detections.
  3. Greedy matching by BEV IoU (threshold default 0.3).
  4. Fused position: weighted average (LIDAR 0.7, camera 0.3).
  5. Unmatched detections from either sensor pass through.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.types.pointcloud import LidarDetection3D
from src.utils.logger import get_logger


@dataclass
class FusedDetection:
    """A detection that may combine LIDAR and camera information."""

    center_3d: np.ndarray | None = None  # (3,) world coords
    dimensions_3d: np.ndarray | None = None  # (3,) l, w, h
    yaw: float = 0.0
    class_name: str = "unknown"
    confidence: float = 0.0
    source: str = "fused"  # "lidar", "camera", or "fused"
    bbox_2d: tuple[float, float, float, float] | None = None  # x1, y1, x2, y2
    lidar_det: LidarDetection3D | None = None
    camera_det: Any | None = None


class LidarCameraFusion:
    """Late-fusion module combining LIDAR 3D and camera 2D detections."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        lidar_weight: float = 0.7,
        camera_weight: float = 0.3,
    ):
        self.iou_threshold = iou_threshold
        self.lidar_weight = lidar_weight
        self.camera_weight = camera_weight
        self.logger = get_logger(__name__)

    def project_to_image(
        self, lidar_dets: list[LidarDetection3D], calibration: Any
    ) -> list[np.ndarray]:
        """Project LIDAR 3D centers to 2D image coordinates.

        Returns a list of (u, v) arrays, one per detection.
        """
        if not lidar_dets or calibration is None:
            return []

        centers = np.array([d.center for d in lidar_dets])  # (N, 3)
        uvs = calibration.project_velo_to_image(centers)  # (N, 2)
        return [uvs[i] for i in range(len(lidar_dets))]

    @staticmethod
    def compute_bev_iou(det_a: LidarDetection3D, det_b: LidarDetection3D) -> float:
        """Compute Bird's-Eye-View IoU between two 3D detections.

        Uses axis-aligned bounding box approximation in the XY plane.
        """
        # det_a bounding box in BEV
        ca = det_a.center[:2]
        la, wa = det_a.dimensions[0] / 2.0, det_a.dimensions[1] / 2.0
        a_min = ca - np.array([la, wa])
        a_max = ca + np.array([la, wa])

        # det_b bounding box in BEV
        cb = det_b.center[:2]
        lb, wb = det_b.dimensions[0] / 2.0, det_b.dimensions[1] / 2.0
        b_min = cb - np.array([lb, wb])
        b_max = cb + np.array([lb, wb])

        # Intersection
        inter_min = np.maximum(a_min, b_min)
        inter_max = np.minimum(a_max, b_max)
        inter_size = np.maximum(inter_max - inter_min, 0.0)
        inter_area = float(inter_size[0] * inter_size[1])

        # Union
        area_a = float((a_max[0] - a_min[0]) * (a_max[1] - a_min[1]))
        area_b = float((b_max[0] - b_min[0]) * (b_max[1] - b_min[1]))
        union_area = area_a + area_b - inter_area

        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    def _camera_det_to_pseudo_lidar(self, cam_det: Any) -> LidarDetection3D | None:
        """Convert a camera detection to a pseudo-LIDAR detection for BEV IoU.

        Expects camera detection to have x, y attributes (ego-frame meters)
        from the orchestrator's per-track conversion.
        """
        x_m = getattr(cam_det, "x", None)
        y_m = getattr(cam_det, "y", None)
        if x_m is None or y_m is None:
            return None

        # Approximate dimensions based on class
        class_name = getattr(cam_det, "class_name", "unknown")
        if class_name in ("car", "vehicle"):
            dims = np.array([4.5, 1.8, 1.5])
        elif class_name in ("truck", "bus"):
            dims = np.array([8.0, 2.5, 3.0])
        elif class_name in ("person", "pedestrian"):
            dims = np.array([0.6, 0.6, 1.7])
        else:
            dims = np.array([2.0, 2.0, 1.5])

        return LidarDetection3D(
            center=np.array([float(x_m), float(y_m), 0.0]),
            dimensions=dims,
            class_name=class_name,
            confidence=getattr(cam_det, "conf", getattr(cam_det, "confidence", 0.5)),
        )

    def fuse(
        self,
        camera_dets: list[Any],
        lidar_dets: list[LidarDetection3D],
        calibration: Any = None,
    ) -> list[FusedDetection]:
        """Fuse camera and LIDAR detections via greedy BEV IoU matching.

        Strategy:
          1. Convert camera detections to pseudo-LIDAR for BEV comparison.
          2. Build cost matrix of BEV IoU.
          3. Greedy assignment (highest IoU first).
          4. Fused: weighted position. Unmatched pass through.
        """
        fused: list[FusedDetection] = []
        matched_lidar = set()
        matched_camera = set()

        # Convert camera dets to pseudo-LIDAR for matching
        pseudo_lidar = []
        for cd in camera_dets:
            pseudo_lidar.append(self._camera_det_to_pseudo_lidar(cd))

        # Build IoU pairs and sort by IoU descending
        pairs: list[tuple[float, int, int]] = []
        for li, ld in enumerate(lidar_dets):
            for ci, pl in enumerate(pseudo_lidar):
                if pl is None:
                    continue
                iou = self.compute_bev_iou(ld, pl)
                if iou >= self.iou_threshold:
                    pairs.append((iou, li, ci))

        # Sort descending by IoU for greedy assignment
        pairs.sort(key=lambda x: x[0], reverse=True)

        for _iou_val, li, ci in pairs:
            if li in matched_lidar or ci in matched_camera:
                continue
            matched_lidar.add(li)
            matched_camera.add(ci)

            ld = lidar_dets[li]
            cd = camera_dets[ci]
            pl = pseudo_lidar[ci]

            # Weighted position fusion
            fused_center = self.lidar_weight * ld.center + self.camera_weight * pl.center

            # Take best class/confidence
            cam_conf = getattr(cd, "conf", getattr(cd, "confidence", 0.0))
            if ld.confidence >= cam_conf:
                class_name = ld.class_name
                confidence = ld.confidence
            else:
                class_name = getattr(cd, "class_name", "unknown")
                confidence = cam_conf

            # Get camera bbox if available
            bbox_2d = None
            if hasattr(cd, "bbox_xyxy"):
                bbox_2d = tuple(cd.bbox_xyxy)
            elif hasattr(cd, "bbox"):
                bbox_2d = tuple(cd.bbox)

            fused.append(
                FusedDetection(
                    center_3d=fused_center,
                    dimensions_3d=ld.dimensions.copy(),
                    yaw=ld.yaw,
                    class_name=class_name,
                    confidence=confidence,
                    source="fused",
                    bbox_2d=bbox_2d,
                    lidar_det=ld,
                    camera_det=cd,
                )
            )

        # Unmatched LIDAR detections pass through
        for li, ld in enumerate(lidar_dets):
            if li in matched_lidar:
                continue
            fused.append(
                FusedDetection(
                    center_3d=ld.center.copy(),
                    dimensions_3d=ld.dimensions.copy(),
                    yaw=ld.yaw,
                    class_name=ld.class_name,
                    confidence=ld.confidence,
                    source="lidar",
                    lidar_det=ld,
                )
            )

        # Unmatched camera detections pass through
        for ci, cd in enumerate(camera_dets):
            if ci in matched_camera:
                continue
            pl = pseudo_lidar[ci]
            bbox_2d = None
            if hasattr(cd, "bbox_xyxy"):
                bbox_2d = tuple(cd.bbox_xyxy)
            elif hasattr(cd, "bbox"):
                bbox_2d = tuple(cd.bbox)

            fused.append(
                FusedDetection(
                    center_3d=pl.center if pl is not None else None,
                    dimensions_3d=pl.dimensions if pl is not None else None,
                    class_name=getattr(cd, "class_name", "unknown"),
                    confidence=getattr(cd, "conf", getattr(cd, "confidence", 0.0)),
                    source="camera",
                    bbox_2d=bbox_2d,
                    camera_det=cd,
                )
            )

        self.logger.debug(
            "Fusion: %d lidar + %d camera -> %d fused (%d matched)",
            len(lidar_dets),
            len(camera_dets),
            len(fused),
            len(matched_lidar),
        )
        return fused
