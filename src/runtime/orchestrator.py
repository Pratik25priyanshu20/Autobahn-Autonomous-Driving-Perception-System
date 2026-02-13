from __future__ import annotations

import time
from collections import deque
from typing import Any

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from src.adas.ttc_filter import TTCFilter
from src.fusion.world_model import DrivableArea, RuntimeStats, WorldModel
from src.perception.detection.yolo import YOLODetector
from src.perception.lanes.lane_detector import CannyHoughLaneDetector
from src.perception.segmentation.deeplabv3_segmenter import DeepLabV3Segmenter
from src.perception.segmentation.postprocess import extract_drivable_area
from src.perception.tracking.deepsort_tracker import DeepSORTTracker
from src.runtime.health_monitor import HealthMonitor
from src.safety.fcw import fcw_state
from src.safety.safety_manager import SafetyManager
from src.safety.ttc import compute_ttc
from src.utils.timing import FPSMeter


class Orchestrator:
    """Central frame-processing engine.

    Coordinates: detection -> tracking -> lanes -> segmentation -> depth
    -> weather -> safety (FCW/LDW/BSD) -> world model construction.
    All new features are gated by config flags (default: disabled).
    """

    def __init__(self, cfg: dict[str, Any], logger):
        self.cfg = cfg
        self.logger = logger
        self.fps_meter = FPSMeter(smoothing=float(cfg.get("performance", {}).get("fps_smoothing", 0.9)))

        # --- Health Monitor (Phase 0.2) ---
        perf_cfg = cfg.get("performance", {})
        self.health_monitor = HealthMonitor({
            "watchdog_ms": perf_cfg.get("watchdog_ms", 100),
            "degraded_after_misses": perf_cfg.get("degraded_after_misses", 3),
        })
        self._base_tracking_interval: int = int(cfg.get("tracking", {}).get("interval", 2))

        # --- Detector (Phase 1.1: config-driven model switch) ---
        det_cfg = cfg.get("perception", {})
        runtime_backend = det_cfg.get("runtime", "pytorch")
        if runtime_backend == "onnx":
            from src.perception.detection.onnx_detector import ONNXDetector
            self.detector = ONNXDetector(
                onnx_path=det_cfg.get("onnx_path", "yolov8n.onnx"),
                provider=det_cfg.get("onnx_provider", "cpu"),
            )
        elif runtime_backend == "tensorrt":
            from src.perception.detection.onnx_detector import ONNXDetector
            self.detector = ONNXDetector(
                onnx_path=det_cfg.get("onnx_path", "yolov8n.onnx"),
                provider="tensorrt",
            )
        else:
            self.detector = YOLODetector(
                model_name=det_cfg.get("detector_model", "yolov8n.pt"),
                device=det_cfg.get("device", None),
            )

        # --- Confidence calibration (Phase 0.5) ---
        self.calibrator = None
        cal_cfg = det_cfg.get("confidence_calibration", {})
        if cal_cfg.get("enabled", False):
            from src.perception.calibration import ConfidenceCalibrator
            self.calibrator = ConfidenceCalibrator(temperature=cal_cfg.get("temperature", 1.0))

        # --- Tracker (Phase 2.1: config-driven switch) ---
        tracker_type = cfg.get("tracking", {}).get("type", "deepsort")
        self.tracking_enabled = bool(cfg.get("tracking", {}).get("enabled", True))
        self.tracking_interval = self._base_tracking_interval
        if tracker_type == "bytetrack":
            from src.perception.tracking.bytetrack_tracker import ByteTrackTracker
            self.tracker = ByteTrackTracker()
        else:
            self.tracker = DeepSORTTracker()
        self._last_tracks: list[Any] = []
        self._last_trajectories: dict[int, list[Any]] = {}

        # --- Lane detector (Phase 1.2: config-driven backend switch) ---
        self.lane_enabled = bool(cfg.get("lane", {}).get("enabled", True))
        lane_backend = cfg.get("lane", {}).get("backend", "canny_hough")
        if self.lane_enabled:
            if lane_backend == "ufldv2":
                from src.perception.lanes.ufld_detector import UFLDv2LaneDetector
                self.lane_detector = UFLDv2LaneDetector(
                    model_path=cfg.get("lane", {}).get("model_path"),
                    device=cfg.get("lane", {}).get("device", "cpu"),
                )
            else:
                self.lane_detector = CannyHoughLaneDetector()
        else:
            self.lane_detector = None
        lane_cfg = cfg.get("lane", {})
        self.lane_min_conf = float(lane_cfg.get("min_confidence", 0.55))
        self.lane_stability_px = float(lane_cfg.get("stability_px", 35))
        self.lane_min_stable_frames = int(lane_cfg.get("min_stable_frames", 5))
        self._lane_center_hist = deque(maxlen=max(20, self.lane_min_stable_frames * 3))

        # --- LDW ---
        ldw_cfg = cfg.get("ldw", {})
        self.ldw_enabled = bool(ldw_cfg.get("enabled", True))
        self.ldw_threshold_px = float(ldw_cfg.get("offset_threshold_px", 80))
        self.ldw_persistence = int(ldw_cfg.get("persistence_frames", 6))
        self._offset_hist = deque(maxlen=max(10, self.ldw_persistence * 2))

        # --- Safety ---
        self.safety = SafetyManager()
        fcw_cfg = cfg.get("fcw", {})
        self.fcw_enabled = bool(fcw_cfg.get("enabled", True))
        self.fcw_px_to_m = float(fcw_cfg.get("px_to_m", 0.05))
        self.fcw_ego_y_ratio = float(fcw_cfg.get("ego_y_ratio", 0.92))
        self.fcw_min_rel_speed = float(fcw_cfg.get("min_rel_speed_mps", 0.3))
        self.fcw_pre_distance_px = float(fcw_cfg.get("pre_distance_px", 220))
        self.fcw_ttc_caution = float(fcw_cfg.get("ttc_caution_s", 4.0))
        self.fcw_ttc_warning = float(fcw_cfg.get("ttc_warning_s", 2.5))
        self.fcw_ttc_critical = float(fcw_cfg.get("ttc_critical_s", 1.5))
        self.lane_width_m = 3.5
        self._track_hist: dict[Any, deque] = {}
        self._track_hist_len = 10
        self.ttc_filter = TTCFilter(alpha=0.3, min_persist_frames=3)

        # --- Segmentation ---
        seg_cfg = cfg.get("segmentation", {})
        self.segmentation_enabled = bool(seg_cfg.get("enabled", False))
        self.segmenter = DeepLabV3Segmenter(device=seg_cfg.get("device")) if self.segmentation_enabled else None

        # --- Depth estimation (Phase 1.3) ---
        self.depth_estimator = None
        depth_cfg = cfg.get("depth", {})
        if depth_cfg.get("enabled", False):
            depth_backend = depth_cfg.get("backend", "midas")
            depth_device = depth_cfg.get("device", "cpu")
            if depth_backend == "depth_anything":
                from src.perception.depth.depth_anything import DepthAnythingV2
                self.depth_estimator = DepthAnythingV2(device=depth_device)
            else:
                from src.perception.depth.midas_depth import MiDASDepth
                self.depth_estimator = MiDASDepth(device=depth_device)

        # --- Mono3D (Phase 1.4) ---
        self.mono3d = None
        if det_cfg.get("mono3d", {}).get("enabled", False):
            from src.perception.detection.mono3d_stub import Mono3DDetector
            self.mono3d = Mono3DDetector()

        # --- Kalman filter (Phase 2.2) ---
        self.kalman_manager = None
        fusion_cfg = cfg.get("fusion", cfg.get("uncertainty", {}))
        kalman_enabled = bool(cfg.get("tracking", {}).get("kalman", True))
        if kalman_enabled:
            from src.fusion.kalman_tracker import KalmanTrackManager
            self.kalman_manager = KalmanTrackManager(
                process_noise=float(fusion_cfg.get("process_noise", fusion_cfg.get("uncertainty", {}).get("process_noise", 0.5))),
                measurement_noise=float(fusion_cfg.get("measurement_noise", fusion_cfg.get("uncertainty", {}).get("measurement_noise", 1.0))),
            )

        # --- Temporal prediction (Phase 2.3) ---
        self.temporal_predictor = None
        if self.kalman_manager is not None:
            from src.fusion.temporal_predictor import TemporalPredictor
            self.temporal_predictor = TemporalPredictor()

        # --- Weather/visibility (Phase 3.3) ---
        self.visibility_detector = None
        weather_cfg = cfg.get("weather", {})
        if weather_cfg.get("enabled", False):
            from src.perception.weather.visibility_detector import VisibilityDetector
            self.visibility_detector = VisibilityDetector(
                dark_threshold=weather_cfg.get("dark_threshold", 60.0),
                glare_threshold=weather_cfg.get("glare_threshold", 210.0),
                fog_contrast_threshold=weather_cfg.get("fog_contrast_threshold", 30.0),
            )

        # --- BSD/RCTA (Phase 3.2) ---
        self.bsd_detector = None
        safety_yaml = cfg.get("bsd", {})
        if safety_yaml.get("enabled", False):
            from src.safety.bsd_rcta import BlindSpotDetector
            self.bsd_detector = BlindSpotDetector(
                blind_spot_x_min=safety_yaml.get("blind_spot_x_range", [1.5, 4.0])[0] if isinstance(safety_yaml.get("blind_spot_x_range"), list) else 1.5,
                blind_spot_x_max=safety_yaml.get("blind_spot_x_range", [1.5, 4.0])[1] if isinstance(safety_yaml.get("blind_spot_x_range"), list) else 4.0,
                ttc_warn_s=safety_yaml.get("ttc_warn_s", 3.0),
            )

        # --- Occupancy grid (Phase 3.1) ---
        self.occupancy_builder = None
        occ_cfg = cfg.get("occupancy_grid", {})
        if occ_cfg.get("enabled", False):
            from src.safety.occupancy_grid import OccupancyGridBuilder
            self.occupancy_builder = OccupancyGridBuilder(
                resolution_m=occ_cfg.get("resolution_m", 0.2),
                max_range_m=occ_cfg.get("max_range_m", 40.0),
            )

        # --- Parallel executor (Phase 4.1) ---
        self.parallel_executor = None
        parallel_cfg = perf_cfg.get("parallel", {})
        if parallel_cfg.get("enabled", False):
            from src.runtime.parallel_executor import ParallelStageExecutor
            self.parallel_executor = ParallelStageExecutor(
                max_workers=int(parallel_cfg.get("max_workers", 4))
            )

        # --- Adaptive skip (Phase 4.3) ---
        self.adaptive_skip = bool(perf_cfg.get("adaptive_skip", False))

        # --- Control (Phase 6.2) ---
        self.controller = None
        ctrl_cfg = cfg.get("control", {})
        if ctrl_cfg.get("enabled", False):
            ctrl_type = ctrl_cfg.get("type", "pure_pursuit")
            if ctrl_type == "mpc":
                from src.control.mpc import MPCController
                self.controller = MPCController()
            else:
                from src.control.pure_pursuit import PurePursuitController
                self.controller = PurePursuitController()

        # --- ISO 26262: ASIL classifier (Phase 4) ---
        self.asil_classifier = None
        safety_yaml = cfg.get("safety", {})
        asil_cfg = safety_yaml.get("asil", {})
        if asil_cfg.get("enabled", False):
            from src.safety.asil_classifier import ASILClassifier
            self.asil_classifier = ASILClassifier()

        # --- ISO 26262: Plausibility checker (Phase 4) ---
        self.plausibility_checker = None
        plaus_cfg = safety_yaml.get("plausibility", {})
        if plaus_cfg.get("enabled", False):
            from src.safety.plausibility_checker import PlausibilityChecker
            self.plausibility_checker = PlausibilityChecker(
                max_velocity_kmh=float(plaus_cfg.get("max_velocity_kmh", 200.0)),
                max_position_jump_m=float(plaus_cfg.get("max_position_jump_m", 10.0)),
                max_bbox_overlap=float(plaus_cfg.get("max_bbox_overlap", 0.8)),
                max_detection_count=int(plaus_cfg.get("max_detection_count", 100)),
            )
        self._prev_tracks_for_plausibility: list[Any] = []

        # --- ISO 26262: DTC logger (Phase 4) ---
        self.dtc_logger = None
        dtc_cfg = safety_yaml.get("dtc", {})
        if dtc_cfg.get("enabled", False):
            from src.safety.dtc_logger import DTCLogger
            self.dtc_logger = DTCLogger(
                output_dir=dtc_cfg.get("output_dir", "results"),
            )

        # --- ISO 26262: Redundant detector (Phase 4) ---
        self.redundant_detector = None
        red_cfg = safety_yaml.get("redundant_detection", {})
        if red_cfg.get("enabled", False):
            from src.safety.redundant_detector import RedundantDetector
            secondary = YOLODetector(
                model_name=red_cfg.get("secondary_model", "yolov8s.pt"),
                device=det_cfg.get("device", None),
            )
            self.redundant_detector = RedundantDetector(
                primary_detector=self.detector,
                secondary_detector=secondary,
                iou_threshold=float(red_cfg.get("iou_threshold", 0.5)),
                min_agreement=float(red_cfg.get("min_agreement", 0.6)),
            )

        self._prev_fcw = "NORMAL"
        self._prev_positions: dict[Any, tuple] = {}

        resize_cfg = cfg.get("video", {}).get("resize", {})
        self.resize_enabled = bool(resize_cfg.get("enabled", False))
        self.resize_w = int(resize_cfg.get("width", 1280))
        self.resize_h = int(resize_cfg.get("height", 720))

        # --- Sensor health monitor (Task 4) ---
        self.sensor_health_monitor = None
        safety_yaml = cfg.get("safety", {})
        sh_cfg = safety_yaml.get("sensor_health", cfg.get("sensor_health", {}))
        if sh_cfg.get("enabled", False):
            from src.safety.sensor_health import SensorHealthMonitor
            br = sh_cfg.get("brightness_range", [40, 220])
            self.sensor_health_monitor = SensorHealthMonitor(
                brightness_range=tuple(br) if isinstance(br, list) else (40.0, 220.0),
                blur_threshold=float(sh_cfg.get("blur_threshold", 100.0)),
                expected_lidar_points=int(sh_cfg.get("expected_lidar_points", 10000)),
                health_threshold=float(sh_cfg.get("health_threshold", 0.5)),
            )

        # --- Saliency / explainability (Task 6) ---
        self.saliency_explainer = None
        explain_cfg = cfg.get("explainability", {})
        if explain_cfg.get("enabled", False):
            from src.perception.explainability.grad_cam import GradCAMExplainer
            self.saliency_explainer = GradCAMExplainer(
                model=getattr(self.detector, "model", None),
                target_layer=explain_cfg.get("target_layer", "model.model.9"),
                num_detections=int(explain_cfg.get("num_detections", 5)),
            )

        # --- Interaction model (Task 7) ---
        self.interaction_model = None
        interact_cfg = cfg.get("interaction", {})
        if interact_cfg.get("enabled", False):
            from src.prediction.interaction_model import InteractionModel
            self.interaction_model = InteractionModel(
                min_gap_s=float(interact_cfg.get("min_gap_s", 3.0)),
                safe_following_s=float(interact_cfg.get("safe_following_distance_s", 2.0)),
                lane_width_m=float(interact_cfg.get("lane_width_m", 3.5)),
                cut_in_lateral_threshold=float(interact_cfg.get("cut_in_lateral_threshold", 0.5)),
            )

        # --- Radar processor (Task 1) ---
        self.radar_processor = None
        radar_cfg = cfg.get("radar", {})
        if radar_cfg.get("enabled", False):
            from src.perception.radar.radar_processor import RadarProcessor
            self.radar_processor = RadarProcessor(
                min_rcs_dbsm=float(radar_cfg.get("min_rcs_dbsm", -10.0)),
                cluster_distance_m=float(radar_cfg.get("cluster_distance_m", 2.0)),
            )

        # --- Radar-camera fusion (Task 1) ---
        self.radar_camera_fusion = None
        rf_cfg = cfg.get("radar_fusion", {})
        if rf_cfg.get("enabled", False):
            import numpy as _np

            from src.fusion.radar_camera_fusion import RadarCameraFusion
            cam_mtx = rf_cfg.get("camera_matrix")
            self.radar_camera_fusion = RadarCameraFusion(
                match_threshold_px=float(rf_cfg.get("match_threshold_px", 100.0)),
                camera_matrix=_np.array(cam_mtx) if cam_mtx is not None else None,
            )

        # --- LIDAR pipeline (Task 3) ---
        self.lidar_processor = None
        lidar_cfg = cfg.get("lidar", {})
        if lidar_cfg.get("enabled", False):
            from src.perception.lidar.point_cloud_processor import PointCloudProcessor
            self.lidar_processor = PointCloudProcessor(
                max_range=float(lidar_cfg.get("max_range_m", 80.0)),
                min_range=float(lidar_cfg.get("min_range_m", 1.0)),
                voxel_size=float(lidar_cfg.get("voxel_size", 0.1)),
                ground_method=lidar_cfg.get("ground_removal", "ransac"),
                cluster_eps=float(lidar_cfg.get("cluster_eps", 0.8)),
                cluster_min_samples=int(lidar_cfg.get("cluster_min_samples", 10)),
            )

        # --- BEV encoder (Task 3) ---
        self.bev_encoder = None
        if lidar_cfg.get("enabled", False):
            from src.perception.lidar.bev_encoder import BEVEncoder
            self.bev_encoder = BEVEncoder()

        # --- LIDAR-camera fusion (Task 3) ---
        self.lidar_camera_fusion = None
        fusion_lidar_cfg = cfg.get("lidar_fusion", {})
        if fusion_lidar_cfg.get("enabled", False):
            from src.fusion.lidar_camera_fusion import LidarCameraFusion
            self.lidar_camera_fusion = LidarCameraFusion(
                iou_threshold=float(fusion_lidar_cfg.get("iou_threshold", 0.3)),
                lidar_weight=float(fusion_lidar_cfg.get("lidar_weight", 0.7)),
                camera_weight=float(fusion_lidar_cfg.get("camera_weight", 0.3)),
            )

    # ── Main frame pipeline ─────────────────────────────────────────

    def process_frame(self, frame_id: int, frame: Any, packet: Any = None) -> WorldModel:
        if cv2 is None:
            raise ImportError("opencv-python is required for orchestrator processing")

        warnings: list[str] = []
        frame_start = time.perf_counter()

        # Stage 1: Preprocessing
        frame, preprocess_ms = self._preprocess(frame)

        # Stage 2: Sensor health
        sensor_health, sensor_health_ms = self._assess_sensors(frame, packet, warnings)

        # Stage 3: Adaptive degradation
        degraded = self.health_monitor.degraded()
        if degraded and self.adaptive_skip:
            self.tracking_interval = self._base_tracking_interval * 2
        else:
            self.tracking_interval = self._base_tracking_interval

        # Stage 4: Detection + lane perception
        perc = self._run_detection_stage(frame, frame_id, degraded)
        detections = perc["detections"]
        lanes = perc["lanes"]
        detect_ms = perc["detect_ms"]
        lane_ms = perc["lane_ms"]
        seg_ms = perc["seg_ms"]
        depth_ms = perc["depth_ms"]
        weather_ms = perc["weather_ms"]

        # Stage 5: Saliency
        saliency_map, saliency_ms = self._run_saliency_stage(frame, detections)

        # Stage 6: Tracking
        tracks, trajectories, track_ms = self._run_tracking_update(frame, frame_id, detections)

        # FPS + info warnings
        fps = self.fps_meter.tick()
        target = float(self.cfg.get("performance", {}).get("target_fps", 20))
        if fps < target * 0.6:
            warnings.append("FPS low (degraded mode active)" if degraded else "FPS low")
        if self.tracking_enabled and self.tracking_interval > 0 and frame_id % self.tracking_interval == 0:
            warnings.append(f"INFO: tracking update | {len(tracks)} tracks")
        else:
            warnings.append(f"INFO: tracking reused | {len(tracks)} tracks")
        warnings.append(f"INFO: {len(detections)} detections")

        # Stage 7: Lane analysis + LDW
        ldw_allowed, lane_departure = self._analyze_lanes(lanes, warnings)

        # Stage 8: Segmentation, depth, weather
        env = self._run_environment_stages(frame, degraded, warnings)
        drivable = env["drivable"]
        seg_conf = env["seg_conf"]
        depth_map = env["depth_map"]
        seg_ms += env["seg_ms"]
        depth_ms += env["depth_ms"]
        weather_ms += env["weather_ms"]

        # Stage 9: Ego-frame conversion + TTC
        alive_ids = self._compute_ego_positions(tracks, frame)

        # Stage 10: Kalman + temporal prediction
        predictions, predictions_topk = self._run_state_estimation(tracks, alive_ids)

        # Stage 11: Interaction model
        interactions, interaction_ms = self._evaluate_interactions(tracks, warnings)

        # Stage 12: BSD/RCTA
        bsd_warnings = self._evaluate_bsd(tracks, warnings)

        # Stage 13: FCW
        fcw_result = self._compute_fcw(tracks, frame, lanes, ldw_allowed, warnings)

        # Stage 14: Occupancy grid
        occupancy = None
        if self.occupancy_builder is not None and depth_map is not None:
            occupancy = self.occupancy_builder.build(depth_map, drivable)

        # Stage 15: LIDAR pipeline
        lidar = self._process_lidar(packet, tracks, warnings)

        # Stage 16: Radar pipeline
        radar = self._process_radar(packet, tracks, warnings)

        # --- Build WorldModel ---
        stages = {
            "preprocess": preprocess_ms,
            "detection": detect_ms,
            "tracking": track_ms,
            "lane": lane_ms,
            "segmentation": seg_ms,
            "depth": depth_ms,
            "weather": weather_ms,
            "fcw": 0.0,
            "lidar": lidar["lidar_ms"],
            "bev": lidar["bev_ms"],
            "fusion": lidar["fusion_ms"],
            "sensor_health": sensor_health_ms,
            "saliency": saliency_ms,
            "interaction": interaction_ms,
            "radar": radar["radar_ms"],
            "radar_fusion": radar["fusion_ms"],
        }

        wm = WorldModel(
            frame_id=frame_id,
            frame=frame,
            detections=detections,
            tracks=tracks,
            trajectories=trajectories,
            lanes=lanes,
            drivable_area=DrivableArea(mask=drivable, confidence=seg_conf or 0.0) if drivable is not None else DrivableArea(),
            fcw=fcw_result["fcw"],
            fcw_pre=fcw_result["fcw_pre"],
            safety=self._build_safety(lanes, fcw_result["fcw"], fcw_result["fcw_pre"], bsd_warnings),
            warnings=warnings,
            runtime=RuntimeStats(fps=fps, stages_ms=stages),
            depth_map=depth_map,
            predictions=predictions,
            predictions_topk=predictions_topk,
            occupancy=occupancy,
            lidar_detections=lidar["detections"],
            fused_detections=lidar["fused"],
            point_cloud=lidar["point_cloud"],
            bev_grid=lidar["bev_grid"],
            radar_detections=radar["detections"],
            radar_fused=radar["fused"],
            sensor_health=sensor_health,
            saliency_map=saliency_map,
            interactions=interactions,
        )

        # Stage 17: Post-safety (plausibility + telemetry)
        self._run_post_safety(wm, tracks, detections, frame_id, fcw_result, warnings)

        # Stage 18: Controller
        if self.controller is not None:
            wm.control = self.controller.plan(wm)

        wm.snapshot()

        # --- Health monitor check ---
        frame_ms = (time.perf_counter() - frame_start) * 1000.0
        self.health_monitor.check_latency(frame_ms)

        assert wm.frame is not None
        assert wm.tracks is not None
        if wm.lanes and wm.lanes.get("ego_offset_px") is not None:
            assert abs(wm.lanes.get("ego_offset_px", 0.0)) < 1000
        if frame_id % 30 == 0:
            self.logger.info("[WORLD] %s", wm.summary())
            if drivable is not None:
                self.logger.info("[WORLD] drivable_area=yes pixels=%d", int(drivable.sum()))
        return wm

    # ── Extracted stage methods ──────────────────────────────────────

    def _preprocess(self, frame: Any) -> tuple[Any, float]:
        """Resize frame if configured. Returns (frame, elapsed_ms)."""
        t0 = cv2.getTickCount()
        if self.resize_enabled:
            frame = cv2.resize(frame, (self.resize_w, self.resize_h), interpolation=cv2.INTER_LINEAR)
        ms = (cv2.getTickCount() - t0) * 1000.0 / cv2.getTickFrequency()
        return frame, ms

    def _assess_sensors(self, frame: Any, packet: Any, warnings: list[str]) -> tuple[dict[str, float], float]:
        """Assess sensor health (Task 4). Returns (health_dict, elapsed_ms)."""
        if self.sensor_health_monitor is None:
            return {}, 0.0
        t = cv2.getTickCount()
        health: dict[str, float] = {}
        cam_h = self.sensor_health_monitor.assess_camera(frame)
        health["camera"] = cam_h.score
        pc = getattr(packet, "point_cloud", None) if packet is not None else None
        if pc is not None:
            health["lidar"] = self.sensor_health_monitor.assess_lidar(pc).score
        rf = getattr(packet, "radar_frame", None) if packet is not None else None
        if rf is not None:
            health["radar"] = self.sensor_health_monitor.assess_radar(rf).score
        ms = (cv2.getTickCount() - t) * 1000.0 / cv2.getTickFrequency()
        if self.sensor_health_monitor.degraded():
            warnings.append("WARNING: Sensor degradation detected")
        return health, ms

    def _run_detection_stage(self, frame: Any, frame_id: int, degraded: bool) -> dict[str, Any]:
        """Run detection + lane perception (parallel or sequential). Returns result dict."""
        if self.parallel_executor is not None and not degraded:
            results = self._run_parallel(frame, frame_id)
            return {
                "detections": results.get("detection", []),
                "lanes": results.get("lane", {}),
                "detect_ms": 0.0,
                "lane_ms": 0.0,
                "seg_ms": 0.0,
                "depth_ms": 0.0,
                "weather_ms": 0.0,
            }

        # Sequential: detection
        t1 = cv2.getTickCount()
        detections = self.detector.infer(frame, conf_thres=self.cfg.get("perception", {}).get("conf_thres", 0.25))
        detect_ms = (cv2.getTickCount() - t1) * 1000.0 / cv2.getTickFrequency()

        # Confidence calibration (Phase 0.5)
        if self.calibrator is not None:
            detections = self.calibrator.calibrate(detections)

        # Lane detection
        lanes: dict[str, Any] = {}
        lane_ms = 0.0
        if self.lane_detector is not None:
            t3 = cv2.getTickCount()
            lanes = self.lane_detector.infer(frame)
            lane_ms = (cv2.getTickCount() - t3) * 1000.0 / cv2.getTickFrequency()

        return {
            "detections": detections,
            "lanes": lanes,
            "detect_ms": detect_ms,
            "lane_ms": lane_ms,
            "seg_ms": 0.0,
            "depth_ms": 0.0,
            "weather_ms": 0.0,
        }

    def _run_saliency_stage(self, frame: Any, detections: list) -> tuple[Any, float]:
        """Run saliency/explainability (Task 6). Returns (saliency_map, elapsed_ms)."""
        if self.saliency_explainer is None or not detections:
            return None, 0.0
        t = cv2.getTickCount()
        saliency_map = self.saliency_explainer.explain(frame, detections)
        ms = (cv2.getTickCount() - t) * 1000.0 / cv2.getTickFrequency()
        return saliency_map, ms

    def _run_tracking_update(self, frame: Any, frame_id: int, detections: list) -> tuple[list, dict, float]:
        """Run tracking with temporal decimation. Returns (tracks, trajectories, elapsed_ms)."""
        if self.tracking_enabled and self.tracking_interval > 0 and frame_id % self.tracking_interval == 0:
            t = cv2.getTickCount()
            tracks, trajectories = self.tracker.update(frame, detections)
            track_ms = (cv2.getTickCount() - t) * 1000.0 / cv2.getTickFrequency()
            self._last_tracks = tracks
            self._last_trajectories = trajectories
            return tracks, trajectories, track_ms
        return self._last_tracks, self._last_trajectories, 0.0

    def _analyze_lanes(self, lanes: dict, warnings: list[str]) -> tuple[bool, str | None]:
        """Apply lane confidence gating, stability analysis, and LDW. Returns (ldw_allowed, lane_departure)."""
        ego_offset = lanes.get("ego_offset_px") if lanes else None
        ldw_allowed = False
        lane_departure = None

        if lanes:
            conf = float(lanes.get("lane_confidence", 0.0))
            center_x = lanes.get("lane_center_x")
            if center_x is not None:
                self._lane_center_hist.append(float(center_x))
            stable = False
            jitter = None
            if len(self._lane_center_hist) >= self.lane_min_stable_frames:
                recent = list(self._lane_center_hist)[-self.lane_min_stable_frames:]
                jitter = max(recent) - min(recent)
                stable = jitter <= self.lane_stability_px
            lanes["lane_center_jitter_px"] = float(jitter) if jitter is not None else None
            lanes["lane_stable"] = stable
            lanes["lane_confidence"] = conf
            ldw_allowed = (conf >= self.lane_min_conf) and stable
            lanes["ldw_allowed"] = ldw_allowed

        if self.ldw_enabled and ego_offset is not None and ldw_allowed:
            self._offset_hist.append(float(ego_offset))
            thr = self.ldw_threshold_px
            recent = list(self._offset_hist)[-self.ldw_persistence:]
            if len(recent) == self.ldw_persistence:
                if all(x > thr for x in recent):
                    lane_departure = "RIGHT"
                elif all(x < -thr for x in recent):
                    lane_departure = "LEFT"
            lanes["lane_departure"] = lane_departure
            lanes["ldw_threshold_px"] = thr

        if ego_offset is not None:
            warnings.append(f"INFO: ego_offset={ego_offset:+.1f}px")
        if lane_departure is not None:
            warnings.append(f"WARNING: LANE DEPARTURE {lane_departure}")

        return ldw_allowed, lane_departure

    def _run_environment_stages(self, frame: Any, degraded: bool, warnings: list[str]) -> dict[str, Any]:
        """Run segmentation, depth, and weather stages. Returns result dict."""
        result: dict[str, Any] = {"drivable": None, "seg_conf": None, "depth_map": None, "seg_ms": 0.0, "depth_ms": 0.0, "weather_ms": 0.0}

        # Segmentation
        if self.segmenter is not None and not (degraded and self.adaptive_skip):
            t = cv2.getTickCount()
            seg_out = self.segmenter.infer(frame)
            result["seg_ms"] = (cv2.getTickCount() - t) * 1000.0 / cv2.getTickFrequency()
            result["drivable"] = extract_drivable_area(seg_out["mask"])
            result["seg_conf"] = seg_out.get("confidence", 0.0)

        # Depth estimation (Phase 1.3)
        if self.depth_estimator is not None and not (degraded and self.adaptive_skip):
            t = cv2.getTickCount()
            depth_out = self.depth_estimator.infer(frame)
            result["depth_ms"] = (cv2.getTickCount() - t) * 1000.0 / cv2.getTickFrequency()
            result["depth_map"] = depth_out.get("depth_map")

        # Weather/visibility (Phase 3.3)
        if self.visibility_detector is not None:
            t = cv2.getTickCount()
            visibility_result = self.visibility_detector.detect(frame)
            result["weather_ms"] = (cv2.getTickCount() - t) * 1000.0 / cv2.getTickFrequency()
            if visibility_result.degraded:
                warnings.append(f"WARNING: visibility={visibility_result.condition}")

        return result

    def _compute_ego_positions(self, tracks: list, frame: Any) -> set:
        """Convert track positions to ego-frame coordinates and compute per-track TTC.

        Returns the set of alive track IDs.
        """
        now_t = time.time()
        h, w = frame.shape[:2]
        for trk in tracks:
            tid = getattr(trk, "track_id", None)
            if tid is None:
                continue
            c = self._track_center(trk)
            if c is None:
                continue
            cx, cy, _, _, x2, y2 = c
            box_h = max(1.0, y2 - c[3])
            if tid not in self._track_hist:
                self._track_hist[tid] = deque(maxlen=self._track_hist_len)
            self._track_hist[tid].append((now_t, cx, cy, box_h))
            dist_proxy_px = h / box_h
            y_m = dist_proxy_px * self.fcw_px_to_m
            x_m = ((cx - (w / 2.0)) / w) * (self.lane_width_m * 4.0)
            trk.x = x_m
            trk.y = y_m
            prev_pos = self._prev_positions.get(tid)
            if prev_pos is not None:
                dt_pos = max(1e-3, now_t - prev_pos[2])
                trk.vx = (x_m - prev_pos[0]) / dt_pos
                trk.vy = (y_m - prev_pos[1]) / dt_pos
            else:
                trk.vx = None
                trk.vy = None
            self._prev_positions[tid] = (x_m, y_m, now_t)
            ttc_obj = None
            if len(self._track_hist[tid]) >= 2:
                t0_h, _, _, h0 = self._track_hist[tid][-2]
                t1_h, _, _, h1 = self._track_hist[tid][-1]
                dt_h = max(1e-3, t1_h - t0_h)
                dist_prev = h / max(1.0, h0)
                closing_rate = (dist_prev - dist_proxy_px) / dt_h
                ttc_obj = compute_ttc(dist_proxy_px, closing_rate)
            trk.ttc = ttc_obj
            trk.risk = fcw_state(ttc_obj) if ttc_obj is not None else None

        # Prune dead tracks
        alive_ids = {getattr(t, "track_id", None) for t in tracks}
        for tid in list(self._track_hist.keys()):
            if tid not in alive_ids:
                del self._track_hist[tid]
        return alive_ids

    def _run_state_estimation(self, tracks: list, alive_ids: set) -> tuple[dict, dict]:
        """Run Kalman filter update + temporal prediction. Returns (predictions, predictions_topk)."""
        if self.kalman_manager is not None:
            for trk in tracks:
                if trk.x is not None and trk.y is not None:
                    kx, ky, kvx, kvy = self.kalman_manager.update_track(trk.track_id, trk.x, trk.y)
                    trk.x = kx
                    trk.y = ky
                    trk.vx = kvx
                    trk.vy = kvy
            self.kalman_manager.prune(alive_ids)

        predictions: dict[int, list[Any]] = {}
        predictions_topk: dict[int, list[Any]] = {}
        if self.temporal_predictor is not None and self.kalman_manager is not None:
            predictions = self.temporal_predictor.predict(self.kalman_manager, alive_ids)
            predictions_topk = self.temporal_predictor.predict_topk(self.kalman_manager, alive_ids)
        return predictions, predictions_topk

    def _evaluate_interactions(self, tracks: list, warnings: list[str]) -> tuple[list, float]:
        """Run interaction model (Task 7). Returns (interactions, elapsed_ms)."""
        if self.interaction_model is None or not tracks:
            return [], 0.0
        t = cv2.getTickCount()
        ego_state = {"x": 0.0, "y": 0.0, "vx": 0.0, "vy": 10.0}
        interactions = self.interaction_model.evaluate(ego_state, tracks)
        ms = (cv2.getTickCount() - t) * 1000.0 / cv2.getTickFrequency()
        for ev in interactions:
            warnings.append(f"WARNING: INTERACTION {ev.type}: {ev.description}")
        return interactions, ms

    def _evaluate_bsd(self, tracks: list, warnings: list[str]) -> list[dict] | None:
        """Run blind spot detection. Returns bsd_warnings list or None."""
        if self.bsd_detector is None:
            return None
        bsd_warnings_list = self.bsd_detector.evaluate(tracks)
        if not bsd_warnings_list:
            return None
        bsd_warnings = [
            {"side": w.side, "track_id": w.track_id, "distance_m": w.distance_m, "ttc_s": w.ttc_s}
            for w in bsd_warnings_list
        ]
        for bw in bsd_warnings_list:
            warnings.append(f"WARNING: BSD {bw.side.upper()} ID={bw.track_id}")
        return bsd_warnings

    def _compute_fcw(self, tracks: list, frame: Any, lanes: dict, ldw_allowed: bool, warnings: list[str]) -> dict[str, Any]:
        """Compute Forward Collision Warning state (full + simple proxy).

        Returns dict with keys: fcw, fcw_pre, ego_lane_only,
        simple_state, simple_ttc, simple_lead, simple_dist, simple_closing.
        """
        h, w = frame.shape[:2]
        fcw = {"state": "NORMAL", "ttc_s": None, "lead_track_id": None, "distance_m": None, "rel_speed_mps": None, "distance_px": None}
        ego_y = h * self.fcw_ego_y_ratio
        ldw_allowed = bool(lanes.get("ldw_allowed", False)) if lanes else False

        # --- FCW Pre-warning ---
        fcw_pre = {"state": "NONE", "ttc_s": None, "lead_track_id": None, "distance_px": None}
        if self.fcw_enabled and tracks:
            best_pre = None
            for trk in tracks:
                tid = getattr(trk, "track_id", None)
                if tid not in self._track_hist:
                    continue
                c = self._track_center(trk)
                if c is None:
                    continue
                cx, cy, *_ = c
                if cy >= ego_y:
                    continue
                dist_px = ego_y - cy
                if dist_px <= 0:
                    continue
                hist = list(self._track_hist[tid])
                if len(hist) < 2:
                    continue
                t0_h, _, cy0, _ = hist[0]
                t1_h, _, cy1, _ = hist[-1]
                dt = max(1e-3, t1_h - t0_h)
                vy_px_s = (cy1 - cy0) / dt
                rel_speed_mps = vy_px_s * self.fcw_px_to_m
                ttc_pre = None
                if rel_speed_mps > 0:
                    ttc_pre = (dist_px * self.fcw_px_to_m) / rel_speed_mps
                if best_pre is None or dist_px < best_pre[0]:
                    best_pre = (dist_px, tid, ttc_pre)
            if best_pre is not None:
                dist_px, tid, ttc_pre = best_pre
                if dist_px < self.fcw_pre_distance_px:
                    fcw_pre.update({
                        "state": "PRE",
                        "lead_track_id": int(tid),
                        "distance_px": float(dist_px),
                        "ttc_s": float(ttc_pre) if ttc_pre is not None else None,
                    })

        # --- Full FCW with ego-lane filtering ---
        ego_lane_only = False
        if self.fcw_enabled and ldw_allowed and lanes and tracks:
            lane_center_x = lanes.get("lane_center_x")
            left = lanes.get("left_line")
            right = lanes.get("right_line")
            lane_width_px = None
            if left and right:
                lane_width_px = abs(float(right[0][0]) - float(left[0][0]))

            corridor_half = self.lane_width_m / 2.0
            def in_corr(trk_obj):
                x_m = getattr(trk_obj, "x", None)
                return x_m is not None and abs(x_m) <= corridor_half

            best = None
            for trk in tracks:
                tid = getattr(trk, "track_id", None)
                if tid is None or tid not in self._track_hist:
                    continue
                if not in_corr(trk):
                    continue
                c = self._track_center(trk)
                if c is None:
                    continue
                cx, cy, x1, y1, x2, y2 = c
                if cy >= ego_y:
                    continue
                if lane_center_x is not None and lane_width_px is not None and lane_width_px > 1 and abs(cx - lane_center_x) > 0.5 * lane_width_px:
                    continue
                hist = list(self._track_hist[tid])
                if len(hist) < 2:
                    continue
                t0_h, cx0, cy0, _ = hist[0]
                t1_h, cx1, cy1, _ = hist[-1]
                dt = max(1e-3, (t1_h - t0_h))
                vy_px_s_local = (cy1 - cy0) / dt
                rel_speed_mps = vy_px_s_local * self.fcw_px_to_m
                if rel_speed_mps < self.fcw_min_rel_speed:
                    continue
                dist_px = (ego_y - cy1)
                dist_m = dist_px * self.fcw_px_to_m
                if dist_m <= 0:
                    continue
                ttc = dist_px / (vy_px_s_local + 1e-9)
                if best is None or ttc < best[0]:
                    best = (ttc, tid, dist_m, rel_speed_mps, (x1, y1, x2, y2), dist_px, vy_px_s_local)
            if best is not None:
                ego_lane_only = True
                ttc, tid, dist_m, rel_speed_mps, bbox, dist_px, vy_px_s = best
                raw_ttc = dist_px / (vy_px_s + 1e-9) if vy_px_s > 0 else None
                raw_state = "NORMAL"
                if raw_ttc is not None:
                    if raw_ttc <= self.fcw_ttc_critical:
                        raw_state = "CRITICAL"
                    elif raw_ttc <= self.fcw_ttc_warning:
                        raw_state = "WARNING"
                    elif raw_ttc <= self.fcw_ttc_caution:
                        raw_state = "CAUTION"
                smoothed_ttc, stable_state = self.ttc_filter.update(raw_ttc, raw_state)
                state = stable_state
                ttc = smoothed_ttc
                fcw.update({
                    "state": state,
                    "ttc_s": float(ttc),
                    "lead_track_id": int(tid),
                    "distance_m": float(dist_m),
                    "rel_speed_mps": float(rel_speed_mps),
                    "lead_bbox": bbox,
                    "distance_px": float(dist_px),
                })
                if state != "NORMAL":
                    warnings.append(f"WARNING: FCW {state} TTC={ttc:.2f}s")

        # --- Simple FCW proxy (fallback/telemetry) ---
        def pick_lead_object(objs):
            if not objs:
                return None
            candidates = [o for o in objs if getattr(o, "class_name", None) in ("car", "truck", "bus", "motorcycle")]
            if not candidates:
                return None
            return max(candidates, key=lambda o: (o.bbox_xyxy[3] if hasattr(o, "bbox_xyxy") else (getattr(o, "bbox", (0, 0, 0, 0))[3])))

        lead = pick_lead_object(tracks)
        fcw_simple_state = "NORMAL"
        fcw_simple_ttc = None
        fcw_simple_lead = None
        fcw_simple_dist = None
        fcw_simple_closing = None
        if lead:
            bbox = getattr(lead, "bbox_xyxy", None) or getattr(lead, "bbox", None)
            if bbox:
                h_curr = max(1.0, bbox[3] - bbox[1])
                dist_proxy = frame.shape[0] / h_curr
                fcw_simple_dist = dist_proxy
                hist = list(self._track_hist.get(getattr(lead, "track_id", None), []))
                closing_rate = None
                if len(hist) >= 2:
                    t0_h, _, _, h0 = hist[-2]
                    t1_h, _, _, _ = hist[-1]
                    dt = max(1e-3, t1_h - t0_h)
                    dist_prev = frame.shape[0] / h0
                    closing_rate = (dist_prev - dist_proxy) / dt
                if closing_rate is not None:
                    fcw_simple_closing = closing_rate
                    fcw_simple_ttc = compute_ttc(dist_proxy, closing_rate)
                    fcw_simple_state = fcw_state(fcw_simple_ttc)
                    fcw_simple_lead = getattr(lead, "track_id", None)

        return {
            "fcw": fcw,
            "fcw_pre": fcw_pre,
            "ego_lane_only": ego_lane_only,
            "simple_state": fcw_simple_state,
            "simple_ttc": fcw_simple_ttc,
            "simple_lead": fcw_simple_lead,
            "simple_dist": fcw_simple_dist,
            "simple_closing": fcw_simple_closing,
        }

    def _process_lidar(self, packet: Any, tracks: list, warnings: list[str]) -> dict[str, Any]:
        """Process LIDAR pipeline: point cloud, BEV encoding, camera fusion."""
        result: dict[str, Any] = {
            "detections": [], "fused": [], "point_cloud": None,
            "bev_grid": None, "lidar_ms": 0.0, "bev_ms": 0.0, "fusion_ms": 0.0,
        }
        pc = getattr(packet, "point_cloud", None) if packet is not None else None
        calib = getattr(packet, "calibration", None) if packet is not None else None

        if self.lidar_processor is None or pc is None:
            return result

        t = cv2.getTickCount()
        result["detections"] = self.lidar_processor.process(pc)
        result["lidar_ms"] = (cv2.getTickCount() - t) * 1000.0 / cv2.getTickFrequency()
        result["point_cloud"] = pc
        warnings.append(f"INFO: LIDAR {len(result['detections'])} detections")

        # BEV encoding
        if self.bev_encoder is not None:
            t = cv2.getTickCount()
            result["bev_grid"] = self.bev_encoder.encode(pc)
            result["bev_ms"] = (cv2.getTickCount() - t) * 1000.0 / cv2.getTickFrequency()

        # LIDAR-camera fusion
        if self.lidar_camera_fusion is not None and tracks:
            t = cv2.getTickCount()
            result["fused"] = self.lidar_camera_fusion.fuse(
                camera_dets=tracks, lidar_dets=result["detections"], calibration=calib,
            )
            result["fusion_ms"] = (cv2.getTickCount() - t) * 1000.0 / cv2.getTickFrequency()
            warnings.append(f"INFO: Fusion {len(result['fused'])} fused detections")

        return result

    def _process_radar(self, packet: Any, tracks: list, warnings: list[str]) -> dict[str, Any]:
        """Process radar pipeline: detection processing and camera fusion."""
        result: dict[str, Any] = {"detections": [], "fused": False, "radar_ms": 0.0, "fusion_ms": 0.0}
        rf = getattr(packet, "radar_frame", None) if packet is not None else None

        if self.radar_processor is None or rf is None:
            return result

        t = cv2.getTickCount()
        result["detections"] = self.radar_processor.process(rf)
        result["radar_ms"] = (cv2.getTickCount() - t) * 1000.0 / cv2.getTickFrequency()
        warnings.append(f"INFO: Radar {len(result['detections'])} detections")

        if self.radar_camera_fusion is not None and tracks:
            t = cv2.getTickCount()
            self.radar_camera_fusion.fuse(tracks, result["detections"])
            result["fusion_ms"] = (cv2.getTickCount() - t) * 1000.0 / cv2.getTickFrequency()
            result["fused"] = True

        return result

    def _run_post_safety(self, wm: WorldModel, tracks: list, detections: list, frame_id: int,
                         fcw_result: dict[str, Any], warnings: list[str]) -> None:
        """Run plausibility checks, DTC logging, and FCW telemetry. Mutates wm."""
        # ISO 26262: Plausibility checks + DTC logging
        if self.plausibility_checker is not None:
            plausibility_violations = self.plausibility_checker.check(
                tracks=tracks,
                detections=detections,
                prev_tracks=self._prev_tracks_for_plausibility or None,
            )
            self._prev_tracks_for_plausibility = list(tracks)

            if plausibility_violations:
                for pv in plausibility_violations:
                    warnings.append(f"WARNING: PLAUSIBILITY {pv.check_name}: {pv.description}")
                    if self.dtc_logger is not None:
                        dtc_code = "DTC_PLC_002" if pv.severity == "critical" else "DTC_PLC_001"
                        self.dtc_logger.log(dtc_code, details={"check": pv.check_name, "description": pv.description}, frame_id=frame_id)

                wm.safety.setdefault("details", {})["plausibility_violations"] = [
                    {"check": v.check_name, "severity": v.severity, "description": v.description}
                    for v in plausibility_violations
                ]

        # FCW proxy telemetry
        fcw_simple_state = fcw_result["simple_state"]
        ego_lane_only = fcw_result["ego_lane_only"]
        if fcw_simple_state != self._prev_fcw:
            self.logger.info(
                "[FCW] state %s -> %s lead=%s ttc=%s ego_lane_only=%s",
                self._prev_fcw, fcw_simple_state, fcw_result["simple_lead"],
                f"{fcw_result['simple_ttc']:.2f}" if fcw_result["simple_ttc"] is not None else None,
                ego_lane_only,
            )
            self._prev_fcw = fcw_simple_state
            wm.safety.setdefault("fcw_event", {}).update({
                "state": fcw_simple_state, "lead_id": fcw_result["simple_lead"],
                "ttc_s": fcw_result["simple_ttc"], "distance_px": fcw_result["simple_dist"],
                "closing_rate": fcw_result["simple_closing"], "ego_lane_only": ego_lane_only,
            })

        wm.safety.setdefault("details", {})["fcw_proxy"] = {
            "state": fcw_simple_state, "ttc_s": fcw_result["simple_ttc"],
            "lead_id": fcw_result["simple_lead"], "distance_px": fcw_result["simple_dist"],
            "closing_rate": fcw_result["simple_closing"],
        }

    # ── Utility methods ──────────────────────────────────────────────

    def _run_parallel(self, frame: Any, frame_id: int) -> dict[str, Any]:
        """Run independent stages in parallel (Phase 4.1)."""
        stages = {}
        stages["detection"] = lambda: self.detector.infer(frame, conf_thres=self.cfg.get("perception", {}).get("conf_thres", 0.25))
        if self.lane_detector is not None:
            stages["lane"] = lambda: self.lane_detector.infer(frame)
        if self.segmenter is not None:
            stages["segmentation"] = lambda: self.segmenter.infer(frame)
        if self.depth_estimator is not None:
            stages["depth"] = lambda: self.depth_estimator.infer(frame)
        return self.parallel_executor.run(stages)

    def _track_center(self, trk):
        if hasattr(trk, "to_ltrb"):
            x1, y1, x2, y2 = trk.to_ltrb()
        elif hasattr(trk, "bbox_xyxy"):
            x1, y1, x2, y2 = trk.bbox_xyxy
        elif hasattr(trk, "bbox"):
            x1, y1, x2, y2 = trk.bbox
        else:
            return None
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        return float(cx), float(cy), float(x1), float(y1), float(x2), float(y2)

    def _build_safety(self, lanes, fcw, fcw_pre, bsd_warnings=None):
        ldw_departure = lanes.get("lane_departure") if lanes else None
        lane_ok = bool(lanes.get("ldw_allowed", False)) if lanes else False
        fcw_state_val = fcw.get("state") if fcw else None
        fcw_ttc = fcw.get("ttc_s") if fcw else None
        fcw_pre_active = fcw_pre.get("state") == "PRE" if fcw_pre else False
        out = self.safety.evaluate(
            ldw_departure=ldw_departure,
            fcw_state=fcw_state_val,
            fcw_ttc_s=fcw_ttc,
            fcw_pre_active=fcw_pre_active,
            lane_ok=lane_ok,
            bsd_warnings=[{"side": w["side"]} for w in bsd_warnings] if bsd_warnings else None,
        )
        return {"state": out.state.value, "message": out.message, "color": out.color, "details": out.details}
