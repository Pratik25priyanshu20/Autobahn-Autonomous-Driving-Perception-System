from __future__ import annotations

from collections import deque
import time
from typing import Any, Dict, List

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from src.adas.ttc_filter import TTCFilter
from src.fusion.world_model import RuntimeStats, WorldModel
from src.perception.detection.yolo import YOLODetector
from src.perception.lanes.lane_detector import CannyHoughLaneDetector
from src.perception.segmentation.deeplabv3_segmenter import DeepLabV3Segmenter
from src.perception.segmentation.postprocess import extract_drivable_area
from src.perception.tracking.deepsort_tracker import DeepSORTTracker
from src.fusion.world_model import DrivableArea
from src.safety.distance import distance_proxy_px
from src.safety.fcw import fcw_state
from src.safety.safety_manager import SafetyManager
from src.safety.ttc import compute_ttc
from src.utils.timing import FPSMeter


class Orchestrator:
    """
    Phase 2 runtime: detection produces a WorldModel; fusion/safety will plug in later.
    """

    def __init__(self, cfg: Dict[str, Any], logger):
        self.cfg = cfg
        self.logger = logger
        self.fps_meter = FPSMeter(smoothing=float(cfg.get("performance", {}).get("fps_smoothing", 0.9)))
        self.detector = YOLODetector(model_name=cfg.get("perception", {}).get("detector_model", "yolov8n.pt"))
        self.tracker = DeepSORTTracker()
        self.tracking_enabled = bool(self.cfg.get("tracking", {}).get("enabled", True))
        self.tracking_interval = int(self.cfg.get("tracking", {}).get("interval", 2))
        self._last_tracks: List[Any] = []
        self._last_trajectories: Dict[int, List[Any]] = {}
        self.lane_enabled = bool(self.cfg.get("lane", {}).get("enabled", True))
        self.lane_detector = CannyHoughLaneDetector() if self.lane_enabled else None
        lane_cfg = self.cfg.get("lane", {})
        self.lane_min_conf = float(lane_cfg.get("min_confidence", 0.55))
        self.lane_stability_px = float(lane_cfg.get("stability_px", 35))
        self.lane_min_stable_frames = int(lane_cfg.get("min_stable_frames", 5))
        self._lane_center_hist = deque(maxlen=max(20, self.lane_min_stable_frames * 3))
        ldw_cfg = self.cfg.get("ldw", {})
        self.ldw_enabled = bool(ldw_cfg.get("enabled", True))
        self.ldw_threshold_px = float(ldw_cfg.get("offset_threshold_px", 80))
        self.ldw_persistence = int(ldw_cfg.get("persistence_frames", 6))
        self._offset_hist = deque(maxlen=max(10, self.ldw_persistence * 2))
        self.safety = SafetyManager()
        fcw_cfg = self.cfg.get("fcw", {})
        self.fcw_enabled = bool(fcw_cfg.get("enabled", True))
        self.fcw_px_to_m = float(fcw_cfg.get("px_to_m", 0.05))
        self.fcw_ego_y_ratio = float(fcw_cfg.get("ego_y_ratio", 0.92))
        self.fcw_min_rel_speed = float(fcw_cfg.get("min_rel_speed_mps", 0.3))
        self.fcw_pre_distance_px = float(fcw_cfg.get("pre_distance_px", 220))
        self.fcw_ttc_caution = float(fcw_cfg.get("ttc_caution_s", 4.0))
        self.fcw_ttc_warning = float(fcw_cfg.get("ttc_warning_s", 2.5))
        self.fcw_ttc_critical = float(fcw_cfg.get("ttc_critical_s", 1.5))
        self.lane_width_m = 3.5
        self._track_hist: Dict[Any, deque] = {}
        self._track_hist_len = 10
        self.ttc_filter = TTCFilter(alpha=0.3, min_persist_frames=3)
        seg_cfg = self.cfg.get("segmentation", {})
        self.segmentation_enabled = bool(seg_cfg.get("enabled", False))
        self.segmenter = DeepLabV3Segmenter(device=seg_cfg.get("device")) if self.segmentation_enabled else None
        self._prev_fcw = "NORMAL"
        self._prev_positions: Dict[Any, tuple] = {}

        resize_cfg = cfg.get("video", {}).get("resize", {})
        self.resize_enabled = bool(resize_cfg.get("enabled", False))
        self.resize_w = int(resize_cfg.get("width", 1280))
        self.resize_h = int(resize_cfg.get("height", 720))

    def process_frame(self, frame_id: int, frame: Any) -> WorldModel:
        if cv2 is None:
            raise ImportError("opencv-python is required for orchestrator processing")

        warnings: List[str] = []

        # Stage: preprocessing
        t0 = cv2.getTickCount()
        if self.resize_enabled:
            frame = cv2.resize(frame, (self.resize_w, self.resize_h), interpolation=cv2.INTER_LINEAR)
        preprocess_ms = (cv2.getTickCount() - t0) * 1000.0 / cv2.getTickFrequency()

        # Stage: detection
        t1 = cv2.getTickCount()
        detections = self.detector.infer(frame, conf_thres=self.cfg.get("perception", {}).get("conf_thres", 0.25))
        detect_ms = (cv2.getTickCount() - t1) * 1000.0 / cv2.getTickFrequency()

        # Stage: tracking (DeepSORT) with temporal decimation
        tracks = self._last_tracks
        trajectories = self._last_trajectories
        track_ms = 0.0
        if self.tracking_enabled and self.tracking_interval > 0 and frame_id % self.tracking_interval == 0:
            t2 = cv2.getTickCount()
            tracks, trajectories = self.tracker.update(frame, detections)
            track_ms = (cv2.getTickCount() - t2) * 1000.0 / cv2.getTickFrequency()
            self._last_tracks = tracks
            self._last_trajectories = trajectories

        fps = self.fps_meter.tick()

        target = float(self.cfg.get("performance", {}).get("target_fps", 20))
        if fps < target * 0.6:
            warnings.append("FPS low (degraded mode planned for Phase 5)")
        if self.tracking_enabled and self.tracking_interval > 0 and frame_id % self.tracking_interval == 0:
            warnings.append(f"INFO: tracking update | {len(tracks)} tracks")
        else:
            warnings.append(f"INFO: tracking reused | {len(tracks)} tracks")
        warnings.append(f"INFO: {len(detections)} detections")

        lanes: Dict[str, Any] = {}
        lane_ms = 0.0
        lane_departure = None
        ego_offset = None
        if self.lane_detector is not None:
            t3 = cv2.getTickCount()
            lanes = self.lane_detector.infer(frame)
            lane_ms = (cv2.getTickCount() - t3) * 1000.0 / cv2.getTickFrequency()
            ego_offset = lanes.get("ego_offset_px")

        # Lane confidence gating and stability
        ldw_allowed = False
        if lanes:
            conf = float(lanes.get("lane_confidence", 0.0))
            center_x = lanes.get("lane_center_x", None)
            if center_x is not None:
                self._lane_center_hist.append(float(center_x))
            stable = False
            jitter = None
            if len(self._lane_center_hist) >= self.lane_min_stable_frames:
                recent = list(self._lane_center_hist)[-self.lane_min_stable_frames :]
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
            recent = list(self._offset_hist)[-self.ldw_persistence :]
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

        seg_ms = 0.0
        drivable = None
        seg_conf = None
        if self.segmenter is not None:
            t_seg = cv2.getTickCount()
            seg_out = self.segmenter.infer(frame)
            seg_ms = (cv2.getTickCount() - t_seg) * 1000.0 / cv2.getTickFrequency()
            drivable = extract_drivable_area(seg_out["mask"])
            seg_conf = seg_out.get("confidence", 0.0)

        stages = {
            "preprocess": preprocess_ms,
            "detection": detect_ms,
            "tracking": track_ms,
            "lane": lane_ms,
            "segmentation": seg_ms,
            "fcw": 0.0,
        }

        now_t = time.time()
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
            # Approximate position in ego frame
            h, w = frame.shape[:2]
            dist_proxy_px = h / box_h
            y_m = dist_proxy_px * self.fcw_px_to_m
            # map image width to roughly two lanes => better lateral separation
            x_m = ((cx - (w / 2.0)) / w) * (self.lane_width_m * 4.0)
            trk.x = x_m
            trk.y = y_m
            # Velocity from previous position
            prev_pos = self._prev_positions.get(tid)
            if prev_pos is not None:
                dt_pos = max(1e-3, now_t - prev_pos[2])
                vx = (x_m - prev_pos[0]) / dt_pos
                vy = (y_m - prev_pos[1]) / dt_pos
                trk.vx = vx
                trk.vy = vy
            else:
                trk.vx = None
                trk.vy = None
            self._prev_positions[tid] = (x_m, y_m, now_t)
            # TTC per object (using bbox height derivative)
            ttc_obj = None
            if len(self._track_hist[tid]) >= 2:
                t0, _, _, h0 = self._track_hist[tid][-2]
                t1, _, _, h1 = self._track_hist[tid][-1]
                dt_h = max(1e-3, t1 - t0)
                dist_prev = h / max(1.0, h0)
                closing_rate = (dist_prev - dist_proxy_px) / dt_h
                ttc_obj = compute_ttc(dist_proxy_px, closing_rate)
            trk.ttc = ttc_obj
            trk.risk = fcw_state(ttc_obj) if ttc_obj is not None else None
        alive_ids = {getattr(t, "track_id", None) for t in tracks}
        for tid in list(self._track_hist.keys()):
            if tid not in alive_ids:
                del self._track_hist[tid]

        fcw = {
            "state": "NORMAL",
            "ttc_s": None,
            "lead_track_id": None,
            "distance_m": None,
            "rel_speed_mps": None,
            "distance_px": None,
        }

        h, w = frame.shape[:2]
        ego_y = h * self.fcw_ego_y_ratio
        ldw_allowed = bool(lanes.get("ldw_allowed", False)) if lanes else False

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
                t0, _, cy0 = hist[0]
                t1, _, cy1 = hist[-1]
                dt = max(1e-3, t1 - t0)
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
                    fcw_pre.update(
                        {
                            "state": "PRE",
                            "lead_track_id": int(tid),
                            "distance_px": float(dist_px),
                            "ttc_s": float(ttc_pre) if ttc_pre is not None else None,
                        }
                    )

        ego_lane_only = False
        if self.fcw_enabled and ldw_allowed and lanes and tracks:
            lane_center_x = lanes.get("lane_center_x", None)
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
                if lane_center_x is not None and lane_width_px is not None and lane_width_px > 1:
                    if abs(cx - lane_center_x) > 0.5 * lane_width_px:
                        continue
                hist = list(self._track_hist[tid])
                if len(hist) < 2:
                    continue
                t0, cx0, cy0 = hist[0]
                t1, cx1, cy1 = hist[-1]
                dt = max(1e-3, (t1 - t0))
                vy_px_s = (cy1 - cy0) / dt
                rel_speed_mps = vy_px_s * self.fcw_px_to_m
                if rel_speed_mps < self.fcw_min_rel_speed:
                    continue
                dist_px = (ego_y - cy1)
                dist_m = dist_px * self.fcw_px_to_m
                if dist_m <= 0:
                    continue
                ttc = dist_px / (vy_px_s + 1e-9)
                if best is None or ttc < best[0]:
                    best = (ttc, tid, dist_m, rel_speed_mps, (x1, y1, x2, y2), dist_px)
            if best is not None:
                ego_lane_only = True
                ttc, tid, dist_m, rel_speed_mps, bbox, dist_px = best
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
                fcw.update(
                    {
                        "state": state,
                        "ttc_s": float(ttc),
                        "lead_track_id": int(tid),
                        "distance_m": float(dist_m),
                        "rel_speed_mps": float(rel_speed_mps),
                        "lead_bbox": bbox,
                        "distance_px": float(dist_px),
                    }
                )
                if state != "NORMAL":
                    warnings.append(f"WARNING: FCW {state} TTC={ttc:.2f}s")

        # Simple FCW proxy based on track geometry (fallback/telemetry)
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
                    t0, _, _, h0 = hist[-2]
                    t1, _, _, _ = hist[-1]
                    dt = max(1e-3, t1 - t0)
                    dist_prev = frame.shape[0] / h0
                    closing_rate = (dist_prev - dist_proxy) / dt
                if closing_rate is not None:
                    fcw_simple_closing = closing_rate
                    fcw_simple_ttc = compute_ttc(dist_proxy, closing_rate)
                    fcw_simple_state = fcw_state(fcw_simple_ttc)
                    fcw_simple_lead = getattr(lead, "track_id", None)
        if fcw_simple_state != self._prev_fcw:
            self.logger.info(
                "[FCW] state %s -> %s lead=%s ttc=%s ego_lane_only=%s",
                self._prev_fcw,
                fcw_simple_state,
                fcw_simple_lead,
                f"{fcw_simple_ttc:.2f}" if fcw_simple_ttc is not None else None,
                ego_lane_only,
            )
            self._prev_fcw = fcw_simple_state
            wm.safety.setdefault("fcw_event", {}).update(
                {
                    "state": fcw_simple_state,
                    "lead_id": fcw_simple_lead,
                    "ttc_s": fcw_simple_ttc,
                    "distance_px": fcw_simple_dist,
                    "closing_rate": fcw_simple_closing,
                    "ego_lane_only": ego_lane_only,
                }
            )

        wm = WorldModel(
            frame_id=frame_id,
            frame=frame,
            detections=detections,
            tracks=tracks,
            trajectories=trajectories,
            lanes=lanes,
            drivable_area=DrivableArea(mask=drivable, confidence=seg_conf or 0.0) if drivable is not None else DrivableArea(),
            fcw=fcw,
            fcw_pre=fcw_pre,
            safety=self._build_safety(lanes, fcw, fcw_pre),
            warnings=warnings,
            runtime=RuntimeStats(fps=fps, stages_ms=stages),
        )
        wm.safety.setdefault("details", {})["fcw_proxy"] = {
            "state": fcw_simple_state,
            "ttc_s": fcw_simple_ttc,
            "lead_id": fcw_simple_lead,
            "distance_px": fcw_simple_dist,
            "closing_rate": fcw_simple_closing,
        }
        wm.snapshot()
        assert wm.frame is not None
        assert wm.tracks is not None
        if wm.lanes and wm.lanes.get("ego_offset_px") is not None:
            assert abs(wm.lanes.get("ego_offset_px", 0.0)) < 1000
        if frame_id % 30 == 0:
            self.logger.info("[WORLD] %s", wm.summary())
            if drivable is not None:
                self.logger.info("[WORLD] drivable_area=yes pixels=%d", int(drivable.sum()))
        return wm

    def _track_center(self, trk):
        if hasattr(trk, "to_ltrb"):
            x1, y1, x2, y2 = trk.to_ltrb()
        elif hasattr(trk, "bbox"):
            x1, y1, x2, y2 = trk.bbox
        else:
            return None
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        return float(cx), float(cy), float(x1), float(y1), float(x2), float(y2)

    def _build_safety(self, lanes, fcw, fcw_pre):
        ldw_departure = lanes.get("lane_departure") if lanes else None
        lane_ok = bool(lanes.get("ldw_allowed", False)) if lanes else False
        fcw_state = fcw.get("state") if fcw else None
        fcw_ttc = fcw.get("ttc_s") if fcw else None
        fcw_pre_active = fcw_pre.get("state") == "PRE" if fcw_pre else False
        out = self.safety.evaluate(
            ldw_departure=ldw_departure,
            fcw_state=fcw_state,
            fcw_ttc_s=fcw_ttc,
            fcw_pre_active=fcw_pre_active,
            lane_ok=lane_ok,
        )
        return {"state": out.state.value, "message": out.message, "color": out.color, "details": out.details}
