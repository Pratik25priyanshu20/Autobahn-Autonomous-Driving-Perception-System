"""Startup config validation for APS++."""
from __future__ import annotations

from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConfigValidationError(ValueError):
    """Raised when config validation fails."""


def _get(cfg: dict, *keys: str, default: Any = None) -> Any:
    """Nested dict access."""
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur


_VALID_RUNTIMES = {"pytorch", "onnx", "tensorrt"}
_VALID_TRACKERS = {"deepsort", "bytetrack"}
_VALID_LANE_BACKENDS = {"canny_hough", "ufldv2"}
_VALID_DEPTH_BACKENDS = {"midas", "depth_anything"}
_VALID_CONTROL_TYPES = {"pure_pursuit", "mpc"}
_VALID_MODES = {"perception_only", "adas", "full_autonomy"}
_VALID_INPUT_SOURCES = {"video", "carla", "kitti", "replay"}


def validate_config(cfg: dict[str, Any], safety_cfg: dict[str, Any] | None = None) -> list[str]:
    """Validate merged config. Returns list of warning/error strings. Raises on critical errors."""
    errors: list[str] = []
    warns: list[str] = []

    # --- Enum checks ---
    runtime = _get(cfg, "perception", "runtime", default="pytorch")
    if runtime not in _VALID_RUNTIMES:
        errors.append(f"perception.runtime '{runtime}' not in {_VALID_RUNTIMES}")

    tracker = _get(cfg, "tracking", "type", default="deepsort")
    if tracker not in _VALID_TRACKERS:
        errors.append(f"tracking.type '{tracker}' not in {_VALID_TRACKERS}")

    lane_backend = _get(cfg, "lane", "backend", default="canny_hough")
    if lane_backend not in _VALID_LANE_BACKENDS:
        errors.append(f"lane.backend '{lane_backend}' not in {_VALID_LANE_BACKENDS}")

    depth_backend = _get(cfg, "depth", "backend", default="midas")
    if depth_backend not in _VALID_DEPTH_BACKENDS:
        errors.append(f"depth.backend '{depth_backend}' not in {_VALID_DEPTH_BACKENDS}")

    ctrl_type = _get(cfg, "control", "type", default="pure_pursuit")
    if ctrl_type not in _VALID_CONTROL_TYPES:
        errors.append(f"control.type '{ctrl_type}' not in {_VALID_CONTROL_TYPES}")

    mode = _get(cfg, "system", "mode", default="perception_only")
    if mode not in _VALID_MODES:
        errors.append(f"system.mode '{mode}' not in {_VALID_MODES}")

    input_source = _get(cfg, "system", "input_source", default="video")
    if input_source not in _VALID_INPUT_SOURCES:
        errors.append(f"system.input_source '{input_source}' not in {_VALID_INPUT_SOURCES}")

    # --- Range checks ---
    conf_thres = _get(cfg, "perception", "conf_thres", default=0.25)
    if not (0.0 <= float(conf_thres) <= 1.0):
        errors.append(f"perception.conf_thres={conf_thres} must be in [0.0, 1.0]")

    target_fps = _get(cfg, "performance", "target_fps", default=20)
    if float(target_fps) <= 0:
        errors.append(f"performance.target_fps={target_fps} must be > 0")

    fcw_ego = _get(cfg, "fcw", "ego_y_ratio", default=0.92)
    if not (0.0 < float(fcw_ego) <= 1.0):
        errors.append(f"fcw.ego_y_ratio={fcw_ego} must be in (0.0, 1.0]")

    ttc_warn = _get(cfg, "fcw", "ttc_warning_s", default=2.5)
    ttc_crit = _get(cfg, "fcw", "ttc_critical_s", default=1.5)
    if float(ttc_crit) >= float(ttc_warn):
        warns.append(f"fcw.ttc_critical_s ({ttc_crit}) should be < ttc_warning_s ({ttc_warn})")

    lane_min_conf = _get(cfg, "lane", "min_confidence", default=0.55)
    if not (0.0 <= float(lane_min_conf) <= 1.0):
        errors.append(f"lane.min_confidence={lane_min_conf} must be in [0.0, 1.0]")

    # --- Dependency checks ---
    if _get(cfg, "radar_fusion", "enabled", default=False) and not _get(cfg, "radar", "enabled", default=False):
        warns.append("radar_fusion.enabled=true but radar.enabled=false — fusion requires radar input")

    if _get(cfg, "lidar_fusion", "enabled", default=False) and not _get(cfg, "lidar", "enabled", default=False):
        warns.append("lidar_fusion.enabled=true but lidar.enabled=false — fusion requires LIDAR input")

    # --- Safety config checks ---
    if safety_cfg is not None:
        br = _get(safety_cfg, "sensor_health", "brightness_range", default=[40, 220])
        if isinstance(br, list) and len(br) == 2 and br[0] >= br[1]:
            errors.append(f"sensor_health.brightness_range min ({br[0]}) must be < max ({br[1]})")
        health_thresh = _get(safety_cfg, "sensor_health", "health_threshold", default=0.5)
        if not (0.0 <= float(health_thresh) <= 1.0):
            errors.append(f"sensor_health.health_threshold={health_thresh} must be in [0.0, 1.0]")

    # --- Log and raise ---
    for w in warns:
        logger.warning("[CONFIG] %s", w)
    if errors:
        for e in errors:
            logger.error("[CONFIG] %s", e)
        raise ConfigValidationError(f"Config validation failed with {len(errors)} error(s): {'; '.join(errors)}")

    return warns
