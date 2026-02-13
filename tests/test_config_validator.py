"""Tests for config validation (Task 11)."""
from __future__ import annotations

import pytest

from src.utils.config_validator import ConfigValidationError, validate_config


def _base_cfg() -> dict:
    return {
        "system": {"mode": "perception_only", "input_source": "video"},
        "perception": {"runtime": "pytorch", "conf_thres": 0.25},
        "tracking": {"type": "deepsort"},
        "lane": {"backend": "canny_hough", "min_confidence": 0.55},
        "depth": {"backend": "midas"},
        "control": {"type": "pure_pursuit"},
        "performance": {"target_fps": 20},
        "fcw": {"ego_y_ratio": 0.92, "ttc_warning_s": 2.5, "ttc_critical_s": 1.5},
    }


class TestValidConfig:
    def test_valid_config_no_errors(self):
        warns = validate_config(_base_cfg())
        assert isinstance(warns, list)

    def test_empty_config_uses_defaults(self):
        validate_config({})


class TestEnumValidation:
    def test_invalid_runtime(self):
        cfg = _base_cfg()
        cfg["perception"]["runtime"] = "invalid"
        with pytest.raises(ConfigValidationError, match="perception.runtime"):
            validate_config(cfg)

    def test_invalid_tracker(self):
        cfg = _base_cfg()
        cfg["tracking"]["type"] = "sort"
        with pytest.raises(ConfigValidationError, match="tracking.type"):
            validate_config(cfg)

    def test_invalid_lane_backend(self):
        cfg = _base_cfg()
        cfg["lane"]["backend"] = "magic"
        with pytest.raises(ConfigValidationError, match="lane.backend"):
            validate_config(cfg)

    def test_invalid_mode(self):
        cfg = _base_cfg()
        cfg["system"]["mode"] = "unknown"
        with pytest.raises(ConfigValidationError, match="system.mode"):
            validate_config(cfg)


class TestRangeValidation:
    def test_conf_thres_out_of_range(self):
        cfg = _base_cfg()
        cfg["perception"]["conf_thres"] = 1.5
        with pytest.raises(ConfigValidationError, match="conf_thres"):
            validate_config(cfg)

    def test_target_fps_zero(self):
        cfg = _base_cfg()
        cfg["performance"]["target_fps"] = 0
        with pytest.raises(ConfigValidationError, match="target_fps"):
            validate_config(cfg)

    def test_ego_y_ratio_out_of_range(self):
        cfg = _base_cfg()
        cfg["fcw"]["ego_y_ratio"] = 1.5
        with pytest.raises(ConfigValidationError, match="ego_y_ratio"):
            validate_config(cfg)

    def test_lane_confidence_negative(self):
        cfg = _base_cfg()
        cfg["lane"]["min_confidence"] = -0.1
        with pytest.raises(ConfigValidationError, match="min_confidence"):
            validate_config(cfg)


class TestDependencyWarnings:
    def test_radar_fusion_without_radar(self):
        cfg = _base_cfg()
        cfg["radar_fusion"] = {"enabled": True}
        cfg["radar"] = {"enabled": False}
        warns = validate_config(cfg)
        assert any("radar_fusion" in w for w in warns)

    def test_lidar_fusion_without_lidar(self):
        cfg = _base_cfg()
        cfg["lidar_fusion"] = {"enabled": True}
        cfg["lidar"] = {"enabled": False}
        warns = validate_config(cfg)
        assert any("lidar_fusion" in w for w in warns)


class TestSafetyConfigValidation:
    def test_invalid_brightness_range(self):
        cfg = _base_cfg()
        safety = {"sensor_health": {"brightness_range": [220, 40]}}
        with pytest.raises(ConfigValidationError, match="brightness_range"):
            validate_config(cfg, safety_cfg=safety)

    def test_invalid_health_threshold(self):
        cfg = _base_cfg()
        safety = {"sensor_health": {"health_threshold": 1.5}}
        with pytest.raises(ConfigValidationError, match="health_threshold"):
            validate_config(cfg, safety_cfg=safety)

    def test_valid_safety_config(self):
        cfg = _base_cfg()
        safety = {"sensor_health": {"brightness_range": [40, 220], "health_threshold": 0.5}}
        validate_config(cfg, safety_cfg=safety)


class TestTTCWarning:
    def test_ttc_critical_ge_warning(self):
        cfg = _base_cfg()
        cfg["fcw"]["ttc_critical_s"] = 3.0
        cfg["fcw"]["ttc_warning_s"] = 2.5
        warns = validate_config(cfg)
        assert any("ttc_critical_s" in w for w in warns)
