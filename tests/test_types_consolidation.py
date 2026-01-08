"""Tests for type consolidation (Phase 0.1)."""
from src.types import (
    Detection,
    Track,
    EgoState,
    LaneGeometry,
    LaneState,
    SafetyStatus,
    SafetyState,
    SafetyOutput,
    SafetyStateEnum,
    WorldModel,
    RuntimeStats,
    DrivableArea,
    FramePacket,
    PerceptionOutput,
    Detection3D,
)


def test_detection_properties():
    d = Detection(x1=10, y1=20, x2=50, y2=80, conf=0.9, class_id=2, class_name="car")
    assert d.bbox == (10, 20, 50, 80)
    assert d.score == 0.9
    assert d.label == "car"
    assert d.width == 40
    assert d.height == 60
    assert d.center == (30.0, 50.0)


def test_track_world_frame_attrs():
    t = Track(track_id=1, bbox_xyxy=(0, 0, 10, 10), class_name="car", conf=0.8)
    t.x = 1.5
    t.y = 10.0
    t.vx = 0.5
    t.vy = -1.0
    t.ttc = 5.0
    t.risk = "NORMAL"
    assert t.x == 1.5
    assert t.vx == 0.5


def test_ego_state():
    e = EgoState(x=1.0, y=2.0, speed=15.0)
    assert e.speed == 15.0


def test_lane_geometry():
    lg = LaneGeometry(confidence=0.9, stable=True)
    assert lg.stable


def test_lane_state():
    ls = LaneState(left_detected=True, right_detected=False, confidence=0.7)
    assert ls.left_detected


def test_world_model_defaults():
    wm = WorldModel(frame_id=1, frame=None)
    assert wm.depth_map is None
    assert wm.predictions == {}
    assert wm.occupancy is None
    assert wm.control is None


def test_safety_state_enum():
    assert SafetyStateEnum.CRITICAL.value == "CRITICAL"


def test_detection3d():
    d = Detection3D(x1=0, y1=0, x2=10, y2=10, conf=0.5, class_id=2, class_name="car", x_m=1.0, y_m=2.0, z_m=5.0)
    assert d.center_3d == (1.0, 2.0, 5.0)


def test_backward_compat_utils_types():
    from src.utils.types import Detection, FramePacket, PerceptionOutput, SafetyStatus, Track, WorldModel
    d = Detection(x1=0, y1=0, x2=1, y2=1, conf=0.5, class_id=0, class_name="person")
    assert d.label == "person"


def test_backward_compat_fusion_types():
    from src.fusion.types import EgoState, LaneGeometry, SafetyState, DrivableArea
    e = EgoState()
    assert e.speed_mps == 0.0
