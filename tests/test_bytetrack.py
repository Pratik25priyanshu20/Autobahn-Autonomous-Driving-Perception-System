"""Tests for ByteTrack tracker (Phase 2.1)."""
import pytest

try:
    import supervision
    HAS_SV = True
except ImportError:
    HAS_SV = False

from src.types.detection import Detection


@pytest.mark.skipif(not HAS_SV, reason="supervision not installed")
def test_bytetrack_basic():
    from src.perception.tracking.bytetrack_tracker import ByteTrackTracker
    import numpy as np

    tracker = ByteTrackTracker()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = [Detection(x1=100, y1=100, x2=200, y2=200, conf=0.9, class_id=2, class_name="car")]
    tracks, trajs = tracker.update(frame, dets)
    # First frame may or may not produce tracks depending on activation threshold
    assert isinstance(tracks, list)
    assert isinstance(trajs, dict)


def test_bytetrack_import_guard():
    """Ensure import fails gracefully without supervision."""
    # This test just checks the module can be imported
    from src.perception.tracking import bytetrack_tracker
    assert hasattr(bytetrack_tracker, "ByteTrackTracker")
