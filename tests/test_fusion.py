from src.fusion.fusion_engine import fuse
from src.utils.types import Detection, PerceptionOutput


def test_fuse_passes_through_objects():
    output = PerceptionOutput(detections=[Detection(x1=0, y1=0, x2=1, y2=1, conf=0.9, class_id=2, class_name="car")])
    world = fuse(output)
    assert world is not None
    assert world.detections[0].label == "car"
