from src.fusion.fusion_engine import fuse
from src.utils.types import Detection, PerceptionOutput


def test_fuse_passes_through_objects():
    output = PerceptionOutput(detections=[Detection((0, 0, 1, 1), 0.9, "car")])
    world = fuse(output)
    assert world is not None
    assert world.objects[0].label == "car"
