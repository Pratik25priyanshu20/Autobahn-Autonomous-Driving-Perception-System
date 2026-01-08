from src.utils.types import Detection, PerceptionOutput


def test_perception_output_defaults():
    output = PerceptionOutput()
    output.detections.append(Detection((0, 0, 1, 1), 0.9, "car"))
    assert output.detections[0].label == "car"
