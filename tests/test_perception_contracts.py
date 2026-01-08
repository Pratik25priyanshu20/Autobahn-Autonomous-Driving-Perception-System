from src.utils.types import Detection, PerceptionOutput


def test_perception_output_defaults():
    output = PerceptionOutput()
    output.detections.append(Detection(x1=0, y1=0, x2=1, y2=1, conf=0.9, class_id=2, class_name="car"))
    assert output.detections[0].label == "car"
