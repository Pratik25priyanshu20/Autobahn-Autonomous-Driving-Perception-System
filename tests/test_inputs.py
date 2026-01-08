from src.inputs.video_input import VideoInput


def test_video_input_configures_path_without_opening():
    vi = VideoInput("/tmp/video.mp4", allow_missing=True)
    assert str(vi.path) == "/tmp/video.mp4"
