"""Tests for data recording and replay (Task 2)."""
from __future__ import annotations

import tempfile
from pathlib import Path

from src.recording.data_recorder import DataRecorder
from src.recording.recording_types import RecordedFrame
from src.recording.replay_input import ReplayInput


class _MockWorldModel:
    def __init__(self, frame_id: int):
        self.frame_id = frame_id
        self.tracks = [_MockTrack(i) for i in range(3)]
        self.lanes = {"ego_offset_px": 5.0, "lane_confidence": 0.8}
        self.fcw = {"state": "NORMAL"}
        self.safety = {"state": "SAFE"}
        self.detections = [1, 2, 3]
        self.sensor_health = {"camera": 0.9}


class _MockTrack:
    def __init__(self, track_id: int):
        self.track_id = track_id
        self.x = 1.0
        self.y = 5.0
        self.vx = 0.5
        self.vy = -1.0
        self.class_name = "car"
        self.conf = 0.85


class TestDataRecorder:
    def test_serialize_track(self):
        trk = _MockTrack(42)
        d = DataRecorder._serialize_track(trk)
        assert d["track_id"] == 42
        assert d["x"] == 1.0
        assert d["class_name"] == "car"

    def test_serialize_wm(self):
        wm = _MockWorldModel(10)
        rf = DataRecorder._serialize_wm(wm)
        assert isinstance(rf, RecordedFrame)
        assert rf.frame_id == 10
        assert len(rf.tracks) == 3
        assert rf.fcw_state == "NORMAL"

    def test_record_and_close(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = DataRecorder(tmpdir, record_interval=1, max_size_mb=10)
            for i in range(5):
                rec.record(_MockWorldModel(i))
            rec.close()
            path = Path(tmpdir) / "recording.apsrec"
            assert path.exists()
            assert path.stat().st_size > 0

    def test_interval_skipping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = DataRecorder(tmpdir, record_interval=3, max_size_mb=10)
            for i in range(10):
                rec.record(_MockWorldModel(i))
            rec.close()
            # Only frames at 0 (count 1=skip), 3 (count 3), 6, 9 should be recorded
            # Actually: frame_count increments, and records when frame_count % interval == 0
            # count: 1,2,3(record),4,5,6(record),7,8,9(record),10 - wait, 10 frames: counts 1-10
            # Records at count 3, 6, 9 = 3 frames
            # Let's verify via replay
            replay = ReplayInput(Path(tmpdir) / "recording.apsrec", playback_speed=100.0)
            replay.start()
            frames = list(replay.frames())
            assert len(frames) == 3
            replay.stop()


class TestRoundTrip:
    def test_record_then_replay(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Record
            rec = DataRecorder(tmpdir, record_interval=1, max_size_mb=10)
            num_frames = 10
            for i in range(num_frames):
                rec.record(_MockWorldModel(i))
            rec.close()

            # Replay
            replay = ReplayInput(Path(tmpdir) / "recording.apsrec", playback_speed=100.0)
            replay.start()
            packets = list(replay.frames())
            assert len(packets) == num_frames

            # Verify frame count
            assert replay.frame_count == num_frames
            replay.stop()

    def test_frame_seeking(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = DataRecorder(tmpdir, record_interval=1)
            for i in range(5):
                rec.record(_MockWorldModel(i))
            rec.close()

            replay = ReplayInput(Path(tmpdir) / "recording.apsrec", playback_speed=100.0)
            replay.start()
            f = replay.get_frame(2)
            assert f is not None
            assert f["frame_id"] == 2
            assert replay.get_frame(100) is None
            replay.stop()


class TestCompressionVerification:
    def test_file_is_compressed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = DataRecorder(tmpdir, record_interval=1)
            for i in range(20):
                rec.record(_MockWorldModel(i))
            rec.close()

            path = Path(tmpdir) / "recording.apsrec"
            raw = path.read_bytes()
            # Verify it's gzip by checking magic bytes
            assert raw[:2] == b"\x1f\x8b"  # gzip magic number
