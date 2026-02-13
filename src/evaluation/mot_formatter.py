"""MOT format parsers for MOT17, KITTI tracking, and APS predictions (Task 5)."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def parse_mot17(path: str | Path) -> list[list[tuple[int, tuple[int, int, int, int]]]]:
    """Parse MOT17 CSV format into per-frame lists of (id, bbox_xyxy).

    MOT17 format: frame_id, track_id, x, y, w, h, conf, class, visibility
    """
    path = Path(path)
    frames_dict: dict[int, list[tuple[int, tuple[int, int, int, int]]]] = {}

    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 6:
                continue
            frame_id = int(row[0])
            track_id = int(row[1])
            x = float(row[2])
            y = float(row[3])
            w = float(row[4])
            h = float(row[5])
            bbox = (int(x), int(y), int(x + w), int(y + h))
            frames_dict.setdefault(frame_id, []).append((track_id, bbox))

    max_frame = max(frames_dict.keys()) if frames_dict else 0
    result = []
    for fid in range(1, max_frame + 1):
        result.append(frames_dict.get(fid, []))
    return result


def parse_kitti_tracking(label_dir: str | Path) -> list[list[tuple[int, tuple[int, int, int, int]]]]:
    """Parse KITTI tracking labels directory into per-frame lists.

    Each file: frame_id type truncated occluded alpha x1 y1 x2 y2 ...
    """
    label_dir = Path(label_dir)
    frames_dict: dict[int, list[tuple[int, tuple[int, int, int, int]]]] = {}

    for label_file in sorted(label_dir.glob("*.txt")):
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 17:
                    continue
                frame_id = int(parts[0])
                track_id = int(parts[1])
                x1 = int(float(parts[6]))
                y1 = int(float(parts[7]))
                x2 = int(float(parts[8]))
                y2 = int(float(parts[9]))
                frames_dict.setdefault(frame_id, []).append((track_id, (x1, y1, x2, y2)))

    if not frames_dict:
        return []

    max_frame = max(frames_dict.keys())
    result = []
    for fid in range(0, max_frame + 1):
        result.append(frames_dict.get(fid, []))
    return result


def format_aps_predictions(
    tracks_by_frame: list[list[Any]],
) -> list[list[tuple[int, tuple[int, int, int, int]]]]:
    """Convert APS internal track format to evaluation format.

    Each track must have track_id and bbox_xyxy attributes.
    """
    result = []
    for frame_tracks in tracks_by_frame:
        frame_data = []
        for trk in frame_tracks:
            tid = getattr(trk, "track_id", None)
            bbox = getattr(trk, "bbox_xyxy", None)
            if tid is not None and bbox is not None:
                frame_data.append((int(tid), tuple(int(x) for x in bbox)))
        result.append(frame_data)
    return result
