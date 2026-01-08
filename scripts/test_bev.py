from __future__ import annotations

"""
Quick BEV unit test (no video required).
Creates a mock world with two objects ahead of ego and writes an image.
"""

from dataclasses import dataclass, field
from pathlib import Path

from src.bev import BEVRenderer
from src.fusion.tracked_object import TrackedObject


@dataclass
class MockWorld:
    objects: list[TrackedObject] = field(default_factory=list)
    lanes: dict | None = None


def main():
    world = MockWorld()
    # Two objects ahead of ego, one centered, one slightly right
    world.objects = [
        TrackedObject(track_id=1, cls="car", bbox=(0, 0, 0, 0), confidence=0.9, x=0.0, y=15.0),
        TrackedObject(track_id=2, cls="car", bbox=(0, 0, 0, 0), confidence=0.9, x=2.0, y=25.0),
    ]

    bev = BEVRenderer(size=500, pixels_per_meter=10.0)
    bev_img = bev.render(world, fcw={"lead_id": 1, "state": "WARNING"})

    out_path = Path("results") / "test_bev.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import cv2  # local import to keep dependency localized

    cv2.imwrite(str(out_path), bev_img)
    print(f"✅ BEV image written to {out_path}")


if __name__ == "__main__":
    main()
