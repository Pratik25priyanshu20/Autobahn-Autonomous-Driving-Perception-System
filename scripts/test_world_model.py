import time

from src.fusion.ego_state import EgoState
from src.fusion.lane_geometry import LaneGeometry
from src.fusion.tracked_object import TrackedObject
from src.fusion.world_model_struct import WorldModel


def main():
    world = WorldModel(timestamp=time.time(), ego=EgoState(speed=12.5))

    obj = TrackedObject(
        track_id=1,
        cls="car",
        bbox=(100, 100, 200, 200),
        confidence=0.91,
        x=5.2,
        y=18.3,
    )
    world.objects.append(obj)

    world.lanes = LaneGeometry(
        left_lane="left_poly",
        right_lane="right_poly",
        ego_offset_px=-32.5,
        confidence=0.98,
    )

    world.confidence["detection"] = 0.91
    world.confidence["lane"] = 0.98

    assert world.timestamp is not None
    assert world.ego.speed > 0
    assert isinstance(world.objects, list)
    assert world.lanes is not None

    print(world.summary())
    print(world)


if __name__ == "__main__":
    main()
