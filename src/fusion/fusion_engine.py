from typing import Optional

from src.fusion.world_model import build_world_model
from src.utils.types import PerceptionOutput, WorldModel


def fuse(perception: PerceptionOutput) -> Optional[WorldModel]:
    if perception is None:
        return None
    return build_world_model(perception)
