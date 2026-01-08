from typing import Iterator

from src.inputs.base_input import BaseInput
from src.utils.types import FramePacket


class CarlaInput(BaseInput):
    def __init__(self, host: str = "localhost", port: int = 2000):
        self.host = host
        self.port = port

    def start(self) -> None:
        # Connect to CARLA client here
        pass

    def frames(self) -> Iterator[FramePacket]:
        # Yield FramePacket objects from CARLA sensors
        return iter(())

    def stop(self) -> None:
        pass
