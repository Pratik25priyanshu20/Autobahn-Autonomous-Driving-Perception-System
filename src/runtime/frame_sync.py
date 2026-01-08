from collections import deque
from typing import Deque, Optional

from src.utils.types import FramePacket


class FrameSync:
    def __init__(self, max_buffer: int = 4):
        self.buffer: Deque[FramePacket] = deque(maxlen=max_buffer)

    def push(self, packet: FramePacket) -> None:
        self.buffer.append(packet)

    def latest(self) -> Optional[FramePacket]:
        return self.buffer[-1] if self.buffer else None
