import abc
from typing import Iterator

from src.utils.types import FramePacket


class BaseInput(abc.ABC):
    @abc.abstractmethod
    def start(self) -> None:
        ...

    @abc.abstractmethod
    def stop(self) -> None:
        ...

    @abc.abstractmethod
    def frames(self) -> Iterator[FramePacket]:
        ...
