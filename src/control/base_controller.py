import abc
from typing import Any


class BaseController(abc.ABC):
    @abc.abstractmethod
    def plan(self, world_model: Any) -> Any:
        ...
