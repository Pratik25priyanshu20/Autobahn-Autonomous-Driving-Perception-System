"""Canonical Detection type for APS++."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Detection:
    """Unified detection output used across the entire stack.

    Fields mirror the YOLO detector output but are backend-agnostic.
    """

    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    class_id: int
    class_name: str

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def score(self) -> float:
        return self.conf

    @property
    def label(self) -> str:
        return self.class_name

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)
