"""Canonical safety types for APS++."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SafetyStateEnum(str, Enum):
    NORMAL = "NORMAL"
    AWARENESS = "AWARENESS"
    CAUTION = "CAUTION"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class SafetyStatus:
    """Output from rule-based safety evaluation."""

    ttc_s: float | None = None
    risk_score: float | None = None
    warnings: list[str] = field(default_factory=list)
    degraded_mode: bool = False


@dataclass
class SafetyState:
    """Simple safety state container."""

    state: str = "NORMAL"
    message: str = "System OK"


@dataclass
class SafetyOutput:
    """Unified safety manager output."""

    state: SafetyStateEnum
    message: str
    color: tuple[int, int, int] = (0, 255, 0)
    details: dict[str, Any] = field(default_factory=dict)
