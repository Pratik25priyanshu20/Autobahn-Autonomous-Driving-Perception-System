"""ISO 26262 ASIL classification for APS++ components.

Assigns Automotive Safety Integrity Levels (QM, A, B, C, D) to each
perception / safety component and determines the required redundancy
and escalation strategy.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.utils.logger import get_logger

logger = get_logger("asil_classifier")


class ASILLevel(str, Enum):
    QM = "QM"  # Quality Management (no safety requirement)
    A = "A"
    B = "B"
    C = "C"
    D = "D"  # Most stringent


@dataclass
class ComponentASIL:
    component: str
    level: ASILLevel
    rationale: str


# Static ASIL assignments for APS++ components
COMPONENT_ASIL_MAP: dict[str, ComponentASIL] = {
    "detection": ComponentASIL("detection", ASILLevel.B, "Object detection failure may cause late braking"),
    "tracking": ComponentASIL("tracking", ASILLevel.B, "Track loss can cause phantom braking or missed obstacles"),
    "lane_detection": ComponentASIL("lane_detection", ASILLevel.A, "Lane departure is comfort, not safety-critical"),
    "fcw": ComponentASIL("fcw", ASILLevel.C, "FCW false negative may lead to rear-end collision"),
    "ttc": ComponentASIL("ttc", ASILLevel.C, "TTC underestimate causes late warning"),
    "bsd": ComponentASIL("bsd", ASILLevel.B, "Blind spot miss causes unsafe lane change"),
    "depth_estimation": ComponentASIL("depth_estimation", ASILLevel.A, "Depth aids distance but is not sole source"),
    "lidar_processing": ComponentASIL("lidar_processing", ASILLevel.B, "LIDAR data loss reduces 3D perception"),
    "fusion": ComponentASIL("fusion", ASILLevel.C, "Fusion errors propagate to all downstream safety"),
    "controller": ComponentASIL("controller", ASILLevel.D, "Control output directly affects vehicle behavior"),
}

# Escalation strategies keyed by ASIL level
_ESCALATION_MAP: dict[ASILLevel, str] = {
    ASILLevel.QM: "none",
    ASILLevel.A: "monitoring",
    ASILLevel.B: "monitoring",
    ASILLevel.C: "redundant",
    ASILLevel.D: "fail_safe",
}


class ASILClassifier:
    """Lookup and query ASIL assignments for APS++ components."""

    def __init__(self, overrides: dict[str, str] | None = None):
        self._map: dict[str, ComponentASIL] = dict(COMPONENT_ASIL_MAP)
        if overrides:
            for comp, level_str in overrides.items():
                try:
                    level = ASILLevel(level_str)
                except ValueError:
                    logger.warning("Invalid ASIL override for %s: %s", comp, level_str)
                    continue
                if comp in self._map:
                    self._map[comp] = ComponentASIL(comp, level, self._map[comp].rationale)
                else:
                    self._map[comp] = ComponentASIL(comp, level, "User override")

    def get_level(self, component: str) -> ASILLevel:
        """Return the ASIL level for *component*, defaulting to QM."""
        entry = self._map.get(component)
        if entry is None:
            return ASILLevel.QM
        return entry.level

    def requires_redundancy(self, component: str) -> bool:
        """Return ``True`` for ASIL-C and ASIL-D components."""
        return self.get_level(component) in (ASILLevel.C, ASILLevel.D)

    def escalation_level(self, component: str) -> str:
        """Return the escalation strategy: none / monitoring / redundant / fail_safe."""
        level = self.get_level(component)
        return _ESCALATION_MAP.get(level, "none")

    def get_all_assignments(self) -> dict[str, ComponentASIL]:
        """Return a copy of the full component -> ASIL mapping."""
        return dict(self._map)
