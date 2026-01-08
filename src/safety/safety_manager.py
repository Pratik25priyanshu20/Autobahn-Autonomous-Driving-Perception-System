from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class SafetyState(str, Enum):
    NORMAL = "NORMAL"
    AWARENESS = "AWARENESS"  # FCW-PRE (heads up)
    CAUTION = "CAUTION"  # mild risk
    WARNING = "WARNING"  # high risk
    CRITICAL = "CRITICAL"  # imminent risk


@dataclass
class SafetyOutput:
    state: SafetyState
    message: str
    color: tuple[int, int, int]
    details: Dict[str, Any]


class SafetyManager:
    """Unifies LDW/FCW signals into a single safety state."""

    def evaluate(
        self,
        ldw_departure: Optional[str],
        fcw_state: Optional[str],
        fcw_ttc_s: Optional[float],
        fcw_pre_active: bool,
        lane_ok: bool,
    ) -> SafetyOutput:
        fcw_state = (fcw_state or "NORMAL").upper()
        state = SafetyState.NORMAL
        msg_parts = []

        if fcw_state == "CRITICAL":
            state = SafetyState.CRITICAL
            msg_parts.append("FCW CRITICAL")
        elif fcw_state == "WARNING":
            state = SafetyState.WARNING
            msg_parts.append("FCW WARNING")
        elif fcw_state == "CAUTION":
            state = SafetyState.CAUTION
            msg_parts.append("FCW CAUTION")
        elif fcw_pre_active:
            state = SafetyState.AWARENESS
            msg_parts.append("FCW PRE")

        if lane_ok and ldw_departure is not None:
            if state in (SafetyState.NORMAL, SafetyState.AWARENESS):
                state = SafetyState.CAUTION
            msg_parts.append(f"LDW {ldw_departure}")

        if fcw_ttc_s is not None and fcw_state in ("CAUTION", "WARNING", "CRITICAL"):
            msg_parts.append(f"TTC={fcw_ttc_s:.2f}s")

        message = "System OK" if not msg_parts else " | ".join(msg_parts)

        if state == SafetyState.NORMAL:
            color = (0, 255, 0)
        elif state == SafetyState.AWARENESS:
            color = (0, 255, 255)
        elif state == SafetyState.CAUTION:
            color = (0, 200, 255)
        elif state == SafetyState.WARNING:
            color = (0, 0, 255)
        else:
            color = (0, 0, 255)

        details = {
            "ldw_departure": ldw_departure,
            "fcw_state": fcw_state,
            "fcw_ttc_s": fcw_ttc_s,
            "fcw_pre_active": fcw_pre_active,
            "lane_ok": lane_ok,
        }

        return SafetyOutput(state=state, message=message, color=color, details=details)
