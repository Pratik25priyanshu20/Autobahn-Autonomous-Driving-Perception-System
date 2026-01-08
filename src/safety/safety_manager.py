from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.types.safety import SafetyOutput, SafetyStateEnum

# Re-export for backward compat
SafetyState = SafetyStateEnum


class SafetyManager:
    """Unifies LDW/FCW/BSD signals into a single safety state."""

    def evaluate(
        self,
        ldw_departure: Optional[str] = None,
        fcw_state: Optional[str] = None,
        fcw_ttc_s: Optional[float] = None,
        fcw_pre_active: bool = False,
        lane_ok: bool = False,
        bsd_warnings: Optional[List[Dict[str, Any]]] = None,
    ) -> SafetyOutput:
        fcw_state = (fcw_state or "NORMAL").upper()
        state = SafetyStateEnum.NORMAL
        msg_parts: List[str] = []

        if fcw_state == "CRITICAL":
            state = SafetyStateEnum.CRITICAL
            msg_parts.append("FCW CRITICAL")
        elif fcw_state == "WARNING":
            state = SafetyStateEnum.WARNING
            msg_parts.append("FCW WARNING")
        elif fcw_state == "CAUTION":
            state = SafetyStateEnum.CAUTION
            msg_parts.append("FCW CAUTION")
        elif fcw_pre_active:
            state = SafetyStateEnum.AWARENESS
            msg_parts.append("FCW PRE")

        if lane_ok and ldw_departure is not None:
            if state in (SafetyStateEnum.NORMAL, SafetyStateEnum.AWARENESS):
                state = SafetyStateEnum.CAUTION
            msg_parts.append(f"LDW {ldw_departure}")

        # Phase 3.2: BSD/RCTA escalation
        if bsd_warnings:
            for bw in bsd_warnings:
                side = bw.get("side", "?")
                msg_parts.append(f"BSD {side.upper()}")
                if state in (SafetyStateEnum.NORMAL, SafetyStateEnum.AWARENESS):
                    state = SafetyStateEnum.CAUTION

        if fcw_ttc_s is not None and fcw_state in ("CAUTION", "WARNING", "CRITICAL"):
            msg_parts.append(f"TTC={fcw_ttc_s:.2f}s")

        message = "System OK" if not msg_parts else " | ".join(msg_parts)

        if state == SafetyStateEnum.NORMAL:
            color = (0, 255, 0)
        elif state == SafetyStateEnum.AWARENESS:
            color = (0, 255, 255)
        elif state == SafetyStateEnum.CAUTION:
            color = (0, 200, 255)
        elif state == SafetyStateEnum.WARNING:
            color = (0, 0, 255)
        else:
            color = (0, 0, 255)

        details: Dict[str, Any] = {
            "ldw_departure": ldw_departure,
            "fcw_state": fcw_state,
            "fcw_ttc_s": fcw_ttc_s,
            "fcw_pre_active": fcw_pre_active,
            "lane_ok": lane_ok,
            "bsd_warnings": bsd_warnings or [],
        }

        return SafetyOutput(state=state, message=message, color=color, details=details)
