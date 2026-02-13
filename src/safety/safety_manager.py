from __future__ import annotations

from typing import Any

from src.types.safety import SafetyOutput, SafetyStateEnum

# Re-export for backward compat
SafetyState = SafetyStateEnum


class SafetyManager:
    """Unifies LDW/FCW/BSD signals into a single safety state."""

    def evaluate_with_asil(
        self,
        ldw_departure: str | None = None,
        fcw_state: str | None = None,
        fcw_ttc_s: float | None = None,
        fcw_pre_active: bool = False,
        lane_ok: bool = False,
        bsd_warnings: list[dict[str, Any]] | None = None,
        plausibility_violations: list[Any] | None = None,
        dtc_logger: Any | None = None,
        frame_id: int = 0,
    ) -> SafetyOutput:
        """Extended evaluation that incorporates plausibility violations and DTC integration.

        Delegates to :meth:`evaluate` for the base safety state and then
        escalates if there are critical plausibility violations.  Any
        violations are logged as DTCs when a *dtc_logger* is provided.
        """
        base = self.evaluate(
            ldw_departure=ldw_departure,
            fcw_state=fcw_state,
            fcw_ttc_s=fcw_ttc_s,
            fcw_pre_active=fcw_pre_active,
            lane_ok=lane_ok,
            bsd_warnings=bsd_warnings,
        )

        violations = plausibility_violations or []
        state = base.state
        msg_parts: list[str] = [base.message] if base.message != "System OK" else []

        has_critical_plausibility = False
        for v in violations:
            severity = getattr(v, "severity", "info")
            check_name = getattr(v, "check_name", "unknown")
            description = getattr(v, "description", "")

            # Log DTC for each violation
            if dtc_logger is not None:
                code = "DTC_PLC_002" if severity == "critical" else "DTC_PLC_001"
                dtc_logger.log(code, details={"check": check_name, "description": description}, frame_id=frame_id)

            if severity == "critical":
                has_critical_plausibility = True
                msg_parts.append(f"PLAUS CRITICAL: {check_name}")
            elif severity == "warning":
                msg_parts.append(f"PLAUS WARN: {check_name}")

        # Escalate state if critical plausibility violation
        if has_critical_plausibility and state in (SafetyStateEnum.NORMAL, SafetyStateEnum.AWARENESS, SafetyStateEnum.CAUTION):
            state = SafetyStateEnum.WARNING

        # Check DTC-level criticality
        if dtc_logger is not None and dtc_logger.has_critical() and state in (SafetyStateEnum.NORMAL, SafetyStateEnum.AWARENESS):
            state = SafetyStateEnum.WARNING
            msg_parts.append("DTC CRITICAL active")

        message = "System OK" if not msg_parts else " | ".join(msg_parts)

        # Re-derive color from possibly escalated state
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

        details = dict(base.details)
        details["plausibility_violations"] = len(violations)
        details["plausibility_critical"] = has_critical_plausibility

        return SafetyOutput(state=state, message=message, color=color, details=details)

    def evaluate(
        self,
        ldw_departure: str | None = None,
        fcw_state: str | None = None,
        fcw_ttc_s: float | None = None,
        fcw_pre_active: bool = False,
        lane_ok: bool = False,
        bsd_warnings: list[dict[str, Any]] | None = None,
    ) -> SafetyOutput:
        fcw_state = (fcw_state or "NORMAL").upper()
        state = SafetyStateEnum.NORMAL
        msg_parts: list[str] = []

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

        details: dict[str, Any] = {
            "ldw_departure": ldw_departure,
            "fcw_state": fcw_state,
            "fcw_ttc_s": fcw_ttc_s,
            "fcw_pre_active": fcw_pre_active,
            "lane_ok": lane_ok,
            "bsd_warnings": bsd_warnings or [],
        }

        return SafetyOutput(state=state, message=message, color=color, details=details)
