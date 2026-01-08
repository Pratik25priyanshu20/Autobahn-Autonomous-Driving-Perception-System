from __future__ import annotations


def fcw_state(ttc: float | None) -> str:
    if ttc is None:
        return "NORMAL"
    if ttc < 1.5:
        return "CRITICAL"
    if ttc < 2.5:
        return "WARNING"
    if ttc < 3.5:
        return "PRE"
    return "NORMAL"
