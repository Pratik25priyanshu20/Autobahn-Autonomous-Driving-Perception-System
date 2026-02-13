"""Diagnostic Trouble Code (DTC) logger for ISO 26262 compliance.

Persists DTC events as JSONL for post-mortem analysis and exposes
runtime queries on active codes.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

logger = get_logger("dtc_logger")


@dataclass
class DTCCode:
    code: str          # e.g. "DTC_DET_001"
    description: str
    severity: str      # "info", "warning", "critical"


# Pre-defined DTC codes
DTC_CODES: dict[str, DTCCode] = {
    "DTC_DET_001": DTCCode("DTC_DET_001", "Detection module timeout", "warning"),
    "DTC_DET_002": DTCCode("DTC_DET_002", "Detection count anomaly", "warning"),
    "DTC_FCW_001": DTCCode("DTC_FCW_001", "FCW false positive rate high", "warning"),
    "DTC_FCW_002": DTCCode("DTC_FCW_002", "FCW module failure", "critical"),
    "DTC_TRK_001": DTCCode("DTC_TRK_001", "Tracker ID discontinuity", "info"),
    "DTC_SEN_001": DTCCode("DTC_SEN_001", "Sensor input degraded", "warning"),
    "DTC_SEN_002": DTCCode("DTC_SEN_002", "Sensor input lost", "critical"),
    "DTC_FUS_001": DTCCode("DTC_FUS_001", "Fusion disagreement above threshold", "warning"),
    "DTC_PLC_001": DTCCode("DTC_PLC_001", "Plausibility violation detected", "warning"),
    "DTC_PLC_002": DTCCode("DTC_PLC_002", "Critical plausibility failure", "critical"),
}


class DTCLogger:
    """Logs DTC events to a JSONL file and tracks active codes at runtime."""

    def __init__(self, output_dir: Path | str):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._output_dir / "dtc_log.jsonl"
        self._active: dict[str, dict[str, Any]] = {}

    def log(self, code: str, details: dict[str, Any] | None = None, frame_id: int = 0) -> None:
        """Record a DTC event."""
        dtc_def = DTC_CODES.get(code)
        severity = dtc_def.severity if dtc_def else "info"
        description = dtc_def.description if dtc_def else code

        record: dict[str, Any] = {
            "timestamp": time.time(),
            "frame_id": frame_id,
            "code": code,
            "description": description,
            "severity": severity,
            "details": details or {},
        }

        # Persist to JSONL
        try:
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError:
            logger.error("Failed to write DTC record for %s", code)

        # Track as active
        self._active[code] = record
        logger.info("[DTC] %s (%s): %s", code, severity, description)

    def get_active(self) -> list[dict[str, Any]]:
        """Return list of currently active DTC records."""
        return list(self._active.values())

    def has_critical(self) -> bool:
        """Return ``True`` if any active DTC has severity 'critical'."""
        return any(r["severity"] == "critical" for r in self._active.values())

    def clear(self, code: str) -> None:
        """Clear (deactivate) a specific DTC code."""
        if code in self._active:
            del self._active[code]
            logger.info("[DTC] cleared %s", code)

    def summary(self) -> dict[str, int]:
        """Return count of active DTCs grouped by severity."""
        counts: dict[str, int] = {"info": 0, "warning": 0, "critical": 0}
        for record in self._active.values():
            sev = record.get("severity", "info")
            counts[sev] = counts.get(sev, 0) + 1
        return counts
