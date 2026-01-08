from __future__ import annotations

from typing import Any, Dict

from src.utils.logger import get_logger


class HealthMonitor:
    """Monitors frame-processing latency and triggers degraded mode."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(__name__)
        self.watchdog_ms: float = float(config.get("watchdog_ms", 100))
        self.degraded_after_misses: int = int(config.get("degraded_after_misses", 3))
        self._consecutive_misses: int = 0
        self._is_degraded: bool = False

    def check_latency(self, latency_ms: float) -> bool:
        """Returns True if within budget, False if budget exceeded."""
        if self.watchdog_ms > 0 and latency_ms > self.watchdog_ms:
            self._consecutive_misses += 1
            if self._consecutive_misses >= self.degraded_after_misses and not self._is_degraded:
                self._is_degraded = True
                self.logger.warning(
                    "Entering degraded mode after %d consecutive misses (%.1f ms > %.1f ms budget)",
                    self._consecutive_misses,
                    latency_ms,
                    self.watchdog_ms,
                )
            return False
        else:
            if self._consecutive_misses > 0:
                self._consecutive_misses = max(0, self._consecutive_misses - 1)
            if self._consecutive_misses == 0 and self._is_degraded:
                self._is_degraded = False
                self.logger.info("Exiting degraded mode — latency recovered")
            return True

    def degraded(self) -> bool:
        """True when the system should run in degraded (reduced-feature) mode."""
        return self._is_degraded

    def reset(self) -> None:
        self._consecutive_misses = 0
        self._is_degraded = False
