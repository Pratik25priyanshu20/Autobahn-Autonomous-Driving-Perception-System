from typing import Any, Dict

from src.utils.logger import get_logger


class HealthMonitor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(__name__)

    def check_latency(self, latency_ms: float) -> bool:
        budget = self.config.get("watchdog_ms", 0)
        if budget and latency_ms > budget:
            self.logger.warning("Latency budget exceeded: %.2f ms", latency_ms)
            return False
        return True

    def degraded(self) -> bool:
        return bool(self.config.get("degrade_on_miss", True))
