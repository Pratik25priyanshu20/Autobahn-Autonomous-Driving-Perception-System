"""Safety response evaluation metrics (Phase 6.4)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SafetyEvent:
    frame_id: int
    timestamp_s: float
    state: str


@dataclass
class SafetyResponseResult:
    mean_response_ms: float = 0.0
    max_response_ms: float = 0.0
    total_events: int = 0
    missed_events: int = 0


class SafetyResponseEvaluator:
    """Measures response latency from ground-truth events to system alerts."""

    def evaluate(
        self,
        gt_events: List[SafetyEvent],
        system_events: List[SafetyEvent],
        max_delay_s: float = 2.0,
    ) -> SafetyResponseResult:
        if not gt_events:
            return SafetyResponseResult()

        delays_ms: List[float] = []
        missed = 0

        for gt in gt_events:
            best_delay = None
            for sys_ev in system_events:
                if sys_ev.state == gt.state and sys_ev.timestamp_s >= gt.timestamp_s:
                    delay = sys_ev.timestamp_s - gt.timestamp_s
                    if delay <= max_delay_s:
                        if best_delay is None or delay < best_delay:
                            best_delay = delay

            if best_delay is not None:
                delays_ms.append(best_delay * 1000.0)
            else:
                missed += 1

        return SafetyResponseResult(
            mean_response_ms=sum(delays_ms) / max(len(delays_ms), 1),
            max_response_ms=max(delays_ms) if delays_ms else 0.0,
            total_events=len(gt_events),
            missed_events=missed,
        )
