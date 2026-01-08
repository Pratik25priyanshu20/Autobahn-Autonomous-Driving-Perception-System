from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator


@contextmanager
def timing(label: str) -> Iterator[None]:
    start = time.perf_counter()
    yield
    duration_ms = (time.perf_counter() - start) * 1000
    print(f"{label} took {duration_ms:.2f} ms")


@dataclass
class StageTimer:
    """Lightweight per-stage timing for a single frame."""

    start_ts: float = field(default_factory=time.perf_counter)
    stages_ms: Dict[str, float] = field(default_factory=dict)

    def mark(self, stage_name: str, stage_start_ts: float) -> None:
        self.stages_ms[stage_name] = (time.perf_counter() - stage_start_ts) * 1000.0


@dataclass
class FPSMeter:
    """Exponential moving average FPS estimator."""

    smoothing: float = 0.9
    fps: float = 0.0
    _last_ts: float = field(default_factory=time.perf_counter)

    def tick(self) -> float:
        now = time.perf_counter()
        dt = max(now - self._last_ts, 1e-9)
        inst_fps = 1.0 / dt
        self.fps = inst_fps if self.fps <= 0 else (self.smoothing * self.fps + (1 - self.smoothing) * inst_fps)
        self._last_ts = now
        return self.fps
