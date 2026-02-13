"""Parallel pipeline stage executor (Phase 4.1).

Runs independent perception stages concurrently using ThreadPoolExecutor.
"""
from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any


class ParallelStageExecutor:
    """Execute independent pipeline stages in parallel."""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers

    def run(self, stages: dict[str, Callable[[], Any]]) -> dict[str, Any]:
        """Run named stages concurrently.

        Args:
            stages: dict of name -> callable (no args, returns result)

        Returns:
            dict of name -> result
        """
        results: dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures: dict[str, Future] = {}
            for name, fn in stages.items():
                futures[name] = executor.submit(fn)
            for name, future in futures.items():
                try:
                    results[name] = future.result()
                except Exception as e:
                    results[name] = {"error": str(e)}
        return results
