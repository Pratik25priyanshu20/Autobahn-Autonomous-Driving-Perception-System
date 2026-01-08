"""Parallel pipeline stage executor (Phase 4.1).

Runs independent perception stages concurrently using ThreadPoolExecutor.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Callable, Dict, List, Tuple


class ParallelStageExecutor:
    """Execute independent pipeline stages in parallel."""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers

    def run(self, stages: Dict[str, Callable[[], Any]]) -> Dict[str, Any]:
        """Run named stages concurrently.

        Args:
            stages: dict of name -> callable (no args, returns result)

        Returns:
            dict of name -> result
        """
        results: Dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures: Dict[str, Future] = {}
            for name, fn in stages.items():
                futures[name] = executor.submit(fn)
            for name, future in futures.items():
                try:
                    results[name] = future.result()
                except Exception as e:
                    results[name] = {"error": str(e)}
        return results
