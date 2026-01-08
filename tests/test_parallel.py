"""Tests for parallel executor (Phase 4.1)."""
import time

from src.runtime.parallel_executor import ParallelStageExecutor


def test_parallel_runs_stages():
    executor = ParallelStageExecutor(max_workers=2)

    def stage_a():
        return "result_a"

    def stage_b():
        return "result_b"

    results = executor.run({"a": stage_a, "b": stage_b})
    assert results["a"] == "result_a"
    assert results["b"] == "result_b"


def test_parallel_handles_errors():
    executor = ParallelStageExecutor(max_workers=2)

    def failing_stage():
        raise ValueError("test error")

    results = executor.run({"fail": failing_stage})
    assert "error" in results["fail"]


def test_parallel_actually_concurrent():
    executor = ParallelStageExecutor(max_workers=4)

    def slow_stage():
        time.sleep(0.1)
        return True

    start = time.perf_counter()
    results = executor.run({f"s{i}": slow_stage for i in range(4)})
    elapsed = time.perf_counter() - start
    assert all(v is True for v in results.values())
    # If truly parallel, 4 x 0.1s stages should complete in ~0.1s, not 0.4s
    assert elapsed < 0.3
