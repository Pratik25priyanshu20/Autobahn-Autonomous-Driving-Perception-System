import time


def main():
    start = time.time()
    # Placeholder benchmark results
    results = {
        "fps": 20.0,
        "latency_ms_p50": 80,
        "latency_ms_p95": 120,
        "dropped_frames": 0,
    }
    print("Benchmark results", results)
    print("Elapsed", time.time() - start)


if __name__ == "__main__":
    main()
