import json
from collections import Counter
from pathlib import Path


def generate_report(run_dir: Path):
    events_file = run_dir / "safety_events.jsonl"
    report_file = run_dir / "safety_report.json"

    states = []
    messages = []
    times = []

    if not events_file.exists():
        print(f"⚠️ No events file found at {events_file}")
        return

    with events_file.open() as f:
        for line in f:
            e = json.loads(line)
            states.append(e.get("state"))
            messages.append(e.get("message"))
            times.append(e.get("time_s"))

    report = {
        "total_events": len(states),
        "state_counts": dict(Counter(states)),
        "messages": dict(Counter(messages)),
        "first_event_time_s": min(times) if times else None,
        "last_event_time_s": max(times) if times else None,
    }

    with report_file.open("w") as f:
        json.dump(report, f, indent=2)

    print(f"✅ Safety report written to {report_file}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python scripts/generate_safety_report.py <run_dir>")
        sys.exit(1)
    generate_report(Path(sys.argv[1]))
