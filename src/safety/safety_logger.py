import json
from pathlib import Path


class SafetyLogger:
    def __init__(self, run_dir: Path):
        self.log_path = run_dir / "safety_events.jsonl"
        self.last_state = None
        self.log_path.touch(exist_ok=True)

    def log(self, frame_idx: int, timestamp_s: float, safety_state: str, message: str, details: dict):
        """Append an event only when safety state changes."""
        if safety_state == self.last_state:
            return
        event = {
            "frame": frame_idx,
            "time_s": round(timestamp_s, 3),
            "state": safety_state,
            "message": message,
            "details": details,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(event) + "\n")
        self.last_state = safety_state
