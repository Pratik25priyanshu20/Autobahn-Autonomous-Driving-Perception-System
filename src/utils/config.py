from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path.resolve()}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get(cfg: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Dot-access helper:
      get(cfg, "runtime.output_dir", "results")
    """
    cur: Any = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


# Backward compatibility with earlier helper name.
load_yaml_config = load_yaml
