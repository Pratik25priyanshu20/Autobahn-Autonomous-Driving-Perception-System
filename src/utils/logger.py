from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def setup_logger(name: str = "aps", log_dir: str | Path = "results", level: int | str = logging.INFO) -> logging.Logger:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    numeric_level = getattr(logging, str(level).upper(), level)
    logger.setLevel(numeric_level)

    # Prevent duplicate handlers across re-runs.
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setLevel(numeric_level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_dir / "run.log", encoding="utf-8")
    fh.setLevel(numeric_level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# Backward compatibility alias.
def get_logger(name: Optional[str] = None, level: str = "INFO", log_file: Optional[Path] = None) -> logging.Logger:
    log_dir = log_file.parent if log_file else "results"
    return setup_logger(name=name or "aps", log_dir=log_dir, level=level)
