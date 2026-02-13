"""Scenario runner for CARLA-style scenario evaluation.

Loads YAML scenario definitions, evaluates pass/fail criteria against
collected metrics, and reports results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.utils.logger import get_logger

logger = get_logger("simulation.scenario_runner")


@dataclass
class ScenarioConfig:
    """Configuration for a single simulation scenario."""

    name: str
    description: str
    category: str  # "cut_in", "pedestrian", "braking", "weather", "sensor_failure", "merge", "highway"
    duration_s: float
    conditions: dict[str, Any] = field(default_factory=dict)
    actors: list[dict[str, Any]] = field(default_factory=list)
    pass_criteria: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> ScenarioConfig:
        """Load a scenario configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Scenario file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(
            name=data["name"],
            description=data["description"],
            category=data["category"],
            duration_s=float(data["duration_s"]),
            conditions=data.get("conditions", {}),
            actors=data.get("actors", []),
            pass_criteria=data.get("pass_criteria", {}),
        )


@dataclass
class ScenarioResult:
    """Result of a single scenario evaluation."""

    scenario_name: str
    passed: bool
    duration_s: float
    metrics: dict[str, float] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)
    timestamp: str = ""
    category: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to a plain dictionary."""
        return {
            "scenario_name": self.scenario_name,
            "passed": self.passed,
            "duration_s": self.duration_s,
            "metrics": self.metrics,
            "violations": self.violations,
            "timestamp": self.timestamp,
            "category": self.category,
        }


class ScenarioRunner:
    """Loads and evaluates YAML scenarios with pass/fail criteria."""

    def __init__(self, scenario_dir: Path = Path("src/simulation/scenarios")) -> None:
        self.scenario_dir = Path(scenario_dir)
        logger.info("ScenarioRunner initialised with dir: %s", self.scenario_dir)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_scenario(self, path: Path) -> ScenarioConfig:
        """Load a single scenario from a YAML path."""
        config = ScenarioConfig.from_yaml(path)
        logger.info("Loaded scenario: %s (%s)", config.name, config.category)
        return config

    def load_all(self, scenario_dir: Path | None = None) -> list[ScenarioConfig]:
        """Load every ``*.yaml`` file in *scenario_dir*."""
        search_dir = Path(scenario_dir) if scenario_dir else self.scenario_dir
        yamls = sorted(search_dir.glob("*.yaml"))
        if not yamls:
            logger.warning("No YAML scenarios found in %s", search_dir)
            return []
        configs: list[ScenarioConfig] = []
        for yp in yamls:
            try:
                configs.append(self.load_scenario(yp))
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to load %s: %s", yp.name, exc)
        return configs

    # ------------------------------------------------------------------
    # Criteria evaluation
    # ------------------------------------------------------------------

    def evaluate_criteria(
        self, config: ScenarioConfig, metrics: dict[str, float]
    ) -> ScenarioResult:
        """Compare *metrics* against the pass criteria in *config*.

        Returns a :class:`ScenarioResult` with pass/fail status and any
        violations found.
        """
        violations: list[str] = []
        criteria = config.pass_criteria

        # --- boolean criteria (e.g. fcw_triggered, no_collision) ---
        for key in ("fcw_triggered", "aeb_triggered", "no_collision"):
            if key in criteria:
                expected = bool(criteria[key])
                actual = bool(metrics.get(key, not expected))
                if actual != expected:
                    violations.append(
                        f"{key}: expected {expected}, got {actual}"
                    )

        # --- upper-bound criteria (metric must be <= threshold) ---
        upper_keys = {
            "fcw_latency_ms_max": "fcw_latency_ms",
            "aeb_latency_ms_max": "aeb_latency_ms",
            "max_lateral_deviation_m": "lateral_deviation_m",
            "max_jerk_m_s3": "jerk_m_s3",
            "max_speed_deviation_kmh": "speed_deviation_kmh",
        }
        for crit_key, metric_key in upper_keys.items():
            if crit_key in criteria:
                threshold = float(criteria[crit_key])
                actual = float(metrics.get(metric_key, threshold + 1))
                if actual > threshold:
                    violations.append(
                        f"{metric_key}: {actual:.2f} exceeds max {threshold:.2f}"
                    )

        # --- lower-bound criteria (metric must be >= threshold) ---
        lower_keys = {
            "min_ttc_s": "ttc_s",
            "detection_rate_min": "detection_rate",
            "tracking_accuracy_min": "tracking_accuracy",
            "min_visibility_score": "visibility_score",
        }
        for crit_key, metric_key in lower_keys.items():
            if crit_key in criteria:
                threshold = float(criteria[crit_key])
                actual = float(metrics.get(metric_key, threshold - 1))
                if actual < threshold:
                    violations.append(
                        f"{metric_key}: {actual:.2f} below min {threshold:.2f}"
                    )

        passed = len(violations) == 0
        now = datetime.now(tz=timezone.utc).isoformat()

        return ScenarioResult(
            scenario_name=config.name,
            passed=passed,
            duration_s=config.duration_s,
            metrics=metrics,
            violations=violations,
            timestamp=now,
            category=config.category,
        )

    # ------------------------------------------------------------------
    # Batch run (load + evaluate with supplied metric-generator)
    # ------------------------------------------------------------------

    def run_all(
        self,
        scenario_dir: Path | None = None,
        metric_fn: Any | None = None,
    ) -> list[ScenarioResult]:
        """Load all scenarios and evaluate them.

        Parameters
        ----------
        scenario_dir:
            Override for the scenario directory.
        metric_fn:
            Optional callable ``(ScenarioConfig) -> Dict[str, float]``.
            If *None*, scenarios are evaluated with empty metrics (all
            criteria will fail).
        """
        configs = self.load_all(scenario_dir)
        results: list[ScenarioResult] = []
        for cfg in configs:
            metrics = metric_fn(cfg) if metric_fn else {}
            result = self.evaluate_criteria(cfg, metrics)
            results.append(result)
            status = "PASS" if result.passed else "FAIL"
            logger.info("[%s] %s", status, cfg.name)
        return results
