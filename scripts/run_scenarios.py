"""CLI for batch scenario execution.

Usage:
    python scripts/run_scenarios.py --scenarios src/simulation/scenarios/ --output results/scenario_results.json
    python scripts/run_scenarios.py --filter weather --output results/weather_results.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is on the Python path so ``src.*`` imports resolve.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from rich.console import Console  # noqa: E402

from src.simulation.test_harness import TestHarness  # noqa: E402

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run APS++ scenario tests")
    parser.add_argument(
        "--scenarios",
        default="src/simulation/scenarios",
        help="Path to scenario YAML directory",
    )
    parser.add_argument(
        "--output",
        default="results/scenario_results.json",
        help="Output results JSON path",
    )
    parser.add_argument(
        "--filter",
        default=None,
        dest="filter_category",
        help="Filter by category (e.g., weather, cut_in, pedestrian, braking, sensor_failure, merge, highway)",
    )
    args = parser.parse_args()

    scenario_dir = Path(args.scenarios)
    output_path = Path(args.output)

    if not scenario_dir.exists():
        console.print(f"[red]Scenario directory not found:[/red] {scenario_dir}")
        sys.exit(1)

    console.rule("[bold cyan]APS++ Scenario Test Runner[/bold cyan]")
    console.print(f"Scenarios : {scenario_dir}")
    console.print(f"Output    : {output_path}")
    if args.filter_category:
        console.print(f"Filter    : {args.filter_category}")
    console.print()

    harness = TestHarness(scenario_dir=scenario_dir, output_dir=output_path.parent)

    results = harness.run_all(filter_category=args.filter_category)

    if not results:
        console.print("[yellow]No scenarios were executed.[/yellow]")
        sys.exit(0)

    harness.print_summary(results)
    harness.export_results(results, output_path)

    console.print()
    console.print(f"[bold green]Results written to {output_path}[/bold green]")


if __name__ == "__main__":
    main()
