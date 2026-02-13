"""Streamlit dashboard for scenario test results visualization.

Run:
    streamlit run src/simulation/metrics_dashboard.py -- --results results/scenario_results.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import streamlit as st

# ======================================================================
# Data loading
# ======================================================================

def load_results(path: Path) -> list[dict[str, Any]]:
    """Load scenario results from a JSON file."""
    path = Path(path)
    if not path.exists():
        st.error(f"Results file not found: {path}")
        return []
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        st.error("Expected a JSON array of scenario results.")
        return []
    return data


# ======================================================================
# Views
# ======================================================================

def render_summary_table(results: list[dict[str, Any]]) -> None:
    """Top-level summary table with pass/fail counts by category."""
    st.subheader("Summary by Category")

    categories: dict[str, dict[str, int]] = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in categories:
            categories[cat] = {"total": 0, "passed": 0, "failed": 0}
        categories[cat]["total"] += 1
        if r.get("passed"):
            categories[cat]["passed"] += 1
        else:
            categories[cat]["failed"] += 1

    rows = []
    for cat, counts in sorted(categories.items()):
        rate = counts["passed"] / max(counts["total"], 1) * 100
        rows.append(
            {
                "Category": cat,
                "Total": counts["total"],
                "Passed": counts["passed"],
                "Failed": counts["failed"],
                "Pass Rate (%)": f"{rate:.1f}",
            }
        )

    st.dataframe(rows, use_container_width=True, hide_index=True)


def render_pass_fail_chart(results: list[dict[str, Any]]) -> None:
    """Bar chart of pass/fail counts by category."""
    st.subheader("Pass / Fail by Category")

    cat_counter: dict[str, dict[str, int]] = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in cat_counter:
            cat_counter[cat] = {"Passed": 0, "Failed": 0}
        if r.get("passed"):
            cat_counter[cat]["Passed"] += 1
        else:
            cat_counter[cat]["Failed"] += 1

    # Build a simple dataframe-style structure for st.bar_chart
    import pandas as pd

    df = pd.DataFrame(cat_counter).T
    df.index.name = "Category"
    st.bar_chart(df, color=["#2ecc71", "#e74c3c"])


def render_per_scenario_metrics(results: list[dict[str, Any]]) -> None:
    """Detailed expandable metrics for each scenario."""
    st.subheader("Per-Scenario Metrics")

    for r in results:
        status_icon = "PASS" if r.get("passed") else "FAIL"
        with st.expander(f"[{status_icon}] {r['scenario_name']} ({r.get('category', '')})"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Metrics:**")
                metrics = r.get("metrics", {})
                for k, v in sorted(metrics.items()):
                    if isinstance(v, float):
                        st.text(f"  {k}: {v:.3f}")
                    else:
                        st.text(f"  {k}: {v}")
            with col2:
                st.markdown("**Violations:**")
                violations = r.get("violations", [])
                if violations:
                    for v in violations:
                        st.markdown(f"- :red[{v}]")
                else:
                    st.markdown(":green[No violations]")

            st.caption(f"Timestamp: {r.get('timestamp', 'N/A')}  |  Duration: {r.get('duration_s', 0):.4f}s")


def render_timeline_view(results: list[dict[str, Any]]) -> None:
    """Timeline of scenario execution ordered by timestamp."""
    st.subheader("Execution Timeline")

    sorted_results = sorted(results, key=lambda r: r.get("timestamp", ""))

    import pandas as pd

    timeline_data = []
    for i, r in enumerate(sorted_results):
        timeline_data.append(
            {
                "Order": i + 1,
                "Scenario": r["scenario_name"],
                "Category": r.get("category", ""),
                "Result": "PASS" if r.get("passed") else "FAIL",
                "Duration (s)": round(r.get("duration_s", 0), 4),
                "Timestamp": r.get("timestamp", ""),
            }
        )

    df = pd.DataFrame(timeline_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_overall_kpi(results: list[dict[str, Any]]) -> None:
    """Render key performance indicators at the top of the dashboard."""
    total = len(results)
    passed = sum(1 for r in results if r.get("passed"))
    failed = total - passed
    pass_rate = passed / max(total, 1) * 100
    n_categories = len({r.get("category", "") for r in results})

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Scenarios", total)
    col2.metric("Passed", passed)
    col3.metric("Failed", failed)
    col4.metric("Pass Rate", f"{pass_rate:.1f}%")
    col5.metric("Categories", n_categories)

    st.divider()


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    st.set_page_config(page_title="APS++ Scenario Dashboard", layout="wide")
    st.title("APS++ Scenario Test Results")
    st.caption("CARLA-style scenario evaluation dashboard for Autobahn Perception Stack")

    # --- Determine results path ---
    # Support both Streamlit CLI args and sidebar file upload
    default_path = Path("results/scenario_results.json")

    # Parse CLI args passed after `--`
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--results", default=str(default_path))
    args, _ = parser.parse_known_args()
    results_path = Path(args.results)

    # Sidebar: allow overriding path or uploading a file
    st.sidebar.header("Data Source")
    path_input = st.sidebar.text_input("Results JSON path", value=str(results_path))
    uploaded = st.sidebar.file_uploader("Or upload results JSON", type=["json"])

    results = json.loads(uploaded.read()) if uploaded is not None else load_results(Path(path_input))

    if not results:
        st.warning("No results loaded. Run scenarios first or provide a valid results file.")
        st.stop()

    # --- Category filter ---
    all_categories = sorted({r.get("category", "unknown") for r in results})
    selected = st.sidebar.multiselect("Filter categories", all_categories, default=all_categories)
    filtered = [r for r in results if r.get("category", "unknown") in selected]

    if not filtered:
        st.warning("No scenarios match the selected filters.")
        st.stop()

    # --- Render views ---
    render_overall_kpi(filtered)
    render_summary_table(filtered)
    render_pass_fail_chart(filtered)
    render_per_scenario_metrics(filtered)
    render_timeline_view(filtered)


if __name__ == "__main__":
    main()
