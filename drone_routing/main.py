"""Entry point for drone delivery route optimization experiments."""

from __future__ import annotations

import logging
from pathlib import Path

from src.constraints import ConstraintConfig
from src.graph import GraphConfig
from src.simulator import ScenarioConfig, run_experiments
from src.visualization import plot_metric_comparison, plot_scenario_routes


def _print_summary(summary: dict[str, dict[str, float]]) -> None:
    print("\nAlgorithm Comparison")
    print("-" * 92)
    print(
        f"{'Algorithm':<12}{'Avg Cost':>12}{'Avg Time(ms)':>15}{'Avg Explored':>15}"
        f"{'Feasible %':>13}{'Avg Delivery Cost':>20}"
    )
    print("-" * 92)
    for algo in ("astar", "bfs", "greedy"):
        row = summary[algo]
        print(
            f"{algo:<12}{row['avg_cost']:>12.2f}{row['avg_time_ms']:>15.2f}"
            f"{row['avg_explored_nodes']:>15.2f}{row['feasibility_rate']:>13.2f}"
            f"{row['avg_delivery_cost']:>20.2f}"
        )
    print("-" * 92)


def main() -> None:
    """Run deterministic experiments and generate result artifacts."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results"

    graph_cfg = GraphConfig(width=8, height=8, seed=11)
    sim_cfg = ScenarioConfig(num_scenarios=10, num_stops=4, seed=90)
    constraints_cfg = ConstraintConfig(
        battery_capacity=120.0,
        consumption_rate=1.0,
        late_penalty=50.0,
        failed_delivery_penalty=200.0,
    )

    scenarios, all_results, summary = run_experiments(graph_cfg, sim_cfg, constraints_cfg, results_dir)
    for scenario in scenarios:
        subset = [r for r in all_results if r.scenario_id == scenario.scenario_id]
        plot_scenario_routes(scenario, subset, results_dir)
    plot_metric_comparison(all_results, summary, results_dir)
    _print_summary(summary)
    print(f"\nArtifacts saved in: {results_dir}")


if __name__ == "__main__":
    main()

