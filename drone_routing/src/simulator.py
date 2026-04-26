"""Scenario simulation and metric aggregation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import statistics
from typing import Callable

import networkx as nx
import numpy as np

from .algorithms import astar, dijkstra, euclidean_heuristic, greedy_bfs
from .constraints import ConstraintConfig, evaluate_path
from .graph import GraphConfig, assign_time_windows, build_city_graph, nearest_neighbor_order, sample_delivery_stops


@dataclass(frozen=True)
class ScenarioConfig:
    """Simulation setup."""

    num_scenarios: int = 10
    num_stops: int = 4
    seed: int = 42
    use_nearest_neighbor_order: bool = True


@dataclass
class Scenario:
    """A generated routing scenario."""

    scenario_id: int
    graph: nx.Graph
    start: str
    stops: list[str]
    ordered_stops: list[str]


@dataclass
class AlgorithmScenarioResult:
    """Per-scenario algorithm result payload."""

    scenario_id: int
    algorithm: str
    feasible: bool
    total_cost: float
    total_time_ms: float
    explored_nodes: int
    lower_bound_cost: float
    delivery_cost_with_penalty: float
    route: list[str] | None
    expansion_histogram: dict[str, int]


def generate_scenarios(graph_cfg: GraphConfig, sim_cfg: ScenarioConfig) -> list[Scenario]:
    """Generate deterministic random scenarios."""
    scenarios: list[Scenario] = []
    rng = np.random.default_rng(sim_cfg.seed)
    for sid in range(sim_cfg.num_scenarios):
        cfg = GraphConfig(**{**asdict(graph_cfg), "seed": int(rng.integers(0, 1_000_000))})
        graph = build_city_graph(cfg)
        feasible_nodes = [n for n, d in graph.nodes(data=True) if not d.get("blocked", False)]
        start = str(rng.choice(feasible_nodes))
        stops = sample_delivery_stops(graph, sim_cfg.num_stops, seed=int(rng.integers(0, 1_000_000)), exclude=[start])
        assign_time_windows(graph, stops, seed=int(rng.integers(0, 1_000_000)))
        ordered = nearest_neighbor_order(graph, start, stops) if sim_cfg.use_nearest_neighbor_order else list(stops)
        scenarios.append(Scenario(scenario_id=sid, graph=graph, start=start, stops=stops, ordered_stops=ordered))
    return scenarios


def _lower_bound_cost(graph: nx.Graph, route_points: list[str]) -> float:
    total = 0.0
    for a, b in zip(route_points, route_points[1:]):
        total += euclidean_heuristic(graph.nodes[a]["pos"], graph.nodes[b]["pos"])
    return total


def _run_leg(
    algo_name: str,
    graph: nx.Graph,
    start: str,
    goal: str,
    constraints: ConstraintConfig,
    battery: float,
    current_time: float,
    expansion_histogram: dict[str, int],
) -> tuple[list[str] | None, float, float, int, bool]:
    if algo_name == "astar":
        return astar(
            graph,
            start,
            goal,
            euclidean_heuristic,
            constraints,
            initial_battery=battery,
            initial_time=current_time,
            track_expansions=expansion_histogram,
        )
    if algo_name == "dijkstra":
        return dijkstra(
            graph,
            start,
            goal,
            constraints,
            initial_battery=battery,
            initial_time=current_time,
            track_expansions=expansion_histogram,
        )
    return greedy_bfs(
        graph,
        start,
        goal,
        euclidean_heuristic,
        constraints,
        initial_battery=battery,
        initial_time=current_time,
        track_expansions=expansion_histogram,
    )


def run_scenario_algorithms(
    scenario: Scenario,
    constraints: ConstraintConfig,
    failure_penalty: float,
) -> list[AlgorithmScenarioResult]:
    """Run A*, Dijkstra, and Greedy for one scenario."""
    results: list[AlgorithmScenarioResult] = []
    algorithms = ["astar", "dijkstra", "greedy"]
    goals = [scenario.start] + scenario.ordered_stops

    for algo_name in algorithms:
        battery = constraints.battery_capacity
        current_time = 0.0
        all_feasible = True
        total_cost = 0.0
        total_runtime_ms = 0.0
        total_explored = 0
        route: list[str] = [scenario.start]
        expansion_histogram: dict[str, int] = {}

        for leg_start, leg_goal in zip(goals, goals[1:]):
            path, cost, elapsed, explored, feasible = _run_leg(
                algo_name,
                scenario.graph,
                leg_start,
                leg_goal,
                constraints,
                battery,
                current_time,
                expansion_histogram,
            )
            total_runtime_ms += elapsed
            total_explored += explored
            if not feasible or path is None:
                all_feasible = False
                break

            leg_ok, _, battery, current_time = evaluate_path(
                scenario.graph,
                path,
                battery,
                current_time,
                constraints,
            )
            if not leg_ok:
                all_feasible = False
                break
            total_cost += cost
            route.extend(path[1:])

        lower_bound = _lower_bound_cost(scenario.graph, goals)
        final_delivery_cost = total_cost if all_feasible else total_cost + failure_penalty
        results.append(
            AlgorithmScenarioResult(
                scenario_id=scenario.scenario_id,
                algorithm=algo_name,
                feasible=all_feasible,
                total_cost=round(total_cost, 4),
                total_time_ms=round(total_runtime_ms, 4),
                explored_nodes=total_explored,
                lower_bound_cost=round(lower_bound, 4),
                delivery_cost_with_penalty=round(final_delivery_cost, 4),
                route=route if all_feasible else None,
                expansion_histogram=expansion_histogram,
            )
        )

    return results


def run_experiments(
    graph_cfg: GraphConfig,
    sim_cfg: ScenarioConfig,
    constraints: ConstraintConfig,
    results_dir: Path,
) -> tuple[list[Scenario], list[AlgorithmScenarioResult], dict[str, dict[str, float]]]:
    """Run all scenarios and aggregate summary metrics."""
    results_dir.mkdir(parents=True, exist_ok=True)
    scenarios = generate_scenarios(graph_cfg, sim_cfg)
    all_results: list[AlgorithmScenarioResult] = []
    for scenario in scenarios:
        all_results.extend(run_scenario_algorithms(scenario, constraints, constraints.failed_delivery_penalty))

    summary: dict[str, dict[str, float]] = {}
    for algo in ("astar", "dijkstra", "greedy"):
        subset = [r for r in all_results if r.algorithm == algo]
        feasible_rate = sum(1 for r in subset if r.feasible) / len(subset) if subset else 0.0
        summary[algo] = {
            "avg_cost": round(statistics.fmean(r.total_cost for r in subset), 4) if subset else 0.0,
            "avg_time_ms": round(statistics.fmean(r.total_time_ms for r in subset), 4) if subset else 0.0,
            "avg_explored_nodes": round(statistics.fmean(r.explored_nodes for r in subset), 4) if subset else 0.0,
            "feasibility_rate": round(feasible_rate * 100.0, 2),
            "avg_delivery_cost": round(statistics.fmean(r.delivery_cost_with_penalty for r in subset), 4)
            if subset
            else 0.0,
        }

    _save_results(all_results, summary, results_dir)
    return scenarios, all_results, summary


def _save_results(
    all_results: list[AlgorithmScenarioResult],
    summary: dict[str, dict[str, float]],
    results_dir: Path,
) -> None:
    csv_path = results_dir / "experiment_results.csv"
    json_path = results_dir / "experiment_results.json"
    summary_path = results_dir / "summary.json"

    rows = []
    for item in all_results:
        row = asdict(item)
        row["route"] = "->".join(item.route) if item.route else ""
        row["expansion_histogram"] = json.dumps(item.expansion_histogram)
        rows.append(row)
    if rows:
        headers = list(rows[0].keys())
        lines = [",".join(headers)]
        for row in rows:
            lines.append(",".join(str(row[h]) for h in headers))
        csv_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(json.dumps([asdict(r) for r in all_results], indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

