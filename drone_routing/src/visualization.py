"""Visualization helpers for scenario routes and metric comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from .simulator import AlgorithmScenarioResult, Scenario


def plot_scenario_routes(
    scenario: Scenario,
    scenario_results: list[AlgorithmScenarioResult],
    output_dir: Path,
) -> None:
    """Plot routes found by each algorithm on the city graph."""
    output_dir.mkdir(parents=True, exist_ok=True)
    graph = scenario.graph
    pos = {n: graph.nodes[n]["pos"] for n in graph.nodes}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for axis, algo in zip(axes, ["astar", "bfs", "greedy"]):
        axis.set_title(f"Scenario {scenario.scenario_id} - {algo.upper()}")
        nx.draw_networkx_edges(graph, pos, ax=axis, alpha=0.2, width=0.8)
        nx.draw_networkx_nodes(graph, pos, ax=axis, node_size=25, node_color="#87ceeb")
        blocked_nodes = [n for n, d in graph.nodes(data=True) if d.get("blocked", False)]
        if blocked_nodes:
            nx.draw_networkx_nodes(graph, pos, nodelist=blocked_nodes, ax=axis, node_size=40, node_color="black")
        nx.draw_networkx_nodes(graph, pos, nodelist=[scenario.start], ax=axis, node_size=85, node_color="green")
        nx.draw_networkx_nodes(graph, pos, nodelist=scenario.ordered_stops, ax=axis, node_size=85, node_color="red")
        result = next((item for item in scenario_results if item.algorithm == algo), None)
        if result and result.route:
            route_edges = list(zip(result.route, result.route[1:]))
            nx.draw_networkx_edges(graph, pos, edgelist=route_edges, ax=axis, edge_color="orange", width=2.6)
        axis.set_xticks([])
        axis.set_yticks([])

    fig.savefig(output_dir / f"routes_scenario_{scenario.scenario_id}.png", dpi=170)
    plt.close(fig)


def plot_metric_comparison(
    all_results: list[AlgorithmScenarioResult],
    summary: dict[str, dict[str, float]],
    output_dir: Path,
) -> None:
    """Plot high-level metric comparison charts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    algorithms = ["astar", "bfs", "greedy"]

    metrics = [
        ("avg_cost", "Average Cost"),
        ("avg_time_ms", "Average Runtime (ms)"),
        ("avg_explored_nodes", "Average Explored States"),
        ("feasibility_rate", "Feasibility Rate (%)"),
        ("avg_delivery_cost", "Avg Delivery Cost (with penalties)"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(22, 5), constrained_layout=True)
    for axis, (metric_key, title) in zip(axes, metrics):
        values = [summary[a][metric_key] for a in algorithms]
        axis.bar(algorithms, values, color=["#4caf50", "#1e88e5", "#ff9800"])
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=20)
    fig.savefig(output_dir / "metrics_comparison.png", dpi=170)
    plt.close(fig)

    # Build a basic explored-node heatmap from aggregate expansion counts.
    heat_counts: dict[str, int] = {}
    for item in all_results:
        for node, count in item.expansion_histogram.items():
            heat_counts[node] = heat_counts.get(node, 0) + count
    if not heat_counts:
        return

    first_route = next((r for r in all_results if r.route), None)
    if first_route is None:
        return

    # Reconstruct graph positions from scenario IDs by finding a result with route.
    # This heatmap uses node IDs as labels to avoid requiring graph context.
    fig_h, ax_h = plt.subplots(figsize=(10, 6), constrained_layout=True)
    nodes = list(heat_counts.keys())
    values = [heat_counts[n] for n in nodes]
    ax_h.bar(range(len(nodes)), values)
    ax_h.set_title("Explored-Node Heatmap (aggregate counts)")
    ax_h.set_xlabel("Node index")
    ax_h.set_ylabel("Expansion count")
    fig_h.savefig(output_dir / "explored_node_heatmap.png", dpi=170)
    plt.close(fig_h)

