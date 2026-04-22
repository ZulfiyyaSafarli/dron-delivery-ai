"""Tests for A*, BFS, and Greedy Best-First search."""

from src.algorithms import astar, bfs, euclidean_heuristic, greedy_bfs
from src.constraints import ConstraintConfig
from src.graph import GraphConfig, build_city_graph


def _simple_graph():
    cfg = GraphConfig(
        width=3,
        height=3,
        seed=1,
        blocked_node_rate=0.0,
        blocked_edge_rate=0.0,
        high_cost_edge_rate=0.0,
    )
    graph = build_city_graph(cfg)
    return graph


def test_astar_finds_feasible_path() -> None:
    graph = _simple_graph()
    constraints = ConstraintConfig(battery_capacity=30.0, consumption_rate=1.0)
    path, cost, _, _, feasible = astar(graph, "n_0_0", "n_2_2", euclidean_heuristic, constraints)
    assert feasible is True
    assert path is not None
    assert path[0] == "n_0_0"
    assert path[-1] == "n_2_2"
    assert cost > 0


def test_bfs_finds_feasible_path() -> None:
    graph = _simple_graph()
    constraints = ConstraintConfig(battery_capacity=30.0, consumption_rate=1.0)
    path, _, _, _, feasible = bfs(graph, "n_0_0", "n_2_2", constraints)
    assert feasible is True
    assert path is not None
    assert path[-1] == "n_2_2"


def test_greedy_finds_feasible_path() -> None:
    graph = _simple_graph()
    constraints = ConstraintConfig(battery_capacity=30.0, consumption_rate=1.0)
    path, _, _, _, feasible = greedy_bfs(graph, "n_0_0", "n_2_2", euclidean_heuristic, constraints)
    assert feasible is True
    assert path is not None
    assert path[-1] == "n_2_2"


def test_all_algorithms_fail_with_insufficient_battery() -> None:
    graph = _simple_graph()
    constraints = ConstraintConfig(battery_capacity=1.0, consumption_rate=5.0)
    a_path, _, _, _, a_ok = astar(graph, "n_0_0", "n_2_2", euclidean_heuristic, constraints)
    b_path, _, _, _, b_ok = bfs(graph, "n_0_0", "n_2_2", constraints)
    g_path, _, _, _, g_ok = greedy_bfs(graph, "n_0_0", "n_2_2", euclidean_heuristic, constraints)
    assert a_ok is False and a_path is None
    assert b_ok is False and b_path is None
    assert g_ok is False and g_path is None

