"""Tests for graph generation and scenario primitives."""

from src.graph import GraphConfig, assign_time_windows, build_city_graph, sample_delivery_stops


def test_build_city_graph_has_nodes_and_edges() -> None:
    cfg = GraphConfig(width=4, height=4, seed=1, blocked_node_rate=0.0, blocked_edge_rate=0.0)
    graph = build_city_graph(cfg)
    assert graph.number_of_nodes() == 16
    assert graph.number_of_edges() > 0
    for _, data in graph.nodes(data=True):
        assert "pos" in data


def test_sample_delivery_stops_excludes_start() -> None:
    cfg = GraphConfig(width=4, height=4, seed=1, blocked_node_rate=0.0, blocked_edge_rate=0.0)
    graph = build_city_graph(cfg)
    start = "n_0_0"
    stops = sample_delivery_stops(graph, num_stops=3, seed=2, exclude=[start])
    assert len(stops) == 3
    assert start not in stops


def test_assign_time_windows_sets_delivery_windows() -> None:
    cfg = GraphConfig(width=4, height=4, seed=1, blocked_node_rate=0.0, blocked_edge_rate=0.0)
    graph = build_city_graph(cfg)
    stops = ["n_0_1", "n_0_2", "n_1_1"]
    assign_time_windows(graph, stops, seed=3)
    for stop in stops:
        assert graph.nodes[stop]["time_window"] is not None

