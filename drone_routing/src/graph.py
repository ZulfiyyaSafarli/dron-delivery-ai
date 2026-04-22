"""Graph generation utilities for drone routing scenarios."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, Sequence

import networkx as nx


@dataclass(frozen=True)
class GraphConfig:
    """Configuration for synthetic city graph generation."""

    width: int = 8
    height: int = 8
    directed: bool = False
    diagonal_edges: bool = False
    min_delay: float = 0.0
    max_delay: float = 2.0
    blocked_node_rate: float = 0.05
    blocked_edge_rate: float = 0.08
    high_cost_edge_rate: float = 0.1
    high_cost_multiplier: float = 3.0
    seed: int = 7


def node_id(x: int, y: int) -> str:
    """Create a stable node identifier."""
    return f"n_{x}_{y}"


def build_city_graph(config: GraphConfig) -> nx.Graph:
    """Build a city graph with coordinates, distance, delay, and costs."""
    graph: nx.Graph
    graph = nx.DiGraph() if config.directed else nx.Graph()
    rng = random.Random(config.seed)

    for x in range(config.width):
        for y in range(config.height):
            nid = node_id(x, y)
            graph.add_node(
                nid,
                pos=(float(x), float(y)),
                blocked=False,
                time_window=None,
            )

    neighbor_steps = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if config.diagonal_edges:
        neighbor_steps.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)])

    for x in range(config.width):
        for y in range(config.height):
            src = node_id(x, y)
            for dx, dy in neighbor_steps:
                nx_pos = x + dx
                ny_pos = y + dy
                if 0 <= nx_pos < config.width and 0 <= ny_pos < config.height:
                    dst = node_id(nx_pos, ny_pos)
                    if graph.has_edge(src, dst):
                        continue
                    distance = math.dist((x, y), (nx_pos, ny_pos))
                    delay = rng.uniform(config.min_delay, config.max_delay)
                    graph.add_edge(
                        src,
                        dst,
                        distance=distance,
                        delay=delay,
                        cost=distance + delay,
                        blocked=False,
                        high_cost=False,
                    )

    apply_obstacles(graph, config, rng)
    return graph


def apply_obstacles(graph: nx.Graph, config: GraphConfig, rng: random.Random | None = None) -> None:
    """Randomly mark nodes/edges as blocked or high cost."""
    local_rng = rng or random.Random(config.seed)
    nodes = list(graph.nodes)
    edges = list(graph.edges)

    blocked_nodes = max(1, int(len(nodes) * config.blocked_node_rate))
    for nid in local_rng.sample(nodes, k=min(blocked_nodes, len(nodes))):
        graph.nodes[nid]["blocked"] = True

    blocked_edges = int(len(edges) * config.blocked_edge_rate)
    for edge in local_rng.sample(edges, k=min(blocked_edges, len(edges))):
        graph.edges[edge]["blocked"] = True

    high_cost_edges = int(len(edges) * config.high_cost_edge_rate)
    for edge in local_rng.sample(edges, k=min(high_cost_edges, len(edges))):
        if graph.edges[edge]["blocked"]:
            continue
        graph.edges[edge]["high_cost"] = True
        graph.edges[edge]["cost"] *= config.high_cost_multiplier


def assign_time_windows(
    graph: nx.Graph,
    delivery_nodes: Sequence[str],
    *,
    seed: int,
    start_max: float = 40.0,
    width_min: float = 12.0,
    width_max: float = 30.0,
) -> None:
    """Assign [start, end] windows to delivery nodes."""
    rng = random.Random(seed)
    for nid in delivery_nodes:
        t_start = rng.uniform(0.0, start_max)
        t_end = t_start + rng.uniform(width_min, width_max)
        graph.nodes[nid]["time_window"] = (t_start, t_end)


def sample_delivery_stops(
    graph: nx.Graph,
    num_stops: int,
    *,
    seed: int,
    exclude: Iterable[str] | None = None,
) -> list[str]:
    """Sample non-blocked delivery stops."""
    excluded = set(exclude or [])
    candidates = [
        nid
        for nid, data in graph.nodes(data=True)
        if not data.get("blocked", False) and nid not in excluded
    ]
    if num_stops > len(candidates):
        raise ValueError("Not enough feasible nodes for requested delivery stops.")
    rng = random.Random(seed)
    return rng.sample(candidates, k=num_stops)


def nearest_neighbor_order(graph: nx.Graph, start: str, stops: Sequence[str]) -> list[str]:
    """Order stops greedily by nearest Euclidean distance from current location."""
    remaining = set(stops)
    ordered: list[str] = []
    current = start
    while remaining:
        cx, cy = graph.nodes[current]["pos"]
        best = min(
            remaining,
            key=lambda node: math.dist((cx, cy), graph.nodes[node]["pos"]),
        )
        ordered.append(best)
        remaining.remove(best)
        current = best
    return ordered

