"""Constraint models and feasibility checks for drone routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx


@dataclass(frozen=True)
class ConstraintConfig:
    """Configurable constraints for a search run."""

    battery_capacity: float = 120.0
    consumption_rate: float = 1.0
    battery_rounding: int = 1
    time_rounding: int = 1
    late_penalty: float = 50.0
    failed_delivery_penalty: float = 200.0


@dataclass
class State:
    """Search state used by all algorithms."""

    node_id: str
    remaining_battery: float
    current_time: float
    path: list[str]
    cost: float


def battery_drain(distance: float, consumption_rate: float) -> float:
    """Compute battery drain on a single edge traversal."""
    return distance * consumption_rate


def apply_time_window(arrival_time: float, window: tuple[float, float] | None) -> tuple[float, bool]:
    """Apply waiting policy and validate delivery time-window."""
    if window is None:
        return arrival_time, True
    window_start, window_end = window
    if arrival_time > window_end:
        return arrival_time, False
    if arrival_time < window_start:
        return window_start, True
    return arrival_time, True


def transition_state(
    graph: "nx.Graph",
    state: State,
    next_node: str,
    cfg: ConstraintConfig,
) -> tuple[State | None, str | None]:
    """Attempt transitioning from current state to next node."""
    if graph.nodes[next_node].get("blocked", False):
        return None, "blocked_node"
    if not graph.has_edge(state.node_id, next_node):
        return None, "missing_edge"

    edge_data = graph.edges[state.node_id, next_node]
    if edge_data.get("blocked", False):
        return None, "blocked_edge"

    distance = float(edge_data["distance"])
    drain = battery_drain(distance, cfg.consumption_rate)
    remaining = state.remaining_battery - drain
    if remaining <= 0:
        return None, "battery_depleted"

    travel_time = float(edge_data["cost"])
    arrival_time = state.current_time + travel_time
    next_time, ok = apply_time_window(arrival_time, graph.nodes[next_node].get("time_window"))
    if not ok:
        return None, "late_arrival"

    return (
        State(
            node_id=next_node,
            remaining_battery=remaining,
            current_time=next_time,
            path=state.path + [next_node],
            cost=state.cost + float(edge_data["cost"]),
        ),
        None,
    )


def state_key(state: State, cfg: ConstraintConfig) -> tuple[str, float, float]:
    """Create rounded key for visited-state pruning."""
    return (
        state.node_id,
        round(state.remaining_battery, cfg.battery_rounding),
        round(state.current_time, cfg.time_rounding),
    )


def evaluate_path(
    graph: "nx.Graph",
    path: list[str],
    initial_battery: float,
    initial_time: float,
    cfg: ConstraintConfig,
) -> tuple[bool, float, float, float]:
    """Evaluate full path feasibility; return feasible, cost, end_battery, end_time."""
    if not path:
        return False, 0.0, initial_battery, initial_time

    state = State(
        node_id=path[0],
        remaining_battery=initial_battery,
        current_time=initial_time,
        path=[path[0]],
        cost=0.0,
    )
    for next_node in path[1:]:
        next_state, _ = transition_state(graph, state, next_node, cfg)
        if next_state is None:
            return False, state.cost, state.remaining_battery, state.current_time
        state = next_state
    return True, state.cost, state.remaining_battery, state.current_time

