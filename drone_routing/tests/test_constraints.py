"""Tests for battery and time-window constraints."""

from src.constraints import ConstraintConfig, State, apply_time_window, battery_drain, transition_state
from src.graph import GraphConfig, build_city_graph


def test_battery_drain() -> None:
    assert battery_drain(10.0, 0.8) == 8.0


def test_apply_time_window_waiting_behavior() -> None:
    adjusted, feasible = apply_time_window(4.0, (5.0, 10.0))
    assert feasible is True
    assert adjusted == 5.0


def test_apply_time_window_late_infeasible() -> None:
    adjusted, feasible = apply_time_window(11.0, (5.0, 10.0))
    assert feasible is False
    assert adjusted == 11.0


def test_transition_state_infeasible_for_low_battery() -> None:
    cfg = GraphConfig(width=3, height=3, seed=10, blocked_node_rate=0.0, blocked_edge_rate=0.0)
    graph = build_city_graph(cfg)
    state = State(
        node_id="n_0_0",
        remaining_battery=0.1,
        current_time=0.0,
        path=["n_0_0"],
        cost=0.0,
    )
    constraints = ConstraintConfig(battery_capacity=20.0, consumption_rate=1.0)
    next_state, reason = transition_state(graph, state, "n_1_0", constraints)
    assert next_state is None
    assert reason == "battery_depleted"

