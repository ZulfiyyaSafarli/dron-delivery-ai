"""Search algorithms for constrained drone routing."""

from __future__ import annotations

from collections import deque
import heapq
import itertools
import logging
import math
import time
from typing import Callable, TYPE_CHECKING

from .constraints import ConstraintConfig, State, state_key, transition_state

if TYPE_CHECKING:
    import networkx as nx


logger = logging.getLogger(__name__)
HeuristicFn = Callable[[tuple[float, float], tuple[float, float]], float]


def euclidean_heuristic(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance heuristic."""
    return math.dist(a, b)


def manhattan_heuristic(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Manhattan distance heuristic."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(
    graph: "nx.Graph",
    start: str,
    goal: str,
    heuristic_fn: HeuristicFn,
    constraints: ConstraintConfig,
    *,
    initial_battery: float | None = None,
    initial_time: float = 0.0,
    track_expansions: dict[str, int] | None = None,
    max_expansions: int = 15000,
) -> tuple[list[str] | None, float, float, int, bool]:
    """Run A* with battery/time-window feasibility pruning."""
    logger.info("A* started from %s to %s", start, goal)
    t0 = time.perf_counter()

    start_state = State(
        node_id=start,
        remaining_battery=initial_battery or constraints.battery_capacity,
        current_time=initial_time,
        path=[start],
        cost=0.0,
    )
    goal_pos = graph.nodes[goal]["pos"]

    pq: list[tuple[float, int, State]] = []
    counter = itertools.count()
    heapq.heappush(pq, (0.0, next(counter), start_state))
    best_cost: dict[tuple[str, float, float], float] = {state_key(start_state, constraints): 0.0}
    explored = 0

    while pq:
        _, _, state = heapq.heappop(pq)
        explored += 1
        if explored >= max_expansions:
            elapsed = (time.perf_counter() - t0) * 1000.0
            logger.warning("A* hit expansion limit (%s) from %s to %s", max_expansions, start, goal)
            return None, 0.0, elapsed, explored, False
        if track_expansions is not None:
            track_expansions[state.node_id] = track_expansions.get(state.node_id, 0) + 1

        if state.node_id == goal:
            elapsed = (time.perf_counter() - t0) * 1000.0
            logger.info("A* found feasible route with cost %.2f", state.cost)
            return state.path, state.cost, elapsed, explored, True

        for nxt in graph.neighbors(state.node_id):
            next_state, reason = transition_state(graph, state, nxt, constraints)
            if next_state is None:
                logger.debug("A* pruned transition %s -> %s due to %s", state.node_id, nxt, reason)
                continue

            key = state_key(next_state, constraints)
            known = best_cost.get(key)
            if known is not None and known <= next_state.cost:
                continue
            best_cost[key] = next_state.cost
            f_score = next_state.cost + heuristic_fn(graph.nodes[nxt]["pos"], goal_pos)
            heapq.heappush(pq, (f_score, next(counter), next_state))

    elapsed = (time.perf_counter() - t0) * 1000.0
    logger.warning("A* failed to find feasible route from %s to %s", start, goal)
    return None, 0.0, elapsed, explored, False


def bfs(
    graph: "nx.Graph",
    start: str,
    goal: str,
    constraints: ConstraintConfig,
    *,
    initial_battery: float | None = None,
    initial_time: float = 0.0,
    track_expansions: dict[str, int] | None = None,
    max_expansions: int = 15000,
) -> tuple[list[str] | None, float, float, int, bool]:
    """Run BFS and return first feasible solution encountered."""
    logger.info("BFS started from %s to %s", start, goal)
    t0 = time.perf_counter()

    start_state = State(
        node_id=start,
        remaining_battery=initial_battery or constraints.battery_capacity,
        current_time=initial_time,
        path=[start],
        cost=0.0,
    )
    queue: deque[State] = deque([start_state])
    visited: set[tuple[str, float, float]] = {state_key(start_state, constraints)}
    explored = 0

    while queue:
        state = queue.popleft()
        explored += 1
        if explored >= max_expansions:
            elapsed = (time.perf_counter() - t0) * 1000.0
            logger.warning("BFS hit expansion limit (%s) from %s to %s", max_expansions, start, goal)
            return None, 0.0, elapsed, explored, False
        if track_expansions is not None:
            track_expansions[state.node_id] = track_expansions.get(state.node_id, 0) + 1
        if state.node_id == goal:
            elapsed = (time.perf_counter() - t0) * 1000.0
            logger.info("BFS found feasible route with cost %.2f", state.cost)
            return state.path, state.cost, elapsed, explored, True

        for nxt in graph.neighbors(state.node_id):
            next_state, reason = transition_state(graph, state, nxt, constraints)
            if next_state is None:
                logger.debug("BFS pruned transition %s -> %s due to %s", state.node_id, nxt, reason)
                continue
            key = state_key(next_state, constraints)
            if key in visited:
                continue
            visited.add(key)
            queue.append(next_state)

    elapsed = (time.perf_counter() - t0) * 1000.0
    logger.warning("BFS failed to find feasible route from %s to %s", start, goal)
    return None, 0.0, elapsed, explored, False


def greedy_bfs(
    graph: "nx.Graph",
    start: str,
    goal: str,
    heuristic_fn: HeuristicFn,
    constraints: ConstraintConfig,
    *,
    initial_battery: float | None = None,
    initial_time: float = 0.0,
    track_expansions: dict[str, int] | None = None,
    max_expansions: int = 15000,
) -> tuple[list[str] | None, float, float, int, bool]:
    """Run Greedy Best-First Search ordered only by heuristic value."""
    logger.info("Greedy BFS started from %s to %s", start, goal)
    t0 = time.perf_counter()

    start_state = State(
        node_id=start,
        remaining_battery=initial_battery or constraints.battery_capacity,
        current_time=initial_time,
        path=[start],
        cost=0.0,
    )
    goal_pos = graph.nodes[goal]["pos"]
    pq: list[tuple[float, int, State]] = []
    counter = itertools.count()
    heapq.heappush(pq, (heuristic_fn(graph.nodes[start]["pos"], goal_pos), next(counter), start_state))
    visited: set[tuple[str, float, float]] = {state_key(start_state, constraints)}
    explored = 0

    while pq:
        _, _, state = heapq.heappop(pq)
        explored += 1
        if explored >= max_expansions:
            elapsed = (time.perf_counter() - t0) * 1000.0
            logger.warning("Greedy BFS hit expansion limit (%s) from %s to %s", max_expansions, start, goal)
            return None, 0.0, elapsed, explored, False
        if track_expansions is not None:
            track_expansions[state.node_id] = track_expansions.get(state.node_id, 0) + 1
        if state.node_id == goal:
            elapsed = (time.perf_counter() - t0) * 1000.0
            logger.info("Greedy BFS found feasible route with cost %.2f", state.cost)
            return state.path, state.cost, elapsed, explored, True

        for nxt in graph.neighbors(state.node_id):
            next_state, reason = transition_state(graph, state, nxt, constraints)
            if next_state is None:
                logger.debug(
                    "Greedy BFS pruned transition %s -> %s due to %s",
                    state.node_id,
                    nxt,
                    reason,
                )
                continue
            key = state_key(next_state, constraints)
            if key in visited:
                continue
            visited.add(key)
            score = heuristic_fn(graph.nodes[nxt]["pos"], goal_pos)
            heapq.heappush(pq, (score, next(counter), next_state))

    elapsed = (time.perf_counter() - t0) * 1000.0
    logger.warning("Greedy BFS failed to find feasible route from %s to %s", start, goal)
    return None, 0.0, elapsed, explored, False

