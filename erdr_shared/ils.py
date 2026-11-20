"""State-dependent perturbation ILS with fixed, accelerated 2-opt and relocate descent.

This is an independent ILS implementation, not Concorde Linkern. Initialization
is candidate-independent and excluded from search time. The external process
watchdog covers loading, initialization and search.
"""
from __future__ import annotations

import json
import math
import random
import time

import numpy as np
from numba import njit

from .support import SearchResult, tour_cost, validate_matrix
from .timing import initialization_phase


def _descent_impl(distance, initial, movable_stop):
    route = initial.copy()
    n = len(route)
    evaluations = 0
    while True:
        best_delta = -1e-12
        first, last = -1, -1
        for i in range(1, movable_stop - 1):
            a, b = route[i - 1], route[i]
            for j in range(i + 1, movable_stop):
                c, d = route[j], route[(j + 1) % n]
                delta = distance[a, c] + distance[b, d] - distance[a, b] - distance[c, d]
                evaluations += 1
                if delta < best_delta:
                    best_delta, first, last = delta, i, j
        if first >= 0:
            route[first:last + 1] = route[first:last + 1][::-1]
            continue
        source, after = -1, -1
        for i in range(1, movable_stop):
            a, b, c = route[i - 1], route[i], route[(i + 1) % n]
            for j in range(movable_stop):
                if j == i or j == i - 1:
                    continue
                u, v = route[j], route[(j + 1) % n]
                delta = distance[a, c] - distance[a, b] - distance[b, c]
                delta += distance[u, b] + distance[b, v] - distance[u, v]
                evaluations += 1
                if delta < best_delta:
                    best_delta, source, after = delta, i, j
        if source < 0:
            return route, evaluations
        node = route[source]
        if source < after:
            for k in range(source, after):
                route[k] = route[k + 1]
            route[after] = node
        else:
            for k in range(source, after + 1, -1):
                route[k] = route[k - 1]
            route[after + 1] = node


_descent_jit = njit(cache=True)(_descent_impl)


def _dynamic_path(distance):
    """Recognize the existing current -> remaining -> origin -> dummy encoding."""
    return (len(distance) >= 3 and distance[-1, 0] == 0
            and distance[-1, -2] == 0
            and np.all(distance[-1, 1:-2] > np.max(distance[:-1, :-1])))


def _validate_route(raw, n, fixed_suffix):
    route = list(raw)
    if (len(route) != n or any(isinstance(x, (bool, np.bool_))
                              or not isinstance(x, (int, np.integer)) for x in route)
            or set(route) != set(range(n)) or route[0] != 0):
        raise ValueError("ILS perturbation must return every integer node exactly once, starting at 0")
    if fixed_suffix and tuple(route[-len(fixed_suffix):]) != fixed_suffix:
        raise ValueError("ILS perturbation changed the fixed terminal/dummy suffix")
    return np.asarray(route, dtype=np.int64)



def _validate_double_bridge(current, proposed, movable_stop):
    if movable_stop < 8:
        if not np.array_equal(current, proposed):
            raise ValueError("Route is too short for four nonadjacent cuts")
        return
    positions = np.empty(len(current), dtype=np.int64)
    positions[current] = np.arange(len(current))
    order = positions[proposed]
    boundaries = np.flatnonzero(np.diff(order) != 1) + 1
    if len(boundaries) != 4:
        raise ValueError("ILS perturbation must be a double-bridge move")
    blocks = np.split(order, boundaries)
    a, c, b, d = int(blocks[0][-1])+1, int(blocks[1][0]), int(blocks[2][0]), int(blocks[4][0])
    if not (1 <= a < b < c < d < movable_stop
            and min(b-a, c-b, d-c, len(current)-d+a) >= 2
            and np.array_equal(order, np.concatenate((np.arange(a), np.arange(c,d),
                np.arange(b,c), np.arange(a,b), np.arange(d,len(current)))))):
        raise ValueError("ILS perturbation must reconnect four nonadjacent cuts without reversals")


def _nearest_neighbor(distance, movable_stop):
    remaining = set(range(1, movable_stop))
    tour = [0]
    while remaining:
        node = min(remaining, key=lambda j: (distance[tour[-1], j], j))
        tour.append(node)
        remaining.remove(node)
    return np.asarray(tour + list(range(movable_stop, len(distance))), dtype=np.int64)


def solve(distance, *, perturbation_rule, iter_limit=500, time_limit_sec=10.,
          seed=1, initial_tour=None, use_numba=True):
    seconds, iterations = float(time_limit_sec), int(iter_limit)
    if not math.isfinite(seconds) or seconds <= 0 or iterations < 0:
        raise ValueError("ILS requires a positive finite search budget and nonnegative iter_limit")

    with initialization_phase():
        matrix = np.asarray(distance, dtype=np.float64).copy()
        np.fill_diagonal(matrix, 0.)
        n = len(matrix)
        if n >= 4:
            validate_matrix(matrix)
        elif (n < 2 or matrix.shape != (n, n) or not np.isfinite(matrix).all()
              or np.any(matrix < 0) or not np.allclose(matrix, matrix.T)):
            raise ValueError("ILS requires a finite nonnegative symmetric matrix with at least two nodes")
        suffix = (n - 2, n - 1) if _dynamic_path(matrix) else ()
        stop = n - len(suffix)
        descent = _descent_jit if use_numba else _descent_impl
        if use_numba:
            probe = np.zeros((4, 4), dtype=np.float64)
            _descent_jit(probe, np.arange(4, dtype=np.int64), 4)
        if initial_tour is None:
            current, _ = descent(matrix, _nearest_neighbor(matrix, stop), stop)
        else:

            current = _validate_route(initial_tour, n, suffix)
        best = current.copy()
        current_cost = best_cost = float(tour_cost(matrix, current))
        rng = random.Random(int(seed))

    calls = evaluations = stagnation = 0
    started = time.perf_counter()
    for iteration in range(1, iterations + 1):
        if time.perf_counter() - started >= seconds:
            break
        state = dict(iteration=iteration, stagnation=stagnation,
                     current_cost=current_cost, best_cost=best_cost,
                     fixed_prefix=(0,), fixed_suffix=suffix, movable_stop=stop)

        raw = perturbation_rule(matrix.copy(), current.tolist(), state, rng)
        calls += 1
        proposed = _validate_route(raw, n, suffix)
        _validate_double_bridge(current, proposed, stop)
        candidate, count = descent(matrix, proposed, stop)
        evaluations += int(count)
        candidate_cost = float(tour_cost(matrix, candidate))
        if not math.isfinite(candidate_cost):
            raise ValueError("Non-finite ILS route cost")
        if candidate_cost < current_cost:
            current, current_cost = candidate.copy(), candidate_cost
        if candidate_cost < best_cost:
            best, best_cost, stagnation = candidate.copy(), candidate_cost, 0
        else:
            stagnation += 1
        elapsed = time.perf_counter() - started
    elapsed = time.perf_counter() - started
    return SearchResult(tuple(map(int, best)), best_cost, evaluations, calls, elapsed)
