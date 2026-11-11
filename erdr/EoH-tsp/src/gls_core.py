"""Full GLS core loop with optional numba JIT acceleration."""

from __future__ import annotations

import random
import time

import numpy as np

try:
    import numba as nb

    NUMBA_AVAILABLE = True
except Exception:  # noqa: BLE001
    nb = None  # type: ignore[assignment]
    NUMBA_AVAILABLE = False

import gls_operators

USE_NUMBA_CACHE = True
_NUMBA_WARMED = False


def _jit_or_plain(func):
    if NUMBA_AVAILABLE:
        return nb.njit(nogil=True, cache=USE_NUMBA_CACHE)(func)
    return func


@_jit_or_plain
def tour_cost_2End(dis_m, tour2End):
    c = 0.0
    s = 0
    e = tour2End[0, 1]
    for _ in range(tour2End.shape[0]):
        c += dis_m[s, e]
        s = e
        e = tour2End[s, 1]
    return c


@_jit_or_plain
def _init_nearest_neighbor_tour(dis_matrix, depot):
    n = dis_matrix.shape[0]
    tour = np.empty(n, dtype=np.int32)
    visited = np.zeros(n, dtype=np.bool_)
    visited[depot] = True
    tour[0] = depot
    for i in range(1, n):
        prev = tour[i - 1]
        min_dist = 1e20
        min_idx = -1
        for j in range(n):
            if visited[j]:
                continue
            d = dis_matrix[prev, j]
            if d < min_dist:
                min_dist = d
                min_idx = j
        tour[i] = min_idx
        visited[min_idx] = True
    return tour


@_jit_or_plain
def _tour_to_route2end(tour):
    n = tour.shape[0]
    route2end = np.empty((n, 2), dtype=np.int32)
    if n == 1:
        route2end[0, 0] = 0
        route2end[0, 1] = 0
        return route2end

    route2end[0, 0] = tour[n - 1]
    route2end[0, 1] = tour[1]
    for i in range(1, n - 1):
        node = tour[i]
        route2end[node, 0] = tour[i - 1]
        route2end[node, 1] = tour[i + 1]

    last = tour[n - 1]
    route2end[last, 0] = tour[n - 2]
    route2end[last, 1] = tour[0]
    return route2end


def nearest_neighbor_2End(dis_matrix, depot):
    dm = np.ascontiguousarray(np.asarray(dis_matrix, dtype=np.float32))
    start = int(depot)
    tour = _init_nearest_neighbor_tour(dm, start)
    return _tour_to_route2end(tour)


@_jit_or_plain
def local_search(init_tour, init_cost, D, N, first_improvement=False):
    cur_route, cur_cost = init_tour, init_cost
    improved = True
    while improved:
        improved = False

        delta, new_tour = gls_operators.two_opt_a2a(cur_route, D, N, first_improvement)
        if delta < 0:
            improved = True
            cur_cost += delta
            cur_route = new_tour

        delta, new_tour = gls_operators.relocate_a2a(cur_route, D, N, first_improvement)
        if delta < 0:
            improved = True
            cur_cost += delta
            cur_route = new_tour

    return cur_route, cur_cost


@_jit_or_plain
def route2tour(route):
    n = route.shape[0]
    tour = np.empty(n, dtype=np.int32)
    s = 0
    for i in range(n):
        nxt = route[s, 1]
        tour[i] = nxt
        s = nxt
    return tour


def warmup_numba_kernels() -> None:
    global _NUMBA_WARMED
    if _NUMBA_WARMED:
        return
    if not NUMBA_AVAILABLE:
        _NUMBA_WARMED = True
        return

    try:
        dm = np.asarray(
            [
                [0.0, 1.0, 2.0],
                [1.0, 0.0, 1.0],
                [2.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        nearest_indices = np.asarray(
            [
                [1, 2],
                [0, 2],
                [0, 1],
            ],
            dtype=np.int32,
        )
        init_tour = nearest_neighbor_2End(dm, 0)
        init_cost = float(tour_cost_2End(dm, init_tour))
        route, _cost = local_search(init_tour, init_cost, dm, nearest_indices, False)
        _ = route2tour(route)
    except Exception:
        pass
    finally:
        _NUMBA_WARMED = True


def guided_local_search(
    coords,
    edge_weight,
    nearest_indices,
    init_tour,
    init_cost,
    t_lim,
    ite_max,
    perturbation_moves,
    first_improvement=False,
    guide_algorithm=None,
):
    random.seed(2024)

    edge_weight = np.ascontiguousarray(np.asarray(edge_weight, dtype=np.float32))
    nearest_indices = np.ascontiguousarray(np.asarray(nearest_indices, dtype=np.int32))
    cur_route = np.ascontiguousarray(np.asarray(init_tour, dtype=np.int32))

    cur_route, cur_cost = local_search(
        cur_route, float(init_cost), edge_weight, nearest_indices, first_improvement
    )
    best_route, best_cost = cur_route, cur_cost

    length = int(edge_weight.shape[0])
    iter_i = 0
    edge_penalty = np.zeros((length, length), dtype=np.float32)

    while iter_i < ite_max and time.time() < t_lim:
        for _move in range(perturbation_moves):
            cur_tour = route2tour(cur_route)
            if guide_algorithm is None:
                raise ValueError("guide_algorithm is required for GLS.")

            # Pass a copy to prevent candidates from mutating the base distance matrix in-place.
            edge_weight_guided = guide_algorithm.update_edge_distance(
                np.array(edge_weight, copy=True),
                cur_tour,
                edge_penalty,
            )
            edge_weight_guided = np.ascontiguousarray(
                np.asarray(edge_weight_guided, dtype=np.float32)
            )
            if edge_weight_guided.shape != edge_weight.shape:
                raise ValueError(
                    "update_edge_distance must return matrix with same shape as edge_weight."
                )

            edge_weight_gap = edge_weight_guided - edge_weight

            for _topid in range(5):
                max_indices = int(np.argmax(edge_weight_gap))
                rows, columns = np.unravel_index(max_indices, edge_weight_gap.shape)
                rows = int(rows)
                columns = int(columns)

                edge_penalty[rows, columns] += 1.0
                edge_penalty[columns, rows] += 1.0

                edge_weight_gap[rows, columns] = -1e30
                edge_weight_gap[columns, rows] = -1e30

                for nid in (rows, columns):
                    delta, new_route = gls_operators.two_opt_o2a_all(
                        cur_route,
                        edge_weight_guided,
                        nearest_indices,
                        nid,
                    )
                    if delta < 0:
                        cur_cost = tour_cost_2End(edge_weight, new_route)
                        cur_route = new_route

                    delta, new_route = gls_operators.relocate_o2a_all(
                        cur_route,
                        edge_weight_guided,
                        nearest_indices,
                        nid,
                    )
                    if delta < 0:
                        cur_cost = tour_cost_2End(edge_weight, new_route)
                        cur_route = new_route

        cur_route, cur_cost = local_search(
            cur_route,
            cur_cost,
            edge_weight,
            nearest_indices,
            first_improvement,
        )
        cur_cost = tour_cost_2End(edge_weight, cur_route)

        if cur_cost < best_cost:
            best_route, best_cost = cur_route, cur_cost

        iter_i += 1
        if iter_i % 50 == 0:
            cur_route, cur_cost = best_route, best_cost

    return best_route, float(best_cost), int(iter_i)
