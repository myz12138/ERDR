import concurrent.futures
import importlib.metadata
import os
import time
from typing import Callable, Optional

import numpy as np
import numpy.typing as npt

_REQUEST_JIT = str(os.environ.get("REEVO_GLS_ENABLE_JIT", "1")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_FORCE_NUMBA_IMPORT = str(os.environ.get("REEVO_GLS_FORCE_NUMBA_IMPORT", "0")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _parse_version_tuple(raw: str) -> tuple[int, ...]:
    parts = []
    for token in str(raw).split("."):
        digits = []
        for ch in token:
            if ch.isdigit():
                digits.append(ch)
            else:
                break
        if not digits:
            break
        parts.append(int("".join(digits)))
    return tuple(parts)


def _numba_import_likely_compatible() -> bool:
    np_ver = _parse_version_tuple(np.__version__)
    if np_ver < (2, 0):
        return True
    try:
        numba_ver = _parse_version_tuple(importlib.metadata.version("numba"))
    except Exception:
        # Unknown version metadata: try import and let runtime fallback if needed.
        return True
    # NumPy 2.0 generally requires newer numba wheels; avoid noisy import crashes
    # for clearly old releases unless user explicitly forces import.
    return numba_ver >= (0, 60)


if _REQUEST_JIT and (_FORCE_NUMBA_IMPORT or _numba_import_likely_compatible()):
    try:
        import numba as nb

        NUMBA_AVAILABLE = True
    except Exception:  # noqa: BLE001
        nb = None  # type: ignore[assignment]
        NUMBA_AVAILABLE = False
else:
    nb = None  # type: ignore[assignment]
    NUMBA_AVAILABLE = False

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def _env_flag(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


_JIT_ENABLED = bool(NUMBA_AVAILABLE and _env_flag("REEVO_GLS_ENABLE_JIT", True))
usecache = bool(_env_flag("REEVO_GLS_NUMBA_CACHE", True))
_NUMBA_WARMED = False


def _jit_or_plain(func: Callable):
    if _JIT_ENABLED:
        return nb.njit(nogil=True, cache=usecache)(func)
    return func


@_jit_or_plain
def _two_opt_once(distmat, tour, fixed_i=0):
    """In-place operation."""
    n = tour.shape[0]
    p = q = 0
    delta = 0
    for i in range(1, n - 1) if fixed_i == 0 else range(fixed_i, fixed_i + 1):
        for j in range(i + 1, n):
            node_i, node_j = tour[i], tour[j]
            node_prev, node_next = tour[i - 1], tour[(j + 1) % n]
            if node_prev == node_j or node_next == node_i:
                continue
            change = (
                distmat[node_prev, node_j]
                + distmat[node_i, node_next]
                - distmat[node_prev, node_i]
                - distmat[node_j, node_next]
            )
            if change < delta:
                p, q, delta = i, j, change
    if delta < -1e-6:
        tour[p : q + 1] = np.flip(tour[p : q + 1])
        return delta
    return 0.0


@_jit_or_plain
def _relocate_once(distmat, tour, fixed_i=0):
    n = distmat.shape[0]
    delta = p = q = 0
    for i in range(1, n) if fixed_i == 0 else range(fixed_i, fixed_i + 1):
        node = tour[i]
        prev_node = tour[i - 1]
        next_node = tour[(i + 1) % n]
        for j in range(n):
            if j == i or j == i - 1:
                continue
            prev_insert = tour[j]
            next_insert = tour[(j + 1) % n]
            cost = (
                -distmat[prev_node, node]
                - distmat[node, next_node]
                - distmat[prev_insert, next_insert]
                + distmat[prev_insert, node]
                + distmat[node, next_insert]
                + distmat[prev_node, next_node]
            )
            if cost < delta:
                delta, p, q = cost, i, j
    if delta >= 0:
        return 0.0
    if p < q:
        tour[p : q + 1] = np.roll(tour[p : q + 1], -1)
    else:
        tour[q : p + 1] = np.roll(tour[q : p + 1], 1)
    return delta


@_jit_or_plain
def _calculate_cost(distmat, tour):
    cost = distmat[tour[-1], tour[0]]
    for i in range(len(tour) - 1):
        cost += distmat[tour[i], tour[i + 1]]
    return cost


@_jit_or_plain
def _local_search(distmat, cur_tour, fixed_i=0, count=1000):
    sum_delta = 0.0
    delta = -1
    while delta < 0 and count > 0:
        delta = 0
        delta += _two_opt_once(distmat, cur_tour, fixed_i)
        delta += _relocate_once(distmat, cur_tour, fixed_i)
        count -= 1
        sum_delta += delta
    return sum_delta


@_jit_or_plain
def _perturbation(distmat, guide, penalty, cur_tour, k, perturbation_moves=30):
    moves = 0
    n = distmat.shape[0]
    while moves < perturbation_moves:
        # Penalize one edge with maximum utility.
        max_util = 0.0
        max_util_idx = 0
        for i in range(n - 1):
            j = i + 1
            u, v = cur_tour[i], cur_tour[j]
            util = guide[u, v] / (1.0 + penalty[u, v])
            if util > max_util:
                max_util_idx, max_util = i, util

        penalty[cur_tour[max_util_idx], cur_tour[max_util_idx + 1]] += 1.0
        edge_weight_guided = distmat + k * penalty

        for fixed_i in (max_util_idx, max_util_idx + 1):
            if fixed_i == 0 or fixed_i + 1 == n:
                continue
            delta = _local_search(edge_weight_guided, cur_tour, fixed_i, 1)
            if delta < 0:
                moves += 1


@_jit_or_plain
def _init_nearest_neighbor(distmat, start):
    n = distmat.shape[0]
    tour = np.zeros(n, dtype=np.uint16)
    visited = np.zeros(n, dtype=np.bool_)
    visited[start] = True
    tour[0] = start
    for i in range(1, n):
        min_dist = np.inf
        min_idx = -1
        for j in range(n):
            if not visited[j] and distmat[tour[i - 1], j] < min_dist:
                min_dist = distmat[tour[i - 1], j]
                min_idx = j
        tour[i] = min_idx
        visited[min_idx] = True
    return tour


@_jit_or_plain
def _guided_local_search(distmat, guide, start, perturbation_moves=30, iter_limit=1000):
    penalty = np.zeros_like(distmat)

    best_tour = _init_nearest_neighbor(distmat, start)
    _local_search(distmat, best_tour, 0, 1000)
    best_cost = _calculate_cost(distmat, best_tour)
    k = 0.1 * best_cost / distmat.shape[0]
    cur_tour = best_tour.copy()

    for _ in range(iter_limit):
        _perturbation(distmat, guide, penalty, cur_tour, k, perturbation_moves)
        _local_search(distmat, cur_tour, 0, 1000)
        cur_cost = _calculate_cost(distmat, cur_tour)
        if cur_cost < best_cost:
            best_tour, best_cost = cur_tour.copy(), cur_cost
    return best_tour


def _guided_local_search_timed(
    distmat: np.ndarray,
    guide: np.ndarray,
    start: int,
    perturbation_moves: int,
    iter_limit: int,
    time_limit_sec: float,
    deadline_perf_counter: Optional[float] = None,
) -> npt.NDArray[np.uint16]:
    dist = np.ascontiguousarray(np.asarray(distmat, dtype=np.float32))
    guide_arr = np.ascontiguousarray(np.asarray(guide, dtype=np.float32))
    start_node = np.uint16(int(start))
    moves = max(int(perturbation_moves), 1)
    max_iter = max(int(iter_limit), 1)
    if deadline_perf_counter is not None:
        try:
            raw_deadline = float(deadline_perf_counter)
        except Exception:
            raw_deadline = float("nan")
        if np.isfinite(raw_deadline):
            deadline = raw_deadline
        else:
            deadline = time.perf_counter() + max(float(time_limit_sec), 1.0e-6)
    else:
        deadline = time.perf_counter() + max(float(time_limit_sec), 1.0e-6)

    if time.perf_counter() >= deadline:
        return np.arange(int(dist.shape[0]), dtype=np.uint16)

    penalty = np.zeros_like(dist, dtype=np.float32)
    best_tour = _init_nearest_neighbor(dist, start_node)
    if time.perf_counter() >= deadline:
        return best_tour

    _local_search(dist, best_tour, 0, 1000)
    if time.perf_counter() >= deadline:
        return best_tour

    best_cost = float(_calculate_cost(dist, best_tour))
    cur_tour = best_tour.copy()
    k = np.float32(0.1 * best_cost / max(1, dist.shape[0]))

    for _ in range(max_iter):
        if time.perf_counter() >= deadline:
            break
        _perturbation(dist, guide_arr, penalty, cur_tour, k, moves)
        _local_search(dist, cur_tour, 0, 1000)
        cur_cost = float(_calculate_cost(dist, cur_tour))
        if cur_cost < best_cost:
            best_tour, best_cost = cur_tour.copy(), cur_cost
    return best_tour


def warmup_numba_kernels() -> None:
    global _NUMBA_WARMED
    if _NUMBA_WARMED:
        return
    if not _JIT_ENABLED:
        _NUMBA_WARMED = True
        return
    try:
        dist = np.asarray(
            [
                [0.0, 1.0, 2.0, 1.5],
                [1.0, 0.0, 1.0, 2.0],
                [2.0, 1.0, 0.0, 1.0],
                [1.5, 2.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        guide = np.asarray(dist, dtype=np.float32)
        _ = guided_local_search(dist, guide, perturbation_moves=1, iter_limit=2, time_limit_sec=0.02)
    except Exception:
        pass
    finally:
        _NUMBA_WARMED = True


def guided_local_search(
    distmat: FloatArray,
    guide: FloatArray,
    perturbation_moves: int = 30,
    iter_limit: int = 1000,
    time_limit_sec: float = 0.0,
    deadline_perf_counter: Optional[float] = None,
) -> npt.NDArray[np.uint16]:
    dist = np.ascontiguousarray(np.asarray(distmat, dtype=np.float32))
    guide_arr = np.ascontiguousarray(np.asarray(guide, dtype=np.float32))
    moves = max(int(perturbation_moves), 1)
    max_iter = max(int(iter_limit), 1)
    max_time = max(float(time_limit_sec), 0.0)
    deadline: Optional[float] = None

    if deadline_perf_counter is not None:
        try:
            raw_deadline = float(deadline_perf_counter)
        except Exception:
            raw_deadline = float("nan")
        if np.isfinite(raw_deadline):
            deadline = raw_deadline

    if max_time > 0.0 or deadline is not None:
        return _guided_local_search_timed(
            distmat=dist,
            guide=guide_arr,
            start=0,
            perturbation_moves=moves,
            iter_limit=max_iter,
            time_limit_sec=max_time,
            deadline_perf_counter=deadline,
        )

    return _guided_local_search(
        distmat=dist,
        guide=guide_arr,
        start=np.uint16(0),
        perturbation_moves=moves,
        iter_limit=max_iter,
    )


def multi_start_guided_local_search(
    dist: FloatArray,
    guide: FloatArray,
    n_starts: int = 10,
    perturbation_moves: int = 30,
    iter_limit: int = 1000,
    time_limit_sec: float = 0.0,
):
    dist_arr = np.ascontiguousarray(np.asarray(dist, dtype=np.float32))
    guide_arr = np.ascontiguousarray(np.asarray(guide, dtype=np.float32))
    start_nodes = np.arange(max(int(n_starts), 1)).astype(np.uint16)
    moves = max(int(perturbation_moves), 1)
    max_iter = max(int(iter_limit), 1)
    max_time = max(float(time_limit_sec), 0.0)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for start in start_nodes:
            if max_time > 0.0:
                future = executor.submit(
                    _guided_local_search_timed,
                    dist_arr,
                    guide_arr,
                    int(start),
                    moves,
                    max_iter,
                    max_time,
                )
            else:
                future = executor.submit(
                    _guided_local_search,
                    dist_arr,
                    guide_arr,
                    start,
                    moves,
                    max_iter,
                )
            futures.append(future)
        tours = [f.result() for f in futures]
        costs = np.array([_calculate_cost(dist_arr, tour) for tour in tours], dtype=float)
        best_tour = tours[int(np.argmin(costs))]
        return best_tour
