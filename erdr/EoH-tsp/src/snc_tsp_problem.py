from __future__ import annotations

import inspect
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from gls_core import (
    NUMBA_AVAILABLE,
    guided_local_search,
    nearest_neighbor_2End,
    tour_cost_2End,
    warmup_numba_kernels,
)


class _GLSPrompts:
    def __init__(self) -> None:
        self.prompt_task = (
            "Task: Given an edge distance matrix and a local optimal route, please help me "
            "design a strategy to update the distance matrix to avoid being trapped in the "
            "local optimum with the final goal of finding a tour with minimized distance. "
            "You should create a heuristic for me to update the edge distance matrix."
        )
        self.prompt_func_name = "update_edge_distance"
        self.prompt_func_inputs = ["edge_distance", "local_opt_tour", "edge_n_used"]
        self.prompt_func_outputs = ["updated_edge_distance"]
        self.prompt_inout_inf = (
            "'local_opt_tour' includes the local optimal tour of node IDs. "
            "'edge_distance' and 'edge_n_used' are matrices. "
            "'edge_n_used' includes the number of each edge used during perturbation."
        )
        self.prompt_other_inf = "All are Numpy arrays."

    def get_task(self):
        return self.prompt_task

    def get_func_name(self):
        return self.prompt_func_name

    def get_func_inputs(self):
        return self.prompt_func_inputs

    def get_func_outputs(self):
        return self.prompt_func_outputs

    def get_inout_inf(self):
        return self.prompt_inout_inf

    def get_other_inf(self):
        return self.prompt_other_inf


class _GuideWrapper:
    def __init__(self, fn: Callable) -> None:
        self.update_edge_distance = fn


class TSPGLSUpdateEdgeProblem:
    """
    GLS evaluator for update_edge_distance(edge_distance, local_opt_tour, edge_n_used).

    Keeps EOH problem interface compatible with static_train + EOH core:
    - self.prompts
    - evaluate(code_string) -> float metric
    - get_last_eval_info()
    """

    LARGE = 1e12

    def __init__(
        self,
        instances: Sequence[Tuple[np.ndarray, np.ndarray]],
        opt_costs: Optional[Sequence[float]] = None,
        n_instance_eval: int | None = None,
        neighbor_k: int = 100,
        group_indices: Optional[Sequence[Sequence[int]]] = None,
        use_grouped_metric: bool = False,
        group_metric_weights: Optional[Dict[str, float]] = None,
        use_gap_metric: bool = True,
        subgroup_tail_basis: str = "gap",
        gap_eps: float = 1e-12,
        time_limit_per_instance: float = 10.0,
        ite_max: int = 1000,
        perturbation_moves: int = 1,
    ) -> None:
        if not instances:
            raise ValueError("instances is empty")

        self.instance_data: List[Tuple[np.ndarray, np.ndarray]] = []
        for idx, (coords, dist) in enumerate(instances):
            c = np.asarray(coords, dtype=np.float32)
            d = np.asarray(dist, dtype=np.float32)
            if c.ndim != 2 or c.shape[1] != 2:
                raise ValueError(f"Invalid coords shape at idx={idx}: {c.shape}")
            if d.ndim != 2 or d.shape[0] != d.shape[1] or d.shape[0] != c.shape[0]:
                raise ValueError(
                    f"Invalid distance matrix shape at idx={idx}: {d.shape}, coords={c.shape}"
                )
            self.instance_data.append((c, d))

        self.n_instance = (
            len(self.instance_data)
            if n_instance_eval is None or int(n_instance_eval) <= 0
            else min(int(n_instance_eval), len(self.instance_data))
        )
        self.group_metric_weights = self._normalize_group_metric_weights(group_metric_weights)
        self.use_grouped_metric = bool(use_grouped_metric)
        self.use_gap_metric = bool(use_gap_metric)
        tail_basis = str(subgroup_tail_basis or "gap").strip().lower()
        if tail_basis not in {"gap", "time"}:
            raise ValueError(
                f"subgroup_tail_basis must be one of: gap, time; got {subgroup_tail_basis!r}"
            )
        self.subgroup_tail_basis = tail_basis
        self.gap_eps = float(max(float(gap_eps), 1e-12))
        metric_base = "gap" if self.use_gap_metric else "cost"
        metric_tail = "mean_std_worst" if self.use_grouped_metric else "mean_only"
        self.metric_mode = f"{metric_base}_{metric_tail}"

        self._reference_costs = self._prepare_reference_costs(opt_costs)
        self.group_indices = self._prepare_group_indices(group_indices)

        self.neighbor_size = int(max(int(neighbor_k), 1))
        self.time_limit_per_instance = float(max(float(time_limit_per_instance), 0.1))
        self.ite_max = int(max(int(ite_max), 1))
        self.perturbation_moves = int(max(int(perturbation_moves), 1))

        self._last_eval_info: Dict[str, Any] = {}
        self.prompts = _GLSPrompts()
        self.jit_kernel_enabled = bool(NUMBA_AVAILABLE)

        warmup_numba_kernels()

    @staticmethod
    def _nearest_neighbor_cost_from_dist(distance_matrix: np.ndarray) -> float:
        d = np.asarray(distance_matrix, dtype=float)
        n = int(d.shape[0])
        if n <= 1:
            return 0.0
        visited = np.zeros(n, dtype=bool)
        cur = 0
        visited[cur] = True
        total = 0.0
        for _ in range(n - 1):
            candidates = np.where(~visited)[0]
            nxt = int(candidates[np.argmin(d[cur, candidates])])
            total += float(d[cur, nxt])
            visited[nxt] = True
            cur = nxt
        total += float(d[cur, 0])
        return float(total)

    def _prepare_reference_costs(self, opt_costs: Optional[Sequence[float]]) -> np.ndarray:
        n_total = len(self.instance_data)
        if opt_costs is None:
            raise ValueError(
                "Concorde reference costs are required for gap evaluation (opt_costs is None)."
            )
        try:
            values = np.asarray([float(x) for x in opt_costs], dtype=float)
        except Exception as e:
            raise ValueError("Invalid opt_costs: failed to parse concorde reference costs.") from e
        if values.size < n_total:
            raise ValueError(
                f"Insufficient concorde reference costs: expected >= {n_total}, got {values.size}."
            )
        refs = values[:n_total].astype(float, copy=True)
        bad = (~np.isfinite(refs)) | (refs <= self.gap_eps)
        if np.any(bad[: self.n_instance]):
            bad_idx = np.where(bad[: self.n_instance])[0].tolist()
            preview = ",".join(str(i) for i in bad_idx[:10])
            raise ValueError(
                "Invalid concorde reference costs. "
                f"Bad {len(bad_idx)}/{self.n_instance} items (first indices: {preview})."
            )
        self._reference_fallback_count = 0
        return refs

    @staticmethod
    def _population_std(values: Sequence[float]) -> float:
        arr = np.asarray([float(x) for x in values], dtype=float)
        if arr.size <= 0:
            return 0.0
        mean_v = float(np.mean(arr))
        sq_mean = float(np.mean((arr - mean_v) ** 2))
        return float(np.sqrt(max(sq_mean, 0.0)))

    @staticmethod
    def _worst_half_mean(values: Sequence[float]) -> tuple[float, int]:
        arr = np.asarray([float(x) for x in values], dtype=float)
        n = int(arr.size)
        if n <= 0:
            return 0.0, 0
        m = max(1, n // 2)
        sorted_desc = np.sort(arr)[::-1]
        return float(np.mean(sorted_desc[:m])), int(m)

    @staticmethod
    def _normalize_group_metric_weights(raw: Any) -> Dict[str, float]:
        default = {"mean": 0.6, "std": 0.2, "worst": 0.2}
        if not isinstance(raw, dict):
            return default

        out: Dict[str, float] = {}
        for key, fallback in default.items():
            value = raw.get(key, fallback)
            try:
                parsed = float(value)
            except Exception:
                parsed = float(fallback)
            out[key] = max(parsed, 0.0)

        total = float(out["mean"] + out["std"] + out["worst"])
        if total <= 0:
            return default
        return {k: float(v / total) for k, v in out.items()}

    def _prepare_group_indices(
        self,
        raw_group_indices: Optional[Sequence[Sequence[int]]],
    ) -> List[List[int]]:
        if not raw_group_indices:
            return [list(range(self.n_instance))]

        groups: List[List[int]] = []
        for gid, idxs in enumerate(raw_group_indices):
            cleaned: List[int] = []
            for idx in idxs:
                try:
                    i = int(idx)
                except Exception as e:
                    raise ValueError(f"Invalid group index at group {gid}: {idx!r}") from e
                if i < 0 or i >= self.n_instance:
                    raise ValueError(
                        f"Group index out of range at group {gid}: {i}, n_instance={self.n_instance}"
                    )
                if i not in cleaned:
                    cleaned.append(i)
            if cleaned:
                groups.append(cleaned)

        if not groups:
            return [list(range(self.n_instance))]

        flat = sorted(i for g in groups for i in g)
        expected = list(range(self.n_instance))
        if flat != expected:
            raise ValueError(
                "group_indices must form a full partition of [0, n_instance_eval). "
                f"Expected {len(expected)} indices, got {len(flat)}."
            )
        return groups

    @staticmethod
    def _compile_function(code: str) -> Tuple[Callable | None, str | None]:
        namespace: Dict[str, object] = {"np": np}
        try:
            exec(code, namespace, namespace)
        except Exception as exc:  # noqa: BLE001
            return None, f"compile_error: {exc}"
        fn = namespace.get("update_edge_distance")
        if fn is None or not callable(fn):
            return None, "compile_error: function `update_edge_distance` not found"
        return fn, None

    @staticmethod
    def _check_signature(fn: Callable) -> Tuple[bool, str | None]:
        expected = ["edge_distance", "local_opt_tour", "edge_n_used"]
        try:
            names = list(inspect.signature(fn).parameters.keys())
        except Exception as exc:  # noqa: BLE001
            return False, f"signature_error: {exc}"
        if names != expected:
            return False, f"signature_mismatch: expected {expected}, got {names}"
        return True, None

    @staticmethod
    def _route_to_sequence(route2end: np.ndarray, n_nodes: int) -> List[int]:
        route2end = np.asarray(route2end, dtype=int)
        seq = [0]
        current = 0
        for _ in range(max(1, n_nodes * 2)):
            nxt = int(route2end[current, 1])
            seq.append(nxt)
            current = nxt
            if current == 0:
                break
        if seq[-1] != 0:
            seq.append(0)
        return seq

    @staticmethod
    def _route_cost_from_sequence(seq: List[int], distance_matrix: np.ndarray) -> float:
        total = 0.0
        for i in range(len(seq) - 1):
            a = int(seq[i])
            b = int(seq[i + 1])
            total += float(distance_matrix[a, b])
        return float(total)

    @staticmethod
    def _route_validity_penalty(seq: List[int], n_nodes: int) -> float:
        penalty = 0.0
        if not seq or seq[0] != 0 or seq[-1] != 0:
            penalty += 500.0
        if len(seq) < 2:
            return penalty + 1000.0

        customers = [x for x in seq[1:-1] if x != 0]
        if any((x < 0 or x >= n_nodes) for x in seq):
            penalty += 500.0
        inner_zero = sum(1 for x in seq[1:-1] if x == 0)
        penalty += 100.0 * float(inner_zero)

        expected = n_nodes - 1
        penalty += 100.0 * abs(len(customers) - expected)

        counts = np.zeros(n_nodes, dtype=int)
        for x in customers:
            if 0 <= x < n_nodes:
                counts[x] += 1
        if n_nodes > 1:
            missing = int(np.sum(counts[1:] == 0))
            duplicates = int(np.sum(np.maximum(0, counts[1:] - 1)))
            penalty += 100.0 * missing + 100.0 * duplicates
        return float(penalty)

    def _evaluate_one_instance(self, fn: Callable, idx: int) -> Dict[str, float]:
        t0 = time.perf_counter()
        coords, dm = self.instance_data[idx]
        n = int(dm.shape[0])
        if n <= 2:
            return {
                "status": "invalid",
                "obj": self.LARGE,
                "gap": self.LARGE,
                "runtime_ms": float((time.perf_counter() - t0) * 1000.0),
            }

        try:
            init_tour = np.ascontiguousarray(nearest_neighbor_2End(dm, 0), dtype=np.int32)
            init_cost = float(tour_cost_2End(dm, init_tour))
            k = max(1, min(int(self.neighbor_size), n - 1))
            nearest_indices = np.ascontiguousarray(
                np.argsort(dm, axis=1)[:, 1 : k + 1].astype(np.int32)
            )
            guide = _GuideWrapper(fn)
            best_route, best_cost, _iter_i = guided_local_search(
                coords,
                dm,
                nearest_indices,
                init_tour,
                init_cost,
                time.time() + self.time_limit_per_instance,
                self.ite_max,
                self.perturbation_moves,
                first_improvement=False,
                guide_algorithm=guide,
            )
            seq = self._route_to_sequence(np.asarray(best_route), n)
            validity_penalty = self._route_validity_penalty(seq, n)
            if validity_penalty > 0.0:
                raise ValueError("invalid_route_structure")

            route_cost = self._route_cost_from_sequence(seq, dm)
            if not np.isfinite(route_cost):
                route_cost = float(best_cost)
            if route_cost <= 1e-12:
                raise ValueError("non_positive_route_cost")

            ref = float(self._reference_costs[idx])
            if not np.isfinite(route_cost) or not np.isfinite(ref) or ref <= self.gap_eps:
                raise ValueError("non_finite_cost_or_ref")

            raw_gap = float((route_cost - ref) / ref)
            if not np.isfinite(raw_gap):
                raise ValueError("non_finite_gap")
            # Avoid tiny negative floating drift around the Concorde baseline.
            gap_tol = float(max(self.gap_eps, 1.0e-7))
            gap = 0.0 if (raw_gap < 0.0 and abs(raw_gap) <= gap_tol) else float(raw_gap)
            return {
                "status": "ok",
                "obj": float(route_cost),
                "gap": float(gap),
                "runtime_ms": float((time.perf_counter() - t0) * 1000.0),
            }
        except Exception:
            return {
                "status": "invalid",
                "obj": self.LARGE,
                "gap": self.LARGE,
                "runtime_ms": float((time.perf_counter() - t0) * 1000.0),
            }

    def _invalid_eval_result(self, *, error: str) -> Dict[str, Any]:
        invalid_metric = float(self.LARGE)
        n_groups = int(max(len(self.group_indices), 1))
        return {
            "status": "invalid",
            "error": str(error),
            "avg_obj": invalid_metric,
            "avg_tsp_time_sec": 0.0,
            "avg_objective": invalid_metric,
            "avg_gap": invalid_metric,
            "avg_gap_ratio": invalid_metric,
            "avg_gap_pct": invalid_metric * 100.0,
            "time_cost_sec": 0.0,
            "n_eval": int(self.n_instance),
            "n_invalid": int(self.n_instance),
            "metric_mode": str(self.metric_mode),
            "grouped_metric_enabled": bool(self.use_grouped_metric),
            "group_metric_weights": dict(self.group_metric_weights),
            "selection_metric_basis": ("gap" if self.use_gap_metric else "cost"),
            "subgroup_objective_mode": ("mean_std_worst" if self.use_grouped_metric else "mean_only"),
            "group_mean_eval": invalid_metric,
            "group_std_eval": (invalid_metric if self.use_grouped_metric else 0.0),
            "group_worst_eval": invalid_metric,
            "group_means": [invalid_metric for _ in range(n_groups)],
            "group_means_gap": [invalid_metric for _ in range(n_groups)],
            "group_means_time_sec": [invalid_metric for _ in range(n_groups)],
            "group_mean_cost": invalid_metric,
            "group_std_cost": (invalid_metric if self.use_grouped_metric else 0.0),
            "group_worst_cost": invalid_metric,
            "group_worst_count": int(max(1, n_groups // 2)),
            "group_mean_gap": invalid_metric,
            "group_std_gap": (invalid_metric if self.use_grouped_metric else 0.0),
            "group_worst_gap": invalid_metric,
            "group_worst_count_gap": int(max(1, n_groups // 2)),
            "group_mean_time_sec": invalid_metric,
            "group_std_time_sec": (invalid_metric if self.use_grouped_metric else 0.0),
            "group_worst_time_sec": invalid_metric,
            "group_worst_count_time": int(max(1, n_groups // 2)),
            "subgroup_tail_basis": (
                str(self.subgroup_tail_basis) if self.use_gap_metric else "cost"
            ),
            "n_groups": n_groups,
            "metric": invalid_metric,
            "jit_kernel_enabled": bool(self.jit_kernel_enabled),
            "reference_cost_source": "concorde_required",
            "n_reference_fallback": int(getattr(self, "_reference_fallback_count", self.n_instance)),
        }

    def evaluate_construct(self, fn: Callable) -> Dict[str, Any]:
        objs = np.full(self.n_instance, self.LARGE, dtype=float)
        gaps = np.full(self.n_instance, self.LARGE, dtype=float)
        runtimes_ms = np.zeros(self.n_instance, dtype=float)
        n_invalid = 0

        for i in range(self.n_instance):
            row = self._evaluate_one_instance(fn, i)
            runtimes_ms[i] = float(row["runtime_ms"])
            if row["status"] != "ok":
                n_invalid += 1
                continue
            objs[i] = float(row["obj"])
            gaps[i] = float(row["gap"])

        avg_tsp_time = float(np.mean(runtimes_ms) / 1000.0) if runtimes_ms.size > 0 else 0.0

        # Any failed sample => infeasible.
        if n_invalid > 0:
            out = self._invalid_eval_result(error=f"n_invalid={n_invalid}")
            out["avg_tsp_time_sec"] = avg_tsp_time
            out["time_cost_sec"] = avg_tsp_time
            out["n_invalid"] = int(n_invalid)
            return out

        avg_obj = float(np.mean(objs))
        avg_gap = float(np.mean(gaps))
        times_sec = runtimes_ms / 1000.0

        group_means_cost: List[float] = []
        group_means_gap: List[float] = []
        group_means_time_sec: List[float] = []
        for idxs in self.group_indices:
            arr_idx = np.asarray(idxs, dtype=int)
            if arr_idx.size == 0:
                continue
            group_means_cost.append(float(np.mean(objs[arr_idx])))
            group_means_gap.append(float(np.mean(gaps[arr_idx])))
            group_means_time_sec.append(float(np.mean(times_sec[arr_idx])))
        if not group_means_cost:
            group_means_cost = [avg_obj]
        if not group_means_gap:
            group_means_gap = [avg_gap]
        if not group_means_time_sec:
            group_means_time_sec = [avg_tsp_time]

        grouped_mean_cost = float(np.mean(group_means_cost))
        grouped_std_cost = self._population_std(group_means_cost)
        grouped_worst_cost, worst_group_count_cost = self._worst_half_mean(group_means_cost)
        grouped_mean_gap = float(np.mean(group_means_gap))
        grouped_std_gap = self._population_std(group_means_gap)
        grouped_worst_gap, worst_group_count_gap = self._worst_half_mean(group_means_gap)
        grouped_mean_time_sec = float(np.mean(group_means_time_sec))
        grouped_std_time_sec = self._population_std(group_means_time_sec)
        grouped_worst_time_sec, worst_group_count_time = self._worst_half_mean(group_means_time_sec)

        if self.use_gap_metric:
            group_mean_eval = float(grouped_mean_gap)
            subgroup_tail_basis_effective = str(self.subgroup_tail_basis)
            if subgroup_tail_basis_effective == "time":
                group_std_eval_raw = float(grouped_std_time_sec)
                group_worst_eval_raw = float(grouped_worst_time_sec)
            else:
                group_std_eval_raw = float(grouped_std_gap)
                group_worst_eval_raw = float(grouped_worst_gap)
            avg_eval = float(avg_gap)
            selection_metric_basis = "gap"
        else:
            group_mean_eval = float(grouped_mean_cost)
            group_std_eval_raw = float(grouped_std_cost)
            group_worst_eval_raw = float(grouped_worst_cost)
            subgroup_tail_basis_effective = "cost"
            avg_eval = float(avg_obj)
            selection_metric_basis = "cost"

        if self.use_grouped_metric:
            group_std_eval = float(group_std_eval_raw)
            group_worst_eval = float(group_worst_eval_raw)
            subgroup_objective_mode = "mean_std_worst"
            metric = float(group_mean_eval)
        else:
            group_std_eval = 0.0
            group_worst_eval = float(group_mean_eval)
            subgroup_objective_mode = "mean_only"
            metric = float(avg_eval)

        return {
            "status": "ok",
            "error": None,
            "avg_obj": avg_obj,
            "avg_tsp_time_sec": avg_tsp_time,
            "avg_objective": avg_obj,
            "avg_gap": avg_gap,
            "avg_gap_ratio": avg_gap,
            "avg_gap_pct": float(avg_gap * 100.0),
            "time_cost_sec": avg_tsp_time,
            "n_eval": int(self.n_instance),
            "n_invalid": int(n_invalid),
            "metric_mode": str(self.metric_mode),
            "grouped_metric_enabled": bool(self.use_grouped_metric),
            "group_metric_weights": dict(self.group_metric_weights),
            "selection_metric_basis": str(selection_metric_basis),
            "subgroup_objective_mode": str(subgroup_objective_mode),
            "group_mean_eval": float(group_mean_eval),
            "group_std_eval": float(group_std_eval),
            "group_worst_eval": float(group_worst_eval),
            "group_means": [float(x) for x in group_means_cost],
            "group_means_gap": [float(x) for x in group_means_gap],
            "group_means_time_sec": [float(x) for x in group_means_time_sec],
            "group_mean_cost": grouped_mean_cost,
            "group_std_cost": grouped_std_cost,
            "group_worst_cost": grouped_worst_cost,
            "group_worst_count": int(worst_group_count_cost),
            "group_mean_gap": grouped_mean_gap,
            "group_std_gap": grouped_std_gap,
            "group_worst_gap": grouped_worst_gap,
            "group_worst_count_gap": int(worst_group_count_gap),
            "group_mean_time_sec": grouped_mean_time_sec,
            "group_std_time_sec": grouped_std_time_sec,
            "group_worst_time_sec": grouped_worst_time_sec,
            "group_worst_count_time": int(worst_group_count_time),
            "subgroup_tail_basis": str(subgroup_tail_basis_effective),
            "n_groups": int(len(group_means_cost)),
            "metric": float(metric),
            "jit_kernel_enabled": bool(self.jit_kernel_enabled),
            "reference_cost_source": "concorde",
            "n_reference_fallback": 0,
        }

    def evaluate(self, code_string: str) -> float | None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                fn, compile_error = self._compile_function(code_string)
                if compile_error is not None or fn is None:
                    self._last_eval_info = self._invalid_eval_result(error=str(compile_error))
                    return float(self.LARGE)

                sig_ok, sig_error = self._check_signature(fn)
                if not sig_ok:
                    self._last_eval_info = self._invalid_eval_result(error=str(sig_error))
                    return float(self.LARGE)

                res = self.evaluate_construct(fn)
                self._last_eval_info = dict(res)
                return float(res.get("metric", self.LARGE))
        except Exception as e:  # noqa: BLE001
            self._last_eval_info = self._invalid_eval_result(error=str(e))
            return float(self.LARGE)

    def get_last_eval_info(self) -> Dict[str, Any]:
        return dict(self._last_eval_info)


# Backward compatibility with previous static_train import path/class names.
TSPSelectNextNodeProblem = TSPGLSUpdateEdgeProblem
TSPGLSStaticProblem = TSPGLSUpdateEdgeProblem
TSPGLSProblem = TSPGLSUpdateEdgeProblem
