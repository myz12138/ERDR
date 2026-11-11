from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Sequence, Tuple

import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.optimization.tsp_gls_2O.get_instance import GetData, TSPInstance
from llm4ad.task.optimization.tsp_gls_2O.template import template_program, task_description

from .gls import guided_local_search_with_time

__all__ = ["TSPGLSMultiObjectiveEvaluation", "load_tsp_dataset"]


def _as_rows(value: Any, name: str) -> list[Any]:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            raise ValueError(f"dataset field {name} must not be a scalar")
        return [value[i] for i in range(value.shape[0])]
    if isinstance(value, (list, tuple)):
        return list(value)
    raise ValueError(f"dataset field {name} must be an array or sequence")


def load_tsp_dataset(data_path: str | Path) -> tuple[list[TSPInstance], list[float]]:
    path = Path(data_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"TSP dataset not found: {path}")
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    if not isinstance(raw, dict) or "coordinate" not in raw:
        raise ValueError("TSP dataset must be a mapping containing coordinate and cost_concorde")
    coordinates = _as_rows(raw["coordinate"], "coordinate")
    distance_raw = raw.get("distance_matrix")
    distances = [None] * len(coordinates) if distance_raw is None else _as_rows(distance_raw, "distance_matrix")
    if len(distances) != len(coordinates):
        raise ValueError("coordinate and distance_matrix counts differ")
    costs_raw = raw.get("cost_concorde")
    if costs_raw is None:
        raise ValueError("TSP dataset requires cost_concorde for ERDR-compatible gap evaluation")
    costs = [float(x) for x in _as_rows(costs_raw, "cost_concorde")]
    if len(costs) != len(coordinates):
        raise ValueError("coordinate and cost_concorde counts differ")
    if any((not np.isfinite(cost)) or cost <= 0.0 for cost in costs):
        raise ValueError("cost_concorde contains non-positive or non-finite values")
    instances = [
        TSPInstance(np.asarray(coords, dtype=np.float64), None if dist is None else np.asarray(dist, dtype=np.float64))
        for coords, dist in zip(coordinates, distances)
    ]
    return instances, costs


def calculate_cost(inst: TSPInstance, path: np.ndarray) -> float:
    return inst.distmat[path, np.roll(path, 1)].sum().item()


def solve_with_time(
    inst: TSPInstance,
    evaluator: Any,
    perturbation_moves: int,
    iter_limit: int,
    time_limit_seconds: float,
) -> Tuple[float, float]:
    try:
        result, running_time = guided_local_search_with_time(
            inst.distmat,
            inst.distmat.copy(),
            evaluator,
            perturbation_moves,
            iter_limit,
            time_limit_seconds,
        )
        return calculate_cost(inst, result), float(running_time)
    except Exception:
        return float("inf"), float("inf")


class TSPGLSMultiObjectiveEvaluation(Evaluation):
    def __init__(
        self,
        *,
        instances: Sequence[TSPInstance] | None = None,
        reference_costs: Sequence[float] | None = None,
        dataset_path: str | Path | None = None,
        n_instance_eval: int | None = None,
        mode: str = "baseline",
        group_indices: Sequence[Sequence[int]] | None = None,
        perturbation_moves: int = 2,
        iter_limit: int = 500,
        time_limit_per_instance: float = 20.0,
        timeout_seconds: float = 1200.0,
    ) -> None:
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds,
        )
        if dataset_path is not None:
            instances, reference_costs = load_tsp_dataset(dataset_path)
        if instances is None:
            n_instance = 16 if n_instance_eval is None else int(n_instance_eval)
            instances = GetData(n_instance, 100).generate_instances()
            reference_costs = None
        if not instances:
            raise ValueError("instances is empty")
        requested = len(instances) if n_instance_eval is None or int(n_instance_eval) <= 0 else int(n_instance_eval)
        self._datasets = list(instances[: max(1, min(requested, len(instances)))])
        self.n_instance = len(self._datasets)
        if reference_costs is None:
            self._reference_costs = None
        else:
            values = np.asarray(reference_costs[: self.n_instance], dtype=float)
            if values.size != self.n_instance or np.any(~np.isfinite(values)) or np.any(values <= 0.0):
                raise ValueError("reference_costs must contain one positive value per instance")
            self._reference_costs = values
        self.mode = str(mode).strip().lower()
        if self.mode not in {"baseline", "erdr"}:
            raise ValueError("mode must be baseline or erdr")
        self.perturbation_moves = max(1, int(perturbation_moves))
        self.iter_limit = max(1, int(iter_limit))
        self.time_limit_per_instance = max(0.1, float(time_limit_per_instance))
        self.group_indices = self._validate_groups(group_indices)
        self.last_metrics: dict[str, Any] = {}

    def _validate_groups(self, group_indices: Sequence[Sequence[int]] | None) -> list[list[int]]:
        if self.mode == "baseline":
            return [list(range(self.n_instance))]
        if not group_indices:
            raise ValueError("ERDR evaluation requires a complete group partition")
        groups = [[int(index) for index in group] for group in group_indices]
        flattened = sorted(index for group in groups for index in group)
        if flattened != list(range(self.n_instance)) or any(not group for group in groups):
            raise ValueError("ERDR group_indices must partition all evaluated instances")
        return groups

    @staticmethod
    def _tail_mean(values: np.ndarray) -> float:
        count = max(1, values.size // 2)
        return float(np.mean(np.sort(values)[-count:]))

    def evaluate_program(self, program_str: str, callable_func: Any) -> np.ndarray:
        costs = np.empty(self.n_instance, dtype=float)
        runtimes = np.empty(self.n_instance, dtype=float)
        for index, instance in enumerate(self._datasets):
            costs[index], runtimes[index] = solve_with_time(
                instance,
                callable_func,
                self.perturbation_moves,
                self.iter_limit,
                self.time_limit_per_instance,
            )
        if np.any(~np.isfinite(costs)) or np.any(~np.isfinite(runtimes)):
            size = 4 if self.mode == "erdr" else 2
            return np.full(size, -np.inf, dtype=float)
        if self._reference_costs is None:
            gaps = costs
        else:
            gaps = (costs - self._reference_costs) / self._reference_costs
        mean_gap = float(np.mean(gaps))
        mean_runtime = float(np.mean(runtimes))
        group_gaps = np.asarray([np.mean(gaps[group]) for group in self.group_indices], dtype=float)
        group_runtimes = np.asarray([np.mean(runtimes[group]) for group in self.group_indices], dtype=float)
        self.last_metrics = {
            "mean_gap": mean_gap,
            "mean_runtime_sec": mean_runtime,
            "group_gap_std": float(np.std(group_gaps)),
            "group_runtime_std": float(np.std(group_runtimes)),
            "group_runtime_tail": self._tail_mean(group_runtimes),
            "n_instance_eval": self.n_instance,
            "n_groups": len(self.group_indices),
            "mode": self.mode,
        }
        if self.mode == "baseline":
            return -np.asarray([mean_gap, mean_runtime], dtype=float)
        return -np.asarray(
            [
                mean_gap,
                mean_runtime,
                float(np.std(group_runtimes)),
                self._tail_mean(group_runtimes),
            ],
            dtype=float,
        )
