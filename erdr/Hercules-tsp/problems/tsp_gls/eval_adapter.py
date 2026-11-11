from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import numpy as np

from gen_inst import TSPInstance, load_dataset
from gls import guided_local_search


LARGE = 1e12


def _calculate_cost(inst: TSPInstance, path: np.ndarray) -> float:
    return float(inst.distmat[path, np.roll(path, 1)].sum().item())


def _reference_cost(inst: TSPInstance, gap_eps: float) -> tuple[float, str]:
    ref_raw = getattr(inst, "reference_cost", None)
    if ref_raw is None:
        raise ValueError("missing concorde reference_cost")
    try:
        ref = float(ref_raw)
    except Exception as e:
        raise ValueError("invalid concorde reference_cost") from e
    if (not np.isfinite(ref)) or ref <= gap_eps:
        raise ValueError("non-positive or non-finite concorde reference_cost")
    return float(ref), "concorde"


def _load_heuristic_function(code_path: Path) -> Callable[[np.ndarray], np.ndarray]:
    if not code_path.exists():
        raise FileNotFoundError(f"code file not found: {code_path}")

    spec = importlib.util.spec_from_file_location("hercules_candidate", str(code_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import candidate code: {code_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    fn = None
    for name in ("heuristics_v2", "heuristics"):
        candidate = getattr(module, name, None)
        if callable(candidate):
            fn = candidate
            break
    if fn is None:
        raise RuntimeError("heuristics_v2 / heuristics not found in candidate code.")
    return fn


def _population_std(values: Sequence[float]) -> float:
    arr = np.asarray([float(x) for x in values], dtype=float)
    if arr.size <= 0:
        return 0.0
    mean_v = float(np.mean(arr))
    sq_mean = float(np.mean((arr - mean_v) ** 2))
    return float(np.sqrt(max(sq_mean, 0.0)))


def _worst_half_mean(values: Sequence[float]) -> tuple[float, int]:
    arr = np.asarray([float(x) for x in values], dtype=float)
    n = int(arr.size)
    if n <= 0:
        return 0.0, 0
    m = max(1, n // 2)
    sorted_desc = np.sort(arr)[::-1]
    return float(np.mean(sorted_desc[:m])), int(m)


def _parse_group_indices(raw: str, n_eval: int) -> List[List[int]]:
    if not str(raw or "").strip():
        return [list(range(n_eval))]
    try:
        obj = json.loads(raw)
    except Exception as e:
        raise ValueError(f"invalid group_indices_json: {e}") from e
    if not isinstance(obj, list):
        raise ValueError("group_indices_json must decode to a list.")
    groups: List[List[int]] = []
    for gid, group in enumerate(obj):
        if not isinstance(group, list):
            raise ValueError(f"group {gid} is not a list.")
        cleaned: List[int] = []
        for idx in group:
            i = int(idx)
            if i < 0 or i >= n_eval:
                raise ValueError(f"group index out of range: {i}, n_eval={n_eval}")
            if i not in cleaned:
                cleaned.append(i)
        if cleaned:
            groups.append(cleaned)
    if not groups:
        return [list(range(n_eval))]

    flat = sorted(i for g in groups for i in g)
    expected = list(range(n_eval))
    if flat != expected:
        raise ValueError(
            "group_indices_json must form a full partition of [0, n_eval). "
            f"Expected {len(expected)} indices, got {len(flat)}."
        )
    return groups


def _invalid_result(
    error: str,
    n_eval: int,
    metric_basis: str,
    subgroup_mode: str,
    subgroup_tail_basis: str = "gap",
) -> Dict[str, Any]:
    return {
        "status": "invalid",
        "error": str(error),
        "selected_metric": float(LARGE),
        "selected_metric_name": ("mean_gap" if metric_basis == "gap" else "mean_cost"),
        "mean_cost": float(LARGE),
        "mean_gap": float(LARGE),
        "mean_gap_pct": float(LARGE * 100.0),
        "mean_time_sec": 0.0,
        "n_eval": int(n_eval),
        "n_invalid": int(n_eval),
        "metric_basis": str(metric_basis),
        "group_evaluation_mode": str(subgroup_mode),
        "subgroup_tail_basis": str(subgroup_tail_basis),
        "group_mean_eval": float(LARGE),
        "group_std_eval": float(LARGE),
        "group_worst_eval": float(LARGE),
        "objective_vector": [float(LARGE), float(LARGE), float(LARGE)],
    }


def evaluate_candidate(
    *,
    dataset: Sequence[TSPInstance],
    heuristics_fn: Callable[[np.ndarray], np.ndarray],
    metric_basis: str,
    subgroup_mode: str,
    subgroup_tail_basis: str,
    group_indices: Sequence[Sequence[int]],
    perturbation_moves: int,
    iter_limit: int,
    time_limit_per_instance: float,
    gap_eps: float,
) -> Dict[str, Any]:
    n_eval = int(len(dataset))
    t_cap = float(time_limit_per_instance)
    if (not np.isfinite(t_cap)) or t_cap <= 0.0:
        raise ValueError(
            f"time_limit_per_instance must be > 0 for strict per-instance timing, got {time_limit_per_instance}"
        )
    costs: List[float] = []
    gaps: List[float] = []
    times: List[float] = []
    n_invalid = 0

    missing_ref_indices: List[int] = []
    for i, inst in enumerate(dataset):
        ref = getattr(inst, "reference_cost", None)
        ok = False
        try:
            rv = float(ref)
            ok = bool(np.isfinite(rv) and rv > float(max(gap_eps, 1e-12)))
        except Exception:
            ok = False
        if not ok:
            missing_ref_indices.append(i)
    if missing_ref_indices:
        preview = ",".join(str(i) for i in missing_ref_indices[:10])
        raise ValueError(
            "Concorde reference costs are required for gap evaluation. "
            f"Missing/invalid {len(missing_ref_indices)}/{n_eval} instances "
            f"(first indices: {preview})."
        )

    for inst in dataset:
        t1 = time.perf_counter()
        deadline = (t1 + t_cap) if t_cap > 0.0 else None
        try:
            guide = heuristics_fn(inst.distmat.copy())
            guide_arr = np.asarray(guide, dtype=np.float32)
            if guide_arr.shape != inst.distmat.shape:
                raise ValueError(
                    f"invalid heuristic matrix shape: {guide_arr.shape}, expected {inst.distmat.shape}"
                )
            tour = guided_local_search(
                inst.distmat,
                guide_arr,
                perturbation_moves=max(int(perturbation_moves), 1),
                iter_limit=max(int(iter_limit), 1),
                time_limit_sec=t_cap,
                deadline_perf_counter=deadline,
            )
            obj = _calculate_cost(inst, np.asarray(tour, dtype=np.int64))
            ref, _ = _reference_cost(inst, float(max(gap_eps, 1e-12)))
            raw_gap = float((obj - ref) / ref)
            if not np.isfinite(obj) or not np.isfinite(raw_gap):
                raise ValueError("non-finite objective or gap")
            # Guard against tiny negative floating drift (e.g. -1e-17) when
            # objective and Concorde reference are numerically equal.
            gap_tol = float(max(gap_eps, 1.0e-12))
            gap = 0.0 if (raw_gap < 0.0 and abs(raw_gap) <= gap_tol) else float(raw_gap)

            costs.append(float(obj))
            gaps.append(float(gap))
            times.append(float(time.perf_counter() - t1))
        except Exception:
            n_invalid += 1
            costs.append(float(LARGE))
            gaps.append(float(LARGE))
            times.append(float(time.perf_counter() - t1))

    mean_cost = float(np.mean(np.asarray(costs, dtype=float)))
    mean_gap = float(np.mean(np.asarray(gaps, dtype=float)))
    mean_time = float(np.mean(np.asarray(times, dtype=float)))

    group_means_cost: List[float] = []
    group_means_gap: List[float] = []
    group_means_time_sec: List[float] = []
    arr_cost = np.asarray(costs, dtype=float)
    arr_gap = np.asarray(gaps, dtype=float)
    arr_time = np.asarray(times, dtype=float)
    for idxs in group_indices:
        idx_arr = np.asarray([int(i) for i in idxs], dtype=int)
        if idx_arr.size <= 0:
            continue
        group_means_cost.append(float(np.mean(arr_cost[idx_arr])))
        group_means_gap.append(float(np.mean(arr_gap[idx_arr])))
        group_means_time_sec.append(float(np.mean(arr_time[idx_arr])))
    if not group_means_cost:
        group_means_cost = [mean_cost]
    if not group_means_gap:
        group_means_gap = [mean_gap]
    if not group_means_time_sec:
        group_means_time_sec = [mean_time]

    group_mean_time_sec = float(np.mean(group_means_time_sec))
    group_std_time_sec = float(_population_std(group_means_time_sec))
    group_worst_time_sec, _ = _worst_half_mean(group_means_time_sec)

    if metric_basis == "gap":
        group_mean_eval = float(np.mean(group_means_gap))
        tail_basis_effective = str(subgroup_tail_basis)
        if tail_basis_effective == "time":
            group_std_eval_raw = float(group_std_time_sec)
            group_worst_eval_raw = float(group_worst_time_sec)
        else:
            group_std_eval_raw = float(_population_std(group_means_gap))
            group_worst_eval_raw, _ = _worst_half_mean(group_means_gap)
        avg_eval = float(mean_gap)
        selected_metric_name = "mean_gap"
    else:
        group_mean_eval = float(np.mean(group_means_cost))
        group_std_eval_raw = float(_population_std(group_means_cost))
        group_worst_eval_raw, _ = _worst_half_mean(group_means_cost)
        tail_basis_effective = "cost"
        avg_eval = float(mean_cost)
        selected_metric_name = "mean_cost"

    if subgroup_mode == "mean_variability_tail":
        group_std_eval = float(group_std_eval_raw)
        group_worst_eval = float(group_worst_eval_raw)
        selected_metric = float(group_mean_eval)
    else:
        group_std_eval = 0.0
        group_worst_eval = float(group_mean_eval)
        selected_metric = float(avg_eval)

    return {
        "status": "ok",
        "error": (None if n_invalid <= 0 else f"n_invalid={n_invalid}"),
        "selected_metric": float(selected_metric),
        "selected_metric_name": str(selected_metric_name),
        "mean_cost": float(mean_cost),
        "mean_gap": float(mean_gap),
        "mean_gap_pct": float(mean_gap * 100.0),
        "mean_time_sec": float(mean_time),
        "n_eval": int(n_eval),
        "n_invalid": int(n_invalid),
        "metric_basis": str(metric_basis),
        "group_evaluation_mode": str(subgroup_mode),
        "subgroup_tail_basis": str(tail_basis_effective),
        "group_mean_eval": float(group_mean_eval),
        "group_std_eval": float(group_std_eval),
        "group_worst_eval": float(group_worst_eval),
        "group_means_cost": [float(x) for x in group_means_cost],
        "group_means_gap": [float(x) for x in group_means_gap],
        "group_means_time_sec": [float(x) for x in group_means_time_sec],
        "group_mean_time_sec": float(group_mean_time_sec),
        "group_std_time_sec": float(group_std_time_sec),
        "group_worst_time_sec": float(group_worst_time_sec),
        "objective_vector": [
            float(group_mean_eval),
            float(group_std_eval),
            float(group_worst_eval),
        ],
        "time_limit_per_instance": float(t_cap),
        "reference_cost_source": "concorde",
        "n_reference_fallback": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one Hercules tsp_gls heuristic on dataset.")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--code-path", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--metric-basis", type=str, choices=["gap", "cost"], default="gap")
    parser.add_argument(
        "--subgroup-mode",
        type=str,
        choices=["mean_only", "mean_variability_tail"],
        default="mean_only",
    )
    parser.add_argument("--subgroup-tail-basis", type=str, choices=["gap", "time"], default="gap")
    parser.add_argument("--group-indices-json", type=str, default="")
    parser.add_argument("--n-instance-eval", type=int, default=0)
    parser.add_argument("--perturbation-moves", type=int, default=30)
    parser.add_argument("--iter-limit", type=int, default=1200)
    parser.add_argument("--time-limit-per-instance", type=float, default=0.0)
    parser.add_argument("--gap-eps", type=float, default=1.0e-12)
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path).resolve()
    code_path = Path(args.code_path).resolve()
    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    try:
        dataset_all = load_dataset(str(dataset_path))
        if int(args.n_instance_eval) > 0:
            n_eval = min(max(int(args.n_instance_eval), 1), len(dataset_all))
            dataset = dataset_all[:n_eval]
        else:
            dataset = dataset_all
            n_eval = len(dataset)

        group_indices = _parse_group_indices(args.group_indices_json, n_eval=n_eval)
        heuristics_fn = _load_heuristic_function(code_path)
        result = evaluate_candidate(
            dataset=dataset,
            heuristics_fn=heuristics_fn,
            metric_basis=str(args.metric_basis),
            subgroup_mode=str(args.subgroup_mode),
            subgroup_tail_basis=str(args.subgroup_tail_basis),
            group_indices=group_indices,
            perturbation_moves=int(args.perturbation_moves),
            iter_limit=int(args.iter_limit),
            time_limit_per_instance=float(args.time_limit_per_instance),
            gap_eps=float(args.gap_eps),
        )
    except Exception as e:
        msg = str(e)
        if (
            ("Concorde reference costs are required" in msg)
            or ("Concorde reference costs are incomplete" in msg)
            or ("time_limit_per_instance must be > 0" in msg)
        ):
            raise
        result = _invalid_result(
            error=str(e),
            n_eval=0,
            metric_basis=str(args.metric_basis),
            subgroup_mode=str(args.subgroup_mode),
            subgroup_tail_basis=str(args.subgroup_tail_basis),
        )

    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(float(result.get("selected_metric", LARGE)))


if __name__ == "__main__":
    main()
