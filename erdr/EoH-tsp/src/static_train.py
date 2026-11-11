from __future__ import annotations

import json
import pickle
import shutil
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from result_utils import save_json, save_jsonl, save_text
from solver_archive import build_solver_archive


def _ensure_requests_stub() -> None:
    try:
        import requests  # noqa: F401
        return
    except Exception:
        pass

    stub = types.ModuleType("requests")

    def _not_available(*args, **kwargs):
        raise RuntimeError("requests package is not installed in current environment.")

    stub.post = _not_available  # type: ignore[attr-defined]
    stub.get = _not_available  # type: ignore[attr-defined]
    sys.modules["requests"] = stub


def _ensure_joblib_stub() -> None:
    try:
        import joblib  # noqa: F401
        return
    except Exception:
        pass

    stub = types.ModuleType("joblib")

    def delayed(func):
        def _wrap(*args, **kwargs):
            return lambda: func(*args, **kwargs)

        return _wrap

    class Parallel:
        def __init__(self, n_jobs=1, timeout=None):
            self.n_jobs = n_jobs
            self.timeout = timeout

        def __call__(self, jobs):
            return [job() for job in jobs]

    stub.delayed = delayed  # type: ignore[attr-defined]
    stub.Parallel = Parallel  # type: ignore[attr-defined]
    sys.modules["joblib"] = stub


def _find_best_individual(
    run_output_dir: Path,
    prefer_population_order: bool = False,
) -> Optional[Dict[str, Any]]:
    pops_best_dir = run_output_dir / "results" / "pops_best"
    if not pops_best_dir.exists():
        return None

    if prefer_population_order:
        latest_row: Optional[Dict[str, Any]] = None
        for path in sorted(pops_best_dir.glob("population_generation_*.json")):
            try:
                with path.open("r", encoding="utf-8") as f:
                    row = json.load(f)
                if not isinstance(row, dict):
                    continue
                if not str(row.get("code", "")).strip():
                    continue
                latest_row = row
            except Exception:
                continue
        if latest_row is not None:
            return latest_row

    best_obj = float("inf")
    best_row: Optional[Dict[str, Any]] = None
    for path in sorted(pops_best_dir.glob("population_generation_*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                row = json.load(f)
            if not isinstance(row, dict):
                continue
            obj = float(row.get("objective", float("inf")))
            if obj < best_obj:
                best_obj = obj
                best_row = row
        except Exception:
            continue
    return best_row


def _collect_search_trace(run_output_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    pops_dir = run_output_dir / "results" / "pops"
    if pops_dir.exists():
        for path in sorted(pops_dir.glob("population_generation_*.json")):
            generation = None
            try:
                generation = int(path.stem.rsplit("_", 1)[-1])
            except Exception:
                generation = None
            try:
                with path.open("r", encoding="utf-8") as f:
                    obj = json.load(f)
            except Exception:
                continue

            candidates: List[Dict[str, Any]] = []
            if isinstance(obj, dict):
                candidates = [obj]
            elif isinstance(obj, list):
                candidates = [x for x in obj if isinstance(x, dict)]
            for rank, row in enumerate(candidates):
                rec = dict(row)
                rec["source_file"] = str(path.resolve())
                rec["source_type"] = "population"
                rec["generation"] = generation
                rec["population_rank"] = int(rank)
                rows.append(rec)
    if rows:
        return rows

    pops_best_dir = run_output_dir / "results" / "pops_best"
    if not pops_best_dir.exists():
        return rows
    for path in sorted(pops_best_dir.glob("population_generation_*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                row = json.load(f)
            if isinstance(row, dict):
                row = dict(row)
                row["source_file"] = str(path.resolve())
                row["source_type"] = "population_best"
                rows.append(row)
        except Exception:
            continue
    return rows


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _clean_run_result_files(run_output_dir: Path) -> None:
    results_dir = run_output_dir / "results"
    for sub in ("pops", "pops_best"):
        p = results_dir / sub
        if not p.exists():
            continue
        for f in p.glob("population_generation_*.json"):
            try:
                f.unlink()
            except Exception:
                continue


def _parse_output_policy(eval_cfg: Dict[str, Any]) -> tuple[bool, int]:
    raw_mode = str(eval_cfg.get("output_mode", "minimal") or "minimal").strip().lower()
    minimal_output = raw_mode != "full"
    try:
        keep_top_k = int(eval_cfg.get("keep_top_k_solvers", eval_cfg.get("archive_top_k", 10)))
    except Exception:
        keep_top_k = 10
    keep_top_k = max(1, keep_top_k)
    return bool(minimal_output), int(keep_top_k)


def _cleanup_minimal_outputs(output_dir: Path, run_output_dir: Path) -> None:
    # Keep only: best_solver.py, solver_archive/, solver_archive.json, final_eval.json.
    for name in (
        "search_trace.jsonl",
        "train_dataset_avg_metrics.json",
        "val_dataset_avg_metrics.json",
        "best_result.json",
        "train_groups_manifest.json",
        "train_instance_features.jsonl",
        "train_instance_meta.jsonl",
    ):
        p = output_dir / name
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass

    groups_dir = output_dir / "train_groups"
    if groups_dir.exists():
        shutil.rmtree(groups_dir, ignore_errors=True)

    if run_output_dir.exists():
        shutil.rmtree(run_output_dir, ignore_errors=True)


def _parse_grouped_evaluation_cfg(eval_cfg: Dict[str, Any]) -> tuple[bool, Dict[str, Any]]:
    raw = eval_cfg.get("grouped_evaluation", {})
    if not isinstance(raw, dict):
        raw = {}

    def _int_or_default(key: str, default: int) -> int:
        try:
            return int(raw.get(key, default))
        except Exception:
            return int(default)

    enabled = bool(raw.get("enabled", False))
    out = {
        "enabled": bool(enabled),
        "group_size": int(max(_int_or_default("group_size", 5), 1)),
        "n_groups_total": int(max(_int_or_default("n_groups_total", 12), 1)),
        # allow 0 to explicitly disable train/val step
        "n_groups_train": int(max(_int_or_default("n_groups_train", 8), 0)),
        "n_groups_val": int(max(_int_or_default("n_groups_val", 4), 0)),
        "split_seed": int(_int_or_default("split_seed", 42)),
    }
    return bool(enabled), out


def _parse_dds_cfg(eval_cfg: Dict[str, Any]) -> Dict[str, Any]:
    raw = eval_cfg.get("dominance_dissimilarity", {})
    if not isinstance(raw, dict):
        raw = {}

    def _f(key: str, default: Any) -> float:
        try:
            return float(raw.get(key, default))
        except Exception:
            try:
                return float(default)
            except Exception:
                return 0.0

    enabled = bool(raw.get("enabled", False))
    return {
        "enabled": bool(enabled),
        "eps_rel": float(max(_f("eps_rel", 1e-3), 0.0)),
        "softmax_tau": float(max(_f("softmax_tau", 1.0), 1e-6)),
        "novelty_weight": float(max(_f("novelty_weight", 0.05), 0.0)),
    }


def _parse_boolish(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _parse_group_evaluation_cfg(eval_cfg: Dict[str, Any], *, erdr_mode: bool) -> Dict[str, Any]:
    default_multi = bool(erdr_mode)
    raw_multi = eval_cfg.get("group_multi_objective", None)
    raw_mode = eval_cfg.get("group_evaluation_mode", None)

    if raw_multi is not None:
        multi_obj = _parse_boolish(raw_multi, default_multi)
    else:
        mode = str(raw_mode or "").strip().lower()
        if mode in {"mean", "mean_only", "avg", "average"}:
            multi_obj = False
        elif mode in {
            "mean_variability_tail",
            "mean+std+worst",
            "multi",
            "multi_objective",
            "multi-objective",
            "mo",
        }:
            multi_obj = True
        else:
            multi_obj = default_multi

    return {
        "group_multi_objective": bool(multi_obj),
        "group_evaluation_mode": ("mean_variability_tail" if bool(multi_obj) else "mean_only"),
    }


def _parse_tail_risk_basis(eval_cfg: Dict[str, Any]) -> str:
    raw = str(eval_cfg.get("tail_risk_basis", "gap") or "gap").strip().lower()
    if raw not in {"gap", "time"}:
        raise RuntimeError(
            f"Invalid evaluation.tail_risk_basis: {raw}. Expected one of: gap, time."
        )
    return raw


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


def _instance_feature_row(instance: tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    _, dist = instance
    d = np.asarray(dist, dtype=float)
    n = int(d.shape[0])
    if n <= 1:
        return {
            "n_nodes": float(n),
            "mean_edge_length": 0.0,
            "edge_cv": 0.0,
            "cluster_ratio": 0.0,
        }

    triu = np.triu_indices(n, k=1)
    edge_vals = d[triu] if triu[0].size > 0 else np.asarray([0.0], dtype=float)
    mean_edge = float(np.mean(edge_vals))
    std_edge = float(np.std(edge_vals))
    edge_cv = float(std_edge / (mean_edge + 1e-12))

    masked = np.array(d, copy=True)
    np.fill_diagonal(masked, np.inf)
    nn = np.min(masked, axis=1)
    nn = nn[np.isfinite(nn)]
    if nn.size <= 0:
        nn = np.asarray([0.0], dtype=float)
    cluster_ratio = float(np.mean(nn) / (mean_edge + 1e-12))
    return {
        "n_nodes": float(n),
        "mean_edge_length": mean_edge,
        "edge_cv": edge_cv,
        "cluster_ratio": cluster_ratio,
    }


def _feature_rank_indices(features: Sequence[Dict[str, float]]) -> List[int]:
    return sorted(
        range(len(features)),
        key=lambda i: (
            float(features[i].get("mean_edge_length", 0.0)),
            float(features[i].get("cluster_ratio", 0.0)),
            float(features[i].get("edge_cv", 0.0)),
            float(features[i].get("n_nodes", 0.0)),
        ),
    )


def _contiguous_groups(n_items: int, group_size: int) -> List[List[int]]:
    n = int(n_items)
    size = int(max(group_size, 1))
    groups: List[List[int]] = []
    start = 0
    while start < n:
        end = min(start + size, n)
        groups.append(list(range(start, end)))
        start = end
    return groups


def _subset_by_group_ids(
    instances: Sequence[tuple[np.ndarray, np.ndarray]],
    opt_costs: Sequence[float],
    group_indices: Sequence[Sequence[int]],
    group_ids: Sequence[int],
) -> tuple[List[tuple[np.ndarray, np.ndarray]], List[float], List[List[int]]]:
    out_instances: List[tuple[np.ndarray, np.ndarray]] = []
    out_costs: List[float] = []
    out_groups: List[List[int]] = []
    cursor = 0
    for gid in group_ids:
        idxs = [int(i) for i in group_indices[int(gid)]]
        local = list(range(cursor, cursor + len(idxs)))
        out_groups.append(local)
        for i in idxs:
            out_instances.append(instances[i])
            out_costs.append(float(opt_costs[i]))
        cursor += len(idxs)
    return out_instances, out_costs, out_groups


def _save_train_groups(
    *,
    output_dir: Path,
    instances: Sequence[tuple[np.ndarray, np.ndarray]],
    opt_costs: Sequence[float],
    features: Sequence[Dict[str, Any]],
    meta: Sequence[Dict[str, Any]],
    group_indices: Sequence[Sequence[int]],
    train_group_ids: Sequence[int],
    val_group_ids: Sequence[int],
    mode: str,
) -> Dict[str, Any]:
    groups_dir = output_dir / "train_groups"
    groups_dir.mkdir(parents=True, exist_ok=True)

    group_files: List[str] = []
    for gid, idxs in enumerate(group_indices):
        idx_list = [int(i) for i in idxs]
        payload = {
            "meta": {
                "group_id": int(gid),
                "group_size": int(len(idx_list)),
                "instance_indices": idx_list,
                "split": (
                    "train"
                    if int(gid) in {int(x) for x in train_group_ids}
                    else ("val" if int(gid) in {int(x) for x in val_group_ids} else "unused")
                ),
            },
            "coordinate": [instances[i][0] for i in idx_list],
            "distance_matrix": [instances[i][1] for i in idx_list],
            "cost": [float(opt_costs[i]) for i in idx_list],
            "instance_features": [features[i] for i in idx_list],
            "instance_meta": [meta[i] for i in idx_list],
        }
        group_path = groups_dir / f"group_{gid:04d}.pkl"
        with group_path.open("wb") as f:
            pickle.dump(payload, f)
        group_files.append(str(group_path.resolve()))

    manifest = {
        "mode": str(mode),
        "n_instances": int(len(instances)),
        "n_groups": int(len(group_indices)),
        "group_size": int(len(group_indices[0])) if group_indices else int(len(instances)),
        "groups": [[int(i) for i in g] for g in group_indices],
        "train_group_ids": [int(x) for x in train_group_ids],
        "val_group_ids": [int(x) for x in val_group_ids],
        "group_files": list(group_files),
        "feature_fields": ["mean_edge_length", "cluster_ratio", "edge_cv", "n_nodes"],
    }
    manifest_path = save_json(output_dir / "train_groups_manifest.json", manifest)
    save_jsonl(output_dir / "train_instance_features.jsonl", list(features))
    save_jsonl(output_dir / "train_instance_meta.jsonl", list(meta))
    return {
        "manifest_path": str(manifest_path.resolve()),
        "group_dir": str(groups_dir.resolve()),
        "n_groups": int(len(group_indices)),
        "group_size": int(manifest["group_size"]),
    }


def run_static_training(
    cfg: Dict[str, Any],
    repo_root: str | Path,
    progress=None,
) -> Dict[str, Any]:
    def _log(msg: str) -> None:
        if progress is not None:
            progress(msg)
        else:
            print(f"[static_train] {msg}", flush=True)

    _ensure_requests_stub()
    _ensure_joblib_stub()
    from eoh import eoh
    from eoh.utils.getParas import Paras
    from data_utils import load_tsp_dataset
    from snc_tsp_problem import TSPSelectNextNodeProblem

    root = Path(repo_root).resolve()
    path_cfg = dict(cfg.get("paths", {}))
    llm_cfg = dict(cfg.get("llm", {}))
    eoh_cfg = dict(cfg.get("eoh", {}))
    eval_cfg = dict(cfg.get("evaluation", {}))
    exp_cfg = dict(cfg.get("exp", {}))
    minimal_output, keep_top_k = _parse_output_policy(eval_cfg)

    train_pkl = Path(path_cfg.get("train_pkl", "data_generate/TSP100_u50/TSP100_u50_concorde.pkl"))
    if not train_pkl.is_absolute():
        train_pkl = (root / train_pkl).resolve()

    output_dir = Path(path_cfg.get("output_dir", "erdr/EoH-tsp/outputs/static"))
    if not output_dir.is_absolute():
        output_dir = (root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_output_dir = output_dir / "eoh_run"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    _clean_run_result_files(run_output_dir)

    _log(f"train_pkl={train_pkl}")
    _log(f"output_dir={output_dir}")

    instances_all, opt_costs_all = load_tsp_dataset(train_pkl)
    n_eval_requested = int(eval_cfg.get("n_instance_eval", len(instances_all)))
    n_eval_requested = min(max(n_eval_requested, 1), len(instances_all))

    erdr_mode, erdr_cfg = _parse_grouped_evaluation_cfg(eval_cfg)
    dds_cfg = _parse_dds_cfg(eval_cfg)
    if erdr_mode and (not bool(dds_cfg.get("enabled", False))):
        dds_cfg["enabled"] = True
        _log("ERDR grouped evaluation enables dominance-dissimilarity selection.")
    dds_enabled = bool(dds_cfg.get("enabled", False))

    use_gap_metric = _parse_boolish(eval_cfg.get("use_gap_metric", True), True)
    if not use_gap_metric:
        _log("static_train enforces GAP-based selection metrics; override use_gap_metric=false -> true.")
        use_gap_metric = True
    gap_eps = float(eval_cfg.get("gap_eps", 1e-12))
    group_evaluation_cfg = _parse_group_evaluation_cfg(eval_cfg, erdr_mode=erdr_mode)
    subgroup_multi_objective = bool(group_evaluation_cfg["group_multi_objective"])
    subgroup_eval_mode = str(group_evaluation_cfg["group_evaluation_mode"])
    subgroup_tail_basis = _parse_tail_risk_basis(eval_cfg)

    neighbor_k = int(eval_cfg.get("gls_neighbor_k", 50))
    gls_time_limit_per_instance = float(eval_cfg.get("gls_time_limit_per_instance", 10.0))
    gls_ite_max = int(eval_cfg.get("gls_ite_max", 1000))
    gls_perturbation_moves = int(eval_cfg.get("gls_perturbation_moves", 1))
    try:
        from gls_core import NUMBA_AVAILABLE as GLS_NUMBA_AVAILABLE  # local import for startup log
    except Exception:
        GLS_NUMBA_AVAILABLE = False
    group_metric_weights = {"mean": 0.6, "std": 0.2, "worst": 0.2}

    trainval_enabled = False
    if erdr_mode:
        group_size = int(max(erdr_cfg.get("group_size", 5), 1))
        n_groups_total = int(max(erdr_cfg.get("n_groups_total", 12), 1))
        n_groups_train = int(max(erdr_cfg.get("n_groups_train", 8), 0))
        n_groups_val = int(max(erdr_cfg.get("n_groups_val", 4), 0))
        split_seed = int(erdr_cfg.get("split_seed", 42))
        required_total = int(group_size * n_groups_total)
        if n_groups_train <= 0 and n_groups_val <= 0:
            raise RuntimeError("grouped_evaluation requires at least one training or validation group.")
        if n_groups_train + n_groups_val != n_groups_total:
            raise RuntimeError("grouped_evaluation requires n_groups_train + n_groups_val to equal n_groups_total.")
        if len(instances_all) < required_total:
            raise RuntimeError(
                f"grouped_evaluation requires at least {required_total} instances, got {len(instances_all)}."
            )

        n_source = int(min(max(n_eval_requested, required_total), len(instances_all)))
        pool_instances = list(instances_all[:n_source])
        if isinstance(opt_costs_all, list):
            pool_opt_costs = [float(x) for x in opt_costs_all[:n_source]]
        else:
            pool_opt_costs = [_nearest_neighbor_cost_from_dist(ins[1]) for ins in pool_instances]

        pool_features = [_instance_feature_row(ins) for ins in pool_instances]
        rng = np.random.default_rng(split_seed)
        selected = rng.permutation(n_source).tolist()[:required_total]

        instances = [pool_instances[i] for i in selected]
        opt_costs = [float(pool_opt_costs[i]) for i in selected]
        instance_features = [pool_features[i] for i in selected]
        instance_meta = [
            {
                "source_type": "grouped_evaluation",
                "selected_order": int(pos),
                "pool_index": int(selected[pos]),
                "split_seed": int(split_seed),
            }
            for pos in range(len(selected))
        ]

        group_indices = _contiguous_groups(len(instances), group_size)
        if len(group_indices) != n_groups_total or any(len(g) != group_size for g in group_indices):
            raise RuntimeError("failed to construct configured groups for grouped_evaluation.")

        group_id_perm = rng.permutation(n_groups_total).tolist()
        train_group_ids = [int(x) for x in group_id_perm[:n_groups_train]]
        val_group_ids = [int(x) for x in group_id_perm[n_groups_train : n_groups_train + n_groups_val]]

        train_instances, train_opt_costs, train_group_indices = _subset_by_group_ids(
            instances=instances,
            opt_costs=opt_costs,
            group_indices=group_indices,
            group_ids=train_group_ids,
        )
        val_instances, val_opt_costs, val_group_indices = _subset_by_group_ids(
            instances=instances,
            opt_costs=opt_costs,
            group_indices=group_indices,
            group_ids=val_group_ids,
        )

        group_artifacts = _save_train_groups(
            output_dir=output_dir,
            instances=instances,
            opt_costs=opt_costs,
            features=instance_features,
            meta=instance_meta,
            group_indices=group_indices,
            train_group_ids=train_group_ids,
            val_group_ids=val_group_ids,
            mode="erdr",
        )

        # If one side is set to 0, skip that split stage.
        if len(train_instances) > 0:
            evolve_instances = train_instances
            evolve_opt_costs = train_opt_costs
            evolve_group_indices = train_group_indices
            evolve_split_name = "train"
        else:
            evolve_instances = val_instances
            evolve_opt_costs = val_opt_costs
            evolve_group_indices = val_group_indices
            evolve_split_name = "val"

        if len(evolve_instances) <= 0:
            raise RuntimeError("grouped_evaluation has no training instances after the split.")

        problem = TSPSelectNextNodeProblem(
            instances=evolve_instances,
            opt_costs=evolve_opt_costs,
            n_instance_eval=len(evolve_instances),
            neighbor_k=neighbor_k,
            time_limit_per_instance=gls_time_limit_per_instance,
            ite_max=gls_ite_max,
            perturbation_moves=gls_perturbation_moves,
            group_indices=evolve_group_indices,
            use_grouped_metric=subgroup_multi_objective,
            group_metric_weights=group_metric_weights,
            use_gap_metric=use_gap_metric,
            subgroup_tail_basis=subgroup_tail_basis,
            gap_eps=gap_eps,
        )
        if len(train_instances) > 0 and len(val_instances) > 0:
            validation_problem = TSPSelectNextNodeProblem(
                instances=val_instances,
                opt_costs=val_opt_costs,
                n_instance_eval=len(val_instances),
                neighbor_k=neighbor_k,
                time_limit_per_instance=gls_time_limit_per_instance,
                ite_max=gls_ite_max,
                perturbation_moves=gls_perturbation_moves,
                group_indices=val_group_indices,
                use_grouped_metric=subgroup_multi_objective,
                group_metric_weights=group_metric_weights,
                use_gap_metric=use_gap_metric,
                subgroup_tail_basis=subgroup_tail_basis,
                gap_eps=gap_eps,
            )
            trainval_enabled = True
        else:
            validation_problem = None
            trainval_enabled = False

        n_eval_train = int(len(evolve_instances))
        n_eval_val = int(len(val_instances) if validation_problem is not None else 0)
        base_pop_size = int(eoh_cfg.get("ec_pop_size", 4))
        effective_pop_size = int(base_pop_size)
        _log(
            "ERDR grouped evaluation partition => "
            f"group_size={group_size}, n_groups_total={n_groups_total}, "
            f"n_groups_train={n_groups_train}, n_groups_val={n_groups_val}, split_seed={split_seed}; "
            f"evolve_split={evolve_split_name}, n_train={n_eval_train}, n_val={n_eval_val}, "
            f"pop={base_pop_size}, "
            f"trainval_enabled={trainval_enabled}, subgroup_eval_mode={subgroup_eval_mode}, "
            f"subgroup_tail_basis={subgroup_tail_basis}, metric=gap, "
            f"gls(time_limit={gls_time_limit_per_instance}, ite_max={gls_ite_max}, "
            f"moves={gls_perturbation_moves}, k={neighbor_k})"
        )
        max_eval_sec = float(n_eval_train * gls_time_limit_per_instance)
        _log(
            f"jit_kernel_enabled={bool(GLS_NUMBA_AVAILABLE)}, "
            f"theoretical_eval_upper_bound_per_candidate≈{max_eval_sec:.1f}s "
            "(before early-stop / convergence / parallel overlap)"
        )
        mode_name = ("erdr" if trainval_enabled else "baseline")
    else:
        instances = list(instances_all[:n_eval_requested])
        if isinstance(opt_costs_all, list):
            opt_costs = [float(x) for x in opt_costs_all[:n_eval_requested]]
        else:
            opt_costs = [_nearest_neighbor_cost_from_dist(ins[1]) for ins in instances]
        instance_features = [_instance_feature_row(ins) for ins in instances]
        instance_meta = [
            {
                "source_type": "static",
                "pool_index": int(i),
            }
            for i in range(len(instances))
        ]
        group_indices = [list(range(len(instances)))]
        train_group_ids = [0]
        val_group_ids: List[int] = []

        group_artifacts = _save_train_groups(
            output_dir=output_dir,
            instances=instances,
            opt_costs=opt_costs,
            features=instance_features,
            meta=instance_meta,
            group_indices=group_indices,
            train_group_ids=train_group_ids,
            val_group_ids=val_group_ids,
            mode="baseline",
        )

        problem = TSPSelectNextNodeProblem(
            instances=instances,
            opt_costs=opt_costs,
            n_instance_eval=len(instances),
            neighbor_k=neighbor_k,
            time_limit_per_instance=gls_time_limit_per_instance,
            ite_max=gls_ite_max,
            perturbation_moves=gls_perturbation_moves,
            group_indices=group_indices,
            use_grouped_metric=subgroup_multi_objective,
            group_metric_weights=group_metric_weights,
            use_gap_metric=use_gap_metric,
            subgroup_tail_basis=subgroup_tail_basis,
            gap_eps=gap_eps,
        )
        validation_problem = None
        n_eval_train = int(len(instances))
        n_eval_val = 0
        base_pop_size = int(eoh_cfg.get("ec_pop_size", 4))
        effective_pop_size = int(base_pop_size)
        trainval_enabled = False
        _log(
            f"static mode: n_eval={n_eval_train}, pop={base_pop_size}, "
            f"dds={dds_enabled}, subgroup_eval_mode={subgroup_eval_mode}, "
            f"subgroup_tail_basis={subgroup_tail_basis}, metric=gap, "
            f"gls(time_limit={gls_time_limit_per_instance}, ite_max={gls_ite_max}, "
            f"moves={gls_perturbation_moves}, k={neighbor_k})"
        )
        max_eval_sec = float(n_eval_train * gls_time_limit_per_instance)
        _log(
            f"jit_kernel_enabled={bool(GLS_NUMBA_AVAILABLE)}, "
            f"theoretical_eval_upper_bound_per_candidate≈{max_eval_sec:.1f}s "
            "(before early-stop / convergence / parallel overlap)"
        )
        mode_name = "baseline"

    paras = Paras()
    eva_timeout_raw = int(exp_cfg.get("eva_timeout", 20))
    min_timeout = int(max(10, 2 * max(n_eval_train, 1)))
    eva_timeout = max(eva_timeout_raw, min_timeout)
    if eva_timeout != eva_timeout_raw:
        _log(
            f"eva_timeout auto-adjusted from {eva_timeout_raw}s to {eva_timeout}s "
            f"for n_eval_train={n_eval_train}"
        )

    selection_method_default = "dds_prob_rank" if dds_enabled else "prob_rank"
    management_method_default = "pop_dds" if dds_enabled else "pop_greedy"
    selection_method = str(eoh_cfg.get("selection", selection_method_default))
    management_method = str(eoh_cfg.get("management", management_method_default))

    paras.set_paras(
        method=str(eoh_cfg.get("method", "eoh")),
        problem="tsp_construct_local",
        selection=selection_method,
        management=management_method,
        llm_use_local=bool(llm_cfg.get("llm_use_local", False)),
        llm_local_url=llm_cfg.get("llm_local_url"),
        llm_api_endpoint=llm_cfg.get("llm_api_endpoint"),
        llm_api_key=llm_cfg.get("llm_api_key"),
        llm_model=llm_cfg.get("llm_model", "gpt-4o-mini"),
        ec_pop_size=int(effective_pop_size),
        ec_n_pop=int(eoh_cfg.get("ec_n_pop", 5)),
        ec_operators=eoh_cfg.get("ec_operators", ["e1", "e2", "m1", "m2", "m3"]),
        ec_operator_weights=eoh_cfg.get("ec_operator_weights", [1.0, 1.0, 1.0, 1.0, 1.0]),
        ec_m=int(eoh_cfg.get("ec_m", 2)),
        exp_n_proc=int(exp_cfg.get("exp_n_proc", 1)),
        exp_debug_mode=bool(exp_cfg.get("exp_debug_mode", False)),
        exp_output_path=str(run_output_dir),
        eva_timeout=int(eva_timeout),
        eva_numba_decorator=bool(exp_cfg.get("eva_numba_decorator", False)),
        dds_eps_rel=float(dds_cfg["eps_rel"]),
        dds_softmax_tau=float(dds_cfg["softmax_tau"]),
        dds_novelty_weight=float(dds_cfg["novelty_weight"]),
        trainval_enabled=bool(trainval_enabled),
        validation_problem=validation_problem,
        trainval_select_size=int(base_pop_size),
        trainval_mode=("validation_guided_reselection" if trainval_enabled else "baseline"),
    )

    _log(
        "start static train: "
        f"mode={mode_name}, n_train={n_eval_train}, n_val={n_eval_val}, "
        f"pop_size={base_pop_size}, "
        f"selection={selection_method}, management={management_method}, "
        f"dds_enabled={dds_enabled}, subgroup_eval_mode={subgroup_eval_mode}, "
        f"subgroup_tail_basis={subgroup_tail_basis}, metric=gap, "
        f"heuristic=update_edge_distance"
    )
    evolution = eoh.EVOL(paras, prob=problem)
    evolution.run()
    _log("evolution finished")

    best = _find_best_individual(
        run_output_dir,
        prefer_population_order=bool(trainval_enabled),
    )
    if best is None or not best.get("code"):
        raise RuntimeError("No valid best solver found in static training results.")

    best_solver_path = save_text(output_dir / "best_solver.py", str(best["code"]).rstrip() + "\n")
    search_trace = _collect_search_trace(run_output_dir)
    search_trace_path = save_jsonl(output_dir / "search_trace.jsonl", search_trace)

    summary_problem = TSPSelectNextNodeProblem(
        instances=problem.instance_data[: problem.n_instance],
        opt_costs=problem._reference_costs[: problem.n_instance],  # type: ignore[attr-defined]
        n_instance_eval=problem.n_instance,  # type: ignore[attr-defined]
        neighbor_k=neighbor_k,
        time_limit_per_instance=gls_time_limit_per_instance,
        ite_max=gls_ite_max,
        perturbation_moves=gls_perturbation_moves,
        group_indices=[list(range(problem.n_instance))],  # type: ignore[attr-defined]
        use_grouped_metric=False,
        group_metric_weights=group_metric_weights,
        use_gap_metric=use_gap_metric,
        subgroup_tail_basis=subgroup_tail_basis,
        gap_eps=gap_eps,
    )
    _ = summary_problem.evaluate(str(best["code"]))
    train_summary_info = summary_problem.get_last_eval_info()
    train_dataset_avg_metrics = {
        "n_eval": int(train_summary_info.get("n_eval", n_eval_train)),
        "avg_cost": _safe_float(train_summary_info.get("avg_obj")),
        "avg_gap": _safe_float(train_summary_info.get("avg_gap")),
        "avg_gap_pct": _safe_float(train_summary_info.get("avg_gap_pct")),
        "avg_tsp_time_sec": _safe_float(train_summary_info.get("avg_tsp_time_sec")),
        "metric_mode": str(train_summary_info.get("metric_mode", "cost")),
        "note": "Train-split average metric on selected best solver.",
    }
    train_dataset_avg_metrics_path = save_json(
        output_dir / "train_dataset_avg_metrics.json",
        train_dataset_avg_metrics,
    )

    val_dataset_avg_metrics = None
    if validation_problem is not None:
        _ = validation_problem.evaluate(str(best["code"]))
        val_info = validation_problem.get_last_eval_info()
        val_dataset_avg_metrics = {
            "n_eval": int(val_info.get("n_eval", n_eval_val)),
            "avg_cost": _safe_float(val_info.get("avg_obj")),
            "avg_gap": _safe_float(val_info.get("avg_gap")),
            "avg_gap_pct": _safe_float(val_info.get("avg_gap_pct")),
            "avg_tsp_time_sec": _safe_float(val_info.get("avg_tsp_time_sec")),
            "metric_mode": str(val_info.get("metric_mode", "cost")),
            "note": "Validation-split average metric on selected best solver.",
        }
        save_json(output_dir / "val_dataset_avg_metrics.json", val_dataset_avg_metrics)

    best_other = best.get("other_inf") if isinstance(best.get("other_inf"), dict) else {}
    raw_objective = _safe_float(best.get("objective"))
    best_metric = raw_objective
    if best_metric is None and isinstance(best_other, dict):
        best_metric = _safe_float(best_other.get("group_mean_eval"))
    if best_metric is None and isinstance(best_other, dict):
        best_metric = _safe_float(best_other.get("avg_gap") if use_gap_metric else best_other.get("avg_obj"))
    if best_metric is None:
        raise RuntimeError("Best solver has invalid avg_cost/objective.")
    best_cost = _safe_float(best_other.get("avg_obj")) if isinstance(best_other, dict) else None
    best_gap = (
        _safe_float(best_other.get("avg_gap"))
        if isinstance(best_other, dict) and best_other.get("avg_gap") is not None
        else None
    )
    if best_gap is None and use_gap_metric:
        best_gap = float(best_metric)

    selection_basis = ("validation_guided_reselection" if trainval_enabled else "final_population")
    best_selection_vector = None
    best_selection_rank_basis = None
    if trainval_enabled and isinstance(best_other, dict):
        best_selection_vector = [
            _safe_float(best_other.get("group_mean_eval")),
            _safe_float(best_other.get("group_std_eval")),
            _safe_float(best_other.get("group_worst_eval")),
        ]
        best_selection_rank_basis = {
            "type": "dds_multiobjective",
            "split": "val",
            "objective_vector_fields": [
                "group_mean_eval",
                "group_std_eval",
                "group_worst_eval",
            ],
            "eps_rel": float(dds_cfg.get("eps_rel", 1e-3)),
            "novelty_weight": float(dds_cfg.get("novelty_weight", 0.05)),
        }

    best_result = {
        "mode": mode_name,
        "best_metric": float(best_metric),
        "best_cost": (float(best_cost) if best_cost is not None else None),
        "best_gap": (float(best_gap) if best_gap is not None else None),
        "best_time_sec": (
            _safe_float(best_other.get("avg_tsp_time_sec"))
            if isinstance(best_other, dict)
            else None
        ),
        "evaluation_profile": (
            ("gap" if use_gap_metric else "cost")
            + "_"
            + subgroup_eval_mode
            + "_"
            + ("dds" if dds_enabled else "single_obj")
        ),
        "selection_method": selection_method,
        "management_method": management_method,
        "selection_basis": selection_basis,
        "best_selection_vector": best_selection_vector,
        "best_selection_rank_basis": best_selection_rank_basis,
        "dominance_dissimilarity": dict(dds_cfg),
        "group_evaluation_mode": subgroup_eval_mode,
        "group_multi_objective": bool(subgroup_multi_objective),
        "tail_risk_basis": str(subgroup_tail_basis),
        "use_gap_metric": bool(use_gap_metric),
        "grouped_evaluation_enabled": bool(erdr_mode),
        "validation_guided_reselection_enabled": bool(trainval_enabled),
        "grouped_evaluation_split_seed": (int(erdr_cfg.get("split_seed", 42)) if erdr_mode else None),
        "validation_population_size": int(base_pop_size),
        "train_pkl": str(train_pkl),
        "output_dir": str(output_dir),
        "run_output_dir": str(run_output_dir),
        "best_solver_path": str(best_solver_path.resolve()),
        "search_trace_path": str(search_trace_path.resolve()),
        "train_dataset_avg_metrics_path": str(train_dataset_avg_metrics_path.resolve()),
        "train_dataset_avg_metrics": train_dataset_avg_metrics,
        "val_dataset_avg_metrics": val_dataset_avg_metrics,
        "n_instances_eval_train": int(n_eval_train),
        "n_instances_eval_val": int(n_eval_val),
        "search_trace_n_rows": int(len(search_trace)),
    }

    archive_info = build_solver_archive(
        search_trace=search_trace,
        output_dir=output_dir,
        top_k=int(keep_top_k),
        use_val_dds_order=bool(trainval_enabled),
        dds_eps_rel=float(dds_cfg.get("eps_rel", 1e-3)),
        dds_novelty_weight=float(dds_cfg.get("novelty_weight", 0.05)),
    )
    best_result["solver_archive_path"] = archive_info["archive_path"]
    best_result["solver_archive_size"] = archive_info["n_solvers"]
    best_result["solver_archive_unique_candidates"] = archive_info.get("n_unique_candidates")
    best_result["solver_archive_total_candidates"] = archive_info.get("n_candidates")
    best_result["train_groups_manifest_path"] = group_artifacts["manifest_path"]
    best_result["train_groups_dir"] = group_artifacts["group_dir"]
    best_result["train_group_size"] = group_artifacts["group_size"]
    best_result["train_n_groups"] = group_artifacts["n_groups"]
    best_result["n_instances_eval_requested"] = int(n_eval_requested)
    final_eval_path = output_dir / "final_eval.json"
    best_result["final_eval_path"] = str(final_eval_path.resolve())
    best_result["artifact_policy"] = {
        "output_mode": ("minimal" if minimal_output else "full"),
        "keep_top_k_solvers": int(keep_top_k),
        "kept_outputs": [
            "best_solver.py",
            "solver_archive/",
            "solver_archive.json",
            "final_eval.json",
        ],
    }
    if minimal_output:
        for key in (
            "search_trace_path",
            "train_dataset_avg_metrics_path",
            "run_output_dir",
            "train_groups_manifest_path",
            "train_groups_dir",
        ):
            best_result.pop(key, None)
    save_json(output_dir / "best_result.json", best_result)
    save_json(final_eval_path, best_result)
    if minimal_output:
        _cleanup_minimal_outputs(output_dir=output_dir, run_output_dir=run_output_dir)
    return best_result
