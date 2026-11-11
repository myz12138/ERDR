from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        v = float(x)
        if v != v:
            return None
        return v
    except Exception:
        return None


def _mean(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _evaluate_one(
    *,
    repo_root: Path,
    solver_py: Path,
    dataset_pkl: Path,
    gls_neighbor_k: int,
    gls_time_limit_per_instance: float,
    gls_ite_max: int,
    gls_perturbation_moves: int,
    gap_eps: float,
) -> Dict[str, Any]:
    import sys

    eoh_plan_dir = Path(__file__).resolve().parents[1]
    if str(eoh_plan_dir / "src") not in sys.path:
        sys.path.insert(0, str(eoh_plan_dir / "src"))

    from data_utils import load_tsp_dataset
    from snc_tsp_problem import TSPSelectNextNodeProblem

    code = solver_py.read_text(encoding="utf-8")
    instances, opt_costs = load_tsp_dataset(dataset_pkl)
    n_total = int(len(instances))
    # Always evaluate all instances contained in the dataset file.
    n_eval = int(n_total)

    sub_instances = list(instances[:n_eval])
    sub_opts = [float(x) for x in opt_costs[:n_eval]] if isinstance(opt_costs, list) else None

    evaluator = TSPSelectNextNodeProblem(
        instances=sub_instances,
        opt_costs=sub_opts,
        n_instance_eval=n_eval,
        neighbor_k=int(max(int(gls_neighbor_k), 1)),
        time_limit_per_instance=float(max(float(gls_time_limit_per_instance), 0.1)),
        ite_max=int(max(int(gls_ite_max), 1)),
        perturbation_moves=int(max(int(gls_perturbation_moves), 1)),
        group_indices=[list(range(n_eval))],
        use_grouped_metric=False,
        group_metric_weights={"mean": 1.0, "std": 0.0, "worst": 0.0},
        use_gap_metric=True,
        gap_eps=float(max(float(gap_eps), 1.0e-12)),
    )
    n_nodes_list = [int(c.shape[0]) for c, _dm in evaluator.instance_data]
    n_nodes_min = int(min(n_nodes_list)) if n_nodes_list else 0
    n_nodes_max = int(max(n_nodes_list)) if n_nodes_list else 0
    n_nodes_mean = (float(sum(n_nodes_list)) / float(len(n_nodes_list))) if n_nodes_list else 0.0
    metric = evaluator.evaluate(code)
    info = evaluator.get_last_eval_info()

    return {
        "mean_gap": _safe_float(info.get("avg_gap")),
        "mean_time_sec": _safe_float(info.get("avg_tsp_time_sec")),
        # actual evaluated sample count for this dataset file
        "n_instance_eval": int(n_eval),
        "n_instances_used_in_eval": int(n_eval),
        # total sample count physically stored in this dataset file
        "n_instance_total_in_dataset": int(n_total),
        "n_instances_in_dynamic_file": int(n_total),
        # clearer alias for downstream tables
        "n_dynamic_instances": int(n_total),
        "n_nodes_per_instance_min": int(n_nodes_min),
        "n_nodes_per_instance_max": int(n_nodes_max),
        "n_nodes_per_instance_mean": float(n_nodes_mean),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one solver on dynamic-perturbed TSP datasets and report gap/time only."
        )
    )
    parser.add_argument("--repo-root", type=str, default=".")
    parser.add_argument(
        "--dynamic-dir",
        type=str,
        required=True,
        help="Directory containing generated dynamic TSP datasets.",
    )
    parser.add_argument("--solver", type=str, required=True, help="Solver .py path to evaluate.")
    parser.add_argument(
        "--dataset-solver-tag",
        type=str,
        required=True,
        help="Dataset tag used in dynamic dataset file names, e.g. baseline or erdr.",
    )
    parser.add_argument("--method-name", type=str, default="method", help="Method label in output rows.")
    parser.add_argument("--ratio-tag", type=str, default="r50", help="e.g. r50 for ratio=0.5")
    parser.add_argument(
        "--n-instance-generated",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--perturb-types",
        type=str,
        default="coord_jitter,node_add,node_remove",
    )
    # Backward-compat only: ignored. Evaluation always uses all instances in dataset.
    parser.add_argument("--n-instance-eval", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--gls-neighbor-k", type=int, default=15)
    parser.add_argument("--gls-time-limit-per-instance", type=float, default=20.0)
    parser.add_argument("--gls-ite-max", type=int, default=500)
    parser.add_argument("--gls-perturbation-moves", type=int, default=2)
    parser.add_argument("--gap-eps", type=float, default=1.0e-12)
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional output summary path (default: <dynamic-dir>/redeploy_eval_summary.json)",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    dynamic_dir = (repo_root / args.dynamic_dir).resolve()
    solver_path = (repo_root / args.solver).resolve()

    if not dynamic_dir.exists():
        raise FileNotFoundError(f"dynamic dir not found: {dynamic_dir}")
    if not solver_path.exists():
        raise FileNotFoundError(f"solver not found: {solver_path}")

    perturb_types = [x.strip() for x in str(args.perturb_types).split(",") if x.strip()]
    if not perturb_types:
        raise ValueError("empty perturb types")
    dataset_solver_tag = str(args.dataset_solver_tag).strip()
    method_name = str(args.method_name).strip() or dataset_solver_tag
    if not dataset_solver_tag:
        raise ValueError("empty dataset-solver-tag")

    rows: List[Dict[str, Any]] = []
    for perturb_type in perturb_types:
        pattern = f"TSP*_dyn_{dataset_solver_tag}_{args.ratio_tag}_{perturb_type}.pkl"
        candidates = sorted(dynamic_dir.glob(pattern))
        if not candidates:
            raise FileNotFoundError(f"dataset missing by pattern: {dynamic_dir / pattern}")
        if len(candidates) > 1:
            names = ", ".join(x.name for x in candidates)
            raise RuntimeError(
                f"ambiguous datasets for perturb_type={perturb_type}, "
                f"ratio_tag={args.ratio_tag}, dataset_solver_tag={dataset_solver_tag}: {names}"
            )
        dataset_pkl = candidates[0]
        dataset_name = dataset_pkl.name

        eval_out = _evaluate_one(
            repo_root=repo_root,
            solver_py=solver_path,
            dataset_pkl=dataset_pkl,
            gls_neighbor_k=int(args.gls_neighbor_k),
            gls_time_limit_per_instance=float(args.gls_time_limit_per_instance),
            gls_ite_max=int(args.gls_ite_max),
            gls_perturbation_moves=int(args.gls_perturbation_moves),
            gap_eps=float(args.gap_eps),
        )
        row = {
            "dataset": str(dataset_pkl),
            "dataset_name": dataset_name,
            "ratio_tag": str(args.ratio_tag),
            "perturb_type": perturb_type,
            "n_instance_generated": int(max(int(args.n_instance_generated), 0)),
            **eval_out,
        }
        rows.append(row)
        print(
            f"[{method_name}][{perturb_type}] "
            f"gap={row.get('mean_gap')} time={row.get('mean_time_sec')}",
            flush=True,
        )

    gap_values: List[float] = []
    time_values: List[float] = []
    for row in rows:
        g = _safe_float(row.get("mean_gap"))
        t = _safe_float(row.get("mean_time_sec"))
        if g is not None:
            gap_values.append(float(g))
        if t is not None:
            time_values.append(float(t))

    summary = {
        "method": method_name,
        "solver_path": str(solver_path),
        "dataset_solver_tag": dataset_solver_tag,
        "dynamic_dir": str(dynamic_dir),
        "ratio_tag": str(args.ratio_tag),
        "params": {
            "repo_root": str(repo_root),
            "dynamic_dir": str(dynamic_dir),
            "solver": str(solver_path),
            "dataset_solver_tag": dataset_solver_tag,
            "method_name": method_name,
            "ratio_tag": str(args.ratio_tag),
            "perturb_types": perturb_types,
            "n_instance_generated": int(max(int(args.n_instance_generated), 0)),
            "n_instance_eval_mode": "all_in_dataset",
            "gls_neighbor_k": int(args.gls_neighbor_k),
            "gls_time_limit_per_instance": float(args.gls_time_limit_per_instance),
            "gls_ite_max": int(args.gls_ite_max),
            "gls_perturbation_moves": int(args.gls_perturbation_moves),
            "gap_eps": float(args.gap_eps),
        },
        "rows": rows,
        "aggregate": {
            "mean_gap": _mean(gap_values) if gap_values else None,
            "mean_time_sec": _mean(time_values) if time_values else None,
        },
    }

    if str(args.output_json).strip():
        out_json = (repo_root / args.output_json).resolve()
    else:
        out_json = dynamic_dir / "redeploy_eval_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[aggregate]", json.dumps(summary["aggregate"], ensure_ascii=False, indent=2), flush=True)
    print(f"[saved] {out_json}", flush=True)


if __name__ == "__main__":
    main()
