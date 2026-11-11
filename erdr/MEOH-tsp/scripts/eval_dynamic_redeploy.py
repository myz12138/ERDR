from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _project_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _evaluate_dataset(dataset: Path, solver: Path, args: argparse.Namespace) -> dict[str, Any]:
    project_dir = _project_dir()
    sys.path.insert(0, str(project_dir))
    from llm4ad.task.optimization.tsp_gls_2O import TSPGLSMultiObjectiveEvaluation

    code = solver.read_text(encoding="utf-8")
    namespace: dict[str, Any] = {"np": np}
    exec(code, namespace, namespace)
    evaluator = TSPGLSMultiObjectiveEvaluation(
        dataset_path=dataset,
        n_instance_eval=0,
        mode="baseline",
        perturbation_moves=args.gls_perturbation_moves,
        iter_limit=args.gls_ite_max,
        time_limit_per_instance=args.gls_time_limit_per_instance,
        timeout_seconds=max(10.0, args.gls_time_limit_per_instance * 2.0),
    )
    score = evaluator.evaluate_program(code, namespace["update_edge_distance"])
    metrics = dict(evaluator.last_metrics)
    return {
        "dataset": str(dataset),
        "dataset_name": dataset.name,
        "ratio_tag": args.ratio_tag,
        "perturb_type": args.perturb_type,
        "n_instance_generated": args.n_instance_generated,
        "n_dynamic_instances": int(metrics.get("n_instance_eval", 0)),
        "n_instances_used_in_eval": int(metrics.get("n_instance_eval", 0)),
        "mean_gap": float(metrics.get("mean_gap", float("inf"))),
        "mean_time_sec": float(metrics.get("mean_runtime_sec", float("inf"))),
        "score": np.asarray(score, dtype=float).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--dynamic-dir", required=True)
    parser.add_argument("--solver", required=True)
    parser.add_argument("--dataset-solver-tag", required=True)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--ratio-tag", required=True)
    parser.add_argument("--perturb-types", required=True)
    parser.add_argument("--n-instance-generated", type=int, required=True)
    parser.add_argument("--gls-time-limit-per-instance", type=float, default=20.0)
    parser.add_argument("--gls-ite-max", type=int, default=500)
    parser.add_argument("--gls-perturbation-moves", type=int, default=2)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    dynamic_dir = (root / args.dynamic_dir).resolve()
    solver = (root / args.solver).resolve()
    if not solver.is_file():
        raise FileNotFoundError(f"solver not found: {solver}")
    rows: list[dict[str, Any]] = []
    for perturb_type in [item.strip() for item in args.perturb_types.split(",") if item.strip()]:
        pattern = f"TSP*_dyn_{args.dataset_solver_tag}_{args.ratio_tag}_{perturb_type}.pkl"
        candidates = sorted(dynamic_dir.glob(pattern))
        if len(candidates) != 1:
            raise FileNotFoundError(f"expected one dynamic dataset for {pattern}, found {len(candidates)}")
        row_args = argparse.Namespace(**vars(args), perturb_type=perturb_type)
        rows.append(_evaluate_dataset(candidates[0], solver, row_args))
    output = (root / args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "method": args.method_name,
        "solver_path": str(solver),
        "dataset_solver_tag": args.dataset_solver_tag,
        "ratio_tag": args.ratio_tag,
        "rows": rows,
        "aggregate": {
            "mean_gap": float(np.mean([row["mean_gap"] for row in rows])),
            "mean_time_sec": float(np.mean([row["mean_time_sec"] for row in rows])),
        },
    }
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
