from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


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


def _load_dataset_shape_info(dataset_pkl: Path) -> tuple[int, int, int, float]:
    with dataset_pkl.open("rb") as f:
        raw = pickle.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"dynamic dataset is not dict format: {dataset_pkl}")
    coords = raw.get("coordinate", None)
    if coords is None:
        raise ValueError(f"dynamic dataset missing coordinate: {dataset_pkl}")
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 3 or arr.shape[2] != 2:
        raise ValueError(f"invalid coordinate shape in dynamic dataset: {arr.shape}")
    n_total = int(arr.shape[0])
    n_nodes = np.asarray([int(arr[i].shape[0]) for i in range(n_total)], dtype=int)
    n_min = int(np.min(n_nodes)) if n_total > 0 else 0
    n_max = int(np.max(n_nodes)) if n_total > 0 else 0
    n_mean = float(np.mean(n_nodes)) if n_total > 0 else 0.0
    return n_total, n_min, n_max, n_mean


def _evaluate_one(
    *,
    repo_root: Path,
    solver_py: Path,
    dataset_pkl: Path,
    metric_basis: str,
    subgroup_mode: str,
    subgroup_tail_basis: str,
    n_instance_eval: int,
    gls_time_limit_per_instance: float,
    gls_iter_limit: int,
    gls_perturbation_moves: int,
    gap_eps: float,
    jit_enable_kernels: bool,
    jit_numba_cache: bool,
    jit_warmup: bool,
    jit_decorate_guide: bool,
) -> Dict[str, Any]:
    project_dir = Path(__file__).resolve().parents[1]
    eval_script = (project_dir / "problems" / "tsp_gls" / "eval_adapter.py").resolve()
    if not eval_script.exists():
        raise FileNotFoundError(f"reevo eval_adapter script not found: {eval_script}")

    tmp_json = dataset_pkl.with_suffix(dataset_pkl.suffix + ".reevo_eval_tmp.json")
    if tmp_json.exists():
        tmp_json.unlink(missing_ok=True)

    cmd = [
        str(Path(sys.executable).resolve()),
        str(eval_script),
        "--dataset-path",
        str(dataset_pkl),
        "--code-path",
        str(solver_py),
        "--output-json",
        str(tmp_json),
        "--metric-basis",
        str(metric_basis),
        "--subgroup-mode",
        str(subgroup_mode),
        "--subgroup-tail-basis",
        str(subgroup_tail_basis),
        "--n-instance-eval",
        str(int(n_instance_eval)),
        "--perturbation-moves",
        str(int(max(int(gls_perturbation_moves), 1))),
        "--iter-limit",
        str(int(max(int(gls_iter_limit), 1))),
        "--time-limit-per-instance",
        str(float(max(float(gls_time_limit_per_instance), 0.1))),
        "--jit-warmup",
        ("1" if bool(jit_warmup) else "0"),
        "--jit-decorate-guide",
        ("1" if bool(jit_decorate_guide) else "0"),
        "--jit-sanitize-guide",
        "1",
        "--jit-sanitize-strict",
        "0",
        "--gap-eps",
        str(float(max(float(gap_eps), 1.0e-12))),
    ]
    env = os.environ.copy()
    env["REEVO_GLS_ENABLE_JIT"] = "1" if bool(jit_enable_kernels) else "0"
    env["REEVO_GLS_NUMBA_CACHE"] = "1" if bool(jit_numba_cache) else "0"

    run = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if run.returncode != 0:
        stderr_tail = (run.stderr or "")[-3000:]
        stdout_tail = (run.stdout or "")[-3000:]
        raise RuntimeError(
            "reevo eval_adapter failed: "
            f"dataset={dataset_pkl.name}, returncode={run.returncode}, "
            f"stdout_tail={stdout_tail!r}, stderr_tail={stderr_tail!r}"
        )
    if not tmp_json.exists():
        raise RuntimeError(f"reevo eval_adapter output json missing: {tmp_json}")

    try:
        payload = json.loads(tmp_json.read_text(encoding="utf-8"))
    finally:
        tmp_json.unlink(missing_ok=True)
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid reevo eval json payload: {tmp_json}")

    n_total, n_nodes_min, n_nodes_max, n_nodes_mean = _load_dataset_shape_info(dataset_pkl)
    n_eval_used = int(payload.get("n_eval", 0))
    if int(n_instance_eval) > 0:
        n_eval_used = min(max(int(n_instance_eval), 1), int(n_total))
    elif n_eval_used <= 0:
        n_eval_used = int(n_total)

    return {
        "selected_metric_name": str(payload.get("selected_metric_name")),
        "selected_metric": _safe_float(payload.get("selected_metric")),
        "mean_gap": _safe_float(payload.get("mean_gap")),
        "mean_gap_pct": _safe_float(payload.get("mean_gap_pct")),
        "mean_time_sec": _safe_float(payload.get("mean_time_sec")),
        "n_invalid": int(payload.get("n_invalid", 0)),
        "n_eval": int(payload.get("n_eval", n_eval_used)),
        "n_instance_eval": int(n_eval_used),
        "n_instances_used_in_eval": int(n_eval_used),
        "n_instance_total_in_dataset": int(n_total),
        "n_instances_in_dynamic_file": int(n_total),
        "n_dynamic_instances": int(n_total),
        "n_nodes_per_instance_min": int(n_nodes_min),
        "n_nodes_per_instance_max": int(n_nodes_max),
        "n_nodes_per_instance_mean": float(n_nodes_mean),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one ReEvo solver on dynamic-perturbed datasets and summarize gap/time."
    )
    parser.add_argument("--repo-root", type=str, default=".")
    parser.add_argument(
        "--dynamic-dir",
        type=str,
        required=True,
        help="Dynamic dataset directory generated by generate_dynamic_perturbed_reevo.py",
    )
    parser.add_argument("--solver", type=str, required=True, help="Solver .py path to evaluate.")
    parser.add_argument(
        "--dataset-solver-tag",
        type=str,
        required=True,
        help="Tag in dynamic filenames, e.g. baseline or erdr.",
    )
    parser.add_argument("--dataset-prefix", type=str, default="", help="Optional fixed dataset prefix.")
    parser.add_argument("--method-name", type=str, default="method", help="Method label in output rows.")
    parser.add_argument("--ratio-tag", type=str, default="r50", help="e.g. r50 for ratio=0.5")
    parser.add_argument("--n-instance-generated", type=int, default=0)
    parser.add_argument("--perturb-types", type=str, default="coord_jitter,node_add,node_remove")
    parser.add_argument("--metric-basis", type=str, choices=["gap", "cost"], default="gap")
    parser.add_argument(
        "--subgroup-mode",
        type=str,
        choices=["mean_only", "mean_std_worst"],
        default="mean_only",
    )
    parser.add_argument("--subgroup-tail-basis", type=str, choices=["gap", "time"], default="time")
    parser.add_argument(
        "--n-instance-eval",
        type=int,
        default=0,
        help="<=0 means evaluate all instances in each dynamic dataset file.",
    )
    parser.add_argument("--gls-time-limit-per-instance", type=float, default=20.0)
    parser.add_argument("--gls-iter-limit", type=int, default=500)
    parser.add_argument("--gls-perturbation-moves", type=int, default=2)
    parser.add_argument("--gap-eps", type=float, default=1.0e-12)
    parser.add_argument("--jit-enable-kernels", type=int, default=1)
    parser.add_argument("--jit-numba-cache", type=int, default=1)
    parser.add_argument("--jit-warmup", type=int, default=1)
    parser.add_argument("--jit-decorate-guide", type=int, default=0)
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
        if str(args.dataset_prefix).strip():
            pattern = f"{str(args.dataset_prefix).strip()}_{dataset_solver_tag}_{args.ratio_tag}_{perturb_type}.pkl"
        else:
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
            metric_basis=str(args.metric_basis),
            subgroup_mode=str(args.subgroup_mode),
            subgroup_tail_basis=str(args.subgroup_tail_basis),
            n_instance_eval=int(args.n_instance_eval),
            gls_time_limit_per_instance=float(args.gls_time_limit_per_instance),
            gls_iter_limit=int(args.gls_iter_limit),
            gls_perturbation_moves=int(args.gls_perturbation_moves),
            gap_eps=float(args.gap_eps),
            jit_enable_kernels=bool(int(args.jit_enable_kernels)),
            jit_numba_cache=bool(int(args.jit_numba_cache)),
            jit_warmup=bool(int(args.jit_warmup)),
            jit_decorate_guide=bool(int(args.jit_decorate_guide)),
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
            "dataset_prefix": str(args.dataset_prefix).strip() or None,
            "method_name": method_name,
            "ratio_tag": str(args.ratio_tag),
            "perturb_types": perturb_types,
            "n_instance_generated": int(max(int(args.n_instance_generated), 0)),
            "n_instance_eval": int(args.n_instance_eval),
            "metric_basis": str(args.metric_basis),
            "subgroup_mode": str(args.subgroup_mode),
            "subgroup_tail_basis": str(args.subgroup_tail_basis),
            "gls_time_limit_per_instance": float(args.gls_time_limit_per_instance),
            "gls_iter_limit": int(args.gls_iter_limit),
            "gls_perturbation_moves": int(args.gls_perturbation_moves),
            "gap_eps": float(args.gap_eps),
            "jit_enable_kernels": bool(int(args.jit_enable_kernels)),
            "jit_numba_cache": bool(int(args.jit_numba_cache)),
            "jit_warmup": bool(int(args.jit_warmup)),
            "jit_decorate_guide": bool(int(args.jit_decorate_guide)),
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
