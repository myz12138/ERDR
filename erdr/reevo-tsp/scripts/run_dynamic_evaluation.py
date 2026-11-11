from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml


def _run(cmd: List[str], cwd: Path) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _cleanup_dynamic_dir(dynamic_dir: Path) -> None:
    # Keep generated dynamic .pkl datasets only.
    for p in dynamic_dir.glob("*.json"):
        p.unlink(missing_ok=True)
    for p in dynamic_dir.glob("*.jsonl"):
        p.unlink(missing_ok=True)
    for p in dynamic_dir.glob("*_meta.json"):
        p.unlink(missing_ok=True)
    concorde_dir = dynamic_dir / "concorde"
    if concorde_dir.exists():
        for f in concorde_dir.rglob("*"):
            if f.is_file():
                f.unlink(missing_ok=True)
        for d in sorted((x for x in concorde_dir.rglob("*") if x.is_dir()), reverse=True):
            d.rmdir()
        concorde_dir.rmdir()


def _ratio_tag(ratio: float) -> str:
    return f"r{int(round(float(ratio) * 100)):02d}"


def _parse_ratio_collection(text: str) -> List[float]:
    vals: List[float] = []
    for part in str(text).split(","):
        p = part.strip()
        if not p:
            continue
        v = float(p)
        if v <= 0.0 or v >= 1.0:
            raise ValueError(f"ratio must be in (0,1): {v}")
        vals.append(v)
    if len(vals) < 2:
        raise ValueError("each ratio collection must contain at least 2 ratios")
    return vals


def _parse_ratio_sets(text: str) -> List[List[float]]:
    groups: List[List[float]] = []
    for chunk in str(text).split(";"):
        c = chunk.strip()
        if not c:
            continue
        groups.append(_parse_ratio_collection(c))
    if not groups:
        raise ValueError("ratio_sets is empty")
    return groups


def _ratio_set_key(ratios: List[float]) -> str:
    tags = [_ratio_tag(x) for x in ratios]
    return "_".join(tags)


def _write_rows_csv(rows: List[Dict[str, Any]], csv_path: Path) -> None:
    fieldnames = [
        "dataset_name",
        "ratio_tag",
        "perturb_type",
        "n_instance_generated",
        "n_dynamic_instances",
        "n_instances_used_in_eval",
        "n_instances_in_dynamic_file",
        "n_nodes_per_instance_min",
        "n_nodes_per_instance_max",
        "n_nodes_per_instance_mean",
        "mean_gap",
        "mean_time_sec",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _safe_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        x = float(v)
        if x != x:
            return None
        return x
    except Exception:
        return None


def _infer_dataset_key(base_pkl: Path) -> str:
    stem = base_pkl.stem.lower()
    for suffix in ("_concorde", "_raw"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    m = re.search(r"tsp(\d+)_u?(\d+)$", stem)
    if m is None:
        m = re.search(r"tsp(\d+)_u?(\d+)", stem)
    if m is not None:
        return f"tsp{int(m.group(1))}_{int(m.group(2))}"
    safe = "".join(ch if ch.isalnum() else "_" for ch in stem).strip("_")
    return safe or "dataset"


def _load_config_defaults(
    config_path: Path | None,
    default_repo_root: Path,
    project_rel: str,
) -> Dict[str, Any]:
    if config_path is None:
        return {}
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    obj = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError(f"config root must be mapping: {config_path}")

    paths = obj.get("paths", {})
    transfer = obj.get("dynamic_evaluation", {})
    evaluation = obj.get("evaluation", {})
    gls = obj.get("gls", {})
    generation = obj.get("generation", {})
    jit = obj.get("jit", {})
    if not isinstance(paths, dict):
        paths = {}
    if not isinstance(transfer, dict):
        transfer = {}
    if not isinstance(evaluation, dict):
        evaluation = {}
    if not isinstance(gls, dict):
        gls = {}
    if not isinstance(generation, dict):
        generation = {}
    if not isinstance(jit, dict):
        jit = {}

    defaults: Dict[str, Any] = {
        "repo_root": paths.get("repo_root", str(default_repo_root)),
        "baseline_solver": paths.get(
            "baseline_solver",
            f"{project_rel}/outputs/reevo_baseline/best_solver.py",
        ),
        "erdr_solver": paths.get(
            "erdr_solver",
            f"{project_rel}/outputs/reevo_erdr/best_solver.py",
        ),
        "base_pkl": paths.get(
            "base_pkl",
            "data_generate/TSP200_u50/TSP200_u50_concorde.pkl",
        ),
        "run_root": paths.get("run_root", f"{project_rel}/data/transfer_eval_runs"),
        "run_name": paths.get("run_name", ""),
        "dataset_prefix": paths.get("dataset_prefix", ""),
        "method": transfer.get("method", "baseline"),
        "ratio_sets": transfer.get("ratio_sets", "0.2,0.5,0.8"),
        "change_types": transfer.get("change_types", "location_change,request_arrival,request_cancellation"),
        "n_instance": transfer.get("n_instance", 50),
        "n_instance_eval": transfer.get("n_instance_eval", 0),
        "seed": transfer.get("seed", 20260524),
        "metric_basis": evaluation.get("metric_basis", "gap"),
        "subgroup_mode": evaluation.get("subgroup_mode", "mean_only"),
        "subgroup_tail_basis": evaluation.get("subgroup_tail_basis", "time"),
        "gls_time_limit_per_instance": gls.get("time_limit_per_instance", 20.0),
        "gls_iter_limit": gls.get("iter_limit", 500),
        "gls_perturbation_moves": gls.get("perturbation_moves", 2),
        "gap_eps": gls.get("gap_eps", 1.0e-12),
        "jitter_sigma": generation.get("jitter_sigma", 0.02),
        "node_frac": generation.get("node_frac", 0.2),
        "concorde_scale": generation.get("concorde_scale", 10000),
        "jit_enable_kernels": jit.get("enable_kernels", True),
        "jit_numba_cache": jit.get("numba_cache", True),
        "jit_warmup": jit.get("warmup", True),
        "jit_decorate_guide": jit.get("decorate_guide", False),
    }
    return defaults


def _to_abs(base: Path, raw_path: str) -> Path:
    p = Path(str(raw_path))
    return p.resolve() if p.is_absolute() else (base / p).resolve()


def _as_rel_if_possible(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except Exception:
        return str(path)


def main() -> None:
    script_path = Path(__file__).resolve()
    reevo_dir = script_path.parents[1]
    workspace_root = reevo_dir.parents[1]
    project_rel = reevo_dir.relative_to(workspace_root).as_posix()

    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument(
        "--config",
        type=str,
        default=f"{project_rel}/configs/dynamic_evaluation.yaml",
    )
    bootstrap_args, _ = bootstrap.parse_known_args()
    config_raw = str(bootstrap_args.config).strip()
    if config_raw:
        config_path = _to_abs(workspace_root, config_raw)
    else:
        config_path = None
    cfg_defaults = _load_config_defaults(config_path, workspace_root, project_rel)

    parser = argparse.ArgumentParser(
        description=(
            "EOH-style single-solver transfer evaluation for ReEvo: "
            "always regenerate dynamic datasets, then evaluate one method only."
        )
    )
    parser.add_argument("--config", type=str, default=config_raw)
    parser.add_argument("--repo-root", type=str, default=str(cfg_defaults.get("repo_root", workspace_root)))
    parser.add_argument(
        "--baseline-solver",
        type=str,
        default=str(
            cfg_defaults.get(
                "baseline_solver",
                f"{project_rel}/outputs/reevo_baseline/best_solver.py",
            )
        ),
    )
    parser.add_argument(
        "--erdr-solver",
        type=str,
        default=str(
            cfg_defaults.get(
                "erdr_solver",
                f"{project_rel}/outputs/reevo_erdr/best_solver.py",
            )
        ),
    )
    parser.add_argument(
        "--run-root",
        type=str,
        default=str(cfg_defaults.get("run_root", f"{project_rel}/data/transfer_eval_runs")),
    )
    parser.add_argument(
        "--base-pkl",
        type=str,
        default=str(cfg_defaults.get("base_pkl", "data_generate/TSP200_u50/TSP200_u50_concorde.pkl")),
    )
    parser.add_argument(
        "--dataset-prefix",
        type=str,
        default=str(cfg_defaults.get("dataset_prefix", "")),
        help="Optional dataset prefix for generated dynamic files; empty means auto.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["baseline", "erdr"],
        default=str(cfg_defaults.get("method", "baseline")),
        help="Run transfer eval for only one solver method per run.",
    )
    parser.add_argument("--run-name", type=str, default=str(cfg_defaults.get("run_name", "")))
    parser.add_argument(
        "--ratio-sets",
        type=str,
        default=str(cfg_defaults.get("ratio_sets", "0.2,0.5,0.8")),
        help="Semicolon-separated ratio collections. Example: 0.2,0.5;0.5,0.8",
    )
    # Backward-compat args: accepted but ignored by new workflow.
    parser.add_argument("--ratios", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--ratio-tag", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--dynamic-dir", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--output-json", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--force-regenerate-dynamic", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--change-types",
        type=str,
        default=str(cfg_defaults.get("change_types", "location_change,request_arrival,request_cancellation")),
    )
    parser.add_argument("--n-instance", type=int, default=int(cfg_defaults.get("n_instance", 50)))
    parser.add_argument(
        "--n-instance-eval",
        type=int,
        default=int(cfg_defaults.get("n_instance_eval", 0)),
        help="<=0 means evaluate all instances in each generated dynamic dataset file.",
    )
    parser.add_argument(
        "--metric-basis",
        type=str,
        choices=["gap", "cost"],
        default=str(cfg_defaults.get("metric_basis", "gap")),
    )
    parser.add_argument(
        "--subgroup-mode",
        type=str,
        choices=["mean_only", "mean_variability_tail"],
        default=str(cfg_defaults.get("subgroup_mode", "mean_only")),
    )
    parser.add_argument(
        "--subgroup-tail-basis",
        type=str,
        choices=["gap", "time"],
        default=str(cfg_defaults.get("subgroup_tail_basis", "time")),
    )
    parser.add_argument(
        "--gls-time-limit-per-instance",
        type=float,
        default=float(cfg_defaults.get("gls_time_limit_per_instance", 20.0)),
    )
    parser.add_argument("--gls-iter-limit", type=int, default=int(cfg_defaults.get("gls_iter_limit", 500)))
    parser.add_argument(
        "--gls-perturbation-moves",
        type=int,
        default=int(cfg_defaults.get("gls_perturbation_moves", 2)),
    )
    parser.add_argument("--gap-eps", type=float, default=float(cfg_defaults.get("gap_eps", 1.0e-12)))
    parser.add_argument("--seed", type=int, default=int(cfg_defaults.get("seed", 20260524)))
    parser.add_argument("--jitter-sigma", type=float, default=float(cfg_defaults.get("jitter_sigma", 0.02)))
    parser.add_argument("--node-frac", type=float, default=float(cfg_defaults.get("node_frac", 0.2)))
    parser.add_argument("--concorde-scale", type=int, default=int(cfg_defaults.get("concorde_scale", 10000)))
    parser.add_argument("--jit-enable-kernels", type=int, default=int(bool(cfg_defaults.get("jit_enable_kernels", True))))
    parser.add_argument("--jit-numba-cache", type=int, default=int(bool(cfg_defaults.get("jit_numba_cache", True))))
    parser.add_argument("--jit-warmup", type=int, default=int(bool(cfg_defaults.get("jit_warmup", True))))
    parser.add_argument("--jit-decorate-guide", type=int, default=int(bool(cfg_defaults.get("jit_decorate_guide", False))))
    args = parser.parse_args()

    root = _to_abs(workspace_root, str(args.repo_root))
    config_used_raw = str(args.config).strip()
    config_used = _to_abs(root, config_used_raw) if config_used_raw else None

    baseline_solver = _to_abs(root, str(args.baseline_solver))
    erdr_solver = _to_abs(root, str(args.erdr_solver))
    base_pkl = _to_abs(root, str(args.base_pkl))

    if not base_pkl.exists():
        raise FileNotFoundError(f"base pkl not found: {base_pkl}")

    change_types = [x.strip() for x in str(args.change_types).split(",") if x.strip()]
    generator_change_types = {"location_change": "coord_jitter", "request_arrival": "node_add", "request_cancellation": "node_remove"}
    if not change_types or any(change not in generator_change_types for change in change_types):
        raise ValueError("change_types must use location_change, request_arrival, request_cancellation")
    perturb_types = [generator_change_types[change] for change in change_types]

    method = str(args.method).strip().lower()
    if method == "baseline":
        selected_solver = baseline_solver
        dataset_solver_tag = "baseline"
    else:
        selected_solver = erdr_solver
        dataset_solver_tag = "erdr"
    if not selected_solver.exists():
        raise FileNotFoundError(f"{method} solver not found: {selected_solver}")

    dataset_key = _infer_dataset_key(base_pkl)
    ratio_sets_text = str(args.ratio_sets).strip()
    if not ratio_sets_text and str(args.ratios).strip():
        ratio_sets_text = str(args.ratios).strip()
    ratio_sets = _parse_ratio_sets(ratio_sets_text)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name_raw = str(args.run_name).strip()
    run_name_base = run_name_raw if run_name_raw else f"dynamic_evaluation_{method}_{dataset_key}_{stamp}"
    run_root = _to_abs(root, str(args.run_root))
    run_root.mkdir(parents=True, exist_ok=True)

    py_exe = Path(sys.executable).resolve()
    gen_script = root / "data_generate" / "generate_dynamic_perturbed_reevo.py"
    eval_script = reevo_dir / "scripts" / "eval_dynamic_redeploy.py"
    created_set_dirs: List[Path] = []

    if not gen_script.exists():
        raise FileNotFoundError(f"generator script not found: {gen_script}")
    if not eval_script.exists():
        raise FileNotFoundError(f"eval script not found: {eval_script}")

    for ratios in ratio_sets:
        set_key = _ratio_set_key(ratios)
        set_run_name = f"{run_name_base}_{set_key}"
        set_dir = run_root / set_run_name
        if set_dir.exists():
            raise FileExistsError(f"run dir already exists: {set_dir}")
        dynamic_dir = set_dir / "dynamic_datasets"
        eval_dir = set_dir / "eval"
        set_dir.mkdir(parents=True, exist_ok=False)
        dynamic_dir.mkdir(parents=True, exist_ok=False)
        eval_dir.mkdir(parents=True, exist_ok=False)
        created_set_dirs.append(set_dir)

        ratios_text = ",".join(f"{x:g}" for x in ratios)
        dynamic_dir_arg = _as_rel_if_possible(dynamic_dir, root)
        gen_cmd = [
            str(py_exe),
            str(gen_script),
            "--repo-root",
            str(root),
            "--base-pkl",
            _as_rel_if_possible(base_pkl, root),
            "--solver-mode",
            "single",
            "--single-solver-id",
            dataset_solver_tag,
            "--single-solver-path",
            _as_rel_if_possible(selected_solver, root),
            "--out-dir",
            dynamic_dir_arg,
            "--n-instance",
            str(int(args.n_instance)),
            "--ratios",
            ratios_text,
            "--perturb-types",
            ",".join(perturb_types),
            "--seed",
            str(int(args.seed)),
            "--gls-time-limit-sec",
            str(float(args.gls_time_limit_per_instance)),
            "--gls-iter-limit",
            str(int(args.gls_iter_limit)),
            "--gls-perturbation-moves",
            str(int(args.gls_perturbation_moves)),
            "--jitter-sigma",
            str(float(args.jitter_sigma)),
            "--node-frac",
            str(float(args.node_frac)),
            "--concorde-scale",
            str(int(args.concorde_scale)),
        ]
        if str(args.dataset_prefix).strip():
            gen_cmd.extend(["--dataset-prefix", str(args.dataset_prefix).strip()])
        _run(gen_cmd, cwd=root)

        combined_rows: List[Dict[str, Any]] = []
        ratio_aggregates: List[Dict[str, Any]] = []
        for ratio in ratios:
            tag = _ratio_tag(ratio)
            ratio_json = eval_dir / f"summary_{tag}.json"
            ratio_json_arg = _as_rel_if_possible(ratio_json, root)
            method_eval_cmd = [
                str(py_exe),
                str(eval_script),
                "--repo-root",
                str(root),
                "--dynamic-dir",
                dynamic_dir_arg,
                "--solver",
                _as_rel_if_possible(selected_solver, root),
                "--dataset-solver-tag",
                dataset_solver_tag,
                "--method-name",
                method,
                "--ratio-tag",
                tag,
                "--n-instance-generated",
                str(int(args.n_instance)),
                "--perturb-types",
                ",".join(perturb_types),
                "--metric-basis",
                str(args.metric_basis),
                "--subgroup-mode",
                str(args.subgroup_mode),
                "--subgroup-tail-basis",
                str(args.subgroup_tail_basis),
                "--n-instance-eval",
                str(int(args.n_instance_eval)),
                "--gls-time-limit-per-instance",
                str(float(args.gls_time_limit_per_instance)),
                "--gls-iter-limit",
                str(int(args.gls_iter_limit)),
                "--gls-perturbation-moves",
                str(int(args.gls_perturbation_moves)),
                "--gap-eps",
                str(float(args.gap_eps)),
                "--jit-enable-kernels",
                str(int(args.jit_enable_kernels)),
                "--jit-numba-cache",
                str(int(args.jit_numba_cache)),
                "--jit-warmup",
                str(int(args.jit_warmup)),
                "--jit-decorate-guide",
                str(int(args.jit_decorate_guide)),
                "--output-json",
                ratio_json_arg,
            ]
            if str(args.dataset_prefix).strip():
                method_eval_cmd.extend(["--dataset-prefix", str(args.dataset_prefix).strip()])
            _run(method_eval_cmd, cwd=root)

            obj = json.loads(ratio_json.read_text(encoding="utf-8"))
            rows = obj.get("rows", [])
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    row_clean = {
                        "dataset_name": row.get("dataset_name"),
                        "ratio_tag": row.get("ratio_tag", tag),
                        "perturb_type": row.get("perturb_type"),
                        "n_instance_generated": row.get("n_instance_generated"),
                        "n_dynamic_instances": row.get("n_dynamic_instances", row.get("n_instance_total_in_dataset")),
                        "n_instances_used_in_eval": row.get("n_instances_used_in_eval", row.get("n_instance_eval")),
                        "n_instances_in_dynamic_file": row.get(
                            "n_instances_in_dynamic_file", row.get("n_instance_total_in_dataset")
                        ),
                        "n_nodes_per_instance_min": row.get("n_nodes_per_instance_min"),
                        "n_nodes_per_instance_max": row.get("n_nodes_per_instance_max"),
                        "n_nodes_per_instance_mean": _safe_float(row.get("n_nodes_per_instance_mean")),
                        "mean_gap": _safe_float(row.get("mean_gap")),
                        "mean_time_sec": _safe_float(row.get("mean_time_sec")),
                    }
                    combined_rows.append(row_clean)
            agg = obj.get("aggregate", {})
            ratio_aggregates.append(
                {
                    "ratio": float(ratio),
                    "ratio_tag": tag,
                    "mean_gap": _safe_float(agg.get("mean_gap")),
                    "mean_time_sec": _safe_float(agg.get("mean_time_sec")),
                }
            )

        _cleanup_dynamic_dir(dynamic_dir)

        combined_csv = eval_dir / "results_table.csv"
        combined_json = eval_dir / "summary.json"
        _write_rows_csv(combined_rows, combined_csv)
        combined_summary = {
            "method": method,
            "solver_path": str(selected_solver),
            "dataset_solver_tag": dataset_solver_tag,
            "ratio_set": [float(x) for x in ratios],
            "ratio_set_key": set_key,
            "params": {
                "config_path": str(config_used) if config_used is not None else None,
                "repo_root": str(root),
                "run_root": str(run_root),
                "run_name_base": run_name_base,
                "set_run_name": set_run_name,
                "run_dir": str(set_dir),
                "set_dir": str(set_dir),
                "dynamic_dir": str(dynamic_dir),
                "eval_dir": str(eval_dir),
                "method": method,
                "baseline_solver": str(baseline_solver),
                "erdr_solver": str(erdr_solver),
                "selected_solver": str(selected_solver),
                "dataset_solver_tag": dataset_solver_tag,
                "dataset_key": dataset_key,
                "base_pkl": str(base_pkl),
                "dataset_prefix": (str(args.dataset_prefix).strip() or None),
                "ratio_set": [float(x) for x in ratios],
                "ratio_set_key": set_key,
                "change_types": change_types,
                "n_instance": int(args.n_instance),
                "n_instance_eval": int(args.n_instance_eval),
                "metric_basis": str(args.metric_basis),
                "subgroup_mode": str(args.subgroup_mode),
                "subgroup_tail_basis": str(args.subgroup_tail_basis),
                "gls_time_limit_per_instance": float(args.gls_time_limit_per_instance),
                "gls_iter_limit": int(args.gls_iter_limit),
                "gls_perturbation_moves": int(args.gls_perturbation_moves),
                "gap_eps": float(args.gap_eps),
                "seed": int(args.seed),
                "jitter_sigma": float(args.jitter_sigma),
                "node_frac": float(args.node_frac),
                "concorde_scale": int(args.concorde_scale),
                "jit_enable_kernels": bool(int(args.jit_enable_kernels)),
                "jit_numba_cache": bool(int(args.jit_numba_cache)),
                "jit_warmup": bool(int(args.jit_warmup)),
                "jit_decorate_guide": bool(int(args.jit_decorate_guide)),
            },
            "ratio_aggregates": ratio_aggregates,
            "rows": combined_rows,
        }
        combined_json.write_text(json.dumps(combined_summary, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"\n[done] method={method} ratio_set={set_key}", flush=True)
        print(f"[done] dynamic_data: {dynamic_dir}", flush=True)
        print(f"[done] eval_json: {combined_json}", flush=True)
        print(f"[done] eval_table: {combined_csv}", flush=True)

    print("\n[done] run dirs:", flush=True)
    for d in created_set_dirs:
        print(f"  - {d}", flush=True)


if __name__ == "__main__":
    main()
