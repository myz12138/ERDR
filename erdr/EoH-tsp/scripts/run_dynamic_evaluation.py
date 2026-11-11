from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
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
    # Keep only generated pkl datasets with embedded concorde costs.
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


def _mirror_input_file(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return dst


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
    dynamic_evaluation = obj.get("dynamic_evaluation", {})
    gls = obj.get("gls", {})
    if not isinstance(paths, dict):
        paths = {}
    if not isinstance(dynamic_evaluation, dict):
        dynamic_evaluation = {}
    if not isinstance(gls, dict):
        gls = {}

    defaults: Dict[str, Any] = {
        "repo_root": paths.get("repo_root", str(default_repo_root)),
        "baseline_solver": paths.get(
            "baseline_solver",
            f"{project_rel}/outputs/eoh_baseline/best_solver.py",
        ),
        "erdr_solver": paths.get(
            "erdr_solver",
            f"{project_rel}/outputs/eoh_erdr/best_solver.py",
        ),
        "base_pkl": paths.get(
            "base_pkl",
            "data_generate/TSP200_u50/TSP200_u50_concorde.pkl",
        ),
        "run_root": paths.get("run_root", f"{project_rel}/data/dynamic_evaluation_runs"),
        "run_name": paths.get("run_name", ""),
        "method": dynamic_evaluation.get("method", "baseline"),
        "ratio_sets": dynamic_evaluation.get("ratio_sets", "0.2,0.5,0.8"),
        "change_types": dynamic_evaluation.get("change_types", "location_change,request_arrival,request_cancellation"),
        "n_instance": dynamic_evaluation.get("n_instance", 50),
        "seed": dynamic_evaluation.get("seed", 20260524),
        "gls_neighbor_k": gls.get("neighbor_k", 20),
        "gls_time_limit_per_instance": gls.get("time_limit_per_instance", 20.0),
        "gls_ite_max": gls.get("ite_max", 500),
        "gls_perturbation_moves": gls.get("perturbation_moves", 2),
        "gap_eps": gls.get("gap_eps", 1.0e-12),
    }
    return defaults


def main() -> None:
    script_path = Path(__file__).resolve()
    project_dir = script_path.parents[1]
    repo_root = project_dir.parents[1]
    project_rel = project_dir.relative_to(repo_root).as_posix()

    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument(
        "--config",
        type=str,
        default=f"{project_rel}/configs/dynamic_evaluation.yaml",
    )
    bootstrap_args, _ = bootstrap.parse_known_args()
    config_raw = str(bootstrap_args.config).strip()
    if config_raw:
        config_path = Path(config_raw)
        if not config_path.is_absolute():
            config_path = (repo_root / config_path).resolve()
    else:
        config_path = None
    cfg_defaults = _load_config_defaults(config_path, repo_root, project_rel)

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one baseline or ERDR solver on dynamic TSP instances."
        )
    )
    parser.add_argument("--config", type=str, default=config_raw)
    parser.add_argument("--repo-root", type=str, default=str(cfg_defaults.get("repo_root", repo_root)))
    parser.add_argument(
        "--baseline-solver",
        type=str,
        default=str(
            cfg_defaults.get(
                "baseline_solver",
                f"{project_rel}/outputs/eoh_baseline/best_solver.py",
            )
        ),
    )
    parser.add_argument(
        "--erdr-solver",
        type=str,
        default=str(
            cfg_defaults.get(
                "erdr_solver",
                f"{project_rel}/outputs/eoh_erdr/best_solver.py",
            )
        ),
    )
    parser.add_argument(
        "--run-root",
        type=str,
        default=str(cfg_defaults.get("run_root", f"{project_rel}/data/dynamic_evaluation_runs")),
    )
    parser.add_argument(
        "--base-pkl",
        type=str,
        default=str(cfg_defaults.get("base_pkl", "data_generate/TSP200_u50/TSP200_u50_concorde.pkl")),
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["baseline", "erdr"],
        default=str(cfg_defaults.get("method", "baseline")),
        help="Evaluate one solver variant per run.",
    )
    parser.add_argument("--run-name", type=str, default=str(cfg_defaults.get("run_name", "")))
    parser.add_argument(
        "--ratio-sets",
        type=str,
        default=str(cfg_defaults.get("ratio_sets", "0.2,0.5,0.8")),
        help="Semicolon-separated ratio collections. Example: 0.2,0.5;0.5,0.8",
    )
    parser.add_argument(
        "--change-types",
        type=str,
        default=str(cfg_defaults.get("change_types", "location_change,request_arrival,request_cancellation")),
    )
    parser.add_argument("--n-instance", type=int, default=int(cfg_defaults.get("n_instance", 50)))
    parser.add_argument("--gls-neighbor-k", type=int, default=int(cfg_defaults.get("gls_neighbor_k", 20)))
    parser.add_argument(
        "--gls-time-limit-per-instance",
        type=float,
        default=float(cfg_defaults.get("gls_time_limit_per_instance", 20.0)),
    )
    parser.add_argument("--gls-ite-max", type=int, default=int(cfg_defaults.get("gls_ite_max", 500)))
    parser.add_argument(
        "--gls-perturbation-moves",
        type=int,
        default=int(cfg_defaults.get("gls_perturbation_moves", 2)),
    )
    parser.add_argument("--gap-eps", type=float, default=float(cfg_defaults.get("gap_eps", 1.0e-12)))
    parser.add_argument("--seed", type=int, default=int(cfg_defaults.get("seed", 20260524)))
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    config_used_raw = str(args.config).strip()
    if config_used_raw:
        config_used = Path(config_used_raw)
        if not config_used.is_absolute():
            config_used = (root / config_used).resolve()
    else:
        config_used = None
    baseline_solver = (root / args.baseline_solver).resolve()
    erdr_solver = (root / args.erdr_solver).resolve()
    base_pkl = Path(str(args.base_pkl))
    if not base_pkl.is_absolute():
        base_pkl = (root / base_pkl).resolve()
    change_types = [x.strip() for x in str(args.change_types).split(",") if x.strip()]
    generator_change_types = {
        "location_change": "coord_jitter",
        "request_arrival": "node_add",
        "request_cancellation": "node_remove",
    }
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
    if not base_pkl.exists():
        raise FileNotFoundError(f"base pkl not found: {base_pkl}")
    dataset_key = _infer_dataset_key(base_pkl)

    ratio_sets = _parse_ratio_sets(str(args.ratio_sets).strip())

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name_raw = str(args.run_name).strip()
    run_name_base = run_name_raw if run_name_raw else f"dynamic_evaluation_{method}_{dataset_key}_{stamp}"
    run_root = (root / args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    py_exe = Path(sys.executable).resolve()
    gen_script = root / "data_generate" / "generate_dynamic_perturbed_eoh7.py"
    eval_script = project_dir / "scripts" / "eval_dynamic_redeploy.py"
    gls_src_dir = project_dir / "src"
    created_set_dirs: List[Path] = []

    for ratios in ratio_sets:
        set_key = _ratio_set_key(ratios)
        set_run_name = f"{run_name_base}_{set_key}"
        set_dir = run_root / set_run_name
        if set_dir.exists():
            raise FileExistsError(f"run dir already exists: {set_dir}")
        dynamic_dir = set_dir / "dynamic_datasets"
        eval_dir = set_dir / "eval"
        input_dir = set_dir / "input_data"
        set_dir.mkdir(parents=True, exist_ok=False)
        dynamic_dir.mkdir(parents=True, exist_ok=False)
        eval_dir.mkdir(parents=True, exist_ok=False)
        input_dir.mkdir(parents=True, exist_ok=False)
        created_set_dirs.append(set_dir)

        local_base_pkl = _mirror_input_file(base_pkl, input_dir)
        ratios_text = ",".join(f"{x:g}" for x in ratios)
        dynamic_dir_rel = dynamic_dir.relative_to(root).as_posix()
        local_base_pkl_rel = local_base_pkl.relative_to(root).as_posix()
        gen_cmd = [
            str(py_exe),
            str(gen_script),
            "--repo-root",
            str(root),
            "--base-pkl",
            local_base_pkl_rel,
            "--solver-mode",
            "single",
            "--single-solver-id",
            dataset_solver_tag,
            "--single-solver-path",
            selected_solver.as_posix(),
            "--gls-src-dir",
            gls_src_dir.as_posix(),
            "--out-dir",
            dynamic_dir_rel,
            "--n-instance",
            str(int(args.n_instance)),
            "--ratios",
            ratios_text,
            "--perturb-types",
            ",".join(perturb_types),
            "--seed",
            str(int(args.seed)),
            "--neighbor-k",
            str(int(args.gls_neighbor_k)),
            "--time-limit-sec",
            str(float(args.gls_time_limit_per_instance)),
            "--ite-max",
            str(int(args.gls_ite_max)),
            "--perturbation-moves",
            str(int(args.gls_perturbation_moves)),
            "--jitter-sigma",
            "0.02",
            "--node-frac",
            "0.2",
            "--concorde-scale",
            "10000",
        ]
        _run(gen_cmd, cwd=root)

        combined_rows: List[Dict[str, Any]] = []
        ratio_aggregates: List[Dict[str, Any]] = []
        for ratio in ratios:
            tag = _ratio_tag(ratio)
            ratio_json = eval_dir / f"summary_{tag}.json"
            ratio_json_rel = ratio_json.relative_to(root).as_posix()
            method_eval_cmd = [
                str(py_exe),
                str(eval_script),
                "--repo-root",
                str(root),
                "--dynamic-dir",
                dynamic_dir_rel,
                "--solver",
                selected_solver.as_posix(),
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
                "--gls-neighbor-k",
                str(int(args.gls_neighbor_k)),
                "--gls-time-limit-per-instance",
                str(float(args.gls_time_limit_per_instance)),
                "--gls-ite-max",
                str(int(args.gls_ite_max)),
                "--gls-perturbation-moves",
                str(int(args.gls_perturbation_moves)),
                "--gap-eps",
                str(float(args.gap_eps)),
                "--output-json",
                ratio_json_rel,
            ]
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
            "solver_path": selected_solver.as_posix(),
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
                "input_dir": str(input_dir),
                "dynamic_dir": str(dynamic_dir),
                "eval_dir": str(eval_dir),
                "method": method,
                "baseline_solver": baseline_solver.as_posix(),
                "erdr_solver": erdr_solver.as_posix(),
                "selected_solver": selected_solver.as_posix(),
                "dataset_solver_tag": dataset_solver_tag,
                "dataset_key": dataset_key,
                "base_pkl": local_base_pkl.as_posix(),
                "base_pkl_source": base_pkl.as_posix(),
                "base_pkl_used": local_base_pkl.as_posix(),
                "ratio_set": [float(x) for x in ratios],
                "ratio_set_key": set_key,
                "change_types": change_types,
                "n_instance": int(args.n_instance),
                "n_instance_eval_mode": "all_in_dataset",
                "gls_neighbor_k": int(args.gls_neighbor_k),
                "gls_time_limit_per_instance": float(args.gls_time_limit_per_instance),
                "gls_ite_max": int(args.gls_ite_max),
                "gls_perturbation_moves": int(args.gls_perturbation_moves),
                "gap_eps": float(args.gap_eps),
                "seed": int(args.seed),
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
