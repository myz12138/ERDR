from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def _project_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("configuration root must be a mapping")
    return value


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _ratio_tag(ratio: float) -> str:
    return f"r{int(round(ratio * 100)):02d}"


def main() -> None:
    project_dir = _project_dir()
    default_repo_root = project_dir.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(project_dir / "configs" / "dynamic_evaluation.yaml"))
    parser.add_argument("--method", choices=["baseline", "erdr"], default="")
    args = parser.parse_args()
    config = _load_config(Path(args.config))
    paths = dict(config.get("paths", {}))
    repo_root = _resolve(default_repo_root, str(paths.get("repo_root", ".")))
    dynamic_cfg = dict(config.get("dynamic_evaluation", {}))
    gls_cfg = dict(config.get("gls", {}))
    method = args.method or str(dynamic_cfg.get("method", "erdr"))
    solver_key = "baseline_solver" if method == "baseline" else "erdr_solver"
    solver = _resolve(repo_root, str(paths[solver_key]))
    base_pkl = _resolve(repo_root, str(paths["base_pkl"]))
    if not solver.is_file() or not base_pkl.is_file():
        raise FileNotFoundError("selected solver and base_pkl must exist before dynamic evaluation")
    count = int(dynamic_cfg.get("n_instance", 50))
    ratios = [float(item.strip()) for item in str(dynamic_cfg.get("ratio_sets", "0.2,0.5,0.8")).split(",") if item.strip()]
    if len(ratios) < 2 or any(ratio <= 0.0 or ratio >= 1.0 for ratio in ratios):
        raise ValueError("ratio_sets must contain at least two values in (0, 1)")
    change_map = {
        "location_change": "coord_jitter",
        "request_arrival": "node_add",
        "request_cancellation": "node_remove",
    }
    changes = [item.strip() for item in str(dynamic_cfg.get("change_types", "")).split(",") if item.strip()]
    if not changes or any(change not in change_map for change in changes):
        raise ValueError("change_types must use location_change, request_arrival, request_cancellation")
    run_root = _resolve(repo_root, str(paths["run_root"]))
    run_dir = run_root / f"dynamic_evaluation_{method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dynamic_dir = run_dir / "dynamic_datasets"
    eval_dir = run_dir / "eval"
    run_dir.mkdir(parents=True, exist_ok=False)
    eval_dir.mkdir(parents=True, exist_ok=False)
    generator = project_dir / "scripts" / "generate_dynamic_tsp.py"
    if not generator.is_file():
        raise FileNotFoundError(f"MEOH dynamic data generator not found: {generator}")
    command = [
        sys.executable,
        str(generator),
        "--repo-root", str(repo_root),
        "--base-pkl", str(base_pkl.relative_to(repo_root).as_posix()),
        "--solver-id", method,
        "--solver-path", str(solver),
        "--out-dir", str(dynamic_dir.relative_to(repo_root).as_posix()),
        "--n-instance", str(count),
        "--ratios", ",".join(f"{ratio:g}" for ratio in ratios),
        "--perturb-types", ",".join(change_map[change] for change in changes),
        "--seed", str(int(dynamic_cfg.get("seed", 20260524))),
        "--neighbor-k", str(int(gls_cfg.get("neighbor_k", 20))),
        "--time-limit-sec", str(float(gls_cfg.get("time_limit_per_instance", 20.0))),
        "--ite-max", str(int(gls_cfg.get("ite_max", 500))),
        "--perturbation-moves", str(int(gls_cfg.get("perturbation_moves", 2))),
        "--jitter-sigma", "0.02",
        "--node-frac", "0.2",
        "--concorde-scale", "10000",
    ]
    subprocess.run(command, cwd=repo_root, check=True)
    summaries = []
    for ratio in ratios:
        output = eval_dir / f"summary_{_ratio_tag(ratio)}.json"
        evaluate_command = [
            sys.executable,
            str(project_dir / "scripts" / "eval_dynamic_redeploy.py"),
            "--repo-root", str(repo_root),
            "--dynamic-dir", str(dynamic_dir.relative_to(repo_root).as_posix()),
            "--solver", str(solver.relative_to(repo_root).as_posix()),
            "--dataset-solver-tag", method,
            "--method-name", method,
            "--ratio-tag", _ratio_tag(ratio),
            "--perturb-types", ",".join(change_map[change] for change in changes),
            "--n-instance-generated", str(count),
            "--gls-time-limit-per-instance", str(float(gls_cfg.get("time_limit_per_instance", 20.0))),
            "--gls-ite-max", str(int(gls_cfg.get("ite_max", 500))),
            "--gls-perturbation-moves", str(int(gls_cfg.get("perturbation_moves", 2))),
            "--output-json", str(output.relative_to(repo_root).as_posix()),
        ]
        subprocess.run(evaluate_command, cwd=repo_root, check=True)
        summaries.append(json.loads(output.read_text(encoding="utf-8")))
    shutil.rmtree(dynamic_dir)
    result = {"method": method, "run_dir": str(run_dir), "summaries": summaries}
    (eval_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
