from __future__ import annotations

import argparse
import subprocess
import sys
import time
import shutil
from pathlib import Path
from typing import Any, Dict

import yaml


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError(f"config root must be mapping: {path}")
    return obj


def _resolve_path(repo_root: Path, value: str) -> Path:
    p = Path(str(value).strip())
    if p.is_absolute():
        return p.resolve()
    if p.parts and p.parts[0] == repo_root.name:
        return (repo_root.parent / p).resolve()
    candidate = (repo_root / p).resolve()
    if candidate.exists():
        return candidate
    parent_candidate = (repo_root.parent / p).resolve()
    if parent_candidate.exists():
        return parent_candidate
    return candidate


def _resolve_config_path(repo_root: Path, value: str) -> Path:
    p = Path(str(value).strip())
    if p.is_absolute():
        return p.resolve()
    if p.parts and p.parts[0] == repo_root.name:
        return (repo_root.parent / p).resolve()
    candidate = (repo_root / p).resolve()
    if candidate.exists():
        return candidate
    parent_candidate = (repo_root.parent / p).resolve()
    if parent_candidate.exists():
        return parent_candidate
    return candidate


def _fmt_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _append_if_present(overrides: list[str], key: str, mapping: Dict[str, Any], value_key: str) -> None:
    if value_key in mapping and mapping.get(value_key) is not None:
        overrides.append(f"{key}={_fmt_override_value(mapping.get(value_key))}")


def _safe_unlink(path: Path, retries: int = 6, backoff_sec: float = 0.2) -> bool:
    for i in range(max(int(retries), 1)):
        try:
            path.unlink(missing_ok=True)
            return True
        except PermissionError:
            if i + 1 < retries:
                time.sleep(backoff_sec * float(i + 1))
                continue
            return False
        except Exception:
            return False
    return False


def _safe_rmtree(path: Path, retries: int = 6, backoff_sec: float = 0.2) -> bool:
    for i in range(max(int(retries), 1)):
        try:
            shutil.rmtree(path, ignore_errors=False)
            return True
        except PermissionError:
            if i + 1 < retries:
                time.sleep(backoff_sec * float(i + 1))
                continue
            return False
        except FileNotFoundError:
            return True
        except Exception:
            return False
    return False


def _enforce_minimal_output_policy(output_dir: Path) -> None:
    keep_names = {
        "best_solver.py",
        "solver_archive",
        "solver_archive.json",
        "final_eval.json",
    }
    if not output_dir.exists():
        return
    for item in output_dir.iterdir():
        if item.name in keep_names:
            continue
        if item.is_dir():
            _safe_rmtree(item)
        else:
            _safe_unlink(item)


def run_train(
    *,
    default_config_rel: str,
    default_problem_cfg: str,
) -> None:
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]
    workspace_root = project_root.parents[1]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(project_root / default_config_rel),
        help="Path to yaml config.",
    )
    args = parser.parse_args()

    cfg_path = _resolve_config_path(workspace_root, str(args.config))
    cfg = _load_yaml(cfg_path)

    llm_cfg = dict(cfg.get("llm", {})) if isinstance(cfg.get("llm"), dict) else {}
    paths_cfg = dict(cfg.get("paths", {})) if isinstance(cfg.get("paths"), dict) else {}
    hercules_cfg = dict(cfg.get("hercules", {})) if isinstance(cfg.get("hercules"), dict) else {}
    eval_cfg = dict(cfg.get("evaluation", {})) if isinstance(cfg.get("evaluation"), dict) else {}
    erdr_cfg_explicit = dict(eval_cfg.get("erdr", {})) if isinstance(eval_cfg.get("erdr"), dict) else {}
    dds_cfg = (
        dict(eval_cfg.get("dominance_dissimilarity", {}))
        if isinstance(eval_cfg.get("dominance_dissimilarity"), dict)
        else {}
    )
    exp_cfg = dict(cfg.get("exp", {})) if isinstance(cfg.get("exp"), dict) else {}

    erdr_cfg: Dict[str, Any] = dict(erdr_cfg_explicit)
    if dds_cfg:
        if "enabled" in dds_cfg and "enabled" not in erdr_cfg:
            erdr_cfg["enabled"] = dds_cfg.get("enabled")
        if "eps_rel" in dds_cfg and "dds_eps_rel" not in erdr_cfg:
            erdr_cfg["dds_eps_rel"] = dds_cfg.get("eps_rel")
        if "novelty_weight" in dds_cfg and "dds_novelty_weight" not in erdr_cfg:
            erdr_cfg["dds_novelty_weight"] = dds_cfg.get("novelty_weight")

    problem_cfg = str(eval_cfg.get("problem_config", default_problem_cfg)).strip() or default_problem_cfg

    output_dir_raw = str(paths_cfg.get("output_dir", "")).strip()
    if not output_dir_raw:
        raise ValueError("paths.output_dir is required.")
    output_dir = _resolve_path(workspace_root, output_dir_raw)

    train_pkl_raw = str(paths_cfg.get("train_pkl", "")).strip()
    if not train_pkl_raw:
        raise ValueError("paths.train_pkl is required.")
    train_pkl = _resolve_path(workspace_root, train_pkl_raw)

    metric_basis = str(eval_cfg.get("metric_basis", "")).strip().lower()
    if not metric_basis:
        if bool(eval_cfg.get("use_gap_metric", True)):
            metric_basis = "gap"
        else:
            metric_basis = "cost"

    mode = str(eval_cfg.get("mode", "")).strip().lower()
    if not mode:
        mode = "erdr" if bool(erdr_cfg.get("enabled", False)) else "baseline"

    overrides: list[str] = [
        "algorithm=hercules-adapter",
        "manual_entry.enabled=false",
        f"problem={problem_cfg}",
        f"hydra.run.dir={output_dir.as_posix()}",
        f"problem.adapter.dataset_path={train_pkl.as_posix()}",
        f"problem.adapter.mode={mode}",
        f"problem.adapter.metric_basis={metric_basis}",
    ]

    _append_if_present(overrides, "llm_client.base_url", llm_cfg, "llm_api_endpoint")
    _append_if_present(overrides, "llm_client.api_key", llm_cfg, "llm_api_key")
    _append_if_present(overrides, "llm_client.model", llm_cfg, "llm_model")
    _append_if_present(overrides, "llm_client.temperature", llm_cfg, "llm_temperature")

    _append_if_present(overrides, "max_fe", hercules_cfg, "max_fe")
    _append_if_present(overrides, "pop_size", hercules_cfg, "pop_size")
    _append_if_present(overrides, "init_pop_size", hercules_cfg, "init_pop_size")
    _append_if_present(overrides, "mutation_rate", hercules_cfg, "mutation_rate")
    _append_if_present(overrides, "diversify_init_pop", hercules_cfg, "diversify_init_pop")

    _append_if_present(overrides, "problem.problem_size", eval_cfg, "problem_size")
    _append_if_present(overrides, "problem.adapter.group_evaluation_mode", eval_cfg, "group_evaluation_mode")
    _append_if_present(overrides, "problem.adapter.tail_risk_basis", eval_cfg, "tail_risk_basis")
    _append_if_present(overrides, "problem.adapter.n_instance_eval", eval_cfg, "n_instance_eval")
    _append_if_present(overrides, "problem.adapter.time_limit_per_instance", eval_cfg, "gls_time_limit_per_instance")
    _append_if_present(overrides, "problem.adapter.gls_perturbation_moves", eval_cfg, "gls_perturbation_moves")
    if "gls_iter_limit" in eval_cfg and eval_cfg.get("gls_iter_limit") is not None:
        _append_if_present(overrides, "problem.adapter.gls_iter_limit", eval_cfg, "gls_iter_limit")
    else:
        _append_if_present(overrides, "problem.adapter.gls_iter_limit", eval_cfg, "gls_ite_max")
    _append_if_present(overrides, "problem.adapter.gap_eps", eval_cfg, "gap_eps")

    _append_if_present(overrides, "problem.adapter.erdr.enabled", erdr_cfg, "enabled")
    _append_if_present(overrides, "problem.adapter.erdr.group_size", erdr_cfg, "group_size")
    _append_if_present(overrides, "problem.adapter.erdr.n_groups_total", erdr_cfg, "n_groups_total")
    _append_if_present(overrides, "problem.adapter.erdr.n_groups_train", erdr_cfg, "n_groups_train")
    _append_if_present(overrides, "problem.adapter.erdr.n_groups_val", erdr_cfg, "n_groups_val")
    _append_if_present(overrides, "problem.adapter.erdr.split_seed", erdr_cfg, "split_seed")
    _append_if_present(overrides, "problem.adapter.erdr.val_interval", erdr_cfg, "val_interval")
    _append_if_present(overrides, "problem.adapter.erdr.train_pool_multiplier", erdr_cfg, "train_pool_multiplier")
    _append_if_present(overrides, "problem.adapter.erdr.dds_eps_rel", erdr_cfg, "dds_eps_rel")
    _append_if_present(overrides, "problem.adapter.erdr.dds_softmax_tau", erdr_cfg, "dds_softmax_tau")
    _append_if_present(overrides, "problem.adapter.erdr.dds_novelty_weight", erdr_cfg, "dds_novelty_weight")

    archive_top_k = eval_cfg.get("archive_top_k")
    if archive_top_k is not None and exp_cfg.get("keep_top_k_solvers") is None:
        exp_cfg["keep_top_k_solvers"] = archive_top_k
    _append_if_present(overrides, "problem.adapter.output_mode", exp_cfg, "output_mode")
    _append_if_present(overrides, "problem.adapter.keep_top_k_solvers", exp_cfg, "keep_top_k_solvers")

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "main.py", *overrides]
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(project_root), check=True)

    output_mode = str(exp_cfg.get("output_mode", "minimal")).strip().lower()
    if output_mode == "minimal":
        _enforce_minimal_output_policy(output_dir)
