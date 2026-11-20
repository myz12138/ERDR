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


def _fmt_override_string(value: Any) -> str:
    text = str(value).replace("\\", "/")
    text = text.replace('"', '\\"')
    return f"\"{text}\""


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


def _enforce_minimal_output_policy(output_dir: Path, project_root: Path) -> None:
    if output_dir.resolve() == project_root.resolve():
        raise RuntimeError("Refusing to clean project root as training output directory.")
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


def _parse_metric_basis(eval_cfg: Dict[str, Any]) -> str:
    metric_basis = str(eval_cfg.get("metric_basis", "")).strip().lower()
    if metric_basis in {"gap", "cost"}:
        return metric_basis
    use_gap = bool(eval_cfg.get("use_gap_metric", True))
    return "gap" if use_gap else "cost"


def run_train(*, default_config_rel: str, default_mode: str) -> None:
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
    reevo_cfg = dict(cfg.get("reevo", {})) if isinstance(cfg.get("reevo"), dict) else {}
    eval_cfg = dict(cfg.get("evaluation", {})) if isinstance(cfg.get("evaluation"), dict) else {}
    exp_cfg = dict(cfg.get("exp", {})) if isinstance(cfg.get("exp"), dict) else {}
    group_cfg = (
        dict(eval_cfg.get("grouped_evaluation", {}))
        if isinstance(eval_cfg.get("grouped_evaluation"), dict)
        else {}
    )
    dds_cfg = (
        dict(eval_cfg.get("dominance_dissimilarity", {}))
        if isinstance(eval_cfg.get("dominance_dissimilarity"), dict)
        else {}
    )

    output_dir_raw = str(paths_cfg.get("output_dir", "")).strip()
    if not output_dir_raw:
        raise ValueError("paths.output_dir is required.")
    output_dir = _resolve_path(workspace_root, output_dir_raw)

    train_pkl_raw = str(paths_cfg.get("train_pkl", "")).strip()
    if not train_pkl_raw:
        raise ValueError("paths.train_pkl is required.")
    train_pkl = _resolve_path(workspace_root, train_pkl_raw)

    mode = str(eval_cfg.get("mode", default_mode)).strip().lower() or str(default_mode).strip().lower()
    if mode not in {"baseline", "erdr"}:
        raise ValueError(f"evaluation.mode must be baseline or erdr, got: {mode}")
    adapter_mode = "static" if mode == "baseline" else "static_risk2"
    metric_basis = _parse_metric_basis(eval_cfg)

    overrides: list[str] = [
        "run=tsp_gls_adapter",
        "problem=tsp_gls",
        "algorithm=reevo",
        f"hydra.run.dir={_fmt_override_string(output_dir.as_posix())}",
        f"run.paths.dataset_path={_fmt_override_string(train_pkl.as_posix())}",
        "run.paths.dataset_paths=[]",
        "run.paths.adapter_artifacts_dir=null",
        f"run.gls_adapter.mode={adapter_mode}",
        f"run.gls_adapter.metric_basis={metric_basis}",
    ]

    _append_if_present(overrides, "run.llm.llm_api_endpoint", llm_cfg, "llm_api_endpoint")
    _append_if_present(overrides, "run.llm.llm_api_key", llm_cfg, "llm_api_key")
    _append_if_present(overrides, "run.llm.model", llm_cfg, "llm_model")
    _append_if_present(overrides, "run.llm.temperature", llm_cfg, "llm_temperature")

    _append_if_present(overrides, "max_fe", reevo_cfg, "max_fe")
    _append_if_present(overrides, "pop_size", reevo_cfg, "pop_size")
    _append_if_present(overrides, "init_pop_size", reevo_cfg, "init_pop_size")
    _append_if_present(overrides, "mutation_rate", reevo_cfg, "mutation_rate")
    _append_if_present(overrides, "timeout", reevo_cfg, "timeout")
    _append_if_present(overrides, "diversify_init_pop", reevo_cfg, "diversify_init_pop")

    if "group_evaluation_mode" in eval_cfg:
        group_evaluation_mode = str(eval_cfg["group_evaluation_mode"])
        adapter_group_mode = "mean_std_worst" if group_evaluation_mode == "mean_variability_tail" else "mean_only"
        overrides.append(f"run.gls_adapter.subgroup_eval_mode={adapter_group_mode}")
    _append_if_present(overrides, "run.gls_adapter.subgroup_tail_basis", eval_cfg, "tail_risk_basis")
    _append_if_present(overrides, "run.gls_adapter.n_instance_eval", eval_cfg, "n_instance_eval")
    _append_if_present(overrides, "run.gls_adapter.time_limit_per_instance", eval_cfg, "gls_time_limit_per_instance")
    _append_if_present(overrides, "run.gls_adapter.perturbation_moves", eval_cfg, "gls_perturbation_moves")
    if "gls_iter_limit" in eval_cfg and eval_cfg.get("gls_iter_limit") is not None:
        _append_if_present(overrides, "run.gls_adapter.iter_limit", eval_cfg, "gls_iter_limit")
    else:
        _append_if_present(overrides, "run.gls_adapter.iter_limit", eval_cfg, "gls_ite_max")
    _append_if_present(overrides, "run.gls_adapter.gap_epsilon", eval_cfg, "gap_eps")
    _append_if_present(overrides, "run.gls_adapter.eval_process_timeout_sec", eval_cfg, "eval_process_timeout_sec")
    _append_if_present(overrides, "run.gls_adapter.jit_enable_kernels", eval_cfg, "jit_enable_kernels")
    _append_if_present(overrides, "run.gls_adapter.jit_numba_cache", eval_cfg, "jit_numba_cache")
    _append_if_present(overrides, "run.gls_adapter.jit_warmup", eval_cfg, "jit_warmup")
    _append_if_present(overrides, "run.gls_adapter.eva_numba_decorator", eval_cfg, "eva_numba_decorator")
    _append_if_present(overrides, "run.gls_adapter.jit_sanitize_guide", eval_cfg, "jit_sanitize_guide")
    _append_if_present(overrides, "run.gls_adapter.jit_sanitize_strict", eval_cfg, "jit_sanitize_strict")

    _append_if_present(overrides, "run.gls_adapter.groups.group_size", group_cfg, "group_size")
    _append_if_present(overrides, "run.gls_adapter.groups.n_groups_total", group_cfg, "n_groups_total")
    _append_if_present(overrides, "run.gls_adapter.groups.n_groups_train", group_cfg, "n_groups_train")
    _append_if_present(overrides, "run.gls_adapter.groups.n_groups_val", group_cfg, "n_groups_val")
    _append_if_present(overrides, "run.gls_adapter.groups.split_seed", group_cfg, "split_seed")
    _append_if_present(overrides, "run.gls_adapter.groups.shuffle_cases", group_cfg, "shuffle_cases")

    _append_if_present(overrides, "run.gls_adapter.population.dds_eps_rel", dds_cfg, "eps_rel")
    _append_if_present(overrides, "run.gls_adapter.population.dds_softmax_tau", dds_cfg, "softmax_tau")
    _append_if_present(overrides, "run.gls_adapter.population.dds_novelty_weight", dds_cfg, "novelty_weight")

    _append_if_present(overrides, "run.gls_adapter.population.train_keep_ratio", eval_cfg, "train_keep_ratio")
    _append_if_present(overrides, "run.gls_adapter.population.train_keep_min", eval_cfg, "train_keep_min")
    _append_if_present(overrides, "run.gls_adapter.population.train_keep_max", eval_cfg, "train_keep_max")
    _append_if_present(overrides, "run.gls_adapter.population.val_keep_ratio", eval_cfg, "val_keep_ratio")
    _append_if_present(overrides, "run.gls_adapter.population.val_keep_min", eval_cfg, "val_keep_min")
    _append_if_present(overrides, "run.gls_adapter.population.val_keep_max", eval_cfg, "val_keep_max")
    _append_if_present(overrides, "run.gls_adapter.population.parent_min", eval_cfg, "parent_min")
    _append_if_present(overrides, "run.gls_adapter.population.parent_max", eval_cfg, "parent_max")
    _append_if_present(overrides, "run.gls_adapter.population.dominance_epsilon", eval_cfg, "dominance_epsilon")
    _append_if_present(overrides, "run.gls_adapter.population.dissimilarity_metric", eval_cfg, "dissimilarity_metric")

    _append_if_present(overrides, "problem.adapter.output_mode", exp_cfg, "output_mode")
    _append_if_present(overrides, "problem.adapter.keep_top_k_solvers", exp_cfg, "keep_top_k_solvers")

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "main.py", *overrides]
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(project_root), check=True)

    output_mode = str(exp_cfg.get("output_mode", "minimal")).strip().lower()
    if output_mode == "minimal":
        _enforce_minimal_output_policy(output_dir, project_root)
