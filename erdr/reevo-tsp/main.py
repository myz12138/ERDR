import hydra
import argparse
import logging 
import os
from pathlib import Path
import subprocess
import json
import shutil
from utils.utils import init_client, print_hyperlink


REEVO_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("REEVO_ROOT", str(REEVO_ROOT))
ROOT_DIR = str(REEVO_ROOT)
logging.basicConfig(level=logging.INFO)


def _patch_hydra_argparse_py314_compat() -> None:
    """
    Hydra 1.3.x uses LazyCompletionHelp as argparse help text.
    Python 3.14 argparse now checks: '%' not in help_string, which
    requires __contains__. Add a tiny compatibility shim at runtime.
    """
    try:
        from hydra._internal.utils import LazyCompletionHelp  # type: ignore

        if not hasattr(LazyCompletionHelp, "__contains__"):
            def _contains(self, item):  # type: ignore[no-redef]
                try:
                    return str(item) in str(self)
                except Exception:
                    return False

            LazyCompletionHelp.__contains__ = _contains  # type: ignore[attr-defined]
    except Exception:
        # Best effort patch; ignore if Hydra internals change.
        pass


_patch_hydra_argparse_py314_compat()


def _patch_argparse_py314_help_compat() -> None:
    """
    Python 3.14 argparse is stricter for help-string containers.
    Hydra 1.3.x may pass lazy help objects; coerce to string early.
    """
    try:
        if getattr(argparse.ArgumentParser._check_help, "__name__", "") == "_patched_check_help_compat":
            return

        _orig_check_help = argparse.ArgumentParser._check_help

        def _patched_check_help_compat(self, action):  # type: ignore[no-redef]
            help_obj = getattr(action, "help", None)
            if help_obj is not None and not isinstance(help_obj, str):
                try:
                    action.help = str(help_obj)
                except Exception:
                    action.help = ""
            return _orig_check_help(self, action)

        argparse.ArgumentParser._check_help = _patched_check_help_compat  # type: ignore[assignment]
    except Exception:
        pass


_patch_argparse_py314_help_compat()


def _to_float_or_inf(value) -> float:
    try:
        v = float(value)
    except Exception:
        return float("inf")
    if v != v:
        return float("inf")
    return v


def _pick_eval_metric(payload: dict) -> float:
    for key in (
        "selected_metric",
        "objective",
        "best_metric",
        "avg_gap",
        "avg_obj",
        "mean_gap_pct",
        "mean_gap",
        "mean_cost",
    ):
        if key in payload:
            val = _to_float_or_inf(payload.get(key))
            if val != float("inf"):
                return val
    return float("inf")


def _finalize_adapter_outputs(workspace_dir: Path, keep_top_k: int, output_mode: str) -> Path:
    adapter_dir = workspace_dir / "adapter_artifacts"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    code_dir = adapter_dir / "code"
    eval_dir = adapter_dir / "eval"
    legacy_eval_dir = adapter_dir / "eval_json"
    if (not eval_dir.exists()) and legacy_eval_dir.exists():
        eval_dir = legacy_eval_dir
    adapter_best_result_path = adapter_dir / "best_result_adapter.json"
    adapter_final_eval_path = adapter_dir / "final_eval.json"
    adapter_manifest_path = adapter_dir / "adapter_manifest.json"

    # EOH-style top-level artifacts under run output dir.
    best_solver_path = workspace_dir / "best_solver.py"
    solver_archive_dir = workspace_dir / "solver_archive"
    solver_archive_json = workspace_dir / "solver_archive.json"
    final_eval_path = workspace_dir / "final_eval.json"

    result = {}
    if adapter_best_result_path.exists():
        with adapter_best_result_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            result = dict(obj)
    elif adapter_final_eval_path.exists():
        with adapter_final_eval_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            result = dict(obj)

    ranked: list[tuple[float, Path]] = []
    code_payload_by_path: dict[str, dict] = {}
    search_trace_rows: list[dict] = []
    if eval_dir.exists() and code_dir.exists():
        for p in eval_dir.glob("*.json"):
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    continue
            except Exception:
                continue
            metric = _pick_eval_metric(payload)
            code_path_raw = str(payload.get("code_path", "")).strip()
            if code_path_raw:
                cp = Path(code_path_raw)
                if not cp.is_absolute():
                    cp = (workspace_dir / cp).resolve()
                else:
                    cp = cp.resolve()
            else:
                cp = (code_dir / f"{p.stem}.py").resolve()
            if cp.exists() and cp.suffix.lower() == ".py":
                ranked.append((metric, cp))
                key = str(cp.resolve())
                prev_payload = code_payload_by_path.get(key)
                prev_metric = _pick_eval_metric(prev_payload) if isinstance(prev_payload, dict) else float("inf")
                if metric <= prev_metric:
                    code_payload_by_path[key] = dict(payload)

                try:
                    code_text = cp.read_text(encoding="utf-8")
                except Exception:
                    code_text = ""
                search_trace_rows.append(
                    {
                        "code": code_text,
                        "objective": float(metric),
                        "other_inf": dict(payload),
                        "source_file": str(p.resolve()),
                        "source_type": "adapter_eval",
                    }
                )

    dedup: dict[str, tuple[float, Path]] = {}
    for metric, cp in ranked:
        key = str(cp.resolve())
        prev = dedup.get(key)
        if prev is None or metric < prev[0]:
            dedup[key] = (metric, cp.resolve())
    ranked = sorted(dedup.values(), key=lambda x: x[0])

    if not ranked and code_dir.exists():
        ranked = [(float("inf"), p.resolve()) for p in sorted(code_dir.glob("*.py"))]

    best_code_path_raw = str(result.get("best_code_path_overall", "")).strip()
    best_code_path = None
    if best_code_path_raw:
        tmp = Path(best_code_path_raw)
        best_code_path = tmp.resolve() if tmp.is_absolute() else (workspace_dir / tmp).resolve()

    keep: list[Path] = [cp for _, cp in ranked[: max(int(keep_top_k), 1)]]
    if best_code_path is not None and best_code_path.exists():
        keep.append(best_code_path.resolve())
    keep_set = {str(p.resolve()) for p in keep}

    if code_dir.exists():
        for p in code_dir.glob("*.py"):
            if str(p.resolve()) not in keep_set:
                try:
                    p.unlink()
                except Exception:
                    pass

    keep_ordered: list[Path] = []
    seen_keep = set()
    for cp in keep:
        key = str(cp.resolve())
        if key in seen_keep:
            continue
        seen_keep.add(key)
        keep_ordered.append(cp.resolve())

    solver_archive_dir.mkdir(parents=True, exist_ok=True)
    for stale in solver_archive_dir.glob("solver_*.py"):
        try:
            stale.unlink()
        except Exception:
            pass

    ranked_metric = {str(cp.resolve()): float(metric) for metric, cp in ranked}
    archive_rows = []
    for idx, cp in enumerate(keep_ordered):
        solver_id = f"solver_{idx}"
        out_path = solver_archive_dir / f"{solver_id}.py"
        shutil.copyfile(cp, out_path)
        payload = code_payload_by_path.get(str(cp.resolve()), {})
        avg_obj = payload.get("mean_cost") if isinstance(payload, dict) else None
        archive_rows.append(
            {
                "solver_id": solver_id,
                "code_path": str(out_path.resolve()),
                "objective": ranked_metric.get(str(cp.resolve()), float("inf")),
                "avg_obj": (_to_float_or_inf(avg_obj) if avg_obj is not None else None),
                "style_tag": "unknown",
            }
        )
    solver_archive_json.write_text(
        json.dumps({"solvers": archive_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if best_code_path is not None and best_code_path.exists():
        shutil.copyfile(best_code_path, best_solver_path)
    elif keep_ordered:
        shutil.copyfile(keep_ordered[0], best_solver_path)

    train_eval = result.get("train_eval", {}) if isinstance(result.get("train_eval"), dict) else {}
    val_eval = result.get("val_eval", {}) if isinstance(result.get("val_eval"), dict) else {}

    train_metrics = {
        "n_eval": int(train_eval.get("n_eval", 0)),
        "avg_cost": (_to_float_or_inf(train_eval.get("mean_cost")) if train_eval else None),
        "avg_gap": (_to_float_or_inf(train_eval.get("mean_gap")) if train_eval else None),
        "avg_gap_pct": (_to_float_or_inf(train_eval.get("mean_gap_pct")) if train_eval else None),
        "avg_tsp_time_sec": (_to_float_or_inf(train_eval.get("mean_time_sec")) if train_eval else None),
        "metric_mode": (
            f"{str(result.get('metric_basis', 'gap'))}_{str(result.get('subgroup_eval_mode', 'mean_only'))}"
        ),
        "note": "Train-split average metric on selected best solver.",
    }
    val_metrics = None
    if val_eval:
        val_metrics = {
            "n_eval": int(val_eval.get("n_eval", 0)),
            "avg_cost": _to_float_or_inf(val_eval.get("mean_cost")),
            "avg_gap": _to_float_or_inf(val_eval.get("mean_gap")),
            "avg_gap_pct": _to_float_or_inf(val_eval.get("mean_gap_pct")),
            "avg_tsp_time_sec": _to_float_or_inf(val_eval.get("mean_time_sec")),
            "metric_mode": (
                f"{str(result.get('metric_basis', 'gap'))}_{str(result.get('subgroup_eval_mode', 'mean_only'))}"
            ),
            "note": "Validation-split average metric on selected best solver.",
        }

    adapter_manifest_obj = {}
    if adapter_manifest_path.exists():
        try:
            manifest_obj = json.loads(adapter_manifest_path.read_text(encoding="utf-8"))
            if isinstance(manifest_obj, dict):
                adapter_manifest_obj = dict(manifest_obj)
        except Exception:
            pass

    best_metric = _to_float_or_inf(result.get("best_obj_overall"))
    if best_metric == float("inf") and train_eval:
        best_metric = _to_float_or_inf(train_eval.get("selected_metric"))
    best_gap = _to_float_or_inf(train_eval.get("mean_gap")) if train_eval else None
    best_cost = _to_float_or_inf(train_eval.get("mean_cost")) if train_eval else None
    best_time_sec = _to_float_or_inf(train_eval.get("mean_time_sec")) if train_eval else None

    mode_raw = str(result.get("mode", "")).strip().lower()
    if mode_raw == "static":
        mode_name = "reevo_baseline"
    elif mode_raw == "static_risk2":
        mode_name = "reevo_erdr"
    else:
        mode_name = mode_raw or "reevo_adapter"

    subgroup_eval_mode_raw = str(result.get("subgroup_eval_mode", "mean_only"))
    subgroup_eval_mode = "mean_variability_tail" if subgroup_eval_mode_raw == "mean_std_worst" else "mean_only"
    subgroup_multi_objective = bool(subgroup_eval_mode == "mean_variability_tail")
    use_gap_metric = bool(str(result.get("metric_basis", "gap")).strip().lower() == "gap")
    dds_enabled = bool(subgroup_multi_objective)
    selection_method = (
        str(adapter_manifest_obj.get("selection_method"))
        if adapter_manifest_obj.get("selection_method") is not None
        else ("dds_pair_select" if mode_raw == "static_risk2" else "random_pair")
    )
    evaluation_profile = (
        ("gap" if use_gap_metric else "cost")
        + "_"
        + subgroup_eval_mode
        + "_"
        + ("dds" if dds_enabled else "single_obj")
    )

    final_obj = {
        "mode": str(mode_name),
        "best_metric": (None if best_metric == float("inf") else float(best_metric)),
        "best_cost": (None if best_cost == float("inf") else float(best_cost)),
        "best_gap": (None if best_gap == float("inf") else float(best_gap)),
        "best_time_sec": (None if best_time_sec == float("inf") else float(best_time_sec)),
        "evaluation_profile": evaluation_profile,
        "selection_method": selection_method,
        "management_method": ("dds_population" if dds_enabled else "greedy_obj"),
        "dominance_dissimilarity": {
            "enabled": bool(dds_enabled),
            "eps_rel": float(adapter_manifest_obj.get("dds_config", {}).get("eps_rel", 1e-3))
            if adapter_manifest_obj
            else 1e-3,
            "softmax_tau": float(adapter_manifest_obj.get("dds_config", {}).get("softmax_tau", 1.0))
            if adapter_manifest_obj
            else 1.0,
            "novelty_weight": float(adapter_manifest_obj.get("dds_config", {}).get("novelty_weight", 0.05))
            if adapter_manifest_obj
            else 0.05,
        },
        "group_evaluation_mode": subgroup_eval_mode,
        "subgroup_multi_objective": bool(subgroup_multi_objective),
        "tail_risk_basis": str(result.get("subgroup_tail_basis", "gap")),
        "use_gap_metric": bool(use_gap_metric),
        "grouped_evaluation_enabled": bool(mode_raw == "static_risk2"),
        "trainval_split_seed": adapter_manifest_obj.get("split_seed"),
        "trainval_population_size": adapter_manifest_obj.get("population_size"),
        "train_pkl": adapter_manifest_obj.get("dataset_source"),
        "output_dir": str(workspace_dir.resolve()),
        "best_solver_path": str(best_solver_path.resolve()) if best_solver_path.exists() else None,
        "best_code_path_overall": str(best_solver_path.resolve()) if best_solver_path.exists() else None,
        "function_evals": result.get("function_evals"),
        "train_dataset_avg_metrics": train_metrics if train_metrics.get("n_eval", 0) > 0 else None,
        "val_dataset_avg_metrics": val_metrics,
        "n_instances_eval_train": int(train_eval.get("n_eval", 0)) if train_eval else 0,
        "n_instances_eval_val": int(val_eval.get("n_eval", 0)) if val_eval else 0,
        "search_trace_n_rows": int(len(search_trace_rows)),
        "solver_archive_path": str(solver_archive_json.resolve()),
        "solver_archive_size": int(len(archive_rows)),
        "solver_archive_unique_candidates": int(len(archive_rows)),
        "solver_archive_total_candidates": int(len(ranked)),
        "final_eval_path": str(final_eval_path.resolve()),
        "artifact_policy": {
            "output_mode": ("full" if str(output_mode).strip().lower() == "full" else "minimal"),
            "keep_top_k_solvers": int(max(int(keep_top_k), 1)),
            "kept_outputs": [
                "best_solver.py",
                "solver_archive/",
                "solver_archive.json",
                "final_eval.json",
            ],
        },
    }

    final_eval_path.write_text(json.dumps(final_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    # Do not enforce strict cleanup in-process on Windows:
    # Hydra/job logging can still hold workspace log handles at this point.
    # Post-run launcher script will enforce strict artifact policy after subprocess exits.
    return final_eval_path

@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg):
    workspace_dir = Path.cwd()
    # Set logging level
    logging.info(f"Workspace: {print_hyperlink(workspace_dir)}")
    logging.info(f"Project Root: {print_hyperlink(ROOT_DIR)}")
    logging.info(f"Using LLM: {cfg.get('model', cfg.llm_client.model)}")
    logging.info(f"Using Algorithm: {cfg.algorithm}")

    client = init_client(cfg)
    # optional clients for operators (ReEvo)
    long_ref_llm = hydra.utils.instantiate(cfg.llm_long_ref) if cfg.get("llm_long_ref") else None
    short_ref_llm = hydra.utils.instantiate(cfg.llm_short_ref) if cfg.get("llm_short_ref") else None
    crossover_llm = hydra.utils.instantiate(cfg.llm_crossover) if cfg.get("llm_crossover") else None
    mutation_llm = hydra.utils.instantiate(cfg.llm_mutation) if cfg.get("llm_mutation") else None
    
    if cfg.algorithm == "reevo":
        from reevo import ReEvo as LHH
    elif cfg.algorithm == "ael":
        from baselines.ael.ga import AEL as LHH
    elif cfg.algorithm == "eoh":
        from baselines.eoh import EoH as LHH
    else:
        raise NotImplementedError

    # Main algorithm
    if cfg.algorithm != "reevo":
        lhh = LHH(cfg, ROOT_DIR, client)
    else:
        lhh = LHH(cfg, ROOT_DIR, client, long_reflector_llm=long_ref_llm, short_reflector_llm=short_ref_llm, 
                  crossover_llm=crossover_llm, mutation_llm=mutation_llm)
        
    best_code_overall, best_code_path_overall = lhh.evolve()
    best_path = best_code_path_overall.replace(".py", ".txt").replace("code", "response")
    logging.info(f"Best Code Path Overall: {print_hyperlink(best_path, best_code_path_overall)}")

    if cfg.algorithm == "reevo" and bool(getattr(lhh, "adapter_enabled", False)):
        adapter_result = getattr(lhh, "adapter_best_result_path", None)
        problem_name = str(cfg.problem.problem_name)
        adapter_cfg = cfg.problem.get("adapter", {})
        if adapter_cfg is not None and hasattr(adapter_cfg, "get"):
            output_mode = str(adapter_cfg.get("output_mode", "minimal"))
        else:
            output_mode = "minimal"
        try:
            if adapter_cfg is not None and hasattr(adapter_cfg, "get"):
                keep_top_k = int(adapter_cfg.get("keep_top_k_solvers", 10))
            else:
                keep_top_k = 10
        except Exception:
            keep_top_k = 10

        if problem_name == "tsp_gls":
            adapter_result = _finalize_adapter_outputs(
                workspace_dir=workspace_dir,
                keep_top_k=max(1, keep_top_k),
                output_mode=output_mode,
            )
            lhh.adapter_best_result_path = adapter_result

        if adapter_result:
            logging.info(f"Adapter summary: {print_hyperlink(adapter_result)}")
            try:
                payload = json.loads(Path(adapter_result).read_text(encoding="utf-8"))
                train_eval = payload.get("train_eval") or payload.get("train_dataset_avg_metrics") or {}
                val_eval = payload.get("val_eval") or payload.get("val_dataset_avg_metrics") or {}
                if train_eval:
                    logging.info(
                        "[Adapter][Train] best_metric=%s avg_gap_pct=%s n_eval=%s",
                        payload.get("best_metric"),
                        train_eval.get("avg_gap_pct", train_eval.get("mean_gap_pct")),
                        train_eval.get("n_eval"),
                    )
                if val_eval:
                    logging.info(
                        "[Adapter][Val] avg_gap_pct=%s n_eval=%s",
                        val_eval.get("avg_gap_pct", val_eval.get("mean_gap_pct")),
                        val_eval.get("n_eval"),
                    )
            except Exception as e:
                logging.warning(f"Failed to read adapter summary json: {e}")
        else:
            logging.warning("Adapter mode enabled but adapter summary file is missing.")
    else:
        # Run validation and redirect stdout to a file "best_code_overall_stdout.txt"
        with open(f"{ROOT_DIR}/problems/{cfg.problem.problem_name}/gpt.py", 'w', encoding="utf-8") as file:
            file.writelines(best_code_overall + '\n')
        test_script = f"{ROOT_DIR}/problems/{cfg.problem.problem_name}/eval.py"
        test_script_stdout = "best_code_overall_val_stdout.txt"
        logging.info(f"Running validation script...: {print_hyperlink(test_script)}")
        with open(test_script_stdout, 'w', encoding="utf-8") as stdout:
            subprocess.run(["python", test_script, "-1", ROOT_DIR, "val"], stdout=stdout)
        logging.info(f"Validation script finished. Results are saved in {print_hyperlink(test_script_stdout)}.")

        # Print the results
        with open(test_script_stdout, 'r', encoding="utf-8") as file:
            for line in file.readlines():
                logging.info(line.strip())

if __name__ == "__main__":
    main()
