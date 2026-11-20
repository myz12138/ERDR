from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _project_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    return config


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _split_erdr(
    instances: list[Any],
    costs: list[float],
    grouped_cfg: dict[str, Any],
) -> tuple[list[Any], list[float], list[list[int]], list[Any], list[float], list[list[int]], dict[str, Any]]:
    group_size = max(1, int(grouped_cfg.get("group_size", 5)))
    n_total = max(1, int(grouped_cfg.get("n_groups_total", 10)))
    n_train = max(0, int(grouped_cfg.get("n_groups_train", 5)))
    n_val = max(0, int(grouped_cfg.get("n_groups_val", 5)))
    if n_train + n_val != n_total or n_train == 0 or n_val == 0:
        raise ValueError("ERDR requires positive train/validation groups summing to n_groups_total")
    required = group_size * n_total
    if len(instances) < required:
        raise ValueError(f"ERDR requires {required} instances, got {len(instances)}")
    seed = int(grouped_cfg.get("split_seed", 42))
    rng = np.random.default_rng(seed)
    chosen = rng.permutation(len(instances))[:required]
    grouped_indices = [chosen[start : start + group_size].tolist() for start in range(0, required, group_size)]
    group_order = rng.permutation(n_total).tolist()

    def collect(group_ids: list[int]) -> tuple[list[Any], list[float], list[list[int]]]:
        flat = [index for group_id in group_ids for index in grouped_indices[group_id]]
        local_groups: list[list[int]] = []
        cursor = 0
        for group_id in group_ids:
            length = len(grouped_indices[group_id])
            local_groups.append(list(range(cursor, cursor + length)))
            cursor += length
        return [instances[i] for i in flat], [costs[i] for i in flat], local_groups

    train_ids = [int(x) for x in group_order[:n_train]]
    val_ids = [int(x) for x in group_order[n_train:]]
    train_instances, train_costs, train_groups = collect(train_ids)
    val_instances, val_costs, val_groups = collect(val_ids)
    metadata = {
        "group_size": group_size,
        "n_groups_total": n_total,
        "n_groups_train": n_train,
        "n_groups_val": n_val,
        "split_seed": seed,
        "train_group_ids": train_ids,
        "validation_group_ids": val_ids,
        "source_indices": [int(x) for x in chosen.tolist()],
    }
    return train_instances, train_costs, train_groups, val_instances, val_costs, val_groups, metadata


def _select_best(method: Any, erdr: bool) -> Any:
    candidates = list(method._population.elitist) or list(method._population.population)
    valid = [candidate for candidate in candidates if candidate.score is not None and np.isfinite(np.asarray(candidate.score)).all()]
    if erdr:
        valid = [
            candidate for candidate in valid
            if getattr(candidate, "validation_score", None) is not None
            and np.isfinite(np.asarray(candidate.validation_score, dtype=float)).all()
        ]
        if not valid:
            raise RuntimeError("no candidate passed ERDR training and validation evaluation")
        return max(valid, key=lambda candidate: float(np.sum(np.asarray(candidate.validation_score, dtype=float))))
    if not valid:
        raise RuntimeError("no candidate passed baseline evaluation")
    return max(valid, key=lambda candidate: float(np.sum(np.asarray(candidate.score, dtype=float))))


def _evaluate_solver(evaluation: Any, solver_code: str) -> tuple[list[float], dict[str, Any]]:
    namespace: dict[str, Any] = {"np": np}
    exec(solver_code, namespace, namespace)
    score = evaluation.evaluate_program(solver_code, namespace["update_edge_distance"])
    return np.asarray(score, dtype=float).tolist(), dict(evaluation.last_metrics)


def run_train(default_config_rel: str, default_mode: str) -> None:
    import argparse

    project_dir = _project_dir()
    repo_root = project_dir.parents[1]
    sys.path.insert(0, str(project_dir))
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(project_dir / default_config_rel))
    args = parser.parse_args()
    cfg = _load_config(args.config)
    method_cfg = dict(cfg.get("method", {}))
    mode = str(method_cfg.get("variant", default_mode)).strip().lower()
    if mode not in {"baseline", "erdr"}:
        raise ValueError("method.variant must be baseline or erdr")

    from llm4ad.base import TextFunctionProgramConverter
    from llm4ad.method.meoh import MEoH, MEoHProfiler
    from llm4ad.task.optimization.tsp_gls_2O import TSPGLSMultiObjectiveEvaluation
    from llm4ad.task.optimization.tsp_gls_2O.evaluation import load_tsp_dataset
    from llm4ad.tools.llm.llm_api_https import HttpsApi

    paths = dict(cfg.get("paths", {}))
    llm_cfg = dict(cfg.get("llm", {}))
    meoh_cfg = dict(cfg.get("meoh", {}))
    evaluation_cfg = dict(cfg.get("evaluation", {}))
    train_path = _resolve(repo_root, str(paths["train_pkl"]))
    output_dir = _resolve(repo_root, str(paths["output_dir"]))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory already contains a run: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_instances, all_costs = load_tsp_dataset(train_path)
    n_eval = min(max(1, int(evaluation_cfg.get("n_instance_eval", 50))), len(all_instances))
    instances = all_instances[:n_eval]
    costs = all_costs[:n_eval]
    common = {
        "perturbation_moves": int(evaluation_cfg.get("gls_perturbation_moves", 2)),
        "iter_limit": int(evaluation_cfg.get("gls_ite_max", 500)),
        "time_limit_per_instance": float(evaluation_cfg.get("gls_time_limit_per_instance", 20.0)),
        "timeout_seconds": float(evaluation_cfg.get("timeout_seconds", 1200.0)),
    }
    validation_evaluation = None
    erdr_metadata: dict[str, Any] | None = None
    if mode == "erdr":
        split = _split_erdr(instances, costs, dict(evaluation_cfg.get("grouped_evaluation", {})))
        train_instances, train_costs, train_groups, val_instances, val_costs, val_groups, erdr_metadata = split
        training_evaluation = TSPGLSMultiObjectiveEvaluation(
            instances=train_instances,
            reference_costs=train_costs,
            mode="erdr",
            group_indices=train_groups,
            **common,
        )
        validation_evaluation = TSPGLSMultiObjectiveEvaluation(
            instances=val_instances,
            reference_costs=val_costs,
            mode="erdr",
            group_indices=val_groups,
            **common,
        )
    else:
        training_evaluation = TSPGLSMultiObjectiveEvaluation(
            instances=instances,
            reference_costs=costs,
            mode="baseline",
            **common,
        )

    host = str(llm_cfg.get("llm_api_host", "")).split("/")[0]
    llm = HttpsApi(
        host=host,
        key=str(llm_cfg.get("llm_api_key", "")),
        model=str(llm_cfg.get("llm_model", "gpt-4o-mini")),
        timeout=float(llm_cfg.get("timeout", 120)),
        temperature=float(llm_cfg.get("llm_temperature", 1.0)),
    )
    num_objs = 4 if mode == "erdr" else 2
    method = MEoH(
        llm=llm,
        evaluation=training_evaluation,
        validation_evaluation=validation_evaluation,
        validation_guided_selection=(mode == "erdr"),
        profiler=MEoHProfiler(log_dir=str(output_dir / "meoh_run"), num_objs=num_objs, log_style="simple"),
        max_sample_nums=int(meoh_cfg.get("max_sample_nums", 1000)),
        max_generations=int(meoh_cfg.get("max_generations", 1000)),
        pop_size=int(meoh_cfg.get("pop_size", 10)),
        selection_num=int(meoh_cfg.get("selection_num", 2)),
        num_samplers=int(meoh_cfg.get("num_samplers", 4)),
        num_evaluators=int(meoh_cfg.get("num_evaluators", 4)),
        num_objs=num_objs,
        debug_mode=bool(meoh_cfg.get("debug_mode", False)),
    )
    method.run()
    best = _select_best(method, erdr=(mode == "erdr"))
    program = TextFunctionProgramConverter.function_to_program(best, training_evaluation.template_program)
    if program is None:
        raise RuntimeError("failed to reconstruct the selected solver")
    solver_code = str(program).rstrip() + "\n"
    best_path = output_dir / "best_solver.py"
    best_path.write_text(solver_code, encoding="utf-8")
    train_score, train_metrics = _evaluate_solver(training_evaluation, solver_code)
    validation_score = None
    validation_metrics = None
    if validation_evaluation is not None:
        validation_score, validation_metrics = _evaluate_solver(validation_evaluation, solver_code)
    result = {
        "method": "MEoH" if mode == "baseline" else "MEoH + ERDR",
        "mode": mode,
        "train_pkl": str(train_path),
        "n_instance_eval": n_eval,
        "best_solver_path": str(best_path),
        "train_score": train_score,
        "train_metrics": train_metrics,
        "validation_score": validation_score,
        "validation_metrics": validation_metrics,
        "erdr": erdr_metadata,
    }
    (output_dir / "final_eval.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
