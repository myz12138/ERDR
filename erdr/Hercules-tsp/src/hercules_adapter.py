from __future__ import annotations

import json
import logging
import os
import pickle
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from omegaconf import DictConfig

from src.hercules import Hercules
from problems.tsp_gls.gen_inst import TSPInstance, generate_datasets, load_dataset
from utils.dds_multiobj import compute_dds_scores, is_feasible_individual, select_population_dds


class HerculesAdapter(Hercules):
    """
    Adapter layer for Hercules on tsp_gls:
    - baseline mode: single-objective mean metric
    - erdr mode: grouped evaluation and validation-guided reselection
    """

    def __init__(
        self,
        cfg: DictConfig,
        root_dir: str,
        generator_llm,
        direction_llm=None,
    ) -> None:
        self._adapter_ready = False
        self.adapter_mode = "baseline"
        self.metric_basis = "gap"
        self.subgroup_eval_mode = "mean_only"
        self.subgroup_tail_basis = "gap"
        self.risk_enabled = False
        self.dds_eps_rel = 1e-3
        self.dds_softmax_tau = 1.0
        self.dds_novelty_weight = 0.05
        self.val_interval = 2
        self.train_pool_size = int(max(int(getattr(cfg, "pop_size", 10)), 1))
        self.train_dataset_path: Optional[Path] = None
        self.val_dataset_path: Optional[Path] = None
        self.train_group_indices: List[List[int]] = []
        self.val_group_indices: List[List[int]] = []
        self.gls_perturbation_moves = 30
        self.gls_iter_limit = 1200
        self.gls_time_limit_per_instance = 5.0
        self.gap_eps = 1.0e-12
        self.n_instance_eval = 0
        self.adapter_dir: Optional[Path] = None
        self.adapter_history_dir: Optional[Path] = None
        self.adapter_eval_dir: Optional[Path] = None
        self.adapter_code_dir: Optional[Path] = None
        self.adapter_log_dir: Optional[Path] = None
        self.adapter_response_dir: Optional[Path] = None
        self._eval_counter = 0
        self._response_counter = 0
        self._generation_id = 0
        super().__init__(cfg, root_dir, generator_llm, direction_llm)
        self._ensure_adapter_ready()
        if self.adapter_mode == "erdr":
            self.population = self._manage_population(
                self.population,
                self.train_pool_size,
                use_multi_objective=(self.subgroup_eval_mode == "mean_variability_tail"),
            )
        else:
            self.population = self._manage_population(self.population, int(max(self.cfg.pop_size, 1)), use_multi_objective=False)

    @staticmethod
    def _normalize_code(code: str) -> str:
        return "\n".join(line.rstrip() for line in str(code).strip().splitlines())

    def _adapter_cfg(self) -> Dict[str, Any]:
        cfg = self.cfg.problem.get("adapter", {})
        if cfg is None:
            return {}
        if hasattr(cfg, "items"):
            return {str(k): v for k, v in cfg.items()}
        if isinstance(cfg, dict):
            return dict(cfg)
        return {}

    @staticmethod
    def _require_concorde_ref(inst: TSPInstance) -> float:
        ref = getattr(inst, "reference_cost", None)
        if ref is None:
            raise ValueError("missing concorde reference_cost in training dataset instance")
        try:
            ref_f = float(ref)
        except Exception as e:
            raise ValueError("invalid concorde reference_cost in training dataset instance") from e
        if (not np.isfinite(ref_f)) or ref_f <= 0.0:
            raise ValueError("non-positive concorde reference_cost in training dataset instance")
        return float(ref_f)

    @classmethod
    def _dump_subset_with_concorde(cls, path: Path, subset: Sequence[TSPInstance]) -> None:
        payload = {
            "coordinate": np.asarray([inst.positions for inst in subset], dtype=np.float64),
            "distance_matrix": np.asarray([inst.distmat for inst in subset], dtype=np.float64),
            "cost_concorde": np.asarray([cls._require_concorde_ref(inst) for inst in subset], dtype=np.float64),
        }
        with path.open("wb") as f:
            pickle.dump(payload, f)

    def _ensure_adapter_ready(self) -> None:
        if self._adapter_ready:
            return

        adapter_cfg = self._adapter_cfg()
        self.adapter_mode = str(adapter_cfg.get("mode", "baseline")).strip().lower()
        if self.adapter_mode not in {"baseline", "erdr"}:
            raise ValueError(f"Invalid adapter.mode: {self.adapter_mode}")

        self.metric_basis = str(adapter_cfg.get("metric_basis", "gap")).strip().lower()
        if self.metric_basis not in {"gap", "cost"}:
            raise ValueError(f"Invalid adapter.metric_basis: {self.metric_basis}")

        self.subgroup_eval_mode = str(adapter_cfg.get("group_evaluation_mode", "mean_only")).strip().lower()
        if self.subgroup_eval_mode not in {"mean_only", "mean_variability_tail"}:
            raise ValueError(f"Invalid adapter.group_evaluation_mode: {self.subgroup_eval_mode}")
        self.subgroup_tail_basis = str(adapter_cfg.get("tail_risk_basis", "gap")).strip().lower()
        if self.subgroup_tail_basis not in {"gap", "time"}:
            raise ValueError(f"Invalid adapter.tail_risk_basis: {self.subgroup_tail_basis}")

        self.n_instance_eval = int(max(int(adapter_cfg.get("n_instance_eval", 0)), 0))
        self.gls_perturbation_moves = int(max(int(adapter_cfg.get("gls_perturbation_moves", 30)), 1))
        self.gls_iter_limit = int(max(int(adapter_cfg.get("gls_iter_limit", 1200)), 1))
        self.gls_time_limit_per_instance = float(max(float(adapter_cfg.get("time_limit_per_instance", 5.0)), 1.0e-6))
        self.gap_eps = float(max(float(adapter_cfg.get("gap_eps", 1.0e-12)), 1.0e-12))

        risk_cfg_raw = adapter_cfg.get("erdr", {})
        risk_cfg = dict(risk_cfg_raw) if isinstance(risk_cfg_raw, dict) or hasattr(risk_cfg_raw, "items") else {}
        self.risk_enabled = bool(risk_cfg.get("enabled", self.adapter_mode == "erdr")) or (
            self.adapter_mode == "erdr"
        )

        self.val_interval = int(max(int(risk_cfg.get("val_interval", 2)), 1))
        self.dds_eps_rel = float(max(float(risk_cfg.get("dds_eps_rel", 1e-3)), 0.0))
        self.dds_softmax_tau = float(max(float(risk_cfg.get("dds_softmax_tau", 1.0)), 1.0e-6))
        self.dds_novelty_weight = float(max(float(risk_cfg.get("dds_novelty_weight", 0.05)), 0.0))

        default_split = "train"
        dataset_path_raw = str(adapter_cfg.get("dataset_path", "") or "").strip()
        if dataset_path_raw:
            dataset_path = Path(dataset_path_raw)
            if not dataset_path.is_absolute():
                # Prefer Hercules-tsp relative path; fallback to workspace root relative path.
                candidate = (Path(self.root_dir) / dataset_path).resolve()
                if not candidate.exists():
                    candidate_parent = (Path(self.root_dir).resolve().parent / dataset_path).resolve()
                    if candidate_parent.exists():
                        candidate = candidate_parent
                dataset_path = candidate
            else:
                dataset_path = dataset_path.resolve()
        else:
            dataset_path = (
                Path(self.root_dir)
                / "problems"
                / self.problem
                / "dataset"
                / f"{default_split}{self.problem_size}_dataset.npy"
            ).resolve()

        if not dataset_path.exists():
            if dataset_path.suffix.lower() == ".npy":
                dataset_dir = dataset_path.parent
                dataset_dir.mkdir(parents=True, exist_ok=True)
                generate_datasets(str(dataset_dir))
            else:
                raise FileNotFoundError(
                    f"Dataset not found: {dataset_path}. "
                    "For custom datasets, please set an existing .npy/.pkl path in adapter.dataset_path."
                )

        data_all = load_dataset(str(dataset_path))
        if not data_all:
            raise RuntimeError(f"Dataset is empty: {dataset_path}")
        n_source = len(data_all)
        if self.n_instance_eval > 0:
            n_source = min(max(self.n_instance_eval, 1), len(data_all))
        data_all = data_all[:n_source]
        for inst in data_all:
            self._require_concorde_ref(inst)

        self.adapter_dir = Path(os.getcwd()).resolve() / "adapter_artifacts"
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        self.adapter_history_dir = self.adapter_dir / "history"
        self.adapter_eval_dir = self.adapter_dir / "eval"
        self.adapter_code_dir = self.adapter_dir / "code"
        self.adapter_log_dir = self.adapter_dir / "logs"
        self.adapter_response_dir = self.adapter_dir / "responses"
        self.adapter_history_dir.mkdir(parents=True, exist_ok=True)
        self.adapter_eval_dir.mkdir(parents=True, exist_ok=True)
        self.adapter_code_dir.mkdir(parents=True, exist_ok=True)
        self.adapter_log_dir.mkdir(parents=True, exist_ok=True)
        self.adapter_response_dir.mkdir(parents=True, exist_ok=True)

        if self.risk_enabled:
            group_size = int(max(int(risk_cfg.get("group_size", 5)), 1))
            n_groups_total = int(max(int(risk_cfg.get("n_groups_total", 12)), 1))
            n_groups_train = int(max(int(risk_cfg.get("n_groups_train", 8)), 0))
            n_groups_val = int(max(int(risk_cfg.get("n_groups_val", 4)), 0))
            split_seed = int(risk_cfg.get("split_seed", 42))
            if n_groups_train <= 0:
                raise RuntimeError("adapter.erdr requires n_groups_train to be greater than zero.")
            if n_groups_train + n_groups_val != n_groups_total:
                raise RuntimeError("adapter.erdr requires n_groups_train + n_groups_val to equal n_groups_total.")

            required_total = int(group_size * n_groups_total)
            if len(data_all) < required_total:
                raise RuntimeError(
                    f"adapter.erdr needs at least {required_total} instances, got {len(data_all)}."
                )
            rng = np.random.default_rng(split_seed)
            selected_indices = rng.permutation(len(data_all)).tolist()[:required_total]
            selected = [data_all[i] for i in selected_indices]

            all_groups: List[List[int]] = []
            start = 0
            while start < required_total:
                end = min(start + group_size, required_total)
                all_groups.append(list(range(start, end)))
                start = end
            if len(all_groups) != n_groups_total or any(len(g) != group_size for g in all_groups):
                raise RuntimeError("failed to construct contiguous groups for adapter.erdr.")

            group_id_perm = rng.permutation(n_groups_total).tolist()
            train_group_ids = [int(x) for x in group_id_perm[:n_groups_train]]
            val_group_ids = [int(x) for x in group_id_perm[n_groups_train : n_groups_train + n_groups_val]]

            train_flat: List[int] = []
            train_groups_local: List[List[int]] = []
            cursor = 0
            for gid in train_group_ids:
                idxs = all_groups[gid]
                train_flat.extend(idxs)
                train_groups_local.append(list(range(cursor, cursor + len(idxs))))
                cursor += len(idxs)

            val_flat: List[int] = []
            val_groups_local: List[List[int]] = []
            cursor = 0
            for gid in val_group_ids:
                idxs = all_groups[gid]
                val_flat.extend(idxs)
                val_groups_local.append(list(range(cursor, cursor + len(idxs))))
                cursor += len(idxs)

            train_subset = [selected[i] for i in train_flat]
            val_subset = [selected[i] for i in val_flat]
            if not train_subset:
                raise RuntimeError("adapter.erdr training subset is empty. Increase n_groups_train.")

            self.train_dataset_path = self.adapter_dir / "dataset_train.pkl"
            self._dump_subset_with_concorde(self.train_dataset_path, train_subset)
            self.train_group_indices = train_groups_local if train_groups_local else [list(range(len(train_subset)))]
            if n_groups_val > 0 and val_subset:
                self.val_dataset_path = self.adapter_dir / "dataset_val.pkl"
                self._dump_subset_with_concorde(self.val_dataset_path, val_subset)
                self.val_group_indices = val_groups_local if val_groups_local else [list(range(len(val_subset)))]
            else:
                self.val_dataset_path = None
                self.val_group_indices = []

            pop_multiplier = int(max(int(risk_cfg.get("train_pool_multiplier", 2)), 1))
            self.train_pool_size = int(max(int(self.cfg.pop_size) * pop_multiplier, int(self.cfg.pop_size)))
        else:
            self.train_dataset_path = self.adapter_dir / "dataset_train.pkl"
            self._dump_subset_with_concorde(self.train_dataset_path, data_all)
            self.val_dataset_path = None
            self.train_group_indices = [list(range(len(data_all)))]
            self.val_group_indices = []
            self.train_pool_size = int(max(int(self.cfg.pop_size), 1))

        manifest = {
            "mode": self.adapter_mode,
            "erdr_enabled": bool(self.risk_enabled),
            "metric_basis": self.metric_basis,
            "group_evaluation_mode": self.subgroup_eval_mode,
            "tail_risk_basis": self.subgroup_tail_basis,
            "selection_method": "dds_prob_rank",
            "split_seed": (int(risk_cfg.get("split_seed", 42)) if self.risk_enabled else None),
            "gls_time_limit_per_instance": float(self.gls_time_limit_per_instance),
            "dataset_path_source": str(dataset_path),
            "train_dataset_path": str(self.train_dataset_path) if self.train_dataset_path else None,
            "val_dataset_path": str(self.val_dataset_path) if self.val_dataset_path else None,
            "n_train_instances": int(sum(len(g) for g in self.train_group_indices)),
            "n_val_instances": int(sum(len(g) for g in self.val_group_indices)),
            "train_pool_size": int(self.train_pool_size),
            "val_interval": int(self.val_interval),
            "dds_eps_rel": float(self.dds_eps_rel),
            "dds_softmax_tau": float(self.dds_softmax_tau),
            "dds_novelty_weight": float(self.dds_novelty_weight),
        }
        (self.adapter_dir / "adapter_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._adapter_ready = True

    def _next_eval_stem(self, split: str, response_id: int) -> str:
        self._eval_counter += 1
        split_tag = str(split).strip().lower()
        return (
            f"c{int(self._eval_counter):05d}_"
            f"i{int(self.iteration):03d}_"
            f"g{int(self._generation_id):03d}_"
            f"{split_tag}_r{int(response_id):03d}"
        )

    def response_to_individual(self, response: str, response_id: int, file_name: str = None) -> dict:
        self._ensure_adapter_ready()
        assert self.adapter_response_dir is not None
        self._response_counter += 1
        resp_name = f"response_{int(self._response_counter):05d}.txt"
        response_path = self.adapter_response_dir / resp_name
        response_path.write_text(str(response or "") + "\n", encoding="utf-8")

        from utils.utils import extract_code_from_generator

        code = extract_code_from_generator(response)
        individual = {
            "stdout_filepath": "",
            "code_path": "",
            "code": code,
            "response_id": int(response_id),
            "response_path": str(response_path.resolve()),
        }
        return individual

    def mark_invalid_individual(self, individual: dict, traceback_msg: str) -> dict:
        individual["exec_success"] = False
        individual["obj"] = float("inf")
        individual["traceback_msg"] = str(traceback_msg)
        individual["eval_info"] = {
            "status": "invalid",
            "error": str(traceback_msg),
            "selected_metric": float("inf"),
            "mean_cost": float("inf"),
            "mean_gap": float("inf"),
            "mean_gap_pct": float("inf"),
            "mean_time_sec": 0.0,
            "n_invalid": 1,
            "objective_vector": [float("inf"), float("inf"), float("inf")],
        }
        individual["objective_vector"] = [float("inf"), float("inf"), float("inf")]
        return individual

    def _eval_meta(self, split: str) -> tuple[Path, List[List[int]]]:
        split_norm = str(split).strip().lower()
        if split_norm == "val":
            if self.val_dataset_path is None:
                raise RuntimeError("validation dataset is unavailable for current adapter mode.")
            return self.val_dataset_path, self.val_group_indices
        return self.train_dataset_path, self.train_group_indices  # type: ignore[return-value]

    def _run_code(self, individual: dict, response_id: int, split: str) -> tuple[subprocess.Popen, Path, Path, Path]:
        self._ensure_adapter_ready()
        assert self.adapter_code_dir is not None
        assert self.adapter_eval_dir is not None
        assert self.adapter_log_dir is not None
        dataset_path, group_indices = self._eval_meta(split)
        stem = self._next_eval_stem(split=split, response_id=response_id)
        code_path = self.adapter_code_dir / f"{stem}.py"
        stdout_path = (self.adapter_log_dir / f"{stem}.log").resolve()
        eval_json_path = self.adapter_eval_dir / f"{stem}.json"

        code_text = str(individual.get("code", "") or "")
        code_path.write_text(code_text + ("\n" if not code_text.endswith("\n") else ""), encoding="utf-8")
        individual["code_path"] = str(code_path)
        individual["stdout_filepath"] = str(stdout_path)

        eval_file_path = Path(self.root_dir) / "problems" / self.problem / "eval_adapter.py"
        cmd = [
            sys.executable,
            "-u",
            str(eval_file_path),
            "--dataset-path",
            str(dataset_path),
            "--code-path",
            str(code_path),
            "--output-json",
            str(eval_json_path),
            "--metric-basis",
            str(self.metric_basis),
            "--subgroup-mode",
            str(self.subgroup_eval_mode),
            "--subgroup-tail-basis",
            str(self.subgroup_tail_basis),
            "--group-indices-json",
            json.dumps(group_indices, ensure_ascii=False),
            "--perturbation-moves",
            str(self.gls_perturbation_moves),
            "--iter-limit",
            str(self.gls_iter_limit),
            "--time-limit-per-instance",
            str(self.gls_time_limit_per_instance),
            "--gap-eps",
            str(self.gap_eps),
        ]
        with open(stdout_path, "w", encoding="utf-8") as f:
            process = subprocess.Popen(cmd, stdout=f, stderr=f)
        return process, eval_json_path, code_path, stdout_path

    def evaluate_population(self, population: List[dict], split: str = "train", count_fe: bool = True) -> List[dict]:
        self._ensure_adapter_ready()
        out: List[dict] = []
        for response_id, ind in enumerate(population):
            row = ind
            if count_fe:
                self.function_evals += 1
            if row.get("code") is None:
                out.append(self.mark_invalid_individual(row, "Invalid response/code."))
                continue

            logging.info(f"Iteration {self.iteration}: Running Code {response_id} [{split}]")
            try:
                proc, eval_json_path, code_path, stdout_path = self._run_code(row, response_id, split)
                proc.communicate()

                if not eval_json_path.exists():
                    out.append(self.mark_invalid_individual(row, "Missing eval_adapter output json."))
                    continue

                eval_info = json.loads(eval_json_path.read_text(encoding="utf-8"))
                status = str(eval_info.get("status", "invalid"))
                metric = float(eval_info.get("selected_metric", float("inf")))
                exec_success = bool(status == "ok" and np.isfinite(metric))

                row["code_path"] = str(code_path)
                row["stdout_filepath"] = str(stdout_path)
                row["eval_info"] = eval_info
                row["obj"] = float(metric if exec_success else float("inf"))
                row["exec_success"] = bool(exec_success)
                row["objective_vector"] = list(eval_info.get("objective_vector", [row["obj"], 0.0, row["obj"]]))
                if not exec_success:
                    row["traceback_msg"] = str(eval_info.get("error", "Invalid evaluation result."))
                out.append(row)
            except Exception as e:
                out.append(self.mark_invalid_individual(row, str(e)))
        return out

    def _manage_population(self, population: Sequence[dict], size: int, use_multi_objective: bool) -> List[dict]:
        target = int(max(size, 0))
        if target <= 0:
            return []
        valid = [ind for ind in population if bool(ind.get("exec_success", False))]
        if not valid:
            return []

        dedup: Dict[str, dict] = {}
        for ind in valid:
            code_norm = self._normalize_code(str(ind.get("code", "")))
            if not code_norm:
                continue
            prev = dedup.get(code_norm)
            if prev is None:
                dedup[code_norm] = ind
                continue
            if float(ind.get("obj", float("inf"))) < float(prev.get("obj", float("inf"))):
                dedup[code_norm] = ind
        unique = list(dedup.values())
        if not unique:
            return []

        if use_multi_objective:
            selected = select_population_dds(
                unique,
                size=target,
                eps_rel=self.dds_eps_rel,
                novelty_weight=self.dds_novelty_weight,
            )
            if selected:
                return selected

        unique.sort(key=lambda x: float(x.get("obj", float("inf"))))
        return unique[: min(target, len(unique))]

    def rank_select(self, population: list[dict]) -> list[dict] | None:
        valid = [ind for ind in population if is_feasible_individual(ind)]
        if len(valid) < 2:
            return None

        scores, _, _, _ = compute_dds_scores(
            population=valid,
            eps_rel=self.dds_eps_rel,
            novelty_weight=self.dds_novelty_weight,
        )
        tau = float(max(self.dds_softmax_tau, 1.0e-6))
        logits = np.asarray(scores, dtype=float) / tau
        if logits.size <= 0:
            return None
        logits = logits - float(np.max(logits))
        weights = np.exp(logits)
        if (not np.isfinite(weights).all()) or float(np.sum(weights)) <= 0.0:
            probs = [1.0 / len(valid)] * len(valid)
        else:
            probs = (weights / float(np.sum(weights))).tolist()

        selected_population = []
        trial = 0
        target_size = int(max(2 * int(self.cfg.pop_size), 2))
        while len(selected_population) < target_size:
            trial += 1
            parent_indices = random.choices(range(len(valid)), weights=probs, k=2)
            if parent_indices[0] == parent_indices[1]:
                if trial > 2000:
                    break
                continue
            p1 = valid[parent_indices[0]]
            p2 = valid[parent_indices[1]]
            if float(p1.get("obj", float("inf"))) == float(p2.get("obj", float("inf"))):
                if trial > 2000:
                    break
                continue
            selected_population.extend([p1, p2])
            if trial > 2000:
                break

        if len(selected_population) < target_size:
            return None
        return selected_population[:target_size]

    def _validation_select_for_next_train(self, population: List[dict], generation_id: int) -> List[dict]:
        if not self.risk_enabled or self.val_dataset_path is None:
            return population
        if (int(generation_id) % int(self.val_interval)) != 0:
            return population

        pop_eval_val = self.evaluate_population(population, split="val", count_fe=False)
        keep_size = int(max(self.cfg.pop_size, 1))
        pop_keep_val = self._manage_population(
            pop_eval_val,
            keep_size,
            use_multi_objective=(self.subgroup_eval_mode == "mean_variability_tail"),
        )
        if not pop_keep_val:
            return population

        pop_keep_train = self.evaluate_population(pop_keep_val, split="train", count_fe=False)
        pop_keep_train = self._manage_population(
            pop_keep_train,
            keep_size,
            use_multi_objective=(self.subgroup_eval_mode == "mean_variability_tail"),
        )
        if not pop_keep_train:
            return population

        assert self.adapter_history_dir is not None
        log_row = {
            "generation": int(generation_id),
            "validation_population_size": int(len(pop_eval_val)),
            "selected_size": int(len(pop_keep_train)),
            "selected_objectives_on_validation": [float(ind.get("obj", 1e10)) for ind in pop_keep_val],
            "selected_objectives_on_train": [float(ind.get("obj", 1e10)) for ind in pop_keep_train],
        }
        (self.adapter_history_dir / f"validation_selection_generation_{generation_id}.json").write_text(
            json.dumps(log_row, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return pop_keep_train

    def update_iter(self) -> None:
        population = self.population
        for response_id in range(len(population)):
            logging.info(
                f"Iteration {self.iteration}, response_id {response_id}: Objective value: {population[response_id].get('obj', float('inf'))}"
            )

        valid = [
            (i, float(ind["obj"]))
            for i, ind in enumerate(population)
            if bool(ind.get("exec_success", False)) and np.isfinite(float(ind.get("obj", float("inf"))))
        ]
        if not valid:
            logging.warning(f"Iteration {self.iteration}: no feasible individuals.")
            self.iteration += 1
            return

        indices, objs = zip(*valid)
        best_obj, best_sample_idx = min(objs), indices[int(np.argmin(objs))]

        if self.best_obj_overall is None or best_obj < self.best_obj_overall:
            self.best_obj_overall = float(best_obj)
            self.best_code_overall = population[best_sample_idx].get("code")
            self.best_code_path_overall = population[best_sample_idx].get("code_path")

        if self.elitist is None or best_obj < float(self.elitist.get("obj", float("inf"))):
            self.elitist = dict(population[best_sample_idx])
            logging.info(f"Iteration {self.iteration}: Elitist: {self.elitist['obj']}")

        logging.info(f"Iteration {self.iteration} finished...")
        logging.info(f"Best obj: {self.best_obj_overall}, Best Code Path: {self.best_code_path_overall}")
        logging.info(f"Function Evals: {self.function_evals}")
        self.iteration += 1

    def _record_best_result(self) -> None:
        if self.best_code_overall is None:
            return
        single = [{"code": self.best_code_overall, "response_id": -1}]
        train_eval = self.evaluate_population(single, split="train", count_fe=False)
        train_info = train_eval[0].get("eval_info") if train_eval else None

        val_info = None
        if self.val_dataset_path is not None:
            val_eval = self.evaluate_population(single, split="val", count_fe=False)
            val_info = val_eval[0].get("eval_info") if val_eval else None

        result = {
            "mode": self.adapter_mode,
            "metric_basis": self.metric_basis,
            "group_evaluation_mode": self.subgroup_eval_mode,
            "tail_risk_basis": self.subgroup_tail_basis,
            "best_obj_overall": self.best_obj_overall,
            "best_code_path_overall": self.best_code_path_overall,
            "function_evals": int(self.function_evals),
            "train_eval": train_info,
            "val_eval": val_info,
        }
        assert self.adapter_dir is not None
        (self.adapter_dir / "best_result_adapter.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def evolve(self):
        while self.function_evals < self.cfg.max_fe:
            self._generation_id += 1
            core_abstraction = ""
            if all([not bool(individual.get("exec_success", False)) for individual in self.population]):
                log_hint = (
                    str(self.adapter_log_dir.resolve())
                    if self.adapter_log_dir is not None
                    else os.getcwd()
                )
                raise RuntimeError(
                    f"All individuals are invalid. Please check adapter logs in {log_hint}."
                )

            population_to_select = (
                self.population
                if (self.elitist is None or self.elitist in self.population)
                else [self.elitist] + self.population
            )
            selected_population = self.rank_select(population_to_select)
            if selected_population is None:
                raise RuntimeError("Selection failed. Please check the population.")

            if self.function_evals <= self.lamda * self.cfg.max_fe:
                core_abstraction = self.core_abstraction()

            short_term_direction_tuple = self.short_term_direction(selected_population, core_abstraction)
            crossed_population = self.crossover(short_term_direction_tuple)
            self.population = self.evaluate_population(crossed_population, split="train", count_fe=True)
            self.update_iter()

            self.long_term_direction([response for response in short_term_direction_tuple[0]], core_abstraction)
            mutated_population = self.mutate()
            self.population.extend(self.evaluate_population(mutated_population, split="train", count_fe=True))

            if self.adapter_mode == "erdr":
                self.population = self._validation_select_for_next_train(self.population, self._generation_id)
            self.update_iter()

        self._record_best_result()
        return self.best_code_overall, self.best_code_path_overall
