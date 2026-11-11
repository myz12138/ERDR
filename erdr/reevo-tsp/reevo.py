from __future__ import annotations

import ast
from collections import Counter
from typing import Any, Optional
import json
import logging
import os
import pickle
from pathlib import Path
import random
import re
import subprocess
import sys
import uuid

import numpy as np
from omegaconf import DictConfig

from utils.utils import *
from utils.llm_client.base import BaseClient


_TOKEN_PATTERN = re.compile(r"[A-Za-z_]\w+|\d+")


class ReEvo:
    def __init__(
        self,
        cfg: DictConfig,
        root_dir: str,
        generator_llm: BaseClient,
        reflector_llm: Optional[BaseClient] = None,
        # Support setting different LLMs for each of the four operators:
        # Short-term Reflection, Long-term Reflection, Crossover, Mutation
        short_reflector_llm: Optional[BaseClient] = None,
        long_reflector_llm: Optional[BaseClient] = None,
        crossover_llm: Optional[BaseClient] = None,
        mutation_llm: Optional[BaseClient] = None,
    ) -> None:
        self.cfg = cfg
        self.generator_llm = generator_llm
        self.reflector_llm = reflector_llm or generator_llm

        self.short_reflector_llm = short_reflector_llm or self.reflector_llm
        self.long_reflector_llm = long_reflector_llm or self.reflector_llm
        self.crossover_llm = crossover_llm or generator_llm
        self.mutation_llm = mutation_llm or generator_llm

        self.root_dir = root_dir

        self.mutation_rate = cfg.mutation_rate
        self.iteration = 0
        self.function_evals = 0
        self.elitist = None
        self.long_term_reflection_str = ""
        self.best_obj_overall = None
        self.best_code_overall = None
        self.best_code_path_overall = None

        # Adapter mode defaults (only active for tsp_gls with cfg.problem.adapter.enabled=true).
        self.adapter_enabled = False
        self._adapter_mode = "static"
        self._adapter_metric_basis = "gap"
        self._adapter_subgroup_eval_mode = "mean_only"
        self._adapter_subgroup_tail_basis = "time"
        self._adapter_gap_eps = 1.0e-12
        self._adapter_n_instance_eval = 0
        self._adapter_perturbation_moves = 30
        self._adapter_iter_limit = 1200
        self._adapter_risk_weight_mean = 1.0
        self._adapter_risk_weight_std = 0.3
        self._adapter_risk_weight_worst = 0.2
        self._adapter_time_limit_per_instance = 5.0
        self._adapter_eval_process_timeout_sec = 0.0
        self._adapter_jit_enable_kernels = True
        self._adapter_jit_numba_cache = True
        self._adapter_jit_warmup = True
        self._adapter_eva_numba_decorator = False
        self._adapter_jit_sanitize_guide = True
        self._adapter_jit_sanitize_strict = False
        self._adapter_dominance_eps = 1.0e-12
        self._adapter_dissimilarity_metric = "l2"
        self._adapter_dds_eps_rel = 1.0e-3
        self._adapter_dds_softmax_tau = 1.0
        self._adapter_dds_novelty_weight = 0.05
        self._adapter_train_keep_ratio = 0.5
        self._adapter_train_keep_min = 1
        self._adapter_train_keep_max = 64
        self._adapter_val_keep_ratio = 0.5
        self._adapter_val_keep_min = 1
        self._adapter_val_keep_max = 64
        self._adapter_parent_min = 1
        self._adapter_parent_max = 16
        self._adapter_train_pool_size = int(max(int(self.cfg.pop_size), 1))
        self._adapter_generation_id = 0
        self._adapter_eval_call_id = 0
        self._adapter_has_val = False
        self._adapter_train_dataset_path: Optional[Path] = None
        self._adapter_val_dataset_path: Optional[Path] = None
        self._adapter_train_group_indices: list[list[int]] = []
        self._adapter_val_group_indices: list[list[int]] = []
        self._adapter_dir: Optional[Path] = None
        self._adapter_code_dir: Optional[Path] = None
        self._adapter_eval_dir: Optional[Path] = None
        self._adapter_history_dir: Optional[Path] = None
        self._adapter_eval_script: Optional[Path] = None
        self.adapter_best_result_path: Optional[Path] = None
        self._adapter_last_eval_rows: list[dict] = []
        self._adapter_best_record: Optional[dict] = None

        self.init_prompt()
        self._init_adapter_mode()
        self.init_population()

    @staticmethod
    def _to_int(value: Any, default: int, min_value: Optional[int] = None) -> int:
        try:
            out = int(value)
        except Exception:
            out = int(default)
        if min_value is not None:
            out = max(int(min_value), out)
        return int(out)

    @staticmethod
    def _to_float(value: Any, default: float, min_value: Optional[float] = None) -> float:
        try:
            out = float(value)
        except Exception:
            out = float(default)
        if min_value is not None:
            out = max(float(min_value), out)
        return float(out)

    @staticmethod
    def _parse_risk_objective_weights(value: Any) -> tuple[float, float, float]:
        default = (1.0, 0.3, 0.2)
        if isinstance(value, dict):
            mean_w = ReEvo._to_float(value.get("mean", default[0]), default[0], min_value=0.0)
            std_w = ReEvo._to_float(value.get("std", default[1]), default[1], min_value=0.0)
            worst_w = ReEvo._to_float(value.get("worst", default[2]), default[2], min_value=0.0)
            return float(mean_w), float(std_w), float(worst_w)
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            mean_w = ReEvo._to_float(value[0], default[0], min_value=0.0)
            std_w = ReEvo._to_float(value[1], default[1], min_value=0.0)
            worst_w = ReEvo._to_float(value[2], default[2], min_value=0.0)
            return float(mean_w), float(std_w), float(worst_w)
        return default

    @staticmethod
    def _build_balanced_groups(n_items: int, n_groups_total: int) -> list[list[int]]:
        n = int(max(n_items, 0))
        g = int(max(n_groups_total, 1))
        if n <= 0:
            return [[] for _ in range(g)]
        if n < g:
            raise ValueError(f"cannot split {n} items into {g} non-empty groups")
        base = n // g
        rem = n % g
        out: list[list[int]] = []
        cursor = 0
        for gid in range(g):
            size = base + (1 if gid < rem else 0)
            out.append(list(range(cursor, cursor + size)))
            cursor += size
        return out

    def init_prompt(self) -> None:
        self.problem = self.cfg.problem.problem_name
        self.problem_desc = self.cfg.problem.description
        self.problem_size = self.cfg.problem.problem_size
        self.func_name = self.cfg.problem.func_name
        self.obj_type = self.cfg.problem.obj_type
        self.problem_type = self.cfg.problem.problem_type

        logging.info("Problem: " + self.problem)
        logging.info("Problem description: " + self.problem_desc)
        logging.info("Function name: " + self.func_name)

        self.prompt_dir = f"{self.root_dir}/prompts"
        self.output_file = f"{self.root_dir}/problems/{self.problem}/gpt.py"

        # Loading all text prompts
        # Problem-specific prompt components
        prompt_path_suffix = "_black_box" if self.problem_type == "black_box" else ""
        problem_prompt_path = f"{self.prompt_dir}/{self.problem}{prompt_path_suffix}"
        self.seed_func = file_to_string(f"{problem_prompt_path}/seed_func.txt")
        self.func_signature = file_to_string(f"{problem_prompt_path}/func_signature.txt")
        self.func_desc = file_to_string(f"{problem_prompt_path}/func_desc.txt")
        if os.path.exists(f"{problem_prompt_path}/external_knowledge.txt"):
            self.external_knowledge = file_to_string(f"{problem_prompt_path}/external_knowledge.txt")
            self.long_term_reflection_str = self.external_knowledge
        else:
            self.external_knowledge = ""

        # Common prompts
        self.system_generator_prompt = file_to_string(f"{self.prompt_dir}/common/system_generator.txt")
        self.system_reflector_prompt = file_to_string(f"{self.prompt_dir}/common/system_reflector.txt")
        self.user_reflector_st_prompt = (
            file_to_string(f"{self.prompt_dir}/common/user_reflector_st.txt")
            if self.problem_type != "black_box"
            else file_to_string(f"{self.prompt_dir}/common/user_reflector_st_black_box.txt")
        )
        self.user_reflector_lt_prompt = file_to_string(f"{self.prompt_dir}/common/user_reflector_lt.txt")
        self.crossover_prompt = file_to_string(f"{self.prompt_dir}/common/crossover.txt")
        self.mutation_prompt = file_to_string(f"{self.prompt_dir}/common/mutation.txt")
        self.user_generator_prompt = file_to_string(f"{self.prompt_dir}/common/user_generator.txt").format(
            func_name=self.func_name,
            problem_desc=self.problem_desc,
            func_desc=self.func_desc,
        )
        self.seed_prompt = file_to_string(f"{self.prompt_dir}/common/seed.txt").format(
            seed_func=self.seed_func,
            func_name=self.func_name,
        )

        # Flag to print prompts
        self.print_crossover_prompt = True
        self.print_mutate_prompt = True
        self.print_short_term_reflection_prompt = True
        self.print_long_term_reflection_prompt = True

    def _resolve_adapter_dataset_path(self, raw_path: str) -> Path:
        text = str(raw_path or "").strip()
        if text:
            dataset_path = Path(text)
            if not dataset_path.is_absolute():
                dataset_path = (Path(self.root_dir) / dataset_path).resolve()
            else:
                dataset_path = dataset_path.resolve()
            return dataset_path
        if self.problem != "tsp_gls":
            raise ValueError(f"adapter mode supports only tsp_gls, got: {self.problem}")
        filename = f"train{self.problem_size}_dataset.pkl"
        return (Path(self.root_dir) / "problems" / self.problem / "dataset" / filename).resolve()

    @staticmethod
    def _has_nonempty_path(value: Any) -> bool:
        text = str(value or "").strip()
        if not text:
            return False
        return text.lower() not in {"null", "none"}

    def _resolve_adapter_dataset_paths(self, raw_paths: Any, raw_single_path: Any) -> list[Path]:
        dataset_paths: list[Path] = []

        if isinstance(raw_paths, (list, tuple)):
            for item in raw_paths:
                if not self._has_nonempty_path(item):
                    continue
                dataset_paths.append(self._resolve_adapter_dataset_path(str(item)))
        elif self._has_nonempty_path(raw_paths):
            dataset_paths.append(self._resolve_adapter_dataset_path(str(raw_paths)))

        if dataset_paths:
            return dataset_paths
        return [self._resolve_adapter_dataset_path(str(raw_single_path or ""))]

    @staticmethod
    def _safe_concorde_ref(value: Any) -> float | None:
        try:
            ref = float(value)
        except Exception:
            return None
        if (not np.isfinite(ref)) or ref <= 0.0:
            return None
        return float(ref)

    def _concorde_sidecar_candidates(self, dataset_path: Path) -> list[Path]:
        sidecar_name = f"{dataset_path.stem}_concorde_full.json"
        return [
            dataset_path.with_name(sidecar_name),
            dataset_path.parent / "concorde" / sidecar_name,
        ]

    def _load_concorde_sidecar_refs(self, dataset_path: Path, expected_len: int) -> Optional[np.ndarray]:
        target_n = int(max(int(expected_len), 0))
        if target_n <= 0:
            return np.asarray([], dtype=float)
        for sidecar in self._concorde_sidecar_candidates(dataset_path):
            if not sidecar.exists():
                continue
            try:
                payload = json.loads(sidecar.read_text(encoding="utf-8"))
                records = payload.get("records", None)
                if not isinstance(records, list) or len(records) < target_n:
                    continue
                out: list[float] = []
                ok = True
                for i in range(target_n):
                    row = records[i]
                    if not isinstance(row, dict) or str(row.get("status", "")) != "ok":
                        ok = False
                        break
                    ref = self._safe_concorde_ref(row.get("opt_float_from_tour"))
                    if ref is None:
                        ok = False
                        break
                    out.append(float(ref))
                if ok and len(out) == target_n:
                    return np.asarray(out, dtype=float)
            except Exception:
                continue
        return None

    def _resolve_tsp_concorde_refs(self, packed: dict[str, Any], dataset_path: Path, n_rows: int) -> np.ndarray:
        target_n = int(max(int(n_rows), 0))
        refs: Optional[np.ndarray] = None
        source_name = ""

        for key in ("cost_concorde", "reference_cost"):
            raw = packed.get(key)
            if raw is None:
                continue
            arr = np.asarray(raw, dtype=float).reshape(-1)
            if arr.size < target_n:
                raise ValueError(
                    f"embedded concorde field '{key}' has insufficient length in {dataset_path}: "
                    f"expected >= {target_n}, got {arr.size}."
                )
            refs = np.asarray(arr[:target_n], dtype=float)
            source_name = key
            break

        if refs is None:
            refs = self._load_concorde_sidecar_refs(dataset_path, expected_len=target_n)
            source_name = "sidecar"

        if refs is None:
            project_name = Path(__file__).resolve().parent.name
            raise ValueError(
                "Concorde reference costs are required for tsp_gls adapter. "
                f"Missing in dataset {dataset_path} and no usable sidecar "
                f"`{project_name}/outputs/concorde_opt/{dataset_path.stem}_concorde_full.json` found."
            )

        bad = np.where((~np.isfinite(refs)) | (refs <= 0.0))[0]
        if bad.size > 0:
            preview = ",".join(str(int(i)) for i in bad[:10])
            raise ValueError(
                f"Invalid concorde references from {source_name} for {dataset_path}: "
                f"{int(bad.size)}/{target_n} bad items (first indices: {preview})."
            )
        return np.asarray(refs, dtype=float)

    def _load_adapter_cases(self, dataset_path: Path) -> list[dict[str, Any]]:
        suffix = dataset_path.suffix.lower()
        if self.problem in {"tsp_gls"}:
            if suffix in {".pkl", ".pickle"}:
                with open(dataset_path, "rb") as f:
                    packed = pickle.load(f)
                if not isinstance(packed, dict) or "distance_matrix" not in packed:
                    raise ValueError(f"unsupported pkl dataset format for adapter: {dataset_path}")

                distance_raw = packed["distance_matrix"]
                if isinstance(distance_raw, np.ndarray) and distance_raw.ndim == 3:
                    distance_list = [np.asarray(distance_raw[i], dtype=np.float32) for i in range(distance_raw.shape[0])]
                else:
                    distance_list = [np.asarray(x, dtype=np.float32) for x in distance_raw]

                coords_raw = packed.get("coordinate")
                n = len(distance_list)
                refs = self._resolve_tsp_concorde_refs(packed, dataset_path, n_rows=n)
                coords = np.asarray(coords_raw, dtype=np.float32) if coords_raw is not None else None

                cases: list[dict[str, Any]] = []
                for i in range(n):
                    dist = np.asarray(distance_list[i], dtype=np.float32)
                    if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
                        raise ValueError(f"invalid distance matrix at index {i}: {dist.shape}")
                    ref = float(refs[i])
                    case: dict[str, Any] = {
                        "distance_matrix": dist,
                        "cost_concorde": ref,
                        "reference_cost": ref,
                    }
                    if coords is not None and i < coords.shape[0]:
                        case["coordinate"] = np.asarray(coords[i], dtype=np.float32)
                    cases.append(case)
                return cases

            raise ValueError(
                f"tsp_gls adapter dataset must be .pkl/.pickle with concorde references, got: {dataset_path.suffix}"
            )

        raise ValueError(f"unsupported adapter problem for dataset loading: {self.problem}")

    def _write_adapter_subset(self, cases: list[dict[str, Any]], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = output_path.suffix.lower()
        if self.problem in {"tsp_gls"}:
            if suffix not in {".pkl", ".pickle"}:
                raise ValueError(
                    "tsp_gls adapter subset must be .pkl/.pickle so concorde references can be embedded."
                )
            payload: dict[str, Any] = {
                "distance_matrix": np.asarray(
                    [np.asarray(case["distance_matrix"], dtype=np.float32) for case in cases],
                    dtype=np.float32,
                )
            }
            refs: list[float] = []
            for i, case in enumerate(cases):
                ref = self._safe_concorde_ref(case.get("cost_concorde", case.get("reference_cost")))
                if ref is None:
                    raise ValueError(
                        f"Missing/invalid concorde reference for tsp_gls adapter subset at index {i}."
                    )
                refs.append(float(ref))
            payload["cost_concorde"] = np.asarray(refs, dtype=float)
            payload["reference_cost"] = np.asarray(refs, dtype=float)
            if cases and all(("coordinate" in case) for case in cases):
                payload["coordinate"] = np.asarray(
                    [np.asarray(case["coordinate"], dtype=np.float32) for case in cases],
                    dtype=np.float32,
                )
            with open(output_path, "wb") as f:
                pickle.dump(payload, f)
            return

        raise ValueError(f"unsupported adapter subset problem: {self.problem}")

    def _init_adapter_mode(self) -> None:
        adapter_cfg = self.cfg.problem.get("adapter", None)
        if adapter_cfg is None:
            return
        if not bool(adapter_cfg.get("enabled", False)):
            return

        if self.problem != "tsp_gls":
            raise ValueError("adapter mode supports only tsp_gls.")

        self.adapter_enabled = True
        self._adapter_mode = str(adapter_cfg.get("mode", "static_risk2")).strip().lower()
        if self._adapter_mode not in {"static", "static_risk2"}:
            raise ValueError("adapter.mode must be one of: static, static_risk2")
        self._adapter_metric_basis = str(adapter_cfg.get("metric_basis", "gap")).strip().lower()
        if self._adapter_metric_basis not in {"gap", "cost"}:
            raise ValueError("adapter.metric_basis must be one of: gap, cost")
        self._adapter_subgroup_eval_mode = str(adapter_cfg.get("subgroup_eval_mode", "mean_std_worst")).strip().lower()
        if self._adapter_subgroup_eval_mode not in {"mean_only", "mean_std_worst"}:
            raise ValueError("adapter.subgroup_eval_mode must be one of: mean_only, mean_std_worst")
        self._adapter_subgroup_tail_basis = str(adapter_cfg.get("subgroup_tail_basis", "time")).strip().lower()
        if self._adapter_subgroup_tail_basis not in {"gap", "time"}:
            raise ValueError("adapter.subgroup_tail_basis must be one of: gap, time")
        (
            self._adapter_risk_weight_mean,
            self._adapter_risk_weight_std,
            self._adapter_risk_weight_worst,
        ) = self._parse_risk_objective_weights(adapter_cfg.get("risk_objective_weights", [1.0, 0.3, 0.2]))
        self._adapter_gap_eps = self._to_float(adapter_cfg.get("gap_epsilon", 1.0e-12), 1.0e-12, min_value=1.0e-12)
        self._adapter_n_instance_eval = self._to_int(adapter_cfg.get("n_instance_eval", 0), 0, min_value=0)
        self._adapter_perturbation_moves = self._to_int(adapter_cfg.get("perturbation_moves", 30), 30, min_value=1)
        self._adapter_iter_limit = self._to_int(adapter_cfg.get("iter_limit", 1200), 1200, min_value=1)
        self._adapter_time_limit_per_instance = self._to_float(
            adapter_cfg.get("time_limit_per_instance", 5.0),
            5.0,
            min_value=1.0e-6,
        )
        self._adapter_eval_process_timeout_sec = self._to_float(
            adapter_cfg.get("eval_process_timeout_sec", 0.0),
            0.0,
        )
        self._adapter_jit_enable_kernels = bool(adapter_cfg.get("jit_enable_kernels", True))
        self._adapter_jit_numba_cache = bool(adapter_cfg.get("jit_numba_cache", True))
        self._adapter_jit_warmup = bool(adapter_cfg.get("jit_warmup", True))
        self._adapter_eva_numba_decorator = bool(adapter_cfg.get("eva_numba_decorator", False))
        self._adapter_jit_sanitize_guide = bool(adapter_cfg.get("jit_sanitize_guide", True))
        self._adapter_jit_sanitize_strict = bool(adapter_cfg.get("jit_sanitize_strict", False))

        groups_cfg = adapter_cfg.get("groups", {})
        pop_cfg = adapter_cfg.get("population", {})
        dds_cfg = adapter_cfg.get("dominance_dissimilarity", {})
        if not isinstance(dds_cfg, dict):
            dds_cfg = {}
        n_groups_total = self._to_int(groups_cfg.get("n_groups_total", 12), 12, min_value=1)
        group_size = self._to_int(groups_cfg.get("group_size", 5), 5, min_value=1)
        n_groups_train = self._to_int(groups_cfg.get("n_groups_train", 8), 8, min_value=0)
        n_groups_val = self._to_int(groups_cfg.get("n_groups_val", 4), 4, min_value=0)
        split_seed = self._to_int(groups_cfg.get("split_seed", 42), 42)
        shuffle_cases = bool(groups_cfg.get("shuffle_cases", True))
        use_group_split = bool(self._adapter_mode == "static_risk2")

        if use_group_split:
            if n_groups_train + n_groups_val != n_groups_total:
                raise ValueError("adapter.groups requires n_groups_train + n_groups_val == n_groups_total")
            if n_groups_train <= 0:
                raise ValueError("adapter.groups.n_groups_train must be > 0")

        self._adapter_train_keep_ratio = self._to_float(pop_cfg.get("train_keep_ratio", 0.5), 0.5, min_value=0.0)
        self._adapter_train_keep_min = self._to_int(pop_cfg.get("train_keep_min", 1), 1, min_value=1)
        self._adapter_train_keep_max = self._to_int(pop_cfg.get("train_keep_max", 64), 64, min_value=1)
        self._adapter_val_keep_ratio = self._to_float(pop_cfg.get("val_keep_ratio", 0.5), 0.5, min_value=0.0)
        self._adapter_val_keep_min = self._to_int(pop_cfg.get("val_keep_min", 1), 1, min_value=1)
        self._adapter_val_keep_max = self._to_int(pop_cfg.get("val_keep_max", 64), 64, min_value=1)
        self._adapter_parent_min = self._to_int(pop_cfg.get("parent_min", 1), 1, min_value=1)
        self._adapter_parent_max = self._to_int(pop_cfg.get("parent_max", 16), 16, min_value=1)
        self._adapter_dominance_eps = self._to_float(pop_cfg.get("dominance_epsilon", 1.0e-12), 1.0e-12, min_value=0.0)
        self._adapter_dissimilarity_metric = str(pop_cfg.get("dissimilarity_metric", "l2")).strip().lower()
        if self._adapter_dissimilarity_metric not in {"l1", "l2"}:
            raise ValueError("adapter.population.dissimilarity_metric must be l1 or l2")
        self._adapter_dds_eps_rel = self._to_float(
            pop_cfg.get("dds_eps_rel", dds_cfg.get("eps_rel", 1.0e-3)),
            1.0e-3,
            min_value=0.0,
        )
        self._adapter_dds_softmax_tau = self._to_float(
            pop_cfg.get("dds_softmax_tau", dds_cfg.get("softmax_tau", 1.0)),
            1.0,
            min_value=1.0e-6,
        )
        self._adapter_dds_novelty_weight = self._to_float(
            pop_cfg.get("dds_novelty_weight", dds_cfg.get("novelty_weight", 0.05)),
            0.05,
            min_value=0.0,
        )

        raw_dataset_paths = adapter_cfg.get("dataset_paths", None)
        raw_dataset_path = adapter_cfg.get("dataset_path", "")
        dataset_paths = self._resolve_adapter_dataset_paths(raw_dataset_paths, raw_dataset_path)

        has_custom_sources = False
        if isinstance(raw_dataset_paths, (list, tuple)):
            has_custom_sources = any(self._has_nonempty_path(x) for x in raw_dataset_paths)
        elif self._has_nonempty_path(raw_dataset_paths):
            has_custom_sources = True
        if self._has_nonempty_path(raw_dataset_path):
            has_custom_sources = True

        dataset_sources: list[Path] = []
        for dataset_path in dataset_paths:
            current_path = dataset_path
            if (not current_path.exists()) and (not has_custom_sources):
                gen_script = (Path(self.root_dir) / "problems" / self.problem / "gen_inst.py").resolve()
                subprocess.run([sys.executable, str(gen_script)], cwd=str(gen_script.parent), check=True)
            if (not current_path.exists()) and (not has_custom_sources):
                suffix = current_path.suffix.lower()
                if suffix in {".npy", ".npz", ".pkl", ".pickle"}:
                    fallback = sorted(current_path.parent.glob(f"*{suffix}"))
                else:
                    fallback = []
                if fallback:
                    logging.warning("adapter dataset %s not found, fallback to %s", current_path, fallback[0])
                    current_path = fallback[0].resolve()
            if not current_path.exists():
                raise FileNotFoundError(f"adapter dataset not found: {current_path}")
            dataset_sources.append(current_path.resolve())

        source_case_counts: dict[str, int] = {}
        cases: list[dict[str, Any]] = []
        for source_path in dataset_sources:
            source_cases = self._load_adapter_cases(source_path)
            source_case_counts[str(source_path)] = int(len(source_cases))
            cases.extend(source_cases)
        if not cases:
            raise ValueError("adapter dataset sources are empty after loading.")

        rng = np.random.default_rng(split_seed)
        # In grouped risk mode, always randomize case order before grouping.
        # This keeps subgroup construction stochastic even if shuffle_cases=false.
        if shuffle_cases or use_group_split:
            perm = rng.permutation(len(cases)).tolist()
            cases = [cases[i] for i in perm]
        if self._adapter_n_instance_eval > 0:
            cases = cases[: self._adapter_n_instance_eval]

        if use_group_split:
            required_total = int(group_size * n_groups_total)
            if len(cases) < required_total:
                raise ValueError(
                    f"adapter grouping needs at least {required_total} cases, got {len(cases)}."
                )
            selected = list(cases)
            all_groups = self._build_balanced_groups(len(selected), n_groups_total)

            def _collect_subset(group_ids: list[int]) -> tuple[list[dict[str, Any]], list[list[int]]]:
                subset: list[dict[str, Any]] = []
                group_indices_local: list[list[int]] = []
                cursor = 0
                for gid in group_ids:
                    idxs = all_groups[gid]
                    group_indices_local.append(list(range(cursor, cursor + len(idxs))))
                    for idx in idxs:
                        subset.append(selected[idx])
                    cursor += len(idxs)
                return subset, group_indices_local

            group_id_perm = rng.permutation(n_groups_total).tolist()
            train_group_ids = [int(x) for x in group_id_perm[:n_groups_train]]
            val_group_ids = [int(x) for x in group_id_perm[n_groups_train : n_groups_train + n_groups_val]]
            train_subset, train_group_indices = _collect_subset(train_group_ids)
            val_subset, val_group_indices = _collect_subset(val_group_ids)
            group_size_effective = int(min(len(g) for g in all_groups))
            group_size_effective_max = int(max(len(g) for g in all_groups))
            n_groups_total_effective = int(n_groups_total)
            n_groups_train_effective = int(n_groups_train)
            n_groups_val_effective = int(n_groups_val)
        else:
            # static mode: single full-train split without train/val group partition.
            train_subset = list(cases)
            train_group_indices = [list(range(len(train_subset)))]
            val_subset = []
            val_group_indices = []
            group_size_effective = int(len(train_subset))
            group_size_effective_max = int(len(train_subset))
            n_groups_total_effective = 1
            n_groups_train_effective = 1
            n_groups_val_effective = 0

        raw_artifacts_dir = str(adapter_cfg.get("artifacts_dir", "") or "").strip()
        if raw_artifacts_dir:
            candidate = Path(raw_artifacts_dir)
            if not candidate.is_absolute():
                self._adapter_dir = (Path(self.root_dir) / candidate).resolve()
            else:
                self._adapter_dir = candidate.resolve()
        else:
            self._adapter_dir = Path(os.getcwd()).resolve() / "adapter_artifacts"
        self._adapter_code_dir = self._adapter_dir / "code"
        self._adapter_eval_dir = self._adapter_dir / "eval_json"
        self._adapter_history_dir = self._adapter_dir / "history"
        self._adapter_dir.mkdir(parents=True, exist_ok=True)
        self._adapter_code_dir.mkdir(parents=True, exist_ok=True)
        self._adapter_eval_dir.mkdir(parents=True, exist_ok=True)
        self._adapter_history_dir.mkdir(parents=True, exist_ok=True)

        source_suffixes = {path.suffix.lower() for path in dataset_sources}
        if len(source_suffixes) != 1:
            raise ValueError(
                "adapter dataset sources must share one suffix type, "
                f"got: {sorted(source_suffixes)}"
            )
        suffix = list(source_suffixes)[0]
        if suffix not in {".npy", ".npz", ".pkl", ".pickle"}:
            raise ValueError(f"unsupported adapter dataset suffix: {suffix}")
        self._adapter_train_dataset_path = self._adapter_dir / f"dataset_train{suffix}"
        self._write_adapter_subset(train_subset, self._adapter_train_dataset_path)

        if len(val_subset) > 0:
            self._adapter_val_dataset_path = self._adapter_dir / f"dataset_val{suffix}"
            self._write_adapter_subset(val_subset, self._adapter_val_dataset_path)
            self._adapter_val_group_indices = val_group_indices
            self._adapter_has_val = True
        else:
            self._adapter_val_dataset_path = None
            self._adapter_val_group_indices = []
            self._adapter_has_val = False

        self._adapter_train_group_indices = train_group_indices
        base_pop_size = int(max(int(self.cfg.pop_size), 1))
        self._adapter_train_pool_size = base_pop_size

        self._adapter_eval_script = (Path(self.root_dir) / "problems" / self.problem / "eval_adapter.py").resolve()

        manifest = {
            "mode": self._adapter_mode,
            "metric_basis": self._adapter_metric_basis,
            "subgroup_eval_mode": self._adapter_subgroup_eval_mode,
            "subgroup_tail_basis": self._adapter_subgroup_tail_basis,
            "risk_objective_weights": {
                "mean": float(self._adapter_risk_weight_mean),
                "std": float(self._adapter_risk_weight_std),
                "worst": float(self._adapter_risk_weight_worst),
            },
            "dds_config": {
                "eps_rel": float(self._adapter_dds_eps_rel),
                "softmax_tau": float(self._adapter_dds_softmax_tau),
                "novelty_weight": float(self._adapter_dds_novelty_weight),
            },
            "dataset_source": (str(dataset_sources[0]) if len(dataset_sources) == 1 else None),
            "dataset_sources": [str(path) for path in dataset_sources],
            "source_case_counts": source_case_counts,
            "n_source_cases_total": int(sum(source_case_counts.values())),
            "train_dataset_path": str(self._adapter_train_dataset_path),
            "val_dataset_path": (str(self._adapter_val_dataset_path) if self._adapter_val_dataset_path else None),
            "n_train_instances": int(sum(len(g) for g in self._adapter_train_group_indices)),
            "n_val_instances": int(sum(len(g) for g in self._adapter_val_group_indices)),
            "population_size": int(self._adapter_train_pool_size),
            "perturbation_moves": int(self._adapter_perturbation_moves),
            "iter_limit": int(self._adapter_iter_limit),
            "time_limit_per_instance": float(self._adapter_time_limit_per_instance),
            "eval_process_timeout_sec": float(self._adapter_eval_process_timeout_sec),
            "jit_enable_kernels": bool(self._adapter_jit_enable_kernels),
            "jit_numba_cache": bool(self._adapter_jit_numba_cache),
            "jit_warmup": bool(self._adapter_jit_warmup),
            "eva_numba_decorator": bool(self._adapter_eva_numba_decorator),
            "jit_sanitize_guide": bool(self._adapter_jit_sanitize_guide),
            "jit_sanitize_strict": bool(self._adapter_jit_sanitize_strict),
            "group_split_enabled": bool(use_group_split),
            "group_size": int(group_size_effective),
            "group_size_max": int(group_size_effective_max),
            "n_groups_total": int(n_groups_total_effective),
            "n_groups_train": int(n_groups_train_effective),
            "n_groups_val": int(n_groups_val_effective),
            "split_seed": int(split_seed),
            "selection_method": ("dds_pair_select" if self._adapter_mode == "static_risk2" else "random_pair"),
            "best_update_method": ("val_dds_cross_generation" if self._adapter_mode == "static_risk2" else "train_obj"),
        }
        manifest_path = self._adapter_dir / "adapter_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info(f"Adapter mode enabled. Manifest: {print_hyperlink(manifest_path)}")

    def init_population(self) -> None:
        if self.adapter_enabled and self._adapter_mode == "static_risk2":
            logging.info("Evaluating seed function [train bootstrap]...")
            code = extract_code_from_generator(self.seed_func).replace("v1", "v2")
            seed_ind = {
                "stdout_filepath": f"problem_iter{self.iteration}_stdout0.txt",
                "code_path": f"problem_iter{self.iteration}_code0.py",
                "code": code,
                "response_id": 0,
            }
            seed_eval = self.evaluate_population(
                [dict(seed_ind)],
                split="train",
                count_fe=False,
                eval_stage="seed_eval",
            )
            if (not seed_eval) or (not bool(seed_eval[0].get("exec_success", False))):
                raise RuntimeError(f"Seed function is invalid. Please check the stdout file in {os.getcwd()}.")
            self.seed_ind = dict(seed_eval[0])

            system = self.system_generator_prompt
            user = self.user_generator_prompt + "\n" + self.seed_prompt + "\n" + self.long_term_reflection_str
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

            responses = self.generator_llm.multi_chat_completion(
                [messages],
                self.cfg.init_pop_size,
                temperature=self.generator_llm.temperature + 0.3,
            )
            population = [self.response_to_individual(response, response_id) for response_id, response in enumerate(responses)]
            population = self.evaluate_population(
                population,
                split="train",
                count_fe=True,
                eval_stage="init_population_eval",
            )
            population = self._manage_population(
                population,
                size=int(max(int(self.cfg.init_pop_size), 1)),
                use_multi_objective=True,
            )
            if len(population) < 2:
                raise RuntimeError("Risk2 initialization failed: train-feasible initial population size is < 2.")

            self.best_obj_overall = None
            self.best_code_overall = None
            self.best_code_path_overall = None
            self.elitist = None
            self._adapter_best_record = None
            self._adapter_generation_id = 0

            self.population = population
            self._adapter_last_eval_rows = list(self.population)
            self.update_iter()
            return

        # Evaluate the seed function, and set it as Elite.
        logging.info("Evaluating seed function...")
        code = extract_code_from_generator(self.seed_func).replace("v1", "v2")
        seed_ind = {
            "stdout_filepath": f"problem_iter{self.iteration}_stdout0.txt",
            "code_path": f"problem_iter{self.iteration}_code0.py",
            "code": code,
            "response_id": 0,
        }
        self.seed_ind = seed_ind
        self.population = self.evaluate_population(
            [seed_ind],
            split="train",
            count_fe=True,
            eval_stage="seed_eval",
        )
        if self.adapter_enabled:
            self._adapter_last_eval_rows = list(self.population)

        # If seed function is invalid, stop.
        if not self.seed_ind["exec_success"]:
            raise RuntimeError(f"Seed function is invalid. Please check the stdout file in {os.getcwd()}.")

        self.update_iter()

        # Generate responses.
        system = self.system_generator_prompt
        user = self.user_generator_prompt + "\n" + self.seed_prompt + "\n" + self.long_term_reflection_str
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        responses = self.generator_llm.multi_chat_completion(
            [messages],
            self.cfg.init_pop_size,
            temperature=self.generator_llm.temperature + 0.3,
        )
        population = [self.response_to_individual(response, response_id) for response_id, response in enumerate(responses)]

        # Run code and evaluate population.
        population = self.evaluate_population(
            population,
            split="train",
            count_fe=True,
            eval_stage="init_population_eval",
        )
        if self.adapter_enabled:
            target_size = self._adapter_train_pool_size if self._adapter_mode == "static_risk2" else int(max(int(self.cfg.pop_size), 1))
            population = self._manage_population(
                population,
                size=target_size,
                use_multi_objective=self._adapter_use_multi_objective(),
            )
            if not population:
                population = list(self.population)

        # Update iteration.
        self.population = population
        if self.adapter_enabled:
            self._adapter_last_eval_rows = list(self.population)
        self.update_iter()

    def response_to_individual(self, response: str, response_id: int, file_name: str = None) -> dict:
        """
        Convert response to individual.
        """
        file_name = f"problem_iter{self.iteration}_response{response_id}.txt" if file_name is None else file_name + ".txt"
        with open(file_name, "w", encoding="utf-8") as file:
            file.writelines(response + "\n")

        code = extract_code_from_generator(response)
        std_out_filepath = (
            f"problem_iter{self.iteration}_stdout{response_id}.txt"
            if file_name is None
            else file_name + "_stdout.txt"
        )

        individual = {
            "stdout_filepath": std_out_filepath,
            "code_path": f"problem_iter{self.iteration}_code{response_id}.py",
            "code": code,
            "response_id": response_id,
        }
        return individual

    def mark_invalid_individual(self, individual: dict, traceback_msg: str) -> dict:
        """
        Mark an individual as invalid.
        """
        individual["exec_success"] = False
        individual["obj"] = float("inf")
        individual["traceback_msg"] = traceback_msg
        if self.adapter_enabled:
            individual["objective_vector"] = [float("inf"), float("inf"), float("inf")]
            individual["eval_info"] = {
                "status": "invalid",
                "error": str(traceback_msg),
                "selected_metric": float("inf"),
                "objective_vector": [float("inf"), float("inf"), float("inf")],
                "n_invalid": 1,
            }
            individual["n_invalid"] = 1
        return individual

    def evaluate_population(
        self,
        population: list[dict],
        split: str = "train",
        count_fe: bool = True,
        eval_stage: str = "",
    ) -> list[dict]:
        """
        Evaluate population by running code and computing objective values.
        """
        if self.adapter_enabled:
            return self._evaluate_population_adapter(
                population,
                split=split,
                count_fe=count_fe,
                eval_stage=eval_stage,
            )

        inner_runs = []
        # Run code to evaluate.
        for response_id in range(len(population)):
            if count_fe:
                self.function_evals += 1
            if population[response_id]["code"] is None:
                population[response_id] = self.mark_invalid_individual(population[response_id], "Invalid response!")
                inner_runs.append(None)
                continue

            logging.info(f"Iteration {self.iteration}: Running Code {response_id}")
            try:
                process = self._run_code(population[response_id], response_id)
                inner_runs.append(process)
            except Exception as e:
                logging.info(f"Error for response_id {response_id}: {e}")
                population[response_id] = self.mark_invalid_individual(population[response_id], str(e))
                inner_runs.append(None)

        # Update population with objective values.
        for response_id, inner_run in enumerate(inner_runs):
            if inner_run is None:
                continue
            try:
                inner_run.communicate(timeout=self.cfg.timeout)
            except subprocess.TimeoutExpired as e:
                logging.info(f"Error for response_id {response_id}: {e}")
                population[response_id] = self.mark_invalid_individual(population[response_id], str(e))
                inner_run.kill()
                continue

            individual = population[response_id]
            stdout_filepath = individual["stdout_filepath"]
            with open(stdout_filepath, "r", encoding="utf-8") as f:
                stdout_str = f.read()
            traceback_msg = filter_traceback(stdout_str)

            if traceback_msg == "":
                try:
                    score = float(stdout_str.split("\n")[-2])
                    individual["obj"] = score if self.obj_type == "min" else -score
                    individual["exec_success"] = True
                except Exception:
                    population[response_id] = self.mark_invalid_individual(
                        population[response_id],
                        "Invalid std out / objective value!",
                    )
            else:
                population[response_id] = self.mark_invalid_individual(population[response_id], traceback_msg)

            logging.info(f"Iteration {self.iteration}, response_id {response_id}: Objective value: {individual['obj']}")
        return population

    def _adapter_eval_meta(self, split: str) -> tuple[Path, list[list[int]]]:
        split_norm = str(split).strip().lower()
        if split_norm == "val" and self._adapter_val_dataset_path is not None:
            return self._adapter_val_dataset_path, self._adapter_val_group_indices
        assert self._adapter_train_dataset_path is not None
        return self._adapter_train_dataset_path, self._adapter_train_group_indices

    def _adapter_process_timeout(self, split: str) -> Optional[float]:
        explicit = float(self._adapter_eval_process_timeout_sec)
        if explicit < 0.0:
            # Negative value disables eval subprocess timeout.
            return None
        if explicit > 0.0:
            return explicit

        base_timeout = float(self.cfg.timeout)
        if self._adapter_time_limit_per_instance <= 0.0:
            return base_timeout

        _, group_indices = self._adapter_eval_meta(split)
        n_eval = int(sum(len(group) for group in group_indices))
        if n_eval <= 0:
            return base_timeout

        estimated = float(n_eval) * float(self._adapter_time_limit_per_instance) + 15.0
        return max(base_timeout, estimated)

    def _run_code_adapter(self, individual: dict, response_id: int, split: str) -> tuple[subprocess.Popen, Path, Path, Path]:
        assert self._adapter_code_dir is not None
        assert self._adapter_eval_dir is not None
        assert self._adapter_eval_script is not None

        dataset_path, group_indices = self._adapter_eval_meta(split)
        stem = f"iter{self.iteration:04d}_gen{self._adapter_generation_id:04d}_{split}_{response_id:04d}_{uuid.uuid4().hex[:8]}"
        code_path = self._adapter_code_dir / f"{stem}.py"
        eval_json_path = self._adapter_eval_dir / f"{stem}.json"
        stdout_path = self._adapter_eval_dir / f"{stem}.stdout.txt"

        code_text = str(individual.get("code", "") or "")
        if not code_text.strip():
            raise ValueError("Empty code for adapter evaluation.")
        code_path.write_text(code_text + ("\n" if not code_text.endswith("\n") else ""), encoding="utf-8")

        cmd = [
            sys.executable,
            "-u",
            str(self._adapter_eval_script),
            "--dataset-path",
            str(dataset_path),
            "--code-path",
            str(code_path),
            "--output-json",
            str(eval_json_path),
            "--metric-basis",
            str(self._adapter_metric_basis),
            "--subgroup-mode",
            str(self._adapter_subgroup_eval_mode),
            "--subgroup-tail-basis",
            str(self._adapter_subgroup_tail_basis),
            "--risk-weight-mean",
            str(self._adapter_risk_weight_mean),
            "--risk-weight-std",
            str(self._adapter_risk_weight_std),
            "--risk-weight-worst",
            str(self._adapter_risk_weight_worst),
            "--group-indices-json",
            json.dumps(group_indices, ensure_ascii=False),
            "--n-instance-eval",
            str(self._adapter_n_instance_eval),
            "--perturbation-moves",
            str(self._adapter_perturbation_moves),
            "--iter-limit",
            str(self._adapter_iter_limit),
            "--time-limit-per-instance",
            str(self._adapter_time_limit_per_instance),
            "--jit-warmup",
            ("1" if self._adapter_jit_warmup else "0"),
            "--jit-decorate-guide",
            ("1" if self._adapter_eva_numba_decorator else "0"),
            "--gap-eps",
            str(self._adapter_gap_eps),
        ]
        if self.problem == "tsp_gls":
            cmd.extend(
                [
                    "--jit-sanitize-guide",
                    ("1" if self._adapter_jit_sanitize_guide else "0"),
                    "--jit-sanitize-strict",
                    ("1" if self._adapter_jit_sanitize_strict else "0"),
                ]
            )
        env = os.environ.copy()
        env["REEVO_GLS_ENABLE_JIT"] = "1" if self._adapter_jit_enable_kernels else "0"
        env["REEVO_GLS_NUMBA_CACHE"] = "1" if self._adapter_jit_numba_cache else "0"
        with open(stdout_path, "w", encoding="utf-8") as f:
            process = subprocess.Popen(cmd, stdout=f, stderr=f, env=env)
        return process, eval_json_path, code_path, stdout_path

    def _evaluate_population_adapter(
        self,
        population: list[dict],
        split: str = "train",
        count_fe: bool = True,
        eval_stage: str = "",
    ) -> list[dict]:
        self._adapter_eval_call_id += 1
        call_id = int(self._adapter_eval_call_id)
        pending: list[tuple[dict, subprocess.Popen, Path, Path, Path]] = []
        out: list[dict] = []
        for response_id, individual in enumerate(population):
            row = individual
            if count_fe:
                self.function_evals += 1
            if row.get("code") is None:
                out.append(self.mark_invalid_individual(row, "Invalid response/code."))
                continue
            logging.info(f"Iteration {self.iteration}: Running Code {response_id} [{split}]")
            try:
                process, eval_json_path, code_path, stdout_path = self._run_code_adapter(row, response_id, split)
                row["code_path"] = str(code_path)
                row["stdout_filepath"] = str(stdout_path)
                pending.append((row, process, eval_json_path, code_path, stdout_path))
            except Exception as e:
                out.append(self.mark_invalid_individual(row, str(e)))

        for row, process, eval_json_path, code_path, stdout_path in pending:
            try:
                timeout_sec = self._adapter_process_timeout(split)
                if timeout_sec is None:
                    process.communicate()
                else:
                    process.communicate(timeout=timeout_sec)
            except subprocess.TimeoutExpired as e:
                process.kill()
                out.append(self.mark_invalid_individual(row, f"Timeout: {e}"))
                continue

            if not eval_json_path.exists():
                out.append(self.mark_invalid_individual(row, "Missing eval_adapter output json."))
                continue

            try:
                eval_info = json.loads(eval_json_path.read_text(encoding="utf-8"))
                status = str(eval_info.get("status", "invalid"))
                n_invalid = int(eval_info.get("n_invalid", 1))
                metric = float(eval_info.get("selected_metric", float("inf")))
                exec_success = bool(status == "ok" and np.isfinite(metric))
                row["code_path"] = str(code_path)
                row["stdout_filepath"] = str(stdout_path)
                row["eval_info"] = eval_info
                row["obj"] = float(metric if exec_success else float("inf"))
                row["exec_success"] = bool(exec_success)
                row["objective_vector"] = list(eval_info.get("objective_vector", [row["obj"], 0.0, row["obj"]]))
                row["n_invalid"] = int(n_invalid)
                row["mean_time_sec"] = float(eval_info.get("mean_time_sec", float("inf")))
                row["mean_gap"] = float(eval_info.get("mean_gap", float("inf")))
                row["std_gap"] = float(eval_info.get("group_std_eval", float("inf")))
                row["worst_gap"] = float(eval_info.get("group_worst_eval", float("inf")))
                if not exec_success:
                    row["traceback_msg"] = str(eval_info.get("error", "Invalid evaluation result."))
                out.append(row)
            except Exception as e:
                out.append(self.mark_invalid_individual(row, str(e)))

        self._log_adapter_guide_jit_stats(
            rows=out,
            split=split,
            eval_stage=eval_stage,
            call_id=call_id,
        )
        return out

    def _run_code(self, individual: dict, response_id: int) -> subprocess.Popen:
        """
        Write code into a file and run eval script.
        """
        logging.debug(f"Iteration {self.iteration}: Processing Code Run {response_id}")

        with open(self.output_file, "w", encoding="utf-8") as file:
            file.writelines(individual["code"] + "\n")

        with open(individual["stdout_filepath"], "w", encoding="utf-8") as f:
            eval_file_path = (
                f"{self.root_dir}/problems/{self.problem}/eval.py"
                if self.problem_type != "black_box"
                else f"{self.root_dir}/problems/{self.problem}/eval_black_box.py"
            )
            process = subprocess.Popen(
                [sys.executable, "-u", eval_file_path, f"{self.problem_size}", self.root_dir, "train"],
                stdout=f,
                stderr=f,
            )

        block_until_running(individual["stdout_filepath"], log_status=True, iter_num=self.iteration, response_id=response_id)
        return process

    def update_iter(self) -> None:
        """
        Update after each iteration.
        """
        population = self.population
        if len(population) <= 0:
            raise RuntimeError("Population is empty.")
        if self.adapter_enabled and self._adapter_mode == "static_risk2":
            self._refresh_risk2_best_state()
            elite_vector = None
            if isinstance(self.elitist, dict):
                try:
                    elite_vector = list(self._objective_vector(self.elitist))
                except Exception:
                    elite_vector = None
            best_record = self._adapter_best_record if isinstance(self._adapter_best_record, dict) else None
            best_vector = best_record.get("objective_vector") if isinstance(best_record, dict) else None
            best_path = self.best_code_path_overall or ""
            if best_path:
                best_path = best_path.replace(".py", ".txt").replace("code", "response")
            logging.info("Current train-DDS elite: %s", elite_vector)
            logging.info(
                "Best val-DDS: %s, Best Code Path: %s",
                best_vector,
                print_hyperlink(best_path, self.best_code_path_overall) if self.best_code_path_overall else "N/A",
            )
            if self.adapter_enabled:
                self._log_adapter_iteration_time_stats()
            logging.info(f"Iteration {self.iteration} finished...")
            logging.info(f"Function Evals: {self.function_evals}")
            self.iteration += 1
            return
        objs = [individual["obj"] for individual in population]
        best_obj, best_sample_idx = min(objs), int(np.argmin(np.asarray(objs, dtype=float)))

        if self.best_obj_overall is None or best_obj < self.best_obj_overall:
            self.best_obj_overall = best_obj
            self.best_code_overall = population[best_sample_idx]["code"]
            self.best_code_path_overall = population[best_sample_idx]["code_path"]

        if self.elitist is None or best_obj < self.elitist["obj"]:
            self.elitist = population[best_sample_idx]
            logging.info(f"Iteration {self.iteration}: Elitist: {self.elitist['obj']}")

        best_path = self.best_code_path_overall.replace(".py", ".txt").replace("code", "response")
        logging.info(
            f"Best obj: {self.best_obj_overall}, Best Code Path: {print_hyperlink(best_path, self.best_code_path_overall)}"
        )
        if self.adapter_enabled:
            self._log_adapter_iteration_time_stats()
        logging.info(f"Iteration {self.iteration} finished...")
        logging.info(f"Function Evals: {self.function_evals}")
        self.iteration += 1

    def _log_adapter_iteration_time_stats(self) -> None:
        rows = list(self._adapter_last_eval_rows or [])
        if not rows:
            return

        total_time = 0.0
        total_eval = 0
        timeout_count = 0
        invalid_count = 0

        for row in rows:
            info = row.get("eval_info")
            info_dict = info if isinstance(info, dict) else {}

            n_eval_raw = info_dict.get("n_eval", 0)
            mean_time_raw = info_dict.get("mean_time_sec", float("nan"))
            n_time_exceed_raw = info_dict.get("n_time_exceed", 0)
            n_invalid_raw = info_dict.get("n_invalid", row.get("n_invalid", (0 if row.get("exec_success", False) else 1)))

            try:
                n_eval = int(n_eval_raw)
            except Exception:
                n_eval = 0
            try:
                mean_time = float(mean_time_raw)
            except Exception:
                mean_time = float("nan")
            try:
                n_time_exceed = int(n_time_exceed_raw)
            except Exception:
                n_time_exceed = 0
            try:
                n_invalid = int(n_invalid_raw)
            except Exception:
                n_invalid = 0

            timeout_count += max(n_time_exceed, 0)
            invalid_count += max(n_invalid, 0)
            if n_eval > 0 and np.isfinite(mean_time):
                total_time += float(mean_time) * float(n_eval)
                total_eval += int(n_eval)

        if total_eval > 0:
            avg_time = float(total_time / float(total_eval))
            logging.info(
                f"Iteration {self.iteration}: Avg time per instance = {avg_time:.6f}s, invalid_count = {invalid_count}, "
                f"time_exceed_count = {timeout_count} "
                f"(aggregated n_eval = {total_eval})"
            )
            return

        fallback_times: list[float] = []
        for row in rows:
            v = row.get("mean_time_sec")
            try:
                fv = float(v)
            except Exception:
                continue
            if np.isfinite(fv):
                fallback_times.append(float(fv))
        if fallback_times:
            avg_time = float(np.mean(np.asarray(fallback_times, dtype=float)))
            logging.info(
                f"Iteration {self.iteration}: Avg time per candidate = {avg_time:.6f}s, invalid_count = {invalid_count}, "
                f"time_exceed_count = {timeout_count}"
            )
        else:
            logging.info(
                f"Iteration {self.iteration}: Avg time unavailable, invalid_count = {invalid_count}, "
                f"time_exceed_count = {timeout_count}"
            )

    def _log_adapter_guide_jit_stats(
        self,
        rows: list[dict],
        split: str,
        eval_stage: str,
        call_id: int,
    ) -> None:
        if not self.adapter_enabled:
            return
        if not rows:
            return

        total_rows = int(len(rows))
        eval_info_rows = 0
        exec_success_rows = 0
        sanitize_enabled = 0
        sanitize_active = 0
        compile_probe_ok = 0
        decorator_active = 0

        for row in rows:
            if bool(row.get("exec_success", False)):
                exec_success_rows += 1
            info = row.get("eval_info")
            if not isinstance(info, dict):
                continue
            eval_info_rows += 1
            if bool(info.get("jit_guide_sanitize_enabled", False)):
                sanitize_enabled += 1
            if bool(info.get("jit_guide_sanitize_active", False)):
                sanitize_active += 1
            if bool(info.get("jit_guide_compile_probe_ok", False)):
                compile_probe_ok += 1
            if bool(info.get("jit_guide_decorator_active", False)):
                decorator_active += 1

        stage_label = str(eval_stage or "").strip() or "unspecified"
        logging.info(
            "Adapter guide-JIT stats: "
            f"stage={stage_label}, split={split}, call={call_id}, "
            f"exec_success={exec_success_rows}/{total_rows}, "
            f"eval_info={eval_info_rows}/{total_rows}, "
            f"sanitize_enabled={sanitize_enabled}/{eval_info_rows}, "
            f"sanitize_active={sanitize_active}/{eval_info_rows}, "
            f"compile_probe_ok={compile_probe_ok}/{eval_info_rows}, "
            f"decorator_active={decorator_active}/{eval_info_rows}"
        )

    def rank_select(self, population: list[dict]) -> Optional[list[dict]]:
        """
        Rank-based selection, select individuals with probability proportional to rank.
        """
        if self.problem_type == "black_box":
            population = [individual for individual in population if individual["exec_success"] and individual["obj"] < self.seed_ind["obj"]]
        else:
            population = [individual for individual in population if individual["exec_success"]]
        if len(population) < 2:
            return None
        population = sorted(population, key=lambda x: x["obj"])
        ranks = [i for i in range(len(population))]
        probs = [1 / (rank + 1 + len(population)) for rank in ranks]
        probs = [prob / sum(probs) for prob in probs]
        selected_population = []
        trial = 0
        while len(selected_population) < 2 * self.cfg.pop_size:
            trial += 1
            parents = np.random.choice(population, size=2, replace=False, p=probs)
            if parents[0]["obj"] != parents[1]["obj"]:
                selected_population.extend(parents)
            if trial > 1000:
                return None
        return selected_population

    def random_select(self, population: list[dict]) -> Optional[list[dict]]:
        """
        Random selection, select individuals with equal probability.
        """
        selected_population = []
        if self.problem_type == "black_box":
            population = [individual for individual in population if individual["exec_success"] and individual["obj"] < self.seed_ind["obj"]]
        else:
            population = [individual for individual in population if individual["exec_success"]]
        if len(population) < 2:
            return None
        trial = 0
        while len(selected_population) < 2 * self.cfg.pop_size:
            trial += 1
            parents = np.random.choice(population, size=2, replace=False)
            if parents[0]["obj"] != parents[1]["obj"]:
                selected_population.extend(parents)
            if trial > 1000:
                return None
        return selected_population

    @staticmethod
    def _normalize_code(code: str) -> str:
        return "\n".join(line.rstrip() for line in str(code or "").strip().splitlines())

    def _adapter_use_multi_objective(self) -> bool:
        return bool(self._adapter_mode == "static_risk2")

    def _objective_vector(self, individual: dict) -> np.ndarray:
        vec = individual.get("objective_vector")
        if isinstance(vec, (list, tuple)) and len(vec) >= 3:
            try:
                arr = np.asarray([float(vec[0]), float(vec[1]), float(vec[2])], dtype=float)
                if np.all(np.isfinite(arr)):
                    return arr
            except Exception:
                pass
        obj = float(individual.get("obj", float("inf")))
        return np.asarray([obj, 0.0, obj], dtype=float)

    def _adapter_is_feasible(self, individual: dict, strict_risk2: bool = False) -> bool:
        if not isinstance(individual, dict):
            return False
        if not str(individual.get("code", "")).strip():
            return False
        if not bool(individual.get("exec_success", False)):
            return False
        obj = float(individual.get("obj", float("inf")))
        if not np.isfinite(obj):
            return False
        vec = self._objective_vector(individual)
        if not np.isfinite(vec).all():
            return False
        if not strict_risk2:
            return True

        info = individual.get("eval_info")
        if isinstance(info, dict):
            status = str(info.get("status", "")).strip().lower()
            if status in {"invalid", "ok_with_penalty"}:
                return False
            try:
                n_invalid = int(info.get("n_invalid", individual.get("n_invalid", 0)))
            except Exception:
                n_invalid = int(individual.get("n_invalid", 0) or 0)
            if n_invalid > 0:
                return False
        else:
            try:
                n_invalid = int(individual.get("n_invalid", 0))
            except Exception:
                n_invalid = 0
            if n_invalid > 0:
                return False
        return True

    @staticmethod
    def _epsilon_from_population(objective_matrix: np.ndarray, eps_rel: float, eps_min: float = 1.0e-9) -> np.ndarray:
        arr = np.asarray(objective_matrix, dtype=float)
        if arr.size <= 0:
            return np.asarray([eps_min, eps_min, eps_min], dtype=float)

        eps_rel_f = float(max(eps_rel, 0.0))
        eps_min_f = float(max(eps_min, 0.0))
        out = np.zeros(arr.shape[1], dtype=float)
        for dim in range(arr.shape[1]):
            vals = arr[:, dim]
            vals = vals[np.isfinite(vals)]
            if vals.size <= 0:
                span = 1.0
            else:
                span = float(np.max(vals) - np.min(vals))
            out[dim] = float(max(span * eps_rel_f, eps_min_f))
        return out

    @staticmethod
    def _code_counter(code: str) -> Counter[str]:
        text = str(code or "")
        if not text.strip():
            return Counter({"__empty_code__": 1})

        try:
            tree = ast.parse(text)
            counter: Counter[str] = Counter()
            for node in ast.walk(tree):
                sig = ast.dump(node, annotate_fields=False, include_attributes=False)
                counter[sig] += 1
            if not counter:
                counter["__empty_ast__"] = 1
            return counter
        except Exception:
            toks = _TOKEN_PATTERN.findall(text)
            if not toks:
                toks = ["__parse_failed__"]
            return Counter(toks)

    @staticmethod
    def _similarity_asymm(counter_a: Counter[str], counter_b: Counter[str]) -> float:
        denom = int(sum(counter_b.values()))
        if denom <= 0:
            return 0.0
        overlap = 0
        for token, count_b in counter_b.items():
            overlap += min(int(counter_a.get(token, 0)), int(count_b))
        return float(max(min(overlap / max(denom, 1), 1.0), 0.0))

    @staticmethod
    def _similarity_sym(counter_a: Counter[str], counter_b: Counter[str]) -> float:
        s_ab = ReEvo._similarity_asymm(counter_a, counter_b)
        s_ba = ReEvo._similarity_asymm(counter_b, counter_a)
        return float(0.5 * (s_ab + s_ba))

    def _compute_dds_scores(self, population: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = int(len(population))
        if n <= 0:
            empty_f = np.zeros(0, dtype=float)
            empty_b = np.zeros(0, dtype=bool)
            empty_i = np.zeros(0, dtype=int)
            return empty_f, np.zeros((0, 3), dtype=float), empty_b, empty_i

        objectives = np.asarray([self._objective_vector(ind) for ind in population], dtype=float)
        eps = self._epsilon_from_population(objectives, eps_rel=self._adapter_dds_eps_rel)
        counters = [self._code_counter(str(ind.get("code", ""))) for ind in population]

        scores = np.zeros(n, dtype=float)
        dominated_counts = np.zeros(n, dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                oi = objectives[i]
                oj = objectives[j]
                i_dom_j = bool(np.all(oi <= oj + eps) and np.any(oi < oj - eps))
                j_dom_i = bool(np.all(oj <= oi + eps) and np.any(oj < oi - eps))
                if i_dom_j and (not j_dom_i):
                    sim = self._similarity_asymm(counters[i], counters[j])
                    scores[j] += float(-sim)
                    dominated_counts[j] += 1
                elif j_dom_i and (not i_dom_j):
                    sim = self._similarity_asymm(counters[j], counters[i])
                    scores[i] += float(-sim)
                    dominated_counts[i] += 1

        nondominated = dominated_counts == 0
        novelty_w = float(max(self._adapter_dds_novelty_weight, 0.0))
        if novelty_w > 0.0 and n > 1:
            nd_indices = np.where(nondominated)[0].tolist()
            for i in nd_indices:
                novelty_vals: list[float] = []
                for j in range(n):
                    if i == j:
                        continue
                    sim = self._similarity_sym(counters[i], counters[j])
                    novelty_vals.append(float(1.0 - sim))
                if novelty_vals:
                    scores[i] += novelty_w * float(np.mean(np.asarray(novelty_vals, dtype=float)))

        return scores, objectives, nondominated, dominated_counts

    def _sort_population_indices_by_dds(self, population: list[dict]) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
        scores, objectives, nondominated, _ = self._compute_dds_scores(population)
        indices = list(range(len(population)))
        indices.sort(
            key=lambda idx: (
                0 if bool(nondominated[idx]) else 1,
                -float(scores[idx]),
                float(objectives[idx][0]),
                float(objectives[idx][2]),
                float(objectives[idx][1]),
            )
        )
        return indices, scores, objectives, nondominated

    def _dds_parent_selection(self, population: list[dict], n_parent: int) -> list[dict]:
        n_target = max(int(n_parent), 1)
        valid = self._deduplicate_feasible(population, strict_risk2=True)
        if not valid:
            return []

        scores, _, _, _ = self._compute_dds_scores(valid)
        tau = float(max(self._adapter_dds_softmax_tau, 1.0e-6))
        logits = np.asarray(scores, dtype=float) / tau
        logits = logits - float(np.max(logits))
        weights = np.exp(logits)
        if (not np.isfinite(weights).all()) or float(np.sum(weights)) <= 0.0:
            probs = [1.0 / len(valid)] * len(valid)
        else:
            probs = (weights / float(np.sum(weights))).tolist()
        return random.choices(valid, weights=probs, k=n_target)

    def _dds_pair_select(self, population: list[dict]) -> Optional[list[dict]]:
        valid = self._deduplicate_feasible(population, strict_risk2=True)
        if len(valid) < 2:
            return None

        scores, _, _, _ = self._compute_dds_scores(valid)
        tau = float(max(self._adapter_dds_softmax_tau, 1.0e-6))
        logits = np.asarray(scores, dtype=float) / tau
        logits = logits - float(np.max(logits))
        weights = np.exp(logits)
        if (not np.isfinite(weights).all()) or float(np.sum(weights)) <= 0.0:
            probs = [1.0 / len(valid)] * len(valid)
        else:
            probs = (weights / float(np.sum(weights))).tolist()

        selected_population: list[dict] = []
        target_size = int(max(2 * int(self.cfg.pop_size), 2))
        probs_arr = np.asarray(probs, dtype=float)
        if (not np.isfinite(probs_arr).all()) or float(np.sum(probs_arr)) <= 0.0:
            probs_arr = np.full(len(valid), 1.0 / float(len(valid)), dtype=float)
        else:
            probs_arr = probs_arr / float(np.sum(probs_arr))

        trial = 0
        trial_limit = max(1000, 200 * target_size)
        while len(selected_population) < target_size:
            trial += 1
            try:
                parent_idx = np.random.choice(len(valid), size=2, replace=False, p=probs_arr)
            except ValueError:
                return None
            parents = [valid[int(parent_idx[0])], valid[int(parent_idx[1])]]
            if self._compare_individuals(parents[0], parents[1]) != 0:
                selected_population.extend(parents)
            if trial > trial_limit:
                return None
        return selected_population[:target_size]

    def _dds_population_management(self, population: list[dict], size: int) -> list[dict]:
        target_size = max(int(size), 0)
        if target_size <= 0:
            return []
        unique = self._deduplicate_feasible(population, strict_risk2=True)
        if not unique:
            return []

        idx_sorted, _, _, _ = self._sort_population_indices_by_dds(unique)
        keep = idx_sorted[: min(target_size, len(idx_sorted))]
        return [unique[idx] for idx in keep]

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        if self._adapter_dissimilarity_metric == "l1":
            return float(np.abs(a - b).sum())
        return float(np.linalg.norm(a - b))

    def _dominates(self, a: dict, b: dict, eps: float) -> bool:
        av = self._objective_vector(a)
        bv = self._objective_vector(b)
        not_worse = bool(np.all(av <= bv + float(eps)))
        strictly_better = bool(np.any(av < bv - float(eps)))
        return not_worse and strictly_better

    def _non_dominated_fronts(self, records: list[dict], eps: float) -> list[list[dict]]:
        dominates_map = {idx: set() for idx in range(len(records))}
        dominated_count = {idx: 0 for idx in range(len(records))}

        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                i_dom_j = self._dominates(records[i], records[j], eps)
                j_dom_i = self._dominates(records[j], records[i], eps)
                if i_dom_j and not j_dom_i:
                    dominates_map[i].add(j)
                    dominated_count[j] += 1
                elif j_dom_i and not i_dom_j:
                    dominates_map[j].add(i)
                    dominated_count[i] += 1

        fronts: list[list[int]] = []
        current = [idx for idx, count in dominated_count.items() if count == 0]
        while current:
            fronts.append(current)
            next_front = []
            for idx in current:
                for dominated in dominates_map[idx]:
                    dominated_count[dominated] -= 1
                    if dominated_count[dominated] == 0:
                        next_front.append(dominated)
            current = next_front
        return [[records[idx] for idx in front] for front in fronts]

    def _pick_by_dissimilarity(
        self,
        front: list[dict],
        already_selected: list[dict],
        remaining: int,
    ) -> list[dict]:
        if remaining <= 0 or not front:
            return []
        if remaining >= len(front):
            return front[:]

        vectors = np.asarray([self._objective_vector(item) for item in front], dtype=float)
        mins = vectors.min(axis=0)
        maxs = vectors.max(axis=0)
        denom = np.where((maxs - mins) > self._adapter_dominance_eps, (maxs - mins), 1.0)
        normalized = (vectors - mins) / denom

        if already_selected:
            anchor_vectors = np.asarray([self._objective_vector(item) for item in already_selected], dtype=float)
            anchors = (anchor_vectors - mins) / denom
        else:
            anchors = np.zeros((0, vectors.shape[1]), dtype=float)

        selected_indices: list[int] = []
        if anchors.shape[0] == 0:
            aggregate = normalized.sum(axis=1)
            selected_indices.append(int(np.argmin(aggregate)))

        while len(selected_indices) < remaining:
            best_idx = None
            best_score = -1.0
            for idx in range(len(front)):
                if idx in selected_indices:
                    continue
                distances = [self._distance(normalized[idx], normalized[si]) for si in selected_indices]
                if anchors.shape[0] > 0:
                    distances.extend(self._distance(normalized[idx], anchor_vec) for anchor_vec in anchors)
                score = min(distances) if distances else 0.0
                if score > best_score:
                    best_score = float(score)
                    best_idx = int(idx)
            if best_idx is None:
                break
            selected_indices.append(best_idx)

        return [front[idx] for idx in selected_indices]

    def _deduplicate_feasible(self, population: list[dict], strict_risk2: bool = False) -> list[dict]:
        by_code: dict[str, dict] = {}
        for ind in population:
            if not self._adapter_is_feasible(ind, strict_risk2=strict_risk2):
                continue
            code_norm = self._normalize_code(str(ind.get("code", "")))
            if not code_norm:
                continue
            prev = by_code.get(code_norm)
            if prev is None:
                by_code[code_norm] = ind
                continue
            if tuple(self._objective_vector(ind)) < tuple(self._objective_vector(prev)):
                by_code[code_norm] = ind
        out = list(by_code.values())
        if strict_risk2:
            out.sort(
                key=lambda x: (
                    tuple(float(v) for v in self._objective_vector(x).tolist()),
                    str(x.get("code_path", "")),
                )
            )
        else:
            out.sort(key=lambda x: (float(x.get("obj", float("inf"))), str(x.get("code_path", ""))))
        return out

    def _compute_keep_n(self, feasible_count: int, stage: str) -> int:
        if stage == "train":
            ratio = self._adapter_train_keep_ratio
            min_keep = self._adapter_train_keep_min
            max_keep = self._adapter_train_keep_max
        else:
            ratio = self._adapter_val_keep_ratio
            min_keep = self._adapter_val_keep_min
            max_keep = self._adapter_val_keep_max
        keep = int(round(float(feasible_count) * float(ratio)))
        keep = max(int(min_keep), keep)
        keep = min(int(max_keep), keep)
        return int(keep)

    def _select_candidates(
        self,
        records: list[dict],
        stage: str,
        target_size: Optional[int] = None,
    ) -> list[dict]:
        strict_risk2 = bool(self._adapter_mode == "static_risk2")
        feasible = self._deduplicate_feasible(records, strict_risk2=strict_risk2)
        if not feasible:
            return []

        if target_size is None:
            keep_n = self._compute_keep_n(len(feasible), stage)
        else:
            keep_n = int(max(int(target_size), 0))
        keep_n = min(keep_n, len(feasible))
        if keep_n <= 0:
            return []

        if self._adapter_mode == "static":
            return sorted(
                feasible,
                key=lambda item: (
                    float(item.get("obj", float("inf"))),
                    float(item.get("mean_time_sec", float("inf"))),
                    str(item.get("code_path", "")),
                ),
            )[:keep_n]

        return self._dds_population_management(feasible, keep_n)

    def _clip_parent_pool(self, population: list[dict]) -> list[dict]:
        unique = self._deduplicate_feasible(
            population,
            strict_risk2=bool(self._adapter_mode == "static_risk2"),
        )
        if not unique:
            return []
        unique = unique[: self._adapter_parent_max]
        return unique

    def _ensure_population_size(self, population: list[dict], backup: list[dict]) -> list[dict]:
        min_needed = max(2, int(self._adapter_parent_min))
        if len(population) >= min_needed:
            return population
        seen = {self._normalize_code(str(ind.get("code", ""))) for ind in population}
        extra: list[dict] = []
        for cand in self._deduplicate_feasible(
            backup,
            strict_risk2=bool(self._adapter_mode == "static_risk2"),
        ):
            norm = self._normalize_code(str(cand.get("code", "")))
            if norm in seen:
                continue
            seen.add(norm)
            extra.append(cand)
            if len(population) + len(extra) >= min_needed:
                break
        merged = population + extra
        if len(merged) > self._adapter_parent_max:
            merged = merged[: self._adapter_parent_max]
        return merged

    def _manage_population(self, population: list[dict], size: int, use_multi_objective: bool) -> list[dict]:
        target = int(max(int(size), 0))
        if target <= 0:
            return []
        feasible = self._deduplicate_feasible(population, strict_risk2=bool(use_multi_objective))
        if not feasible:
            return []
        if not use_multi_objective:
            return feasible[: min(target, len(feasible))]
        return self._dds_population_management(feasible, target)

    def _refresh_risk2_best_state(self) -> None:
        if self._adapter_mode != "static_risk2":
            return
        current_best = self._manage_population(
            list(self.population),
            size=1,
            use_multi_objective=True,
        )
        if current_best:
            self.elitist = dict(current_best[0])

    def _update_risk2_best_from_val_rows(self, population: list[dict]) -> None:
        if self._adapter_mode != "static_risk2":
            return
        if not population:
            return
        batch_best = self._manage_population(
            list(population),
            size=1,
            use_multi_objective=True,
        )
        if not batch_best:
            return
        candidate = dict(batch_best[0])
        if isinstance(self._adapter_best_record, dict):
            winner_rows = self._manage_population(
                [dict(self._adapter_best_record), candidate],
                size=1,
                use_multi_objective=True,
            )
            if not winner_rows:
                return
            winner = dict(winner_rows[0])
        else:
            winner = dict(candidate)

        self._adapter_best_record = dict(winner)
        self.best_code_overall = winner.get("code")
        self.best_code_path_overall = winner.get("code_path")
        self.best_obj_overall = float(winner.get("obj", float("inf")))

    def _select_next_population_from_val_rows(
        self,
        val_rows: list[dict],
        target_size: Optional[int] = None,
    ) -> list[dict]:
        if self._adapter_mode != "static_risk2":
            return list(val_rows)
        keep_size = int(max(int(self.cfg.pop_size), 1) if target_size is None else max(int(target_size), 0))
        return self._manage_population(
            list(val_rows),
            size=keep_size,
            use_multi_objective=True,
        )

    def _compare_individuals(self, ind1: dict, ind2: dict) -> int:
        vec1 = tuple(float(x) for x in self._objective_vector(ind1).tolist())
        vec2 = tuple(float(x) for x in self._objective_vector(ind2).tolist())
        if vec1 < vec2:
            return -1
        if vec2 < vec1:
            return 1
        if not (self.adapter_enabled and self._adapter_mode == "static_risk2"):
            obj1 = float(ind1.get("obj", float("inf")))
            obj2 = float(ind2.get("obj", float("inf")))
            if obj1 < obj2:
                return -1
            if obj2 < obj1:
                return 1
        path1 = str(ind1.get("code_path", ""))
        path2 = str(ind2.get("code_path", ""))
        if path1 < path2:
            return -1
        if path2 < path1:
            return 1
        return 0

    def gen_short_term_reflection_prompt(self, ind1: dict, ind2: dict) -> tuple[list[dict], str, str]:
        """
        Short-term reflection before crossovering two individuals.
        """
        if self._compare_individuals(ind1, ind2) <= 0:
            better_ind, worse_ind = ind1, ind2
        else:
            better_ind, worse_ind = ind2, ind1

        worse_code = filter_code(worse_ind["code"])
        better_code = filter_code(better_ind["code"])

        system = self.system_reflector_prompt
        user = self.user_reflector_st_prompt.format(
            func_name=self.func_name,
            func_desc=self.func_desc,
            problem_desc=self.problem_desc,
            worse_code=worse_code,
            better_code=better_code,
        )
        message = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        if self.print_short_term_reflection_prompt:
            self.print_short_term_reflection_prompt = False
        return message, worse_code, better_code

    def short_term_reflection(self, population: list[dict]) -> tuple[list[list[dict]], list[str], list[str]]:
        """
        Short-term reflection before crossovering two individuals.
        """
        messages_lst = []
        worse_code_lst = []
        better_code_lst = []
        for i in range(0, len(population), 2):
            parent_1 = population[i]
            parent_2 = population[i + 1]
            messages, worse_code, better_code = self.gen_short_term_reflection_prompt(parent_1, parent_2)
            messages_lst.append(messages)
            worse_code_lst.append(worse_code)
            better_code_lst.append(better_code)

        response_lst = self.short_reflector_llm.multi_chat_completion(messages_lst)
        return response_lst, worse_code_lst, better_code_lst

    def long_term_reflection(self, short_term_reflections: list[str]) -> None:
        """
        Long-term reflection before mutation.
        """
        system = self.system_reflector_prompt
        user = self.user_reflector_lt_prompt.format(
            problem_desc=self.problem_desc,
            prior_reflection=self.long_term_reflection_str,
            new_reflection="\n".join(short_term_reflections),
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        if self.print_long_term_reflection_prompt:
            self.print_long_term_reflection_prompt = False

        self.long_term_reflection_str = self.long_reflector_llm.multi_chat_completion([messages])[0]

        file_name = f"problem_iter{self.iteration}_short_term_reflections.txt"
        with open(file_name, "w", encoding="utf-8") as file:
            file.writelines("\n".join(short_term_reflections) + "\n")

        file_name = f"problem_iter{self.iteration}_long_term_reflection.txt"
        with open(file_name, "w", encoding="utf-8") as file:
            file.writelines(self.long_term_reflection_str + "\n")

    def crossover(self, short_term_reflection_tuple: tuple[list[list[dict]], list[str], list[str]]) -> list[dict]:
        reflection_content_lst, worse_code_lst, better_code_lst = short_term_reflection_tuple
        messages_lst = []
        for reflection, worse_code, better_code in zip(reflection_content_lst, worse_code_lst, better_code_lst):
            system = self.system_generator_prompt
            func_signature0 = self.func_signature.format(version=0)
            func_signature1 = self.func_signature.format(version=1)
            user = self.crossover_prompt.format(
                user_generator=self.user_generator_prompt,
                func_signature0=func_signature0,
                func_signature1=func_signature1,
                worse_code=worse_code,
                better_code=better_code,
                reflection=reflection,
                func_name=self.func_name,
            )
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            messages_lst.append(messages)

            if self.print_crossover_prompt:
                self.print_crossover_prompt = False

        response_lst = self.crossover_llm.multi_chat_completion(messages_lst)
        crossed_population = [self.response_to_individual(response, response_id) for response_id, response in enumerate(response_lst)]
        assert len(crossed_population) == self.cfg.pop_size
        return crossed_population

    def mutate(self) -> list[dict]:
        """Elitist-based mutation. We only mutate the best individual to generate n_pop new individuals."""
        system = self.system_generator_prompt
        func_signature1 = self.func_signature.format(version=1)
        user = self.mutation_prompt.format(
            user_generator=self.user_generator_prompt,
            reflection=self.long_term_reflection_str + self.external_knowledge,
            func_signature1=func_signature1,
            elitist_code=filter_code(self.elitist["code"]),
            func_name=self.func_name,
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if self.print_mutate_prompt:
            self.print_mutate_prompt = False
        responses = self.mutation_llm.multi_chat_completion([messages], int(self.cfg.pop_size * self.mutation_rate))
        population = [self.response_to_individual(response, response_id) for response_id, response in enumerate(responses)]
        return population

    def _record_best_result_adapter(self) -> None:
        if self.adapter_enabled and self._adapter_mode == "static_risk2" and isinstance(self._adapter_best_record, dict):
            self.best_code_overall = self._adapter_best_record.get("code")
            self.best_code_path_overall = self._adapter_best_record.get("code_path")
            self.best_obj_overall = float(self._adapter_best_record.get("obj", float("inf")))
        if not self.adapter_enabled or self.best_code_overall is None:
            return
        assert self._adapter_dir is not None
        single = [{"code": self.best_code_overall, "response_id": -1}]
        train_eval = self.evaluate_population(
            single,
            split="train",
            count_fe=False,
            eval_stage="record_best_train_eval",
        )
        train_info = train_eval[0].get("eval_info") if train_eval else None

        val_info = None
        if self._adapter_has_val:
            val_eval = self.evaluate_population(
                single,
                split="val",
                count_fe=False,
                eval_stage="record_best_val_eval",
            )
            val_info = val_eval[0].get("eval_info") if val_eval else None

        result = {
            "mode": self._adapter_mode,
            "metric_basis": self._adapter_metric_basis,
            "subgroup_eval_mode": self._adapter_subgroup_eval_mode,
            "subgroup_tail_basis": self._adapter_subgroup_tail_basis,
            "best_obj_overall": self.best_obj_overall,
            "best_code_path_overall": self.best_code_path_overall,
            "best_selection_basis": "val_dds" if self._adapter_mode == "static_risk2" else "train_obj",
            "best_selection_vector": (
                self._adapter_best_record.get("objective_vector")
                if isinstance(self._adapter_best_record, dict)
                else None
            ),
            "best_selection_metric": (
                float(self._adapter_best_record.get("obj", float("inf")))
                if isinstance(self._adapter_best_record, dict)
                else None
            ),
            "function_evals": int(self.function_evals),
            "train_eval": train_info,
            "val_eval": val_info,
        }
        out_path = self._adapter_dir / "best_result_adapter.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        self.adapter_best_result_path = out_path
        logging.info(f"Adapter best result saved: {print_hyperlink(out_path)}")

    def _evolve_default(self) -> tuple[str, str]:
        while self.function_evals < self.cfg.max_fe:
            if all([not individual["exec_success"] for individual in self.population]):
                raise RuntimeError(f"All individuals are invalid. Please check the stdout files in {os.getcwd()}.")
            population_to_select = (
                self.population
                if (self.elitist is None or self.elitist in self.population)
                else [self.elitist] + self.population
            )
            selected_population = self.random_select(population_to_select)
            if selected_population is None:
                raise RuntimeError("Selection failed. Please check the population.")
            short_term_reflection_tuple = self.short_term_reflection(selected_population)
            crossed_population = self.crossover(short_term_reflection_tuple)
            self.population = self.evaluate_population(crossed_population, split="train", count_fe=True)
            self.update_iter()
            self.long_term_reflection([response for response in short_term_reflection_tuple[0]])
            mutated_population = self.mutate()
            self.population.extend(self.evaluate_population(mutated_population, split="train", count_fe=True))
            self.update_iter()
        return self.best_code_overall, self.best_code_path_overall

    def _evolve_adapter(self) -> tuple[str, str]:
        while self.function_evals < self.cfg.max_fe:
            self._adapter_generation_id += 1
            if all([not bool(individual.get("exec_success", False)) for individual in self.population]):
                raise RuntimeError(f"All individuals are invalid. Please check the adapter stdout files in {os.getcwd()}.")

            population_to_select = (
                self.population
                if (self.elitist is None or self.elitist in self.population)
                else [self.elitist] + self.population
            )
            if self._adapter_mode == "static_risk2":
                selected_population = self._dds_pair_select(population_to_select)
                if selected_population is None:
                    raise RuntimeError("Risk2 train-DDS parent selection failed.")
                short_term_reflection_tuple = self.short_term_reflection(selected_population)
                crossed_population = self.crossover(short_term_reflection_tuple)
                crossed_population = self.evaluate_population(
                    crossed_population,
                    split="train",
                    count_fe=True,
                    eval_stage="crossover_eval",
                )
                self.long_term_reflection([response for response in short_term_reflection_tuple[0]])
                if self.elitist is None:
                    raise RuntimeError("Risk2 train-DDS elite selection failed: no elitist is available for mutation.")
                mutated_population = self.mutate()
                mutated_population = self.evaluate_population(
                    mutated_population,
                    split="train",
                    count_fe=True,
                    eval_stage="mutation_eval",
                )

                candidate_pool_train: list[dict] = []
                if self._adapter_is_feasible(self.elitist, strict_risk2=True):
                    candidate_pool_train.append(dict(self.elitist))
                candidate_pool_train.extend(dict(ind) for ind in crossed_population)
                candidate_pool_train.extend(dict(ind) for ind in mutated_population)
                if not candidate_pool_train:
                    raise RuntimeError("Risk2 candidate pool is empty after crossover and mutation.")

                candidate_pool_val = self.evaluate_population(
                    [dict(ind) for ind in candidate_pool_train],
                    split="val",
                    count_fe=False,
                    eval_stage="candidate_pool_val_eval",
                )
                next_population_val = self._select_next_population_from_val_rows(
                    candidate_pool_val,
                    target_size=int(max(self.cfg.pop_size, 1)),
                )
                if not next_population_val:
                    raise RuntimeError("Risk2 val-DDS population selection failed.")
                self._update_risk2_best_from_val_rows(next_population_val)

                next_population_train = self.evaluate_population(
                    [dict(ind) for ind in next_population_val],
                    split="train",
                    count_fe=False,
                    eval_stage="next_generation_train_eval",
                )
                next_population_train = self._manage_population(
                    next_population_train,
                    size=int(max(self.cfg.pop_size, 1)),
                    use_multi_objective=True,
                )
                if len(next_population_train) < 2:
                    raise RuntimeError("Risk2 next-generation train population size is < 2.")

                self.population = [dict(ind) for ind in next_population_train]
                self._adapter_last_eval_rows = list(self.population)
                self.update_iter()
                continue

            selected_population = self.random_select(population_to_select)
            if selected_population is None:
                logging.warning("Selection failed for crossover; fallback to mutation-only update for this generation.")
                mutated_population = self.mutate()
                mutated_population = self.evaluate_population(
                    mutated_population,
                    split="train",
                    count_fe=True,
                    eval_stage="mutation_only_eval",
                )
                iter_eval_rows = list(mutated_population)
                combined = list(mutated_population)
                if self.elitist is not None:
                    combined.append(dict(self.elitist))
            else:
                short_term_reflection_tuple = self.short_term_reflection(selected_population)
                crossed_population = self.crossover(short_term_reflection_tuple)
                crossed_population = self.evaluate_population(
                    crossed_population,
                    split="train",
                    count_fe=True,
                    eval_stage="crossover_eval",
                )

                # Keep ReEvo's original two-stage rhythm:
                # update once after crossover evaluation, then again after mutation + management.
                self.population = list(crossed_population)
                self._adapter_last_eval_rows = list(crossed_population)
                self.update_iter()

                self.long_term_reflection([response for response in short_term_reflection_tuple[0]])
                mutated_population = self.mutate()
                mutated_population = self.evaluate_population(
                    mutated_population,
                    split="train",
                    count_fe=True,
                    eval_stage="mutation_eval",
                )

                iter_eval_rows = list(crossed_population) + list(mutated_population)
                combined = list(crossed_population) + list(mutated_population)
                if self.elitist is not None:
                    combined.append(dict(self.elitist))

            train_selected = self._select_candidates(
                combined,
                stage="train",
                target_size=self._adapter_train_pool_size,
            )
            if not train_selected:
                train_selected = self._manage_population(
                    combined,
                    size=self._adapter_train_pool_size,
                    use_multi_objective=self._adapter_use_multi_objective(),
                )
            if not train_selected:
                raise RuntimeError("Adapter train selection failed: no feasible candidates.")

            next_population = self._clip_parent_pool(train_selected)
            next_population = self._clip_parent_pool(next_population)
            next_population = self._ensure_population_size(next_population, train_selected)
            if not next_population:
                raise RuntimeError("Population became empty after adapter management.")

            self.population = next_population
            self._adapter_last_eval_rows = list(iter_eval_rows)
            self.update_iter()

        self._record_best_result_adapter()
        return self.best_code_overall, self.best_code_path_overall

    def evolve(self) -> tuple[str, str]:
        if self.adapter_enabled:
            return self._evolve_adapter()
        return self._evolve_default()
