from __future__ import annotations

import ast
import math
import re
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


_TOKEN_PATTERN = re.compile(r"[A-Za-z_]\w+|\d+")


def _safe_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return float(out)


def is_feasible_individual(individual: Dict[str, Any]) -> bool:
    if not isinstance(individual, dict):
        return False
    if not bool(individual.get("exec_success", False)):
        return False
    obj = _safe_float(individual.get("obj"), float("inf"))
    if not np.isfinite(obj):
        return False
    info = individual.get("eval_info")
    if isinstance(info, dict):
        try:
            n_invalid = int(info.get("n_invalid", 0))
        except Exception:
            n_invalid = 0
        if n_invalid > 0:
            return False
    return True


def objective_vector(individual: Dict[str, Any]) -> np.ndarray:
    info = individual.get("eval_info")
    if isinstance(info, dict):
        vec = info.get("objective_vector")
        if isinstance(vec, (list, tuple)) and len(vec) >= 3:
            a = _safe_float(vec[0], float("inf"))
            b = _safe_float(vec[1], 0.0)
            c = _safe_float(vec[2], float("inf"))
            return np.asarray([a, b, c], dtype=float)

        mean_eval = _safe_float(info.get("group_mean_eval"), float("nan"))
        std_eval = _safe_float(info.get("group_std_eval"), float("nan"))
        worst_eval = _safe_float(info.get("group_worst_eval"), float("nan"))
        if np.isfinite(mean_eval):
            if not np.isfinite(std_eval):
                std_eval = 0.0
            if not np.isfinite(worst_eval):
                worst_eval = mean_eval
            return np.asarray([mean_eval, std_eval, worst_eval], dtype=float)

    obj = _safe_float(individual.get("obj"), float("inf"))
    return np.asarray([obj, 0.0, obj], dtype=float)


def _epsilon_from_population(
    objective_matrix: np.ndarray,
    eps_rel: float = 1e-3,
    eps_min: float = 1e-9,
) -> np.ndarray:
    arr = np.asarray(objective_matrix, dtype=float)
    if arr.size <= 0:
        return np.asarray([eps_min, eps_min, eps_min], dtype=float)

    eps_rel_f = float(max(eps_rel, 0.0))
    eps_min_f = float(max(eps_min, 0.0))
    out = np.zeros(arr.shape[1], dtype=float)
    for d in range(arr.shape[1]):
        vals = arr[:, d]
        vals = vals[np.isfinite(vals)]
        if vals.size <= 0:
            span = 1.0
        else:
            span = float(np.max(vals) - np.min(vals))
        out[d] = float(max(span * eps_rel_f, eps_min_f))
    return out


def _dominates(a: np.ndarray, b: np.ndarray, eps: np.ndarray) -> bool:
    return bool(np.all(a <= b + eps) and np.any(a < b - eps))


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


def _similarity_asymm(counter_a: Counter[str], counter_b: Counter[str]) -> float:
    denom = int(sum(counter_b.values()))
    if denom <= 0:
        return 0.0
    overlap = 0
    for token, count_b in counter_b.items():
        overlap += min(int(counter_a.get(token, 0)), int(count_b))
    return float(max(min(overlap / max(denom, 1), 1.0), 0.0))


def _similarity_sym(counter_a: Counter[str], counter_b: Counter[str]) -> float:
    s_ab = _similarity_asymm(counter_a, counter_b)
    s_ba = _similarity_asymm(counter_b, counter_a)
    return float(0.5 * (s_ab + s_ba))


def compute_dds_scores(
    population: Sequence[Dict[str, Any]],
    eps_rel: float = 1e-3,
    novelty_weight: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = int(len(population))
    if n <= 0:
        empty_f = np.zeros(0, dtype=float)
        empty_b = np.zeros(0, dtype=bool)
        empty_i = np.zeros(0, dtype=int)
        return empty_f, np.zeros((0, 3), dtype=float), empty_b, empty_i

    objectives = np.asarray([objective_vector(ind) for ind in population], dtype=float)
    eps = _epsilon_from_population(objectives, eps_rel=eps_rel)
    counters = [_code_counter(str(ind.get("code", ""))) for ind in population]

    scores = np.zeros(n, dtype=float)
    dominated_counts = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            oi = objectives[i]
            oj = objectives[j]
            if _dominates(oi, oj, eps):
                sim = _similarity_asymm(counters[i], counters[j])
                scores[j] += float(-sim)
                dominated_counts[j] += 1
            elif _dominates(oj, oi, eps):
                sim = _similarity_asymm(counters[j], counters[i])
                scores[i] += float(-sim)
                dominated_counts[i] += 1

    nondominated = dominated_counts == 0
    novelty_w = float(max(novelty_weight, 0.0))
    if novelty_w > 0.0 and n > 1:
        nd_indices = np.where(nondominated)[0].tolist()
        for i in nd_indices:
            novelty_vals: List[float] = []
            for j in range(n):
                if i == j:
                    continue
                sim = _similarity_sym(counters[i], counters[j])
                novelty_vals.append(float(1.0 - sim))
            if novelty_vals:
                scores[i] += novelty_w * float(np.mean(np.asarray(novelty_vals, dtype=float)))

    return scores, objectives, nondominated, dominated_counts


def sort_population_indices_by_dds(
    population: Sequence[Dict[str, Any]],
    eps_rel: float = 1e-3,
    novelty_weight: float = 0.05,
) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray]:
    scores, objectives, nondominated, _ = compute_dds_scores(
        population=population,
        eps_rel=eps_rel,
        novelty_weight=novelty_weight,
    )
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


def _normalize_code(code: str) -> str:
    return "\n".join(line.rstrip() for line in str(code).strip().splitlines())


def _deduplicate_by_code(population: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_code: Dict[str, Dict[str, Any]] = {}
    ordered: List[str] = []
    for ind in population:
        if not is_feasible_individual(ind):
            continue
        code_norm = _normalize_code(str(ind.get("code", "")))
        if not code_norm:
            continue
        prev = by_code.get(code_norm)
        if prev is None:
            by_code[code_norm] = ind
            ordered.append(code_norm)
            continue
        if tuple(objective_vector(ind)) < tuple(objective_vector(prev)):
            by_code[code_norm] = ind
    return [by_code[k] for k in ordered if k in by_code]


def select_population_dds(
    population: Sequence[Dict[str, Any]],
    size: int,
    eps_rel: float = 1e-3,
    novelty_weight: float = 0.05,
) -> List[Dict[str, Any]]:
    target_size = int(max(size, 0))
    if target_size <= 0:
        return []

    unique = _deduplicate_by_code(population)
    if not unique:
        return []

    idx_sorted, _, _, _ = sort_population_indices_by_dds(
        population=unique,
        eps_rel=eps_rel,
        novelty_weight=novelty_weight,
    )
    keep = idx_sorted[: min(target_size, len(idx_sorted))]
    return [unique[idx] for idx in keep]

