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

    # Basic numeric sanity.
    obj = _safe_float(individual.get("objective"), float("inf"))
    if not np.isfinite(obj):
        return False

    other = individual.get("other_inf")
    if not isinstance(other, dict):
        # Keep backward compatibility for tasks where other_inf is unavailable.
        return True

    status = str(other.get("status", "")).strip().lower()
    if status in {"invalid", "ok_with_penalty"}:
        return False

    try:
        n_invalid = int(other.get("n_invalid", 0))
    except Exception:
        n_invalid = 0
    if n_invalid > 0:
        return False

    return True


def objective_vector(individual: Dict[str, Any]) -> np.ndarray:
    base_obj = _safe_float(individual.get("objective"), float("inf"))
    other = individual.get("other_inf")
    if not isinstance(other, dict):
        other = {}

    grouped_enabled = bool(other.get("grouped_metric_enabled", True))
    subgroup_mode = str(other.get("subgroup_objective_mode", "")).strip().lower()
    if subgroup_mode in {"mean_only", "mean"}:
        grouped_enabled = False

    mean_eval = _safe_float(other.get("group_mean_eval"), float("nan"))
    std_eval = _safe_float(other.get("group_std_eval"), float("nan"))
    worst_eval = _safe_float(other.get("group_worst_eval"), float("nan"))

    if not np.isfinite(mean_eval):
        basis = str(other.get("selection_metric_basis", "cost")).strip().lower()
        if basis == "gap":
            mean_eval = _safe_float(other.get("group_mean_gap"), base_obj)
            std_eval = _safe_float(other.get("group_std_gap"), 0.0)
            worst_eval = _safe_float(other.get("group_worst_gap"), mean_eval)
        else:
            mean_eval = _safe_float(other.get("group_mean_cost"), base_obj)
            std_eval = _safe_float(other.get("group_std_cost"), 0.0)
            worst_eval = _safe_float(other.get("group_worst_cost"), mean_eval)

    if not np.isfinite(mean_eval):
        mean_eval = base_obj
    if not np.isfinite(std_eval):
        std_eval = 0.0
    if not np.isfinite(worst_eval):
        worst_eval = mean_eval

    if not grouped_enabled:
        std_eval = 0.0
        worst_eval = mean_eval

    return np.asarray([mean_eval, std_eval, worst_eval], dtype=float)


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
