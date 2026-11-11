from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from ..dds_utils import is_feasible_individual, objective_vector, sort_population_indices_by_dds


def _normalize_code(code: str) -> str:
    return "\n".join(line.rstrip() for line in str(code).strip().splitlines())


def _is_valid(individual: Dict) -> bool:
    if not isinstance(individual, dict):
        return False
    if not str(individual.get("code", "")).strip():
        return False
    if not is_feasible_individual(individual):
        return False
    vec = objective_vector(individual)
    return bool(np.isfinite(vec).all())


def _lex_key(individual: Dict) -> tuple:
    vec = objective_vector(individual)
    raw_obj = individual.get("objective")
    try:
        obj = float(raw_obj)
    except Exception:
        obj = float("inf")
    return (float(vec[0]), float(vec[2]), float(vec[1]), float(obj))


def _deduplicate_by_code(population: Sequence[Dict]) -> List[Dict]:
    by_code: Dict[str, Dict] = {}
    ordered: List[str] = []
    for ind in population:
        if not _is_valid(ind):
            continue
        code_norm = _normalize_code(str(ind.get("code", "")))
        if not code_norm:
            continue
        prev = by_code.get(code_norm)
        if prev is None:
            by_code[code_norm] = ind
            ordered.append(code_norm)
            continue
        if _lex_key(ind) < _lex_key(prev):
            by_code[code_norm] = ind
    return [by_code[k] for k in ordered if k in by_code]


def population_management(pop, size, eps_rel=1e-3, novelty_weight=0.05):
    target_size = max(int(size), 0)
    if target_size <= 0:
        return []

    unique = _deduplicate_by_code(pop)
    if not unique:
        return []

    idx_sorted, _, _, _ = sort_population_indices_by_dds(
        population=unique,
        eps_rel=float(max(eps_rel, 0.0)),
        novelty_weight=float(max(novelty_weight, 0.0)),
    )
    keep = idx_sorted[: min(target_size, len(idx_sorted))]
    return [unique[idx] for idx in keep]
