import random

import numpy as np

from ..dds_utils import compute_dds_scores, objective_vector, is_feasible_individual


def _is_valid(individual):
    if not isinstance(individual, dict):
        return False
    if not str(individual.get("code", "")).strip():
        return False
    if not is_feasible_individual(individual):
        return False
    vec = objective_vector(individual)
    return bool(np.isfinite(vec).all())


def parent_selection(pop, m, eps_rel=1e-3, softmax_tau=1.0, novelty_weight=0.05):
    n_parent = max(int(m), 1)
    valid = [ind for ind in pop if _is_valid(ind)]
    if not valid:
        # Strict feasibility gate: invalid/penalized candidates are not allowed
        # in the parent pool.
        return []

    scores, _, _, _ = compute_dds_scores(
        population=valid,
        eps_rel=float(max(eps_rel, 0.0)),
        novelty_weight=float(max(novelty_weight, 0.0)),
    )
    tau = float(max(softmax_tau, 1e-6))
    logits = np.asarray(scores, dtype=float) / tau
    logits = logits - float(np.max(logits))
    weights = np.exp(logits)
    if (not np.isfinite(weights).all()) or float(np.sum(weights)) <= 0.0:
        probs = [1.0 / len(valid)] * len(valid)
    else:
        probs = (weights / float(np.sum(weights))).tolist()
    return random.choices(valid, weights=probs, k=n_parent)
