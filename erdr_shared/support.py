from dataclasses import dataclass
import importlib.util
import numpy as np

@dataclass(frozen=True)
class SearchResult:
    tour: tuple
    cost: float
    evaluations: int
    iterations: int
    elapsed_sec: float

def validate_matrix(matrix):
    n = len(matrix)
    if matrix.shape != (n, n) or not np.isfinite(matrix).all() or np.any(matrix < 0) or not np.allclose(matrix, matrix.T):
        raise ValueError("Expected a finite, nonnegative, symmetric distance matrix")

def tour_cost(matrix, tour):
    route = np.asarray(tour, dtype=int)
    return float(matrix[route, np.roll(route, -1)].sum())

def load_rule(path):
    spec = importlib.util.spec_from_file_location("ils_candidate", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("perturb_tour_v2", "perturb_tour_v1", "perturb_tour"):
        rule = getattr(module, name, None)
        if callable(rule):
            return rule
    raise ValueError("Candidate does not define perturb_tour(distance, current_tour, search_state, rng)")
