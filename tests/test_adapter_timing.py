import ast
from contextlib import nullcontext
import types
import unittest
from pathlib import Path

import numpy as np


class AdapterTimingTest(unittest.TestCase):
    def test_initialization_excluded_guidance_and_search_included(self):
        root = Path(__file__).resolve().parents[1]
        for backbone in ("Hercules-tsp", "reevo-tsp"):
            with self.subTest(backbone=backbone):
                source = root / "erdr" / backbone / "problems/tsp_gls/eval_adapter.py"
                tree = ast.parse(source.read_text(encoding="utf-8"))
                names = {"evaluate_candidate", "_population_std", "_worst_half_mean", "_risk_weighted_metric"}
                body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
                body.extend(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names)
                clock = [0.0]
                def init(dist, start):
                    clock[0] += 100.0
                    return np.arange(len(dist), dtype=np.uint16)
                def descent(*args):
                    clock[0] += 50.0
                def guide(dist):
                    clock[0] += 2.0
                    return np.ones_like(dist)
                def search(dist, weights, **kwargs):
                    self.assertEqual(kwargs["deadline_perf_counter"], 160.0)
                    np.testing.assert_array_equal(kwargs["initial_tour"], np.arange(4))
                    clock[0] += 3.0
                    return kwargs["initial_tour"]
                def cost(*args):
                    clock[0] += 20.0
                    return 4.0
                scope = dict(initialization_phase=nullcontext, np=np, time=types.SimpleNamespace(perf_counter=lambda: clock[0]), ILS_ENABLED=False,
                             LARGE=1e10, _init_nearest_neighbor=init, _local_search=descent,
                             guided_local_search=search, _calculate_cost=cost,
                             _reference_cost=lambda *args: (4.0, "test"))
                exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(source), "exec"), scope)
                instance = types.SimpleNamespace(distmat=np.ones((4, 4))-np.eye(4), reference_cost=4.0)
                kwargs = dict(dataset=[instance], heuristics_fn=guide, metric_basis="gap", subgroup_mode="none",
                              subgroup_tail_basis="time", group_indices=[[0]], perturbation_moves=1,
                              iter_limit=1, time_limit_per_instance=10.0, gap_eps=1e-9)
                if backbone == "reevo-tsp":
                    kwargs.update(references=np.array([4.0]), risk_weight_mean=1.0, risk_weight_std=1.0,
                                  risk_weight_worst=1.0, reference_cost_source="test")
                result = scope["evaluate_candidate"](**kwargs)
                self.assertEqual(result["n_invalid"], 0)
                self.assertEqual(result["mean_time_sec"], 5.0)


if __name__ == "__main__":
    unittest.main()
