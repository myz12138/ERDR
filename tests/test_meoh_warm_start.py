import sys
import tempfile
import pickle
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'erdr/MEOH-tsp'))
from llm4ad.task.optimization.tsp_gls_2O.evaluation import load_tsp_dataset, solve_with_time, TSPGLSMultiObjectiveEvaluation
from llm4ad.base.evaluate import SecureEvaluator
from llm4ad.task.optimization.tsp_gls_2O.get_instance import TSPInstance
import multiprocessing
import time
from erdr_shared.dynamic import snapshot
from erdr_shared.support import tour_cost


class MeohWarmStartChecks(unittest.TestCase):
    def test_secure_evaluator_real_search_and_hang(self):
        coords = np.random.default_rng(18).uniform(size=(10,2))
        instance = TSPInstance(coords)
        evaluator = TSPGLSMultiObjectiveEvaluation(instances=[instance], reference_costs=[1.0],
            iter_limit=1, perturbation_moves=1, time_limit_per_instance=1.0)
        secure = SecureEvaluator(evaluator)
        code = "def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):\n    return edge_distance.copy()\n"
        result = secure.evaluate_program(code)
        self.assertIsNotNone(result)
        self.assertTrue(np.isfinite(result).all())
        before = {p.pid for p in multiprocessing.active_children()}
        evaluator.batch_time_budget = 0.5
        started = time.monotonic()
        hung = "def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):\n    while True:\n        pass\n"
        self.assertIsNone(secure.evaluate_program(hung))
        self.assertLess(time.monotonic()-started, 12)
        self.assertEqual(before, {p.pid for p in multiprocessing.active_children()})

    def test_loader_and_solver_preserve_start(self):
        coords=np.random.default_rng(29).uniform(size=(20,2))
        matrix,warm,positions=snapshot(coords,np.arange(20),.5,'request_arrival',return_coordinates=True)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            path=Path(folder)/'dataset.pkl'
            with path.open('wb') as handle:
                pickle.dump(dict(coordinate=[positions],distance_matrix=[matrix],initial_tour=[warm],cost_concorde=[1.]),handle)
            instances,_=load_tsp_dataset(path)
            np.testing.assert_array_equal(instances[0].initial_tour,warm)
            def identity(d,t,e):return d
            cost,elapsed=solve_with_time(instances[0],identity,2,0,2)
            self.assertAlmostEqual(cost,tour_cost(matrix,warm),places=5)
            self.assertGreaterEqual(elapsed,0)
            def failure(d,t,e):raise RuntimeError('invalid candidate')
            cost,elapsed=solve_with_time(instances[0],failure,2,1,2)
            self.assertTrue(np.isinf(cost))
            self.assertEqual(elapsed,4)


if __name__=='__main__':unittest.main()
