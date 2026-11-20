import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'erdr/EoH-tsp/src'))
from snc_tsp_problem import TSPGLSUpdateEdgeProblem
from erdr_shared.dynamic import snapshot
from erdr_shared.support import tour_cost
from gls_core import _tour_to_route2end, route2tour, tour_cost_2End
import gls_operators


class EohWarmStartChecks(unittest.TestCase):
    def test_operator_costs_and_dummy_neighbors(self):
        rng=np.random.default_rng(81)
        for _ in range(10):
            coords=rng.uniform(size=(20,2))
            matrix,warm,_=snapshot(coords,np.arange(20),.2,'request_arrival',return_coordinates=True)
            matrix=matrix.astype(np.float32)
            warm[1:-2]=rng.permutation(warm[1:-2])
            neighbors=np.argsort(matrix,axis=1).astype(np.int32)
            initial=_tour_to_route2end(warm.astype(np.int32))
            for name in ['two_opt_o2a_all','relocate_o2a_all']:
                for node in range(len(warm)):
                    route=initial.copy()
                    before=tour_cost_2End(matrix,route)
                    delta,result=getattr(gls_operators,name)(route,matrix,neighbors,node)
                    after=tour_cost_2End(matrix,result)
                    self.assertAlmostEqual(after-before,delta,places=4)
                    self.assertEqual(set(result[-1].tolist()),{0,len(warm)-2})
                    self.assertEqual(sorted(route2tour(result)),list(range(len(warm))))

    def setUp(self):
        coords=np.random.default_rng(23).uniform(size=(20,2))
        self.matrix,self.warm,self.coords=snapshot(coords,np.arange(20),.5,'request_arrival',return_coordinates=True)
        self.problem=TSPGLSUpdateEdgeProblem([(self.coords,self.matrix)],opt_costs=[1.],ite_max=1,time_limit_per_instance=2)
        self.problem.initial_tours=[self.warm]

    def test_process_evaluation_preserves_warm_route(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'erdr/EoH-tsp'))
        from eoh.methods.eoh.evaluator_process import evaluate_metric
        from ils_tsp_problem import TSPILSProblem
        from erdr_shared.task import SEED
        for backend in ['gls', 'ils']:
            problem = self.problem if backend == 'gls' else TSPILSProblem(
                [(self.coords,self.matrix)], opt_costs=[1.], ite_max=1,time_limit_per_instance=2)
            problem.initial_tours = [self.warm]
            problem.ite_max = 0
            code = SEED if backend == 'ils' else "def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):\n    return edge_distance.copy()\n"
            evaluate_metric(problem, code)
            info = problem.get_last_eval_info()
            self.assertEqual(info['status'], 'ok', info)
            self.assertAlmostEqual(info['avg_obj'],tour_cost(self.matrix,self.warm),places=5)

    def test_exact_initial_objective(self):
        self.problem.ite_max=0
        def identity(d,t,e):return d
        row=self.problem._evaluate_one_instance(identity,0)
        self.assertEqual(row['status'],'ok')
        self.assertAlmostEqual(row['obj'],tour_cost(self.matrix,self.warm),places=5)

    def test_failure_penalty(self):
        def failure(d,t,e):raise RuntimeError('invalid candidate')
        row=self.problem._evaluate_one_instance(failure,0)
        self.assertEqual(row['status'],'invalid')
        self.assertEqual(row['runtime_ms'],4000.)

    def test_guidance_excludes_dummy(self):
        calls=[]
        def guide(d,t,e):
            self.assertEqual(d.shape,(len(self.warm)-1,len(self.warm)-1))
            self.assertEqual(e.shape,d.shape)
            self.assertEqual(sorted(t),list(range(len(d))))
            calls.append(True)
            return d*1.01
        row=self.problem._evaluate_one_instance(guide,0)
        self.assertEqual(row['status'],'ok')
        self.assertTrue(calls)


if __name__=='__main__':unittest.main()
