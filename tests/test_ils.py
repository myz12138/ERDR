"""Deterministic kernel/protocol checks. No LLM calls or generated results."""
import unittest
import numpy as np
from erdr_shared.ils import solve, _descent_impl, _descent_jit, _validate_double_bridge
from erdr_shared.ils_seed import perturb_tour
from erdr_shared.dynamic import snapshot, load_sources, evaluate_one
from erdr_shared.support import tour_cost


class ILSChecks(unittest.TestCase):
    def setUp(self):
        self.coords = np.random.default_rng(12).uniform(0, 1, (14, 2))
        self.distance = np.linalg.norm(self.coords[:,None]-self.coords[None,:],axis=2)

    def test_route_and_incumbent(self):
        route = np.arange(14)
        result = solve(self.distance,perturbation_rule=perturb_tour,initial_tour=route,iter_limit=8,seed=3)
        self.assertEqual(sorted(result.tour),route.tolist())
        self.assertEqual(result.tour[0],0)
        self.assertLessEqual(result.cost,tour_cost(self.distance,route)+1e-10)
        self.assertAlmostEqual(result.cost,tour_cost(self.distance,result.tour))
        self.assertEqual(result.iterations,8)

    def test_all_dynamic_scenarios(self):
        for progress in [.2,.5,.8]:
            for change in ['location_change','request_arrival','request_cancellation']:
                matrix,warm=snapshot(self.coords,np.arange(14),progress,change,seed=19)
                result=solve(matrix,perturbation_rule=perturb_tour,initial_tour=warm,iter_limit=5)
                self.assertEqual(result.tour[-2:],(len(matrix)-2,len(matrix)-1))
                self.assertEqual(result.tour[0],0)
                self.assertLessEqual(result.cost,tour_cost(matrix,warm)+1e-10)
                a,b=snapshot(self.coords,np.arange(14),progress,change,seed=19)
                np.testing.assert_array_equal(matrix,a)
                np.testing.assert_array_equal(warm,b)

    def test_double_bridge_validation(self):
        import random
        for n in range(8, 22):
            current = np.arange(n)
            for seed in range(20):
                candidate = np.asarray(perturb_tour(None, current, {"movable_stop":n}, random.Random(seed)))
                _validate_double_bridge(current, candidate, n)
                edges_before = {tuple(sorted((current[i], current[(i+1)%n]))) for i in range(n)}
                edges_after = {tuple(sorted((candidate[i], candidate[(i+1)%n]))) for i in range(n)}
                self.assertEqual(len(edges_before-edges_after), 4)
        current = np.arange(14)
        for candidate in [current, np.r_[0, 2, 1, np.arange(3,14)], np.r_[0,np.arange(13,0,-1)]]:
            with self.assertRaises(ValueError):
                _validate_double_bridge(current, candidate, 14)

    def test_invalid_candidate_rejected(self):
        def invalid(d,t,s,r): return [0]*len(t)
        with self.assertRaises(ValueError):solve(self.distance,perturbation_rule=invalid,iter_limit=1)

    def test_zero_change_ratio(self):
        reference, warm = snapshot(self.coords,np.arange(14),.5,'location_change',sigma=0)
        for change in ['request_arrival','request_cancellation']:
            matrix, route = snapshot(self.coords,np.arange(14),.5,change,ratio=0)
            np.testing.assert_array_equal(matrix,reference)
            np.testing.assert_array_equal(route,warm)

    def test_nonfinite_change_rejected(self):
        for parameter in ['progress','ratio','sigma']:
            args = dict(progress=.5,ratio=.2,sigma=.02)
            args[parameter] = float('nan')
            with self.assertRaises(ValueError):
                snapshot(self.coords,np.arange(14),change='request_arrival',**args)

    def test_exact_initial_route_and_zero_iterations(self):
        warm=np.arange(14)
        result=solve(self.distance,perturbation_rule=perturb_tour,initial_tour=warm,iter_limit=0)
        self.assertEqual(result.tour,tuple(warm))
        self.assertEqual(result.cost,tour_cost(self.distance,warm))

    def test_kernel_equivalence(self):
        a,c1=_descent_impl(self.distance,np.arange(14),14)
        b,c2=_descent_jit(self.distance,np.arange(14),14)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(c1,c2)

    def test_descent_reaches_both_neighborhood_optima(self):
        for stop in [12,14]:
            route,_=_descent_impl(self.distance,np.arange(14),stop)
            cost=tour_cost(self.distance,route)
            self.assertEqual(route[0],0)
            np.testing.assert_array_equal(route[stop:],np.arange(14)[stop:])
            for i in range(1,stop):
                for j in range(i+1,stop):
                    candidate=route.copy()
                    candidate[i:j+1]=candidate[i:j+1][::-1]
                    self.assertGreaterEqual(tour_cost(self.distance,candidate)+1e-10,cost)
                for j in range(stop):
                    if j in [i,i-1]:continue
                    candidate=route.tolist()
                    node=candidate.pop(i)
                    candidate.insert(j if i<j else j+1,node)
                    self.assertGreaterEqual(tour_cost(self.distance,candidate)+1e-10,cost)

    def test_fixed_suffix_rejected(self):
        matrix,warm=snapshot(self.coords,np.arange(14),.5,'request_arrival')
        def invalid(d,t,s,r):
            t[-1],t[1]=t[1],t[-1]
            return t
        with self.assertRaises(ValueError):solve(matrix,perturbation_rule=invalid,initial_tour=warm,iter_limit=1)

    def test_read_only_inputs(self):
        before=self.distance.copy()
        solve(self.distance,perturbation_rule=perturb_tour,iter_limit=2)
        np.testing.assert_array_equal(before,self.distance)

if __name__=='__main__':unittest.main()
