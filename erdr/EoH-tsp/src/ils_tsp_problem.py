import inspect
import numpy as np
from snc_tsp_problem import TSPGLSUpdateEdgeProblem
from erdr_shared.ils import solve
from erdr_shared.task import DESCRIPTION, ARGS

class ILSPrompts:
    def get_task(self): return DESCRIPTION
    def get_func_name(self): return "perturb_tour"
    def get_func_inputs(self): return list(ARGS)
    def get_func_outputs(self): return ["perturbed_tour"]
    def get_inout_inf(self): return DESCRIPTION
    def get_other_inf(self): return "Use the supplied random.Random rng. Return a permutation preserving fixed endpoints."

class TSPILSProblem(TSPGLSUpdateEdgeProblem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompts = ILSPrompts()

    @staticmethod
    def _compile_function(code):
        namespace = {"np": np}
        try:
            exec(code, namespace)
            fn = namespace.get("perturb_tour")
            if not callable(fn): raise ValueError("perturb_tour not found")
            return fn, None
        except Exception as exc:
            return None, str(exc)

    @staticmethod
    def _check_signature(fn):
        ok = tuple(inspect.signature(fn).parameters) == ARGS
        return ok, None if ok else "Expected perturb_tour(distance, current_tour, search_state, rng)"

    def _evaluate_one_instance(self, fn, idx):
        try:
            warm_routes = getattr(self, "initial_tours", None)
            result = solve(self.instance_data[idx][1], perturbation_rule=fn,
                           iter_limit=self.ite_max, time_limit_sec=self.time_limit_per_instance, seed=idx+1,
                           initial_tour=None if warm_routes is None else warm_routes[idx])
            gap = (result.cost - self._reference_costs[idx]) / self._reference_costs[idx]
            return dict(status="ok", obj=result.cost, gap=float(gap), runtime_ms=result.elapsed_sec*1000)
        except Exception:
            return dict(status="invalid", obj=self.LARGE, gap=self.LARGE, runtime_ms=2000*self.time_limit_per_instance)
