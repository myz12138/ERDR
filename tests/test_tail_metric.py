import ast
import unittest
from pathlib import Path
import numpy as np


class TailMetricTest(unittest.TestCase):
    def test_odd_and_even_group_counts(self):
        root = Path(__file__).resolve().parents[1]
        files = ["erdr/EoH-tsp/src/snc_tsp_problem.py",
                 "erdr/Hercules-tsp/problems/tsp_gls/eval_adapter.py",
                 "erdr/reevo-tsp/problems/tsp_gls/eval_adapter.py",
                 "erdr/MEOH-tsp/llm4ad/task/optimization/tsp_gls_2O/evaluation.py"]
        for filename in files:
            tree = ast.parse((root/filename).read_text(encoding="utf-8"))
            fn = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                      and node.name in ["_worst_half_mean", "_tail_mean"])
            fn.decorator_list = []
            module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),fn],type_ignores=[])
            scope = {"np": np}
            exec(compile(ast.fix_missing_locations(module), filename, "exec"),scope)
            for values, expected in [([1.,2.,3.,4.,5.],4.0),([1.,2.,3.,4.],3.5),([2.],2.0)]:
                with self.subTest(file=filename, values=values):
                    value = scope[fn.name](np.array(values))
                    if isinstance(value, tuple):
                        self.assertEqual(value[1],(len(values)+1)//2)
                        value=value[0]
                    self.assertEqual(value,expected)


if __name__ == "__main__":
    unittest.main()
