import multiprocessing
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "erdr/EoH-tsp/eoh/methods/eoh"))
from evaluator_process import evaluate_in_process


class Evaluator:
    def evaluate(self, code):
        if code == "initialize":
            from erdr_shared.timing import initialization_phase
            with initialization_phase():
                time.sleep(0.6)
            code = "3"
        if code == "hang":
            while True:
                time.sleep(0.05)
        if code == "raise":
            raise ValueError("invalid candidate")
        self.result = float(code) ** 2
        return self.result

    def get_last_eval_info(self):
        return {"objective": self.result}


class ProcessEvaluationTest(unittest.TestCase):
    def test_returns_objective_and_metadata(self):
        result, info = evaluate_in_process(Evaluator(), "3", 10)
        self.assertEqual(result, 9.0)
        self.assertEqual(info, {"objective": 9.0})

    def test_initialization_excluded(self):
        result, info = evaluate_in_process(Evaluator(), "initialize", 0.2)
        self.assertEqual(result, 9.0)

    def test_propagates_failure(self):
        with self.assertRaisesRegex(RuntimeError, "invalid candidate"):
            evaluate_in_process(Evaluator(), "raise", 10)

    def test_terminates_hung_candidate(self):
        before = {child.pid for child in multiprocessing.active_children()}
        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            evaluate_in_process(Evaluator(), "hang", 1)
        self.assertLess(time.monotonic() - started, 5)
        after = {child.pid for child in multiprocessing.active_children()}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
