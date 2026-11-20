import json
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from erdr_shared.dynamic import snapshot
from erdr_shared.support import tour_cost


class AdapterCliTest(unittest.TestCase):
    def test_warm_route_through_gls_and_ils_entry_points(self):
        root = Path(__file__).resolve().parents[1]
        coords = np.random.default_rng(4).uniform(size=(20,2))
        matrix, warm, positions = snapshot(coords, np.arange(20), .5, "request_arrival", return_coordinates=True)
        initial_cost = tour_cost(matrix, warm)
        with tempfile.TemporaryDirectory(dir=root / "tests") as folder:
            folder = Path(folder)
            dataset = folder / "dynamic.pkl"
            with dataset.open("wb") as handle:
                pickle.dump(dict(coordinate=[positions], distance_matrix=[matrix], initial_tour=[warm],
                                 cost_concorde=[initial_cost]), handle)
            for name in ["Hercules-tsp", "reevo-tsp"]:
                for backend in ["gls", "ils"]:
                    with self.subTest(method=name, backend=backend):
                        code = (root / "erdr" / name / "prompts" / ("tsp_"+backend) / "seed_func.txt").read_text(encoding="utf-8")
                        code = code.replace("```python", "").replace("```", "").replace("heuristics_v1", "heuristics_v2")
                        solver = folder / "candidate.py"
                        solver.write_text("import numpy as np\n"+code, encoding="utf-8")
                        result_path = folder / "result.json"
                        result_path.unlink(missing_ok=True)
                        script = root / "erdr" / name / "problems/tsp_gls/eval_adapter.py"
                        env = os.environ.copy()
                        env["ERDR_SEARCH_BACKEND"] = backend
                        env["PYTHONPATH"] = str(root)+os.pathsep+env.get("PYTHONPATH", "")
                        command = [sys.executable, str(script), "--dataset-path", str(dataset),
                                   "--code-path", str(solver), "--output-json", str(result_path),
                                   "--iter-limit", "0", "--time-limit-per-instance", "2"]
                        completed = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=45)
                        self.assertEqual(completed.returncode, 0, completed.stderr)
                        result = json.loads(result_path.read_text(encoding="utf-8"))
                        self.assertEqual(result["status"], "ok", result)
                        self.assertEqual(result["n_invalid"], 0, result)
                        self.assertAlmostEqual(result["mean_cost"], initial_cost, places=5)
                        self.assertGreaterEqual(result["mean_time_sec"], 0)


if __name__ == "__main__":
    unittest.main()
