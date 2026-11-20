import csv
import json
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from erdr_shared.dynamic import snapshot
from erdr_shared.support import tour_cost


class DynamicCliTest(unittest.TestCase):
    def test_gls_dynamic_evaluation_entries(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root / "tests") as directory:
            folder = Path(directory)
            coords = np.random.default_rng(4).uniform(size=(20,2))
            for tag, change in [('coord_jitter','location_change'),('node_add','request_arrival'),('node_remove','request_cancellation')]:
                matrix,warm,positions=snapshot(coords,np.arange(20),.5,change,return_coordinates=True)
                with (folder/f"TSP20_dyn_baseline_r50_{tag}.pkl").open('wb') as handle:
                    pickle.dump(dict(coordinate=[positions],distance_matrix=[matrix],initial_tour=[warm],
                                     cost_concorde=[tour_cost(matrix,warm)]),handle)
            for name in ["EoH-tsp", "reevo-tsp", "Hercules-tsp", "MEOH-tsp"]:
                with self.subTest(backbone=name):
                    solver = folder/'solver.py'
                    if name in ['reevo-tsp','Hercules-tsp']:
                        code='import numpy as np\ndef heuristics_v2(distance_matrix):\n    return distance_matrix.copy()\n'
                    else:
                        code='def update_edge_distance(edge_distance,local_opt_tour,edge_n_used):\n    return edge_distance.copy()\n'
                    solver.write_text(code,encoding='utf-8')
                    output=folder/'summary.json'
                    output.unlink(missing_ok=True)
                    script=root/'erdr'/name/'scripts/eval_dynamic_redeploy.py'
                    option='--gls-iter-limit' if name in ['reevo-tsp','Hercules-tsp'] else '--gls-ite-max'
                    command=[sys.executable,str(script),'--repo-root',str(root),'--dynamic-dir',str(folder),
                        '--solver',str(solver),'--dataset-solver-tag','baseline','--method-name',name,
                        '--ratio-tag','r50','--n-instance-generated','1','--perturb-types','coord_jitter,node_add,node_remove',
                        '--gls-time-limit-per-instance','1',option,'1','--gls-perturbation-moves','1','--output-json',str(output)]
                    env=dict(os.environ)
                    env['PYTHONPATH']=str(root)+os.pathsep+env.get('PYTHONPATH','')
                    env.pop('ERDR_SEARCH_BACKEND',None)
                    result=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True,timeout=45)
                    self.assertEqual(result.returncode,0,result.stderr)
                    rows=json.loads(output.read_text(encoding='utf-8'))['rows']
                    self.assertEqual(len(rows),3)
                    for row in rows:
                        self.assertTrue(np.isfinite(row['mean_gap']),row)
                        self.assertLessEqual(row['mean_gap'],1e-5,row)
                        self.assertGreaterEqual(row['mean_time_sec'],0,row)
                        self.assertEqual(row['n_instances_used_in_eval'],1,row)

    def test_four_backbones_share_paired_ils_snapshots(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root / "tests") as directory:
            folder = Path(directory)
            dataset = folder / "source.pkl"
            coords = np.column_stack((np.linspace(0,1,20), np.zeros(20)))
            with dataset.open("wb") as handle:
                pickle.dump(dict(coordinate=[coords], optimal_tour_concorde=[list(range(20))]), handle)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(root)+os.pathsep+env.get("PYTHONPATH", "")
            hashes = []
            for name in ["EoH-tsp", "reevo-tsp", "Hercules-tsp", "MEOH-tsp"]:
                with self.subTest(backbone=name):
                    output = folder / name
                    config = dict(dataset_path=str(dataset), output_dir=str(output), n_instances=1,
                        seed=1, progress=[.5], sigma=.02, change_ratio=.2,
                        solvers=dict(baseline=str(root/"erdr_shared/ils_seed.py"), erdr=str(root/"erdr_shared/ils_seed.py")),
                        search=dict(framework="ils",max_iterations=0,time_limit_seconds=1,watchdog_seconds=15))
                    config_path = folder / "config.yaml"
                    config_path.write_text(yaml.safe_dump(config),encoding="utf-8")
                    script = root/"erdr"/name/"scripts/run_dynamic_evaluation_ils.py"
                    result = subprocess.run([sys.executable,str(script),"--config",str(config_path)],
                        cwd=root,env=env,capture_output=True,text=True,timeout=60)
                    self.assertEqual(result.returncode,0,result.stderr)
                    records = [json.loads(line) for line in (output/"instances.jsonl").read_text(encoding="utf-8").splitlines()]
                    self.assertEqual(len(records),3)
                    hashes.append([record["snapshot_sha256"] for record in records])
                    for record in records:
                        for variant in ["baseline","erdr"]:
                            self.assertEqual(record[variant]["status"],"ok",record)
                            self.assertAlmostEqual(record[variant]["obj"],record["initial_obj"])
                    with (output/"summary.csv").open(encoding="utf-8") as handle:
                        rows=list(csv.DictReader(handle))
                    self.assertEqual(len(rows),6)
                    self.assertTrue(all(int(row["n_paired"])==1 for row in rows))
            for value in hashes[1:]:
                self.assertEqual(value,hashes[0])


if __name__ == "__main__":
    unittest.main()
