import runpy
import pickle
import tempfile
import unittest
from pathlib import Path
import numpy as np
from types import SimpleNamespace


class DynamicGenerationChecks(unittest.TestCase):
    def test_all_backbones_share_snapshots(self):
        root=Path(__file__).resolve().parents[1]
        modules=[runpy.run_path(str(root/'data_generate'/f'generate_dynamic_perturbed_{name}.py')) for name in ['eoh7','reevo','hercules']]
        meoh=runpy.run_path(str(root/'erdr/MEOH-tsp/scripts/generate_dynamic_tsp.py'))
        coords=np.random.default_rng(7).uniform(size=(1,20,2))
        distance=np.linalg.norm(coords[:,:,None,:]-coords[:,None,:,:],axis=-1)
        tour=np.arange(20)
        for progress in [.2,.5,.8]:
            for change in ['coord_jitter','node_add','node_remove']:
                args=dict(coords_all=coords,dms_all=distance,solver=SimpleNamespace(solver_id='baseline'),ratio=progress,
                          perturb_type=change,seed=13,jitter_sigma=.02,node_frac=.2)
                results=[]
                for index,module in enumerate(modules):
                    extra=dict(source_tours=[tour],gls_core_mod=None,neighbor_k=20,time_limit_sec=10,ite_max=500,perturbation_moves=2) if index==0 else dict(base_cache=[dict(order=tour)])
                    results.append(module['_build_dataset_for_setting'](**args,**extra)[0])
                results.append(meoh['_generate_dataset'](coords,distance,None,progress,change,13,10,500,2,.02,.2,[tour])[0])
                for result in results[1:]:
                    for key in results[0]:np.testing.assert_array_equal(results[0][key],result[key])

    def test_reference_writer_preserves_warm_routes_and_rejects_old_cache(self):
        root = Path(__file__).resolve().parents[1]
        writer = runpy.run_path(str(root / "data_generate/compute_concorde_opt.py"))["_dump_output_pkl"]
        validators = [runpy.run_path(str(root / "data_generate" / f"generate_dynamic_perturbed_{name}.py"))["_dataset_has_cost_concorde"]
                      for name in ["eoh7", "reevo", "hercules"]]
        coords = np.random.default_rng(1).uniform(size=(5,2))
        matrix = np.linalg.norm(coords[:,None]-coords[None,:], axis=2)
        warm = np.array([0,2,1,3,4])
        with tempfile.TemporaryDirectory(dir=root / "tests") as folder:
            path = Path(folder) / "snapshot.pkl"
            raw = dict(coordinate=np.array([coords]), distance_matrix=np.array([matrix]),
                       initial_tour=np.array([warm]), dynamic_protocol="concorde_shared_v1")
            writer(output_pkl=path, raw_obj=raw, coords_rows=[coords], dist_rows=[matrix],
                   cost_concorde=[1.0], tour_concorde=[list(range(5))])
            with path.open("rb") as handle:
                saved = pickle.load(handle)
            np.testing.assert_array_equal(saved["initial_tour"], [warm])
            np.testing.assert_array_equal(saved["distance_matrix"], [matrix])
            for check in validators:
                self.assertTrue(check(path))
            saved.pop("dynamic_protocol")
            with path.open("wb") as handle:
                pickle.dump(saved, handle)
            for check in validators:
                self.assertFalse(check(path))

    def test_shared_concorde_snapshot(self):
        module = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'data_generate/generate_dynamic_perturbed_eoh7.py'))
        coords = np.random.default_rng(4).uniform(size=(1,20,2))
        distances = np.linalg.norm(coords[:,:,None,:]-coords[:,None,:,:],axis=-1)
        tour = np.arange(20)[::-1].copy()
        def forbidden(*args):
            raise AssertionError('Generation must not call an evolved solver')
        args = dict(coords_all=coords,source_tours=[tour],dms_all=distances,ratio=.5,
                    perturb_type='node_add',seed=4,gls_core_mod=None,neighbor_k=20,
                    time_limit_sec=10,ite_max=500,perturbation_moves=2,jitter_sigma=.02,node_frac=.2)
        results=[]
        for tag in ['baseline','erdr']:
            solver=module['SolverSpec'](tag,Path('unused'),forbidden)
            data,meta=module['_build_dataset_for_setting'](solver=solver,**args)
            results.append(data)
            np.testing.assert_array_equal(data['coordinate'][0,0],coords[0,tour[9]])
            np.testing.assert_array_equal(data['coordinate'][0,-2],coords[0,tour[0]])
            np.testing.assert_array_equal(data['initial_tour'][0],np.arange(data['coordinate'].shape[1]))
        for key in results[0]:
            np.testing.assert_array_equal(results[0][key],results[1][key])


if __name__=='__main__':unittest.main()
