import importlib
import sys
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from erdr_shared.dynamic import snapshot

def main():
    target = root / sys.argv[1]
    if 'MEOH-tsp' in str(target):
        sys.path.insert(0,str(root/'erdr/MEOH-tsp'))
        module = vars(importlib.import_module('llm4ad.task.optimization.tsp_gls_2O.gls'))
    else:
        sys.path.insert(0,str(target.parent))
        module = vars(importlib.import_module('gls'))
    rng=np.random.default_rng(31)
    for dynamic in [False,True]:
        for _ in range(12):
            coords=rng.uniform(size=(15,2))
            if dynamic:
                matrix,route=snapshot(coords,np.arange(15),.2,'request_arrival')
                stop=len(route)-2
            else:
                matrix=np.linalg.norm(coords[:,None]-coords[None,:],axis=-1)
                route=np.arange(len(coords))
                stop=len(route)
            matrix=matrix.astype(np.float32)
            route=route.astype(np.uint16)
            route[1:stop]=rng.permutation(route[1:stop])
            for name in ['_two_opt_once','_relocate_once']:
                candidate=route.copy()
                before=float(matrix[candidate,np.roll(candidate,-1)].sum())
                delta=module[name](matrix,candidate,np.uint16(0))
                after=float(matrix[candidate,np.roll(candidate,-1)].sum())
                np.testing.assert_allclose(after-before,delta,atol=2e-5)
                assert after<=before+2e-5
                assert candidate[0]==0
                np.testing.assert_array_equal(candidate[stop:],route[stop:])
                np.testing.assert_array_equal(np.sort(candidate),np.arange(len(candidate)))
            if dynamic:
                candidate=route.copy()
                penalty=np.zeros_like(matrix)
                if 'MEOH' in sys.argv[1]:
                    def guide(d,t,e):
                        assert d.shape==(len(matrix)-1,len(matrix)-1)
                        assert sorted(t.tolist())==list(range(len(matrix)-1))
                        return d*1.01
                    module['_perturbation'](matrix,matrix.copy(),penalty,candidate,guide,2)
                else:
                    module['_perturbation'](matrix,matrix.copy(),penalty,candidate,np.float32(.1),np.uint32(2))
                if 'MEOH' not in sys.argv[1]:
                    np.testing.assert_array_equal(penalty,penalty.T)
                    guided=matrix+np.float32(.1)*penalty
                    np.testing.assert_array_equal(guided,guided.T)
                    for fixed_i in range(stop):
                        checked=candidate.copy()
                        before=float(guided[checked,np.roll(checked,-1)].sum())
                        delta=module['_two_opt_once'](guided,checked,np.uint16(fixed_i))
                        after=float(guided[checked,np.roll(checked,-1)].sum())
                        np.testing.assert_allclose(after-before,delta,atol=2e-5)
                assert candidate[0]==0
                np.testing.assert_array_equal(candidate[stop:],route[stop:])
                np.testing.assert_array_equal(penalty[-1],np.zeros(len(matrix)))
                np.testing.assert_array_equal(penalty[:,-1],np.zeros(len(matrix)))
    print(sys.argv[1], 'operator checks passed')

if __name__=='__main__':main()
