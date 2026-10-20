"""Dense small-survivor checks for the actual N=5,...,10 winning graphs."""
from pathlib import Path
from itertools import product
import json
import numpy as np
from stabilizer_engine import sensor_stabilizers,pure_profile,qmatrix
from graph_sensing_core import graph_state,conditional_sensor_states,qufim,I,PAULI,tensor


def main():
    out=Path(__file__).resolve().parent/'results';rng=np.random.default_rng(18724)
    choices={};generators={}
    for k in range(1,5):
        entries=[(a,(1,)+sg) for a in product(range(3),repeat=k) for sg in product((-1,1),repeat=k-1)]
        choices[k]=(np.array([[3*i+x for i,x in enumerate(a)] for a,sg in entries]),np.array([sg for a,sg in entries]))
        generators[k]=[tensor([PAULI[a]/2 if j==i else I for j in range(k)]) for i in range(k) for a in 'XYZ']
    cases=[];unique={}
    for n in range(5,11):
        data=json.load((out/f'results_N{n}.json').open())
        for m in (1,2):
            candidates=[r for r in data['optimal'] if r['m']==m and r['gain_over_best_previous']>1.000001]
            if candidates:
                r=max(candidates,key=lambda r:r['gain_over_best_previous']);unique[(n,m,tuple(map(tuple,r['edges'])))]=r
    for (n,m,edges),record in unique.items():
        psi=graph_state(n+m,edges)
        for basis in product('XY',repeat=m):
            profile=pure_profile(n,sensor_stabilizers(n,m,edges,basis))
            for k in range(1,5):
                for trial in range(3):
                    survivors=tuple(sorted(map(int,rng.choice(n,k,replace=False))))
                    mask=sum(1<<i for i in survivors);d=rng.random(k)
                    for signs,p,rho in conditional_sensor_states(psi,n,m,survivors,basis)[:2]:
                        full=qufim(rho,generators[k]);ids,sg=choices[k];v=d[None,:]*sg
                        qs=full[ids[:,:,None],ids[:,None,:]]
                        actual=float(np.einsum('ai,aij,aj->a',v,qs,v).max())
                        expected=float(d@qmatrix(n,profile[mask-1])[np.ix_(survivors,survivors)]@d)
                        error=abs(actual-expected);assert error<1e-8,(n,m,basis,survivors,error)
                        cases.append(dict(n=n,m=m,survivor_count=k,error=error))
    payload=dict(graph_count=len(unique),dense_outcome_cases=len(cases),
                 max_absolute_error=max(c['error'] for c in cases),records=cases)
    (out/'winner_validation.json').write_text(json.dumps(payload,indent=2))
    print(json.dumps({k:v for k,v in payload.items() if k!='records'}))


if __name__=='__main__':main()
