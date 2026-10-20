"""Independent dense-state tests of stabilizer reduction and QFIM duality."""
from pathlib import Path
from itertools import combinations,product
import json
import numpy as np
from stabilizer_engine import sensor_stabilizers,pure_profile,qmatrix,graph_profile,optimize_profile
from graph_sensing_core import graph_state,conditional_sensor_states,qufim,I,PAULI,tensor
from clique_reference import information


def main():
    rng=np.random.default_rng(18723);records=[];max_error=0.
    choices={};generators={}
    for k in range(1,5):
        choices[k]=[(np.array([3*i+a for i,a in enumerate(axes)]),np.array((1,)+sign))
                    for axes in product(range(3),repeat=k) for sign in product((-1,1),repeat=k-1)]
        generators[k]=[tensor([PAULI[a]/2 if j==i else I for j in range(k)]) for i in range(k) for a in 'XYZ']
    for n in (3,4,5):
        for m in range(4):
            for trial in range(12):
                edges=[e for e in combinations(range(n+m),2) if rng.random()<(.15+.07*(trial%6))]
                bases=tuple(rng.choice(list('XY'),m));profile=pure_profile(n,sensor_stabilizers(n,m,edges,bases))
                k=int(rng.integers(1,min(n,4)+1));survivors=tuple(sorted(map(int,rng.choice(n,k,replace=False))))
                mask=sum(1<<i for i in survivors);d=rng.random(k)
                for signs,p,rho in conditional_sensor_states(graph_state(n+m,edges),n,m,survivors,bases)[:3]:
                    full=qufim(rho,generators[k]);maximum=-1
                    for ix,sg in choices[k]:
                        v=d*sg;maximum=max(maximum,float(v@full[np.ix_(ix,ix)]@v))
                    expected=d@qmatrix(n,profile[mask-1])[np.ix_(survivors,survivors)]@d
                    error=abs(maximum-expected);max_error=max(max_error,error)
                    assert error<1e-8,(n,m,edges,bases,survivors,maximum,expected)
                    records.append(dict(n=n,m=m,trial=trial,survivors=survivors,error=error))
    comparisons=[]
    for n in range(3,7):
        for m in range(n+1):
            edges=list(combinations(range(n),2))+[(i,n+i) for i in range(m)]
            profile,audit=graph_profile(n,m,edges)
            for eta in (.2,.5,.8):
                result=optimize_profile(n,profile,eta);ref=information(n,m,eta)['feff']
                error=abs(result['feff']-ref)/max(ref,1e-15)
                assert error<2e-5,(n,m,eta,result['feff'],ref,error)
                comparisons.append(dict(n=n,m=m,eta=eta,error=error,gap=result['relative_gap']))
    out=Path(__file__).resolve().parent/'results';out.mkdir(exist_ok=True)
    payload=dict(dense_state_cases=len(records),max_dense_absolute_error=max_error,
                 reference_comparisons=len(comparisons),max_reference_relative_error=max(r['error'] for r in comparisons),
                 max_primal_dual_gap=max(r['gap'] for r in comparisons),dense_records=records,reference_records=comparisons)
    (out/'engine_validation.json').write_text(json.dumps(payload,indent=2))
    print(json.dumps({k:v for k,v in payload.items() if not isinstance(v,list)}))


if __name__=='__main__':main()
