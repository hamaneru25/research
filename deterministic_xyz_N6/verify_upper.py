"""Independent exhaustive local-axis/sign checks for N=6 support bounds."""
from itertools import product
import gzip,json
import numpy as np
from search import ROOT,N
from algebra import sensor_stabilizers,pure_profile
from record_io import read_graph_records
from dense_reference import graph_state,conditional_sensor_states,qufim,PAULI,I,tensor


def main():
    records=read_graph_records(ROOT)
    rng=np.random.default_rng(62026);checks=[];maximum=0.
    for m in range(5):
        eligible=[r for r in records if r['m']==m]
        for trial in range(3):
            r=eligible[int(rng.integers(len(eligible)))];bases=''.join(rng.choice(list('XYZ'),m))
            live=list(range(N)) if trial==0 else sorted(map(int,rng.choice(N,3,replace=False)))
            mask=sum(1<<i for i in live);k=len(live)
            expected=pure_profile(N,sensor_stabilizers(N,m,r['edges'],bases))[mask-1].bit_count()
            generators=[tensor([PAULI[a]/2 if j==i else I for j in range(k)]) for i in range(k) for a in 'XYZ']
            ix=np.array([[3*i+a for i,a in enumerate(axes)] for axes in product(range(3),repeat=k)])
            signs=np.array([(1,)+s for s in product((-1,1),repeat=k-1)])
            for outcome,p,rho in conditional_sensor_states(graph_state(N+m,r['edges']),N,m,live,bases)[:3]:
                full=qufim(rho,generators);matrices=full[ix[:,:,None],ix[:,None,:]]
                direct=float(np.max(np.einsum('si,aij,sj->as',signs,matrices,signs)))
                error=abs(expected-direct);assert error<1e-8;maximum=max(maximum,error)
                checks.append(dict(graph_id=r['id'],m=m,bases=bases,survivors=live,outcome=list(outcome),
                                   axis_tuples=3**k,relative_sign_tuples=2**(k-1),expected=expected,direct=direct,error=error))
    result=dict(n=6,dense_axis_sign_outcomes=len(checks),max_error=maximum,checks=checks,no_timing_benchmark=True)
    (ROOT/'results/upper_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}))


if __name__=='__main__':main()
