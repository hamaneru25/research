"""Independent direct-axis checks of graph QFI upper bounds; no timing study."""
from itertools import combinations,product
from pathlib import Path
import json,hashlib,gzip
import numpy as np
from algebra import sensor_stabilizers,pure_profile
from dense_reference import graph_state,conditional_sensor_states,qufim,PAULI,I,tensor
from catalogue import ROOT


def main():
    rng=np.random.default_rng(20261020);records=[];maximum=0.
    for n,m in [(3,0),(3,1),(3,2),(4,1),(4,2),(4,3),(5,1),(5,2),(5,3)]:
        for trial in range(4):
            edges=[e for e in combinations(range(n+m),2) if rng.random()<(.2+.15*(trial%3))]
            bases=tuple(rng.choice(list('XYZ'),m))
            if m and trial%2==0:bases=('Z',)+bases[1:]
            k=n if trial==0 else min(3,n)
            survivors=tuple(sorted(map(int,rng.choice(n,k,replace=False))));mask=sum(1<<i for i in survivors)
            profile=pure_profile(n,sensor_stabilizers(n,m,edges,bases))
            expected=profile[mask-1].bit_count()
            generators=[tensor([PAULI[a]/2 if j==i else I for j in range(k)]) for i in range(k) for a in 'XYZ']
            ix=np.array([[3*i+a for i,a in enumerate(axes)] for axes in product(range(3),repeat=k)])
            signs=np.array([(1,)+tail for tail in product((-1,1),repeat=k-1)])
            for outcome,p,rho in conditional_sensor_states(graph_state(n+m,edges),n,m,survivors,bases)[:3]:
                full=qufim(rho,generators);matrices=full[ix[:,:,None],ix[:,None,:]]
                maximum_direct=float(np.max(np.einsum('si,aij,sj->as',signs,matrices,signs)))
                error=abs(maximum_direct-expected);maximum=max(maximum,error)
                assert error<1e-8,(n,m,bases,survivors,outcome,maximum_direct,expected)
                records.append(dict(n=n,m=m,trial=trial,axes=''.join(bases),survivors=survivors,error=error))
    sources=json.loads((ROOT/'data/sources.json').read_text())
    for source in sources:
        p=ROOT/'data'/f"vncorbits{source['vertices']}.g6"
        assert hashlib.sha256(p.read_bytes()).hexdigest()==source['sha256']
    certificate=json.loads((ROOT/'results/summary.json').read_text())['certificate']
    assert len(certificate)==50 and all(r['exact_gap']=='0' for r in certificate)
    result=dict(direct_axis_outcome_cases=len(records),maximum_upper_support_error=maximum,
                source_sha256_checks=len(sources),exact_grid_certificate_rows=len(certificate),
                no_setting_probability_optimization=True,no_timing_benchmark=True,records=records)
    (ROOT/'results/independent_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))


if __name__=='__main__':main()
