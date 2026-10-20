"""Independent dense-state, all-X/Y-bases/all-local-Clifford verification.

Validates the analytical reduction for N=3,4 and M=0,...,N.
CVXPY/CLARABEL solve the full conditional QFIM candidate problem.
"""
from itertools import combinations,product
from pathlib import Path
import json
import numpy as np
import cvxpy as cp
from graph_sensing_core import I,PAULI,tensor,graph_state,conditional_sensor_states,qufim,effective_information
from compare_fixed_graphs import information,matrix


class Factory:
    def __init__(self):self.cache={};self.generators={};self.ids={};self.signs={}
    def candidates(self,rho):
        k=int(round(np.log2(len(rho))))
        key=(k,np.rint(rho.real*2**k).astype(np.int16).tobytes(),
             np.rint(rho.imag*2**k).astype(np.int16).tobytes())
        if key not in self.cache:
            if k not in self.generators:
                self.generators[k]=[tensor([PAULI[a]/2 if j==i else I for j in range(k)]) for i in range(k) for a in 'XYZ']
                choices=[(a,(1,)+s) for a in product(range(3),repeat=k) for s in product((1,-1),repeat=k-1)]
                self.ids[k]=np.array([[3*i+a[i] for i in range(k)] for a,s in choices])
                self.signs[k]=np.array([s for a,s in choices])
            full=qufim(rho,self.generators[k]);ids=self.ids[k];ss=self.signs[k]
            qs=full[ids[:,:,None],ids[:,None,:]]*ss[:,:,None]*ss[:,None,:]
            assert np.max(np.abs(qs-np.rint(qs)))<1e-8
            self.cache[key]=frozenset(tuple(map(int,np.rint(q).ravel())) for q in qs)
        return self.cache[key]


def verify():
    fac=Factory();records=[];max_error=0.;branches=0
    for n in (3,4):
        for m in range(n+1):
            edges=list(combinations(range(n),2))+[(i,n+i) for i in range(m)]
            psi=graph_state(n+m,edges)
            survivors=[s for k in range(1,n+1) for s in combinations(range(n),k)]
            qs=[];loss=[]
            for j,s in enumerate(survivors):
                union=set()
                for b in product('XY',repeat=m):
                    sets=[]
                    for r,p,rho in conditional_sensor_states(psi,n,m,s,b):
                        branches+=1;sets.append(fac.candidates(rho))
                    assert len(set(sets))==1
                    union.update(sets[0])
                for key in sorted(union):
                    q=np.zeros((n,n));q[np.ix_(s,s)]=np.array(key).reshape(len(s),len(s))
                    qs.append(q);loss.append(j)
            qs=np.array(qs);loss=np.array(loss)
            x=cp.Variable(len(qs));t=cp.Variable();coeff=cp.Parameter(len(survivors),nonneg=True)
            f=cp.reshape(qs.reshape(len(qs),n*n).T@cp.multiply(coeff[loss],x),(n,n),order='C')
            w=np.ones((n,1))/n
            groups=np.array([loss==j for j in range(len(survivors))],float)
            constraints=[x>=0,groups@x==1,cp.bmat([[f,w],[w.T,cp.reshape(t,(1,1),order='C')]])>>0]
            problem=cp.Problem(cp.Minimize(t),constraints)
            for eta in (.2,.4,.5,.7,.9):
                coeff.value=np.array([eta**(len(s)-1)*(1-eta)**(n-len(s)) for s in survivors])
                problem.solve(solver='CLARABEL',tol_gap_abs=1e-10,tol_gap_rel=1e-10,tol_feas=1e-10,max_iter=300)
                assert problem.status in ('optimal','optimal_inaccurate')
                xx=np.maximum(x.value,0)
                for j in range(len(survivors)):xx[loss==j]/=xx[loss==j].sum()
                ff=eta*np.einsum('a,aij->ij',xx*coeff.value[loss],qs)
                actual=effective_information(ff);expected=information(n,m,eta)['feff']
                rel=abs(actual-expected)/max(expected,1e-12)
                max_error=max(max_error,rel)
                assert rel<2e-6,(n,m,eta,actual,expected)
                analytic_matrix=matrix(n,m,eta)
                assert abs(effective_information(analytic_matrix)-expected)<1e-8
                records.append(dict(n=n,m=m,eta=eta,dense_sdp=actual,analytical=expected,relative_error=rel))
            print(json.dumps({'verified_n':n,'m':m,'candidate_variables':len(qs)}),flush=True)
    result={'verified_cases':len(records),'nonzero_branches':branches,'max_relative_error':max_error,'records':records}
    output=Path(__file__).resolve().parent/'results';output.mkdir(exist_ok=True)
    (output/'validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'cases':len(records),'branches':branches,'max_relative_error':max_error}),flush=True)


if __name__=='__main__':verify()
