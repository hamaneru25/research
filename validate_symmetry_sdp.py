"""Enumerate all signed Pauli-axis subsets for each pure surviving clique.

Role symmetry averages those candidates before a full c-optimal SDP. This
independently tests the one-dimensional cross-correlation optimization for
N=4,...,10. Mixed branches have diagonal maximum QFIM as derived in README.
"""
from math import comb
from pathlib import Path
import json
import numpy as np
import cvxpy as cp
from compare_fixed_graphs import information
from graph_sensing_core import effective_information


def counts(k):
    return [(p-a,p+a) for p in range(k+1) for a in range(k-p+1)]


def make_solver(n,m):
    q=n-m;qs=[];groups=[]
    for h in range(m+1):
        unique={}
        for (ah,sh) in counts(h):
            for (au,su) in counts(q):
                f=np.zeros((n,n))
                ch=(ah*ah-sh)/(h*(h-1)) if h>1 else 0
                cu=(au*au-su)/(q*(q-1)) if q>1 else 0
                cross=ah*au/(h*q) if h>0 and q>0 else 0
                for i in range(n):
                    for j in range(n):
                        if i==j:continue
                        if i<m and j<m:f[i,j]=h*(h-1)/(m*(m-1))*ch if m>1 else 0
                        elif i>=m and j>=m:f[i,j]=cu
                        else:f[i,j]=h/m*cross
                unique[tuple(np.round(f,12).ravel())]=f
        for f in unique.values():qs.append(f);groups.append(h)
    qs=np.array(qs);groups=np.array(groups)
    x=cp.Variable(len(qs));t=cp.Variable();p=cp.Parameter(m+1,nonneg=True)
    diag=cp.Parameter(n,nonneg=True)
    f=cp.diag(diag)+cp.reshape(qs.reshape(len(qs),n*n).T@cp.multiply(p[groups],x),(n,n),order='C')
    w=np.ones((n,1))/n
    g=np.array([groups==h for h in range(m+1)],float)
    constraints=[x>=0,g@x==1,cp.bmat([[f,w],[w.T,cp.reshape(t,(1,1),order='C')]])>>0]
    problem=cp.Problem(cp.Minimize(t),constraints)
    return qs,groups,x,p,diag,problem


def main():
    records=[];max_error=0.
    for n in range(4,11):
        for m in range(1,n+1):
            q=n-m;qs,groups,x,p,diag,problem=make_solver(n,m)
            for eta in (.2,.5,.65,.85,.99):
                p.value=np.array([comb(m,h)*eta**(q+h-1)*(1-eta)**(m-h) for h in range(m+1)])
                du=1 if q==1 else 1-(1-eta)**(n-1)
                diag.value=np.array([1.]*m+[du]*q)
                problem.solve(solver='CLARABEL',tol_gap_abs=1e-10,tol_gap_rel=1e-10,tol_feas=1e-10,max_iter=300)
                assert problem.status in ('optimal','optimal_inaccurate')
                xx=np.maximum(x.value,0)
                for h in range(m+1):xx[groups==h]/=xx[groups==h].sum()
                f=eta*(np.diag(diag.value)+np.einsum('a,aij->ij',xx*p.value[groups],qs))
                actual=effective_information(f);expected=information(n,m,eta)['feff']
                error=abs(actual-expected)/expected;max_error=max(max_error,error)
                assert error<2e-6,(n,m,eta,actual,expected)
                records.append(dict(n=n,m=m,eta=eta,sdp=actual,analytical=expected,relative_error=error))
            print(json.dumps({'n':n,'m':m,'variables':len(qs)}),flush=True)
    output=Path(__file__).resolve().parent/'results';output.mkdir(exist_ok=True)
    (output/'symmetry_validation.json').write_text(json.dumps(dict(cases=len(records),max_relative_error=max_error,records=records),indent=2))
    print(json.dumps(dict(cases=len(records),max_relative_error=max_error)),flush=True)


if __name__=='__main__':main()
