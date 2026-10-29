"""Final winners only: independent SLD-action calculation for all loss records.

Unlike the search moment-minus-correction formula, explicitly construct L_i U
on the support, including its orthogonal-null-space component, and evaluate
Re Tr(rho L_i L_j). No comparisons are made during the numerical search.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,time,itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from engine import effective
from search import ROOT,expand
from dense_engine import layout,pauli_actions,batches

def verify(record):
    winner=record['best'];n=record['n'];eta=record['eta'];started=time.monotonic()
    if winner['family']==4:
        # Canonical GHZ: only the intact record contributes, Q=eta^N J_N.
        q=eta**n*np.ones((n,n));r=float(effective(q,np.ones(n)/n)/n)
        return {'n':n,'eta':eta,'family':4,'R_recomputed':r,'R_error':abs(r-winner['R']),'seconds':time.monotonic()-started}
    if 'sizes'in winner:
        weights=expand(winner['sizes'],winner['group_weights']);axes=[a for a,s in zip(winner['group_axes'],winner['sizes'])for _ in range(s)]
    else:weights=np.array(winner['weights_rad']);axes=winner['axes']
    psi=np.exp(1j*layout(n)[1]@weights)/np.sqrt(1<<n)
    axis=np.array(['XYZ'.index(a[-1])for a in axes]);sign=np.array([1 if a[0]=='+'else -1 for a in axes]);avg=np.zeros((n,n))
    # The full-survival branch is handled by the same SLD-support expression.
    groups=list(layout(n)[2]);groups.append(([tuple(range(n))],np.array([(1<<n)-1]),np.arange(1<<n)[None,:,None],np.array([np.arange(3*n)])))
    for sets,masks,index,_ in batches(groups,chunk=16):
        kept=np.array(sets);k=len(sets[0]);A=psi[index];U,s,_=np.linalg.svd(A,full_matrices=False);vals=s*s
        perm,phase=pauli_actions(k);indices=3*np.arange(k)[None,:]+axis[kept]
        PV=phase[indices]*sign[kept][:,:,None]/2
        GV=PV[:,:,:,None]*U[np.arange(len(sets))[:,None,None],perm[indices],:]
        B=U.conj().transpose(0,2,1)[:,None]@GV
        den=vals[:,:,None]+vals[:,None,:]
        ratio=np.divide(vals[:,:,None]-vals[:,None,:],den,out=np.zeros_like(den),where=den>1e-13)
        support=2j*ratio[:,None]*B
        outside=GV-U[:,None]@B
        LU=U[:,None]@support-2j*outside
        q=np.einsum('biar,bjar,br->bij',LU.conj(),LU,vals).real
        probability=eta**k*(1-eta)**(n-k)
        for b,live in enumerate(sets):avg[np.ix_(live,live)]+=probability*q[b]
    r=float(effective(avg,np.ones(n)/n)/n)
    if 'sizes'in winner:
        C=np.zeros((len(winner['sizes']),n));offset=0
        for g,size in enumerate(winner['sizes']):C[g,offset:offset+size]=1;offset+=size
        qerr=float(np.max(abs(C@avg@C.T-np.array(winner['Q_group']))))
    else:qerr=float(np.max(abs(avg-np.array(winner['Qbar']))))
    return {'n':n,'eta':eta,'family':winner['family'],'R_recomputed':r,'R_error':abs(r-winner['R']),'Qbar_error':qerr,'seconds':time.monotonic()-started}

if __name__=='__main__':
    records=json.loads((ROOT/'results/best.json').read_text());results=[]
    with ProcessPoolExecutor(max_workers=3)as pool:
        fs=[pool.submit(verify,r)for r in records]
        for i,f in enumerate(as_completed(fs)):
            results.append(f.result());print(json.dumps({'done':i+1,'total':len(fs),'last':results[-1]}),flush=True)
    results.sort(key=lambda r:(r['n'],r['eta']));report={'scope':'only the final selected winner at each N/eta; all loss records; independent SLD-action formula','count':len(results),'max_R_error':max(r['R_error']for r in results),'max_Qbar_error':max(r.get('Qbar_error',0)for r in results),'results':results}
    report['passed']=report['max_R_error']<1e-7 and report['max_Qbar_error']<1e-8
    (ROOT/'results/validation.json').write_text(json.dumps(report,indent=2)+'\n');assert report['passed'],report
