"""Independent physical circuits and general mixed-state SLD verification."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse,csv,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from engine import qfi_tensor,fixed_clifford_qfim,best_clifford,WeightObjective,loss_weights,effective_information

ROOT=Path(__file__).resolve().parent
from weighted_graph_sensors_N5.engine import qfi_tensor as reference_tensor,best_clifford as reference_clifford


I=np.eye(2,dtype=complex)
X=np.array([[0,1],[1,0]],complex)
Y=np.array([[0,-1j],[1j,0]],complex)
Z=np.diag([1,-1]).astype(complex)
H=(X+Z)/np.sqrt(2)
SDG=np.diag([1,-1j])


def kron(ops):
    out=np.ones((1,1),complex)
    for op in ops:out=np.kron(out,op)
    return out


def dense_profile(n,weights,axes,signs,theta=None,cutoff=1e-14):
    # Build each two-qubit controlled-phase separately from |+>^n.
    psi=np.ones([2]*n,dtype=complex)/np.sqrt(1<<n)
    from itertools import combinations
    for angle,(i,j) in zip(weights,combinations(range(n),2)):
        index=[slice(None)]*n;index[i]=index[j]=1
        psi[tuple(index)]*=np.exp(1j*angle)
    gates=[]
    for axis,sign in zip(axes,signs):
        c=[H,H@SDG,I][int(axis)]
        gates.append(c if sign==1 else X@c)
    psi=kron(gates)@psi.ravel()
    if theta is not None:
        psi=kron([np.diag(np.exp([-1j*t/2,1j*t/2])) for t in theta])@psi
    out=np.zeros((1<<n,n,n))
    for s in range(1,1<<n):
        kept=[i for i in range(n) if s>>i&1];lost=[i for i in range(n) if not s>>i&1]
        amp=psi.reshape([2]*n).transpose(kept+lost).reshape(1<<len(kept),-1)
        rho=amp@amp.conj().T
        vals,vecs=np.linalg.eigh(rho);denom=vals[:,None]+vals[None,:]
        inverse=np.divide(1,denom,out=np.zeros_like(denom),where=denom>cutoff)
        ds=[]
        for j in range(len(kept)):
            G=kron([Z/2 if i==j else I for i in range(len(kept))])
            ds.append(vecs.conj().T@(-1j*(G@rho-rho@G))@vecs)
        q=2*np.einsum('iab,jab,ab->ij',ds,np.conjugate(ds),inverse).real
        out[s][np.ix_(kept,kept)]=q
    return out



def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=ROOT/'results')
    p.add_argument('--seed',type=int,default=20261023)
    p.add_argument('--random-checks',type=int,default=4)
    p.add_argument('--gradient-checks',type=int,default=2)
    p.add_argument('--kernel-only',action='store_true')
    a=p.parse_args();rng=np.random.default_rng(a.seed)
    errors={'maximum_reference_tensor_error':0.,'maximum_physical_QFIM_error':0.,
            'maximum_Clifford_score_error':0.,'maximum_gradient_error':0.,'maximum_conjugation_score_error':0.}
    kernel_counts={}
    for n in [5,6,7]:
        m=n*(n-1)//2;cases=[rng.uniform(-np.pi,np.pi,m) for _ in range(a.random_checks)]
        cases.extend([np.zeros(m),np.full(m,np.pi)])
        near_singular=np.zeros(m);near_singular[:n-1]=np.pi-1e-4
        cases.append(near_singular)
        for x in cases:
            T=qfi_tensor(n,x);ref=reference_tensor(n,x)
            errors['maximum_reference_tensor_error']=max(errors['maximum_reference_tensor_error'],float(np.max(np.abs(T-ref))))
            axes=rng.integers(0,3,n);signs=rng.choice([-1,1],n)
            physical=dense_profile(n,x,axes,signs)
            expected=fixed_clifford_qfim(T,n,axes,signs)
            errors['maximum_physical_QFIM_error']=max(errors['maximum_physical_QFIM_error'],float(np.max(np.abs(physical-expected))))
            for eta in [.05,.65,1.]:
                avg=np.einsum('s,sij->ij',loss_weights(n,[eta])[0],T)
                refavg=np.einsum('s,sij->ij',loss_weights(n,[eta])[0],ref)
                value=best_clifford(avg,n)[0];other=reference_clifford(refavg,n)[0]
                errors['maximum_Clifford_score_error']=max(errors['maximum_Clifford_score_error'],abs(value-other))
            negative=qfi_tensor(n,-x)
            avg=np.einsum('s,sij->ij',loss_weights(n,[.7])[0],T)
            negavg=np.einsum('s,sij->ij',loss_weights(n,[.7])[0],negative)
            errors['maximum_conjugation_score_error']=max(errors['maximum_conjugation_score_error'],abs(best_clifford(avg,n)[0]-best_clifford(negavg,n)[0]))
        for _ in range(a.gradient_checks):
            x=rng.uniform(.2,6,m);T=qfi_tensor(n,x)
            avg=np.einsum('s,sij->ij',loss_weights(n,[.65])[0],T)
            _,axes,signs=best_clifford(avg,n)
            obj=WeightObjective(n,.65,axes=axes,signs=signs);_,analytic=obj(x);numeric=[]
            for j in range(m):
                d=np.eye(m)[j]*1e-5
                numeric.append((obj(x+d)[0]-obj(x-d)[0])/2e-5)
            errors['maximum_gradient_error']=max(errors['maximum_gradient_error'],float(np.max(np.abs(analytic-numeric))))
        kernel_counts[str(n)]={'physical_protocols':len(cases),'loss_masks_per_protocol':1<<n,'full_edge_gradient_checks':a.gradient_checks}
    # Backward compatibility with every saved N5 weighted winner.
    old=json.loads((ROOT.parent/'weighted_graph_sensors_N5/results/protocols.json').read_text());n5max=0.
    for r in old.values():
        T=qfi_tensor(5,r['weights_rad'])
        avg=np.einsum('s,sij->ij',loss_weights(5,[r['eta']])[0],T)
        n5max=max(n5max,abs(best_clifford(avg,5)[0]-r['score']))
    errors['maximum_N5_regression_score_error']=n5max
    limits={k:(1e-6 if 'gradient' in k else 1e-8) for k in errors}
    if any(errors[k]>limits[k] for k in errors):raise ArithmeticError(errors)
    result={'verified_utc':datetime.now(timezone.utc).isoformat(),'seed':a.seed,'kernel_checks':kernel_counts,
            **errors,'N5_saved_winners_checked':len(old)}
    if not a.kernel_only:
        rows=list(csv.DictReader((a.results/'best_by_n_eta.csv').open()))
        protocols=json.loads((a.results/'protocols.json').read_text());profiles=np.load(a.results/'qfim_profiles.npz')
        oldrows=list(csv.DictReader((ROOT.parent/'fixed_clifford_sensors_N6_7/results/best_by_n_eta.csv').open()))
        baseline={(int(r['n']),round(float(r['eta']),10)):float(r['information_per_photon']) for r in oldrows}
        summary=json.loads((a.results/'summary.json').read_text())
        for name,digest in summary['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ArithmeticError('Source changed after search: '+name)
        winner_error=0.;score_error=0.;aggregate_error=0.;phase_error=0.;baseline_error=0.;matched=0;phase_checks=0;mask_checks=0
        for row in rows:
            r=protocols[row['protocol_id']];n=int(row['n']);eta=float(row['eta'])
            Q=dense_profile(n,r['weights_rad'],r['axes_indices'],r['signs'])
            winner_error=max(winner_error,float(np.max(np.abs(Q-profiles[row['protocol_id']]))));mask_checks+=1<<n
            avg=np.einsum('s,sij->ij',loss_weights(n,[eta])[0],Q)
            score=float(effective_information(avg)/n)
            score_error=max(score_error,abs(score-float(row['best_found'])))
            aggregate_error=max(aggregate_error,float(np.max(np.abs(avg-np.array(r['average_QFIM'])))))
            counts=np.array([s.bit_count() for s in range(1<<n)])
            aggregated=np.array([Q[counts==k].sum(axis=0) for k in range(n+1)])
            aggregate_error=max(aggregate_error,float(np.max(np.abs(aggregated-r['qfim_by_survivor_count']))))
            key=n,round(eta,10)
            if key in baseline:
                baseline_error=max(baseline_error,abs(baseline[key]-float(row['clifford_baseline'])));matched+=1
            if abs(eta-.7)<1e-10 or abs(eta-.85)<1e-10:
                rotated=dense_profile(n,r['weights_rad'],r['axes_indices'],r['signs'],theta=rng.uniform(-2,2,n))
                phase_error=max(phase_error,float(np.max(np.abs(rotated-Q))));phase_checks+=1
        if max(winner_error,score_error,aggregate_error,phase_error,baseline_error)>1e-8:
            raise ArithmeticError({'winners':winner_error,'score':score_error,'aggregate':aggregate_error,'phase':phase_error,'baseline':baseline_error})
        result.update({'saved_winners_checked':len(rows),'saved_loss_masks_checked':mask_checks,
            'maximum_saved_QFIM_error':winner_error,'maximum_saved_score_error':score_error,
            'maximum_aggregate_QFIM_error':aggregate_error,'nonzero_independent_phase_checks':phase_checks,
            'maximum_nonzero_phase_QFIM_error':phase_error,'earlier_baseline_points_matched':matched,
            'maximum_earlier_baseline_error':baseline_error,'all_checks_passed':True})
        (a.results/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    else:(a.results/'kernel_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
