"""Independent weighted-gate circuit, partial traces and mixed-state SLD."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
from engine import (qfi_tensor,fixed_clifford_qfim,fixed_score,loss_weights,
                    best_clifford,layout,WeightObjective,effective_information)
ROOT=Path(__file__).resolve().parent
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
    p.add_argument('--seed',type=int,default=20261022)
    p.add_argument('--random-checks',type=int,default=12)
    p.add_argument('--kernel-only',action='store_true')
    p.add_argument('--seed-file',type=Path)
    a=p.parse_args();rng=np.random.default_rng(a.seed);n=5
    maximum=0.;checks=0
    examples=[]
    for k in range(a.random_checks):
        examples.append({'weights':rng.uniform(0,2*np.pi,10),'axes':rng.integers(0,3,n),'signs':rng.choice([-1,1],n)})
    if a.seed_file:
        examples+=list(json.loads(a.seed_file.read_text()).values())
    for ex in examples:
        expected=fixed_clifford_qfim(qfi_tensor(n,ex['weights']),n,ex['axes'],ex['signs'])
        physical=dense_profile(n,ex['weights'],ex['axes'],ex['signs'])
        error=float(np.max(np.abs(expected-physical)));maximum=max(maximum,error);checks+=1
        if error>1e-9:raise ArithmeticError(f'Physical mixed-state check failed: {error}')
    gradient_error=0.
    for _ in range(3):
        weights=rng.uniform(.2,6,10)
        avg=np.einsum('s,sij->ij',loss_weights(n,[.65])[0],qfi_tensor(n,weights))
        optimum,axes,signs=best_clifford(avg,n)
        objective=WeightObjective(n,.65,axes=axes,signs=signs)
        _,analytic=objective(weights);step=1e-5;numerical=[]
        for edge in range(10):
            delta=np.eye(10)[edge]*step
            numerical.append((objective(weights+delta)[0]-objective(weights-delta)[0])/(2*step))
        gradient_error=max(gradient_error,float(np.max(np.abs(analytic-numerical))))
    if gradient_error>1e-6:raise ArithmeticError('Weight gradient check failed')
    # A separate direct 6**5 batch checks both the axis and sign reductions.
    from itertools import product
    unsigned=np.array(list(product(range(3),repeat=n)))
    all_signs=np.array(list(product((1,-1),repeat=n)))
    gather=3*np.arange(n)+unsigned
    q=avg[gather[:,:,None],gather[:,None,:]]
    signed=(q[:,None]*all_signs[None,:,:,None]*all_signs[None,:,None,:]).reshape(-1,n,n)
    literal=float(np.max(effective_information(signed)/n))
    if abs(literal-optimum)>1e-10:raise ArithmeticError('Finite Clifford enumeration mismatch')
    result={'verified_utc':datetime.now(timezone.utc).isoformat(),'seed':a.seed,'kernel_physical_protocols':checks,
            'loss_patterns_per_protocol':1<<n,'maximum_kernel_QFIM_error':maximum,
            'weight_gradient_checks':3,'maximum_weight_gradient_error':gradient_error,
            'literal_signed_Clifford_assignments_checked':len(signed),
            'maximum_Clifford_enumeration_error':abs(literal-optimum)}
    if not a.kernel_only:
        protocols=json.loads((a.results/'protocols.json').read_text());rows=list(csv.DictReader((a.results/'best_by_eta.csv').open()))
        max_winner=0.;max_score=0.;max_phase=0.;phase_cases=0
        for row in rows:
            proto=protocols[row['protocol_id']]
            dense=dense_profile(n,proto['weights_rad'],proto['axes_indices'],proto['signs'])
            max_winner=max(max_winner,float(np.max(np.abs(dense-np.array(proto['qfi_by_survivor_mask'])))))
            avg=np.einsum('s,sij->ij',loss_weights(n,[float(row['eta'])])[0],dense)
            actual=float(effective_information(avg)/n)
            max_score=max(max_score,abs(actual-float(row['best_found'])))
            if row['updated']=='True' and phase_cases<3:
                rotated=dense_profile(n,proto['weights_rad'],proto['axes_indices'],proto['signs'],theta=rng.uniform(-2,2,n))
                max_phase=max(max_phase,float(np.max(np.abs(rotated-dense))));phase_cases+=1
        if max_winner>1e-9 or max_score>1e-8 or max_phase>1e-9:raise ArithmeticError('Saved witness validation failed')
        old={float(r['eta']):float(r['information_per_photon']) for r in csv.DictReader((ROOT.parent/'fixed_clifford_sensors'/'results'/'best_by_n_eta.csv').open()) if int(r['n'])==n}
        matched=0
        for row in rows:
            eta=float(row['eta'])
            if eta in old:
                if abs(old[eta]-float(row['clifford_baseline']))>1e-10:raise ArithmeticError('Earlier baseline mismatch')
                matched+=1
        result.update({'saved_winners_checked':len(rows),'maximum_saved_QFIM_error':max_winner,
                       'maximum_saved_score_error':max_score,'nonzero_independent_phase_checks':phase_cases,
                       'maximum_nonzero_phase_QFIM_error':max_phase,'earlier_baseline_points_matched':matched,'all_checks_passed':True})
        (a.results/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
