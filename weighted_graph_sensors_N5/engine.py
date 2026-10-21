"""Dense mixed-state SLD QFI for weighted graphs, all local Clifford axes."""
from functools import lru_cache
from itertools import combinations,product
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fixed_clifford_sensors.engine import search_catalogue,loss_weights,effective_information

PAULIS=np.array([[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]],dtype=complex)
I=np.eye(2,dtype=complex)


def kron_all(ops):
    out=np.ones((1,1),complex)
    for op in ops:out=np.kron(out,op)
    return out


@lru_cache(None)
def layout(n):
    pairs=list(combinations(range(n),2))
    bits=(np.arange(1<<n)[:,None]>>np.arange(n-1,-1,-1))&1
    features=np.column_stack([bits[:,i]*bits[:,j] for i,j in pairs]) if pairs else np.empty((1<<n,0))
    groups=[]
    for k in range(1,n):
        sets=list(combinations(range(n),k));masks=[sum(1<<i for i in kept) for kept in sets]
        indices=[]
        for kept in sets:
            lost=[i for i in range(n) if i not in kept]
            indices.append(np.arange(1<<n).reshape([2]*n).transpose(list(kept)+lost).reshape(1<<k,-1))
        ops=np.array([kron_all([P if j==i else I for j in range(k)]) for i in range(k) for P in PAULIS])
        groups.append((sets,np.array(masks),np.array(indices),ops))
    full_ops=np.array([kron_all([P if j==i else I for j in range(n)]) for i in range(n) for P in PAULIS])
    unsigned=np.array(list(product(range(3),repeat=n)))
    gather=3*np.arange(n)+unsigned
    signs=np.array([(1,)+s for s in product((1,-1),repeat=n-1)])
    return pairs,features,groups,full_ops,unsigned,gather,signs


def weighted_state(n,weights):
    return np.exp(1j*layout(n)[1]@np.asarray(weights))/np.sqrt(1<<n)


def qfi_tensor(n,weights,return_aux=False):
    """Return QFI for all local Pauli/2 directions and every survivor mask.

    For each partial trace use the general mixed-state spectral formula;
    its spectrum is NOT assumed flat. Lost-site rows and columns are zero.
    """
    _,_,groups,full_ops,_,_,_=layout(n)
    psi=weighted_state(n,weights);out=np.zeros((1<<n,3*n,3*n));aux=[]
    acted=full_ops@psi
    means=(acted@psi.conj()).real
    out[-1]=(acted.conj()@acted.T).real-np.outer(means,means)
    for sets,masks,index,ops in groups:
        amp=psi[index];rho=amp@amp.conj().transpose(0,2,1)
        vals,vecs=np.linalg.eigh(rho);vals=np.maximum(vals,0)
        if return_aux:aux.append((sets,masks,index,ops,amp,vals,vecs))
        denom=vals[:,:,None]+vals[:,None,:]
        spectral=np.divide(.5*(vals[:,:,None]-vals[:,None,:])**2,denom,
                           out=np.zeros_like(denom),where=denom>1e-12)
        transformed=vecs.conj().transpose(0,2,1)[:,None]@ops[None]@vecs[:,None]
        flat=transformed.reshape(len(sets),len(ops),-1)
        q=((flat*spectral.reshape(len(sets),1,-1))@flat.conj().transpose(0,2,1)).real
        for b,(kept,mask) in enumerate(zip(sets,masks)):
            indices=[3*i+a for i in kept for a in range(3)]
            out[mask][np.ix_(indices,indices)]=q[b]
    out=(out+out.transpose(0,2,1))/2
    return (out,(psi,aux)) if return_aux else out


def best_clifford(tensor,n,eig_rtol=1e-10,range_tol=1e-8):
    """Exact finite enumeration of 6**n axes at the supplied weights/eta.

    Enumerate 3**n unsigned axes, diagonalize each Q once, and test all
    relative signs. A simultaneous sign flip leaves the objective invariant.
    """
    unsigned,gather,signs=layout(n)[4:]
    q=tensor[gather[:,:,None],gather[:,None,:]]
    vals,vecs=np.linalg.eigh(q)
    scale=np.maximum(np.max(np.abs(vals),axis=1),1e-300)
    if np.min(vals/scale[:,None]) < -1e-7:raise ArithmeticError('Non-PSD QFIM')
    active=vals>eig_rtol*scale[:,None]
    coordinates=np.einsum('bji,sj->bsi',vecs,signs/n)
    identifiable=np.linalg.norm(np.where(active[:,None,:],0,coordinates),axis=2)<=range_tol
    inverse=np.divide(1,vals,out=np.zeros_like(vals),where=active)
    costs=np.einsum('bsi,bi->bs',coordinates**2,inverse)
    scores=np.divide(1,n*costs,out=np.zeros_like(costs),where=identifiable&(costs>0))
    index=np.unravel_index(np.argmax(scores),scores.shape)
    axes=unsigned[index[0]];sgn=signs[index[1]]
    return float(scores[index]), axes.copy(),sgn.copy()


def fixed_clifford_qfim(tensor,n,axes,signs):
    indices=3*np.arange(n)+np.asarray(axes)
    return tensor[...,indices[:,None],indices[None,:]]*np.asarray(signs)[:,None]*np.asarray(signs)[None,:]


def fixed_score(tensor,n,axes,signs,eig_rtol=1e-10,range_tol=1e-8):
    return float(effective_information(fixed_clifford_qfim(tensor,n,axes,signs),eig_rtol,range_tol)/n)


def directional_qfi_gradient(n,aux,axes,signs,v,probabilities):
    """Derivative of v^T Q(eta) v with respect to edge angles.

    With d=-i[G,rho] and rho L+L rho=2d, the SLD identity gives
    partial Q_G = Tr[(partial rho)(2i[G,L]-L^2)]. No derivative of
    eigenvectors is required. Applies on constant-rank smooth strata.
    """
    psi,groups=aux
    features=layout(n)[1]
    dpsi=1j*features.T*psi[None,:]
    coeff=np.zeros((n,3));coeff[np.arange(n),axes]=v*np.asarray(signs)
    G=np.einsum('a,aij->ij',coeff.ravel(),layout(n)[3])/2
    Gpsi=G@psi;mean=np.vdot(psi,Gpsi).real
    hpsi=G@Gpsi-2*mean*Gpsi
    grad=probabilities[-1]*8*(dpsi.conj()@hpsi).real
    for sets,masks,index,ops,amp,vals,vecs in groups:
        local_coeff=np.array([coeff[list(kept)].ravel() for kept in sets])
        G=np.einsum('ba,aij->bij',local_coeff,ops)/2
        Ge=vecs.conj().transpose(0,2,1)@G@vecs
        denom=vals[:,:,None]+vals[:,None,:]
        ratio=np.divide(vals[:,:,None]-vals[:,None,:],denom,out=np.zeros_like(denom),where=denom>1e-12)
        Le=2j*ratio*Ge
        L=vecs@Le@vecs.conj().transpose(0,2,1)
        M=2j*(G@L-L@G)-L@L
        damp=dpsi[:,index]
        drho=damp@amp.conj().transpose(0,2,1)[None]+amp[None]@damp.conj().transpose(0,1,3,2)
        derivatives=np.einsum('ebij,bji->eb',drho,M).real
        grad+=derivatives@probabilities[masks]
    return grad


class WeightObjective:
    """Ten edge weights continuous; all local Clifford choices enumerated."""
    def __init__(self,n,eta,ridge=1e-10,axes=None,signs=None):
        self.n=n;self.probabilities=loss_weights(n,[eta])[0];self.ridge=ridge
        self.axes=axes;self.signs=signs;self.calls=0

    def __call__(self,weights):
        n=self.n
        T,aux=qfi_tensor(n,weights,return_aux=True)
        avg=np.einsum('s,sij->ij',self.probabilities,T)
        if self.axes is None:
            _,axes,signs=best_clifford(avg,n)
        else:axes,signs=np.asarray(self.axes),np.asarray(self.signs)
        Q=fixed_clifford_qfim(avg,n,axes,signs)
        v=np.linalg.solve(Q+self.ridge*np.eye(n),np.full(n,1/n))
        cost=v.sum()/n
        grad=directional_qfi_gradient(n,aux,axes,signs,v,self.probabilities)/(n*cost*cost)
        self.calls+=1
        return -1/(n*cost),-grad
