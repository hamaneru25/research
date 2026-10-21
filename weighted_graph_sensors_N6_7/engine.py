"""Low-rank mixed-state SLD QFI; exact finite local-Clifford enumeration.

No stabilizer or weighted-graph LC reduction. The reduced density operator
has rank at most 2**min(kept,lost); its null-space contribution is evaluated
from pure-state Pauli moments, without diagonalizing the null space.
"""
from functools import lru_cache
from itertools import combinations, product
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fixed_clifford_sensors.engine import loss_weights, effective_information
from weighted_graph_sensors_N5.engine import fixed_clifford_qfim
from weighted_graph_sensors_N5.engine import best_clifford as stable_clifford


@lru_cache(None)
def pauli_actions(k):
    """P @ V = phase[:,None] * V[permutation], ordered X,Y,Z per site."""
    x=np.arange(1<<k);perms=[];phases=[]
    for i in range(k):
        bit=1<<(k-1-i);z=1-2*((x>> (k-1-i))&1)
        perms.extend([x^bit,x^bit,x]);phases.extend([np.ones(len(x)), -1j*z,z])
    return np.array(perms),np.array(phases,dtype=complex)


@lru_cache(None)
def layout(n):
    pairs=list(combinations(range(n),2))
    bits=(np.arange(1<<n)[:,None]>>np.arange(n-1,-1,-1))&1
    features=np.column_stack([bits[:,i]*bits[:,j] for i,j in pairs])
    groups=[]
    for k in range(1,n):
        sets=list(combinations(range(n),k))
        masks=np.array([sum(1<<i for i in kept) for kept in sets])
        indices=[]
        for kept in sets:
            lost=[i for i in range(n) if i not in kept]
            indices.append(np.arange(1<<n).reshape([2]*n).transpose(list(kept)+lost).reshape(1<<k,-1))
        local=np.array([[3*i+a for i in kept for a in range(3)] for kept in sets])
        groups.append((sets,masks,np.array(indices),local))
    return pairs,features,groups


@lru_cache(None)
def clifford_layout(n):
    unsigned=np.array(list(product(range(3),repeat=n)))
    gather=3*np.arange(n)+unsigned
    signs=np.array([(1,)+s for s in product((1,-1),repeat=n-1)])
    i,j=np.triu_indices(n,1)
    return unsigned,gather,signs,i,j,(signs[:,i]*signs[:,j]).T


def best_clifford(tensor,n,eig_rtol=1e-10,range_tol=1e-8):
    unsigned,gather,signs,i,j,products=clifford_layout(n)
    q=tensor[gather[:,:,None],gather[:,None,:]]
    vals,vecs=np.linalg.eigh(q)
    scale=np.maximum(np.max(np.abs(vals),axis=1),1e-300)
    if np.min(vals/scale[:,None]) < -1e-7:raise ArithmeticError('Non-PSD QFIM')
    active=vals>eig_rtol*scale[:,None]
    # A trace-plus-offdiagonal quadratic form loses precision when large
    # inverse entries cancel near a rank change. Use the original stable
    # eigen-coordinate evaluation on such inputs, including all signs.
    smallest=np.min(np.where(active,vals,np.inf),axis=1)
    condition=np.divide(scale,smallest,out=np.zeros_like(scale),where=np.isfinite(smallest))
    if np.any(condition>1e6):
        return stable_clifford(tensor,n,eig_rtol,range_tol)
    inv=np.divide(1,vals,out=np.zeros_like(vals),where=active)
    pinv=(vecs*inv[:,None,:])@vecs.transpose(0,2,1)
    costs=(np.trace(pinv,axis1=1,axis2=2)[:,None]+2*pinv[:,i,j]@products)/(n*n)
    identifiable=np.ones(costs.shape,dtype=bool)
    singular=np.where(~np.all(active,axis=1))[0]
    if len(singular):
        coordinates=np.einsum('bji,sj->bsi',vecs[singular],signs/n)
        identifiable[singular]=np.linalg.norm(np.where(active[singular,None,:],0,coordinates),axis=2)<=range_tol
    scores=np.divide(1,n*costs,out=np.zeros_like(costs),where=identifiable&(costs>0))
    idx=np.unravel_index(np.argmax(scores),scores.shape)
    return float(scores[idx]),unsigned[idx[0]].copy(),signs[idx[1]].copy()


def qfi_tensor(n,weights,return_aux=False):
    features=layout(n)[1]
    psi=np.exp(1j*features@np.asarray(weights))/np.sqrt(1<<n)
    perm,phase=pauli_actions(n)
    acted=phase*psi[perm]
    moments=(acted.conj()@acted.T).real
    means=(acted@psi.conj()).real
    out=np.zeros((1<<n,3*n,3*n));out[-1]=moments-np.outer(means,means)
    aux=[]
    for sets,masks,index,local in layout(n)[2]:
        k=len(sets[0]);amp=psi[index]
        # SVD computes just the occupied support, including at rank changes.
        U,s,_=np.linalg.svd(amp,full_matrices=False)
        vals=s*s
        # Keep the full Schmidt width. Small/zero values are masked below;
        # retaining them avoids per-mask rank branches and is numerically safe.
        V=U
        perm,phase=pauli_actions(k)
        PV=phase[None,:,:,None]*V[:,perm,:]
        B=V.conj().transpose(0,2,1)[:,None]@PV
        denom=vals[:,:,None]+vals[:,None,:]
        correction=np.divide(2*vals[:,:,None]*vals[:,None,:],denom,
                             out=np.zeros_like(denom),where=denom>1e-14)
        flat=B.reshape(len(sets),3*k,-1)
        subtract=((flat*correction.reshape(len(sets),1,-1))@flat.conj().transpose(0,2,1)).real
        for b,mask in enumerate(masks):
            loc=local[b];out[mask][np.ix_(loc,loc)]=moments[np.ix_(loc,loc)]-subtract[b]
        if return_aux:aux.append((sets,masks,index,amp,vals,V,PV,B))
    out=(out+out.transpose(0,2,1))/2
    # Remove cancellation noise in exactly insensitive Pauli directions.
    out[np.abs(out)<1e-13]=0
    return (out,(psi,aux)) if return_aux else out


def directional_gradient(n,aux,axes,signs,v,probabilities):
    psi,groups=aux
    dpsi=1j*layout(n)[1].T*psi[None,:]
    coeff=np.zeros((n,3));coeff[np.arange(n),axes]=v*np.asarray(signs)
    perm,phase=pauli_actions(n)
    def act_full(w):return np.einsum('a,ad,ad->d',coeff.ravel(),phase,w[perm])/2
    Gpsi=act_full(psi);mean=np.vdot(psi,Gpsi).real
    hpsi=act_full(Gpsi)-2*mean*Gpsi
    grad=probabilities[-1]*8*(dpsi.conj()@hpsi).real
    for sets,masks,index,amp,vals,V,PV,B in groups:
        k=len(sets[0]);c=np.array([coeff[list(kept)].ravel() for kept in sets])/2
        GV=np.einsum('ba,badr->bdr',c,PV)
        Ge=np.einsum('ba,barq->brq',c,B)
        denom=vals[:,:,None]+vals[:,None,:]
        ratio=np.divide(vals[:,:,None]-vals[:,None,:],denom,
                        out=np.zeros_like(denom),where=denom>1e-14)
        Le=2j*ratio*Ge
        # L on support plus its exact coupling to the orthogonal null space.
        outside=GV-V@Ge
        L=V@Le@V.conj().transpose(0,2,1)-2j*(outside@V.conj().transpose(0,2,1)-V@outside.conj().transpose(0,2,1))
        perm,phase=pauli_actions(k)
        GL=np.einsum('ba,ad,badj->bdj',c,phase,L[:,perm,:])
        # GL - LG = GL - (GL)^dagger because G and L are Hermitian.
        M=2j*(GL-GL.conj().transpose(0,2,1))-L@L
        damp=dpsi[:,index]
        ma=M@amp
        derivatives=2*np.einsum('ebdl,bdl->eb',damp.conj(),ma).real
        grad+=derivatives@probabilities[masks]
    return grad


class WeightObjective:
    def __init__(self,n,eta,ridge=1e-10,axes=None,signs=None):
        self.n=n;self.probabilities=loss_weights(n,[eta])[0];self.ridge=ridge
        self.axes=axes;self.signs=signs;self.calls=0
    def __call__(self,weights):
        n=self.n;T,aux=qfi_tensor(n,weights,return_aux=True)
        avg=np.einsum('s,sij->ij',self.probabilities,T)
        if self.axes is None:_,axes,signs=best_clifford(avg,n)
        else:axes,signs=np.asarray(self.axes),np.asarray(self.signs)
        Q=fixed_clifford_qfim(avg,n,axes,signs)
        v=np.linalg.solve(Q+self.ridge*np.eye(n),np.full(n,1/n))
        cost=v.sum()/n
        grad=directional_gradient(n,aux,axes,signs,v,self.probabilities)/(n*cost*cost)
        self.calls+=1
        return -1/(n*cost),-grad
