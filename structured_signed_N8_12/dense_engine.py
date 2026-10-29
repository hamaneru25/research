"""Low-rank Pauli/2 QFIM kernel adapted from weighted_graph_sensors_N6_7/engine.py."""
from functools import lru_cache
from itertools import combinations
import numpy as np

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


def batches(groups,chunk=32):
    for sets,masks,index,local in groups:
        for start in range(0,len(sets),chunk):
            yield sets[start:start+chunk],masks[start:start+chunk],index[start:start+chunk],local[start:start+chunk]

def qfi_tensor(n,weights,return_aux=False):
    features=layout(n)[1]
    psi=np.exp(1j*features@np.asarray(weights))/np.sqrt(1<<n)
    perm,phase=pauli_actions(n)
    acted=phase*psi[perm]
    moments=(acted.conj()@acted.T).real
    means=(acted@psi.conj()).real
    out=np.zeros((1<<n,3*n,3*n));out[-1]=moments-np.outer(means,means)
    aux=[]
    for sets,masks,index,local in batches(layout(n)[2]):
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



