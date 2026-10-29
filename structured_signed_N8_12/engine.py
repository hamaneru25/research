"""Exact mixed-state QFIM in permutation-symmetric vertex groups.

Independent phases, flagged IID erasure, fixed group-uniform Clifford axes.
The average-phase bound reduces to group collective phases by permutation
symmetry; this does NOT assume the unknown sensor phases are equal.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import itertools,math
from functools import lru_cache
import numpy as np

def spin(n):
    z=np.diag((n-2*np.arange(n+1))/2)
    off=np.sqrt((np.arange(n)+1)*(n-np.arange(n)))/2
    up=np.diag(off,1)
    return (up+up.T,-1j*(up-up.T),z)

def kron(ops):
    a=np.ones((1,1),complex)
    for op in ops:a=np.kron(a,op)
    return a

def effective(q,w):
    vals,vec=np.linalg.eigh((q+q.swapaxes(-1,-2))/2)
    coords=np.einsum('...ij,i->...j',vec,w)
    active=vals>np.maximum(1e-12,np.max(np.abs(vals),axis=-1,keepdims=True)*1e-10)
    valid=np.sum(np.where(active,0,coords**2),axis=-1)<1e-14
    cost=np.divide(coords**2,vals,out=np.zeros_like(vals),where=active).sum(-1)
    return np.divide(1,cost,out=np.zeros_like(cost),where=valid&(cost>1e-14))

class Kernel:
    def __init__(self,sizes):
        self.sizes=tuple(sizes);self.n=sum(sizes);self.g=len(sizes);self.branches=[]
        self.pairs=list(itertools.combinations_with_replacement(range(self.g),2))
        for kept in itertools.product(*(range(s+1)for s in sizes)):
            if not sum(kept):continue
            lost=tuple(s-k for s,k in zip(sizes,kept))
            live=np.array(list(itertools.product(*(range(k+1)for k in kept))))
            env=np.array(list(itertools.product(*(range(k+1)for k in lost))))
            c_live=np.array([math.prod(math.comb(k,int(a))for k,a in zip(kept,row))for row in live])
            c_env=np.array([math.prod(math.comb(k,int(a))for k,a in zip(lost,row))for row in env])
            amp=np.sqrt(c_live[:,None]*c_env[None,:]/2.**self.n)
            total=live[:,None,:]+env[None,:,:]
            features=np.array([total[:,:,i]*(total[:,:,i]-1)/2 if i==j else total[:,:,i]*total[:,:,j]for i,j in self.pairs])
            generators=np.array([kron([spin(k)[axis]if j==i else np.eye(k+1)for j,k in enumerate(kept)])for i in range(self.g)for axis in range(3)])
            mult=math.prod(math.comb(s,k)for s,k in zip(sizes,kept))
            self.branches.append((sum(kept),mult,amp,features,generators))
        axes=np.array(list(itertools.product(range(3),repeat=self.g)))
        signs=np.array([(1,)+s for s in itertools.product((1,-1),repeat=self.g-1)])
        self.gather=3*np.arange(self.g)+axes
        self.signs=signs;self.axes=axes

    def profile(self,weights):
        result=np.zeros((self.n+1,3*self.g,3*self.g))
        for count,mult,amp,feat,ops in self.branches:
            a=amp*np.exp(1j*np.einsum('p,pij->ij',weights,feat))
            u,s,_=np.linalg.svd(a,full_matrices=False)
            vals=s*s;active=vals>1e-13;vals=vals[active];u=u[:,active]
            act=ops@u
            moments=np.einsum('idr,jdr,r->ij',act.conj(),act,vals).real
            inside=np.einsum('dr,idt->irt',u.conj(),act)
            f=vals[:,None]*vals[None,:]/(vals[:,None]+vals[None,:])
            correction=np.einsum('rt,irt,jrt->ij',f,inside,inside.conj()).real
            q=4*moments-8*correction
            result[count]+=mult*(q+q.T)/2
        return result

    def evaluate(self,weights,etas):
        profile=self.profile(weights)
        k=np.arange(self.n+1)
        prob=np.asarray(etas)[:,None]**k*(1-np.asarray(etas)[:,None])**(self.n-k)
        tensor=np.einsum('ek,kij->eij',prob,profile)
        gather=self.gather
        q=tensor[:,gather[:,:,None],gather[:,None,:]]
        # Sign changes of the generators are evaluated by changing the target
        # covector in the unsigned-axis QFIM, avoiding redundant diagonalizations.
        vals,vec=np.linalg.eigh(q)
        target=self.signs*np.array(self.sizes)[None,:]/self.n
        coord=np.einsum('eaji,si->easj',vec.swapaxes(-1,-2),target)
        tol=np.maximum(1e-12,np.max(np.abs(vals),axis=-1,keepdims=True)*1e-10)
        active=vals>tol
        valid=np.sum(np.where(active[:,:,None,:],0,coord**2),axis=-1)<1e-14
        cost=np.divide(coord**2,vals[:,:,None,:],out=np.zeros_like(coord),where=active[:,:,None,:]).sum(-1)
        scores=np.divide(1,cost*self.n,out=np.zeros_like(cost),where=valid&(cost>1e-14))
        flat=scores.reshape(len(etas),-1);choice=flat.argmax(-1)
        out=[]
        for e,c in enumerate(choice):
            ai,si=divmod(int(c),len(self.signs))
            sign=self.signs[si]
            out.append({'R':float(flat[e,c]),'group_axes':[('+'if s>0 else '-')+'XYZ'[a]for a,s in zip(self.axes[ai],sign)],'Q_group':(q[e,ai]*sign[:,None]*sign[None,:]).tolist()})
        return out

def ghz(n,eta):return n*eta**n
