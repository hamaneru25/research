"""Independent density-matrix SLD check for each saved protocol."""
from expanded import D
import itertools,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
P={'X':np.array([[0,1],[1,0]]),'Y':np.array([[0,-1j],[1j,0]]),'Z':np.diag([1,-1])}
def tensor(x):
    r=np.ones((1,1),complex)
    for p in x:r=np.kron(r,p)
    return r
def independently(row):
    n,l=row['N'],row['L'];a=row['allocation'];t=row['eta']
    nd=np.repeat(np.arange(n),a)
    b=np.array(list(itertools.product(range(2),repeat=l)))
    ph=sum(v*b[:,i]*b[:,j] for v,(i,j) in zip(row['weights'],itertools.combinations(range(l),2)))
    psi=np.exp(1j*ph).reshape([2]*l)/np.sqrt(2**l)
    q=np.zeros((n,n))
    for k in range(1,l+1):
        for s in itertools.combinations(range(l),k):
            w=t**k*(1-t)**(l-k)
            if not w:continue
            arr=psi.transpose(s+tuple(i for i in range(l) if i not in s)).reshape(2**k,-1)
            rho=arr@arr.conj().T
            vals,u=np.linalg.eigh(rho);vals=np.maximum(vals,0)
            g=np.zeros((n,2**k,2**k),complex)
            for ix,i in enumerate(s):
                lab=row['axes'][i];sign=1 if lab[0]=='+' else -1
                g[nd[i]]+=sign*tensor([P[lab[-1]] if j==ix else np.eye(2) for j in range(k)])/2
            d=-1j*(g@rho-rho@g);d=u.conj().T@d@u
            den=vals[:,None]+vals[None,:]
            coef=np.divide(2,den,out=np.zeros_like(den),where=den>1e-13)
            q+=w*np.einsum('iab,jab,ab->ij',d,d.conj(),coef).real
    v,u=np.linalg.eigh(q)
    active=v>max(1e-13,v[-1]*2e-12)
    co=u.sum(axis=0)/n
    y=1/sum(co[active]**2/v[active]) if np.any(active) and np.linalg.norm(co[~active])<1e-9 else 0.
    return q.tolist(),float(y),float(abs(y-row['F_eff']))
