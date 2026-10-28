"""Independent full-Hilbert-space checks of the symmetry-compressed kernel."""
import itertools,json,math
from pathlib import Path
import numpy as np
from search import Kernel,candidates,effective2,ghz_best

def kron(ops):
    out=np.ones((1,1),complex)
    for op in ops:out=np.kron(out,op)
    return out

def dense(n,h,q,phi,hubaxis,leafaxis,leafsign,eta):
    X=np.array([[0,1],[1,0]],complex);Y=np.array([[0,-1j],[1j,0]]);Z=np.diag([1,-1]);P=[X,Y,Z]
    H=np.array([[1,1],[1,-1]],complex)/np.sqrt(2);S=np.diag([1,1j]);gates=[H,H@Z,H@S.conj().T,H@S,np.eye(2),X]
    bits=(np.arange(1<<n)[:,None]>>np.arange(n-1,-1,-1))&1
    signs=[1]*(h-q)+[-1]*q
    phase=phi*(bits[:,:h]@np.array(signs))*bits[:,h:].sum(1)
    psi=np.exp(1j*phase)/np.sqrt(1<<n)
    axes=[2*hubaxis+(int(sign<0) if hubaxis in (1,2) else 0) for sign in signs]+[2*leafaxis+int(leafsign<0)]*(n-h)
    psi=kron([gates[a] for a in axes])@psi
    result=np.zeros((n,n))
    for mask in range(1,1<<n):
        kept=[i for i in range(n) if mask>>i&1];lost=[i for i in range(n) if not mask>>i&1];m=len(kept)
        amp=psi.reshape([2]*n).transpose(kept+lost).reshape(1<<m,-1);rho=amp@amp.conj().T
        vals,v=np.linalg.eigh(rho);den=vals[:,None]+vals[None,:]
        factor=np.divide(2.,den,out=np.zeros_like(den),where=den>1e-12)
        ds=[]
        for i in range(m):
            g=kron([Z/2 if i==j else np.eye(2) for j in range(m)])
            ds.append(v.conj().T@(-1j*(g@rho-rho@g))@v)
        mat=np.einsum('ab,iab,jab->ij',factor,np.array(ds),np.array(ds).conj()).real
        result[np.ix_(kept,kept)]+=eta**m*(1-eta)**(n-m)*mat
    return result

def main():
    checks=[]
    cases=[(5,2,1,.9358262683387703,2,1,-1,.65),(6,3,1,.713,1,0,1,.4),(6,4,2,1.21,0,2,-1,.75),(7,2,1,1.08,2,1,-1,.55),(5,1,0,math.pi,2,0,1,.85)]
    for n,h,q,phi,ha,la,ls,eta in cases:
        a=min(h,n-h);kernel=Kernel(n,a);k=np.arange(n+1)
        t=np.einsum('k,kij->ij',eta**k*(1-eta)**(n-k),kernel.profile(phi))
        cs,labels=candidates(n,a,phi);index=labels.index((h,q,ha,la,ls));c=cs[index]
        compressed=c@t@c.T
        full=dense(n,h,q,phi,ha,la,ls,eta)
        groups=(list(range(h)),list(range(h,n))) if h==a else (list(range(h,n)),list(range(h)))
        reference=np.array([[full[np.ix_(i,j)].sum() for j in groups] for i in groups])
        error=float(np.max(np.abs(reference-compressed)))
        w=np.array([len(g) for g in groups])/n
        eig,vec=np.linalg.eigh(full);coords=vec.T@(np.ones(n)/n);active=eig>1e-10
        direct=1/np.sum(coords[active]**2/eig[active]) if np.linalg.norm(coords[~active])<1e-8 else 0.
        score=float(effective2(compressed,w)/n)
        assert error<1e-8 and abs(score-direct/n)<1e-8
        checks.append({'n':n,'h':h,'negative_hubs':q,'phi':phi,'eta':eta,'Qbar_group_max_error':error,'R_compressed':score,'R_dense':direct/n})
    assert abs(checks[0]['R_compressed']-.9441448843105996)<1e-9
    assert abs(checks[-1]['R_compressed']-5*.85**5)<1e-9
    (Path(__file__).parent/'validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
