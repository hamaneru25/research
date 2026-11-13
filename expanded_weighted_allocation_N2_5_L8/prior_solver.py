"""Finite multistart, full individual controlled-phase weights and Clifford enumeration.

The node phases are independent; the readout target is their unweighted mean.
All probabilities count generated, not just surviving, photons.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,csv,itertools,json,math,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from kernel import qfi_tensor,layout,directional_gradient

D=Path(__file__).resolve().parent
ETAS=np.arange(1,21)/20
TOL=2e-12

def blocks(items):
    if not items:yield ();return
    i,*tail=items
    for p in blocks(tail):
        yield ((i,),)+p
        for j in range(len(p)):yield p[:j]+((i,)+p[j],)+p[j+1:]

def scores(q):
    """Vectorized SLD bound, allowing singular identifiable mean directions."""
    n=q.shape[-1]
    a=q[...,0,0]; b=q[...,0,1]; c=q[...,1,1]
    if n==2:
        det=a*c-b*b; total=a+c-2*b
        scale=np.maximum(a+c,1e-300)
    else:
        d=q[...,0,2]; e=q[...,1,2]; f=q[...,2,2]
        A=c*f-e*e; B=d*e-b*f; C=b*e-c*d
        E=a*f-d*d; H=b*d-a*e; I=a*c-b*b
        det=a*A+b*B+d*C
        total=A+E+I+2*(B+C+H)
        scale=np.maximum(a+c+f,1e-300)
    good=(det>1e-7*scale**n)&(total>0)
    out=np.divide(n*n*det,total,out=np.zeros_like(det),where=good)
    if np.any(~good):
        v,u=np.linalg.eigh(q[~good])
        live=v>np.maximum(1e-13,v[:,-1:]*TOL)
        y=u.sum(axis=-2)/n
        valid=np.sum(np.where(live,0,y*y),axis=-1)<1e-18
        cost=np.divide(y*y,v,out=np.zeros_like(v),where=live).sum(axis=-1)
        out[~good]=np.divide(1,cost,out=np.zeros_like(cost),where=valid&(cost>0))
    return out

def ghz_states(alloc):
    n=len(alloc); l=sum(alloc); nodes=np.repeat(np.arange(n),alloc)
    pairs=layout(l)[0]
    rec=[]
    for p in blocks(list(range(l))):
        ss=list(itertools.product(*[(1,) if any(b[0]==i for b in p) else (-1,1) for i in range(l)]))
        sg=np.array(ss)
        v=np.stack([np.stack([sg[:,[i for i in b if nodes[i]==j]].sum(axis=1) for j in range(n)],axis=1) for b in p],axis=1)
        q=np.einsum('ek,ski,skj->esij',ETAS[:,None]**np.array([len(b) for b in p]),v,v)
        z=scores(q)
        w=np.array([np.pi if any((i==b[0] and j in b) or (j==b[0] and i in b) for b in p) else 0. for i,j in pairs])
        axes=np.array([2 if any(i==b[0] for b in p) else 0 for i in range(l)])
        rec.append((z,w,axes,sg,p))
    return rec

class Model:
    def __init__(self,alloc):
        self.alloc=tuple(alloc); self.n=len(alloc);self.l=sum(alloc)
        self.nmap=np.repeat(np.arange(self.n),alloc)
        self.m=np.eye(self.n)[self.nmap]
        self.edges=layout(self.l)[0];self.d=len(self.edges)
        self.labels=np.array(list(itertools.product(range(6),repeat=self.l-1)))
        self.labels=np.concatenate([np.repeat(np.arange(3),len(self.labels))[:,None],np.tile(self.labels,(3,1))],axis=1)
        # 0,1,2 = +X,+Y,+Z; 3,4,5 = -X,-Y,-Z. Global sign immaterial.
        self.gather=self.labels%3+3*np.arange(self.l)
        self.sign=np.where(self.labels<3,1.,-1.)
        self.u,self.v=np.triu_indices(self.l)
        self.pair_axis=self.gather[:,self.u]*(3*self.l)+self.gather[:,self.v]
        self.signprod=self.sign[:,self.u]*self.sign[:,self.v]
        k=np.array([s.bit_count() for s in range(1<<self.l)])
        self.p=ETAS[:,None]**k*(1-ETAS[:,None])**(self.l-k)
        self.ghz=ghz_states(alloc)
        self.baseline=np.max([a[0].max(axis=1) for a in self.ghz],axis=0)
        self.calls=0;self.axis_calls=0
    def tensor(self,x,aux=False):
        self.calls+=1
        return qfi_tensor(self.l,x,return_aux=aux)
    def aggregate(self,t,idx):
        out=self.p[idx]@t.reshape(len(t),-1)
        return out.reshape(-1,3*self.l,3*self.l)
    def all_axes(self,q):
        self.axis_calls+=len(q)
        a=q.reshape(len(q),-1)[:,self.pair_axis]*self.signprod
        shape=(len(q),len(self.labels),self.n,self.n)
        node=np.zeros(shape)
        for ix,(i,j) in enumerate(zip(self.u,self.v)):
            u,v=self.nmap[i],self.nmap[j]
            node[:,:,u,v]+=a[:,:,ix]
            if i!=j:node[:,:,v,u]+=a[:,:,ix]
        z=scores(node)
        k=z.argmax(axis=1)
        return z[np.arange(len(z)),k],self.labels[k]
    def fixed(self,t,ie,labs):
        g=labs%3+3*np.arange(self.l);sg=np.where(labs<3,1.,-1.)
        a=self.aggregate(t,[ie])[0];a=a[np.ix_(g,g)]*sg[:,None]*sg[None,:]
        return self.m.T@a@self.m
    def fit(self,x,labs,ie,iterations):
        if not iterations:return x,float(scores(self.fixed(self.tensor(x),ie,labs)[None])[0])
        l,n=self.l,self.n; a=labs%3;s=np.where(labs<3,1.,-1.)
        w=np.ones(n)/n
        best=[-1.,np.array(x)]
        def f(y):
            t,aux=self.tensor(y,True);Q=self.fixed(t,ie,labs)
            r=float(scores(Q[None])[0])
            if r>best[0]:best[:]=[r,y.copy()]
            ev=np.linalg.eigvalsh(Q)
            if ev[0]<=max(1e-11,ev[-1]*1e-9):
                return -r,np.zeros(self.d)
            v=np.linalg.solve(Q,w)
            grad=directional_gradient(l,aux,a,s,v[self.nmap],self.p[ie])*r*r
            return -r,-grad
        minimize(f,np.asarray(x),jac=True,method='L-BFGS-B',options={'maxiter':iterations,'maxfun':2*iterations+16,'ftol':3e-12,'gtol':2e-6,'maxls':18})
        return best[1],best[0]
    def seeds(self,nsob):
        uniq={}
        # Keep every distinct GHZ graph as an available warm start;
        # analytic signed GHZ winners are retained separately even at rank changes.
        def put(w):uniq[tuple(np.round(wrap(w),9))]=np.asarray(w)
        for z,w,a,s,p in self.ghz:
            if np.any(z.max(axis=1)>=self.baseline-1e-10):
                put(w)
                for d in [.18,-.18]:
                    rng=np.random.default_rng(1003+len(uniq)*19+self.l)
                    put(w+d*rng.uniform(-1,1,self.d))
        put(np.zeros(self.d))
        for mask in range(1,1<<(self.l-1)):
            cut=np.array([((mask>>i)^(mask>>j))&1 for i,j in self.edges])
            for z in [.12,.38,.70,1.1]:put(cut*z)
            for z in [.22,.65]:
                # Balanced local sign patterns for cross edges, which are not generally equivalent
                # at fixed Clifford axes due to accompanying local Z rotations.
                g=np.array([1 if (mask>>i)&1 else 0 for i in range(self.l)])
                ids=np.where(g==1)[0];s=np.ones(self.l);s[ids[:len(ids)//2]]=-1
                put(cut*z*np.array([s[i]*s[j] for i,j in self.edges]))
        if nsob:
            rng=qmc.Sobol(self.d,scramble=True,seed=739+self.l*137+sum(i*v for i,v in enumerate(self.alloc)))
            p=rng.random_base2(int(math.log2(nsob)))
            for i,row in enumerate(p):put((2*row-1)*([.35,1.15,np.pi][i%3]))
        return list(uniq.values())

def wrap(x):return (np.asarray(x)+np.pi)%(2*np.pi)-np.pi

def record(value,x,labs,model,ie,source):
    return dict(N=model.n,L=model.l,allocation=list(model.alloc),eta=float(ETAS[ie]),F_eff=float(value),
                R=float(value/model.l),CZ_R=float(model.baseline[ie]/model.l),
                weights=wrap(x).tolist(),axes=[('+' if a<3 else '-')+'XYZ'[a%3] for a in labs],source=source)

def run(job):
    alloc,indices,nsob,steps,mstarts=job
    m=Model(alloc);ids=np.array(indices)
    out=[]; top=[[] for _ in ids]
    for j,ie in enumerate(ids):
        z,x,a,sg,p=max(m.ghz,key=lambda r:r[0][ie].max())
        s=sg[z[ie].argmax()];s=s*s[0]
        lab=a+3*(s<0)
        out.append(record(z[ie].max(),x,lab,m,ie,'CZ_GHZ'))
    for x in m.seeds(nsob):
        z,lab=m.all_axes(m.aggregate(m.tensor(x),ids))
        for j,ie in enumerate(ids):
            value=z[j]
            if value>out[j]['F_eff']+5e-11:out[j]=record(value,x,lab[j],m,ie,'seed')
            if len(top[j])<9 or value>top[j][-1][0]:
                top[j].append((value,x,lab[j]))
                top[j].sort(key=lambda r:r[0],reverse=True)
                top[j]=top[j][:9]
    for j,ie in enumerate(ids):
        if not steps:continue
        starts=[]
        for value,x,lab in top[j]:
            if any(np.linalg.norm(wrap(x-y))<.16 and np.array_equal(lab,b) for y,b in starts):continue
            starts.append((x,lab))
            if len(starts)>=mstarts:break
        for x,lab in starts:
            x,value=m.fit(x,lab,ie,steps)
            z,ls=m.all_axes(m.aggregate(m.tensor(x),[ie]))
            if z[0]>out[j]['F_eff']+5e-11:out[j]=record(z[0],x,ls[0],m,ie,'refined')
            if not np.array_equal(lab,ls[0]) and z[0]>value+1e-7:
                x,_=m.fit(x,ls[0],ie,steps)
                z,ls=m.all_axes(m.aggregate(m.tensor(x),[ie]))
                if z[0]>out[j]['F_eff']+5e-11:out[j]=record(z[0],x,ls[0],m,ie,'refined_after_axis')
    return out,m.calls,m.axis_calls

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workers',type=int,default=7);p.add_argument('--sobol',type=int,default=64)
    p.add_argument('--steps',type=int,default=50);p.add_argument('--starts',type=int,default=3)
    p.add_argument('--etas',type=int,nargs='+',default=list(range(20)))
    p.add_argument('--max-l',type=int,default=6);p.add_argument('--out',type=Path,default=D/'result.json')
    a=p.parse_args()
    ref=list(csv.DictReader((D/'cz_reference.csv').open()))
    allocs=sorted({tuple(map(int,r['allocation'].split('+'))) for r in ref if int(r['L'])<=a.max_l},key=lambda z:(-sum(z),len(z),z))
    jobs=[(x,a.etas,a.sobol,a.steps,a.starts) for x in allocs]
    st=time.monotonic(); rows=[]; calls=axis=0
    with ProcessPoolExecutor(a.workers) as e:
        for f in as_completed([e.submit(run,j) for j in jobs]):
            rr,n,m=f.result()
            rows+=rr;calls+=n;axis+=m
            print(rr[0]['allocation'],n,round(time.monotonic()-st,1),flush=True)
    rmap={(r['N'],tuple(r['allocation']),r['eta']):r for r in rows}
    er=max(abs(x['CZ_R']-float(y['R'])) for y in ref if (x:=rmap.get((int(y['N']),tuple(map(int,y['allocation'].split('+'))),float(y['eta'])))) is not None)
    if er>2e-9:raise ValueError(('CZ mismatch',er))
    a.out.write_text(json.dumps({'rows':sorted(rows,key=lambda r:(r['N'],r['eta'],r['L'],r['allocation'])),
                                'baseline_error':er,'weight_evaluations':calls,'axis_sweeps':axis,
                                'settings':{'sobol':a.sobol,'steps':a.steps,'starts':a.starts},
                                'seconds':time.monotonic()-st},indent=2))
    print('done',len(rows),calls,axis,er,flush=True)
if __name__=='__main__':main()
