"""Weighted graph sensing, 2--5 nodes and at most 8 generated photons."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import itertools as it, numpy as np, math, json, time, argparse
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor,as_completed
from scipy.optimize import minimize
from scipy.stats import qmc
from kernel import qfi_tensor,layout,directional_gradient
from prior_solver import ETAS,blocks,wrap,scores as small_scores
D=Path(__file__).resolve().parent
AX=np.array(['+X','+Y','+Z','-X','-Y','-Z'])
def allocs(l,n,lo=1):
    if n==1:
        if l>=lo: yield (l,)
    else:
        for x in range(lo,l//n+1):
            for s in allocs(l-x,n-1,x):yield (x,)+s
def score(q):
    if q.shape[-1]<4:return small_scores(q)
    n=q.shape[-1]
    # If full rank, compute the inverse quadratic form directly.
    v,u=np.linalg.eigh(q)
    a=v>np.maximum(1e-13,v[...,-1:]*2e-12)
    y=u.sum(axis=-2)/n
    ok=np.sum(np.where(a,0,y*y),axis=-1)<1e-18
    d=np.divide(y*y,v,out=np.zeros_like(v),where=a).sum(axis=-1)
    return np.divide(1,d,out=np.zeros_like(d),where=ok&(d>0))
def atomic(path,v):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=1));tmp.replace(path)
def labs_for_sign(axes,s):
    return (np.array(axes)+3*(np.array(s)*s[0]<0)).astype(np.int16)
def prior_rows():
    return json.loads((D/'prior_verified.json').read_text())['rows']
class Model:
    def __init__(self,s):
        self.s=tuple(s); self.n=len(s);self.l=sum(s)
        self.nodes=np.repeat(np.arange(self.n),s);self.M=np.eye(self.n)[self.nodes]
        self.edges=layout(self.l)[0]; self.d=len(self.edges)
        self.i,self.j=np.array(self.edges).T
        self.ui,self.uj=np.triu_indices(self.l)
        self.di,self.dj=self.nodes[self.ui],self.nodes[self.uj]
        self.groups=[(a,b,np.flatnonzero((self.di==a)&(self.dj==b))) for a in range(self.n) for b in range(a,self.n)]
        self.count=np.array([v.bit_count() for v in range(2**self.l)])
        self.prob=ETAS[:,None]**self.count*(1-ETAS[:,None])**(self.l-self.count)
        self.calls=0
    def tensor(self,x,aux=False):
        self.calls+=1;return qfi_tensor(self.l,x,return_aux=aux)
    def agg(self,t,ies):
        return (self.prob[ies]@t.reshape(len(t),-1)).reshape(-1,3*self.l,3*self.l)
    def allq(self,q,labs):
        g=labs%3+3*np.arange(self.l); s=np.where(labs<3,1.,-1.)
        a=q[:,g[:,self.ui],g[:,self.uj]]*(s[:,self.ui]*s[:,self.uj])[None,:,:]
        out=np.zeros((len(q),len(labs),self.n,self.n))
        for u,v,inds in self.groups:
            val=a[:,:,inds].sum(axis=-1)
            if u==v:
                val=val+a[:,:,inds[self.ui[inds]!=self.uj[inds]]].sum(axis=-1)
            out[:,:,u,v]=out[:,:,v,u]=val
        return out
    def fixed(self,q,lab):
        g=lab%3+3*np.arange(self.l);s=np.where(lab<3,1.,-1.)
        return self.M.T@(q[np.ix_(g,g)]*s[:,None]*s[None,:])@self.M
    def pick(self,q,labs):
        z=score(self.allq(q,labs));ids=z.argmax(axis=1);return z[np.arange(len(q)),ids],labs[ids]
    def fit(self,x,lab,ie,steps):
        l=self.l; n=self.n
        sg=np.where(lab<3,1,-1);ax=lab%3
        w=np.ones(n)/n; best=[-1,np.array(x)]
        def f(y):
            t,aux=self.tensor(y,True);q=self.fixed(self.agg(t,[ie])[0],lab)
            z=float(score(q[None])[0])
            if z>best[0]:best[:]=[z,y.copy()]
            ev=np.linalg.eigvalsh(q)
            if ev[0]<=max(1e-11,1e-9*ev[-1]):return -z,np.zeros(self.d)
            v=np.linalg.solve(q,w)[self.nodes]
            return -z,-directional_gradient(l,aux,ax,sg,v,self.prob[ie])*z*z
        minimize(f,np.asarray(x),jac=True,method='L-BFGS-B',options={'maxiter':steps,'maxfun':2*steps+12,'ftol':1e-10,'gtol':2e-6})
        return wrap(best[1]),best[0]
    def rec(self,z,x,lab,ie,source):
        return {'N':self.n,'L':self.l,'allocation':self.s,'eta':float(ETAS[ie]),'F_eff':float(z),
                'R':float(z/self.l),'weights':wrap(x).tolist(),'axes':AX[lab].tolist(),'source':source}
    def structured(self,ie,rng):
        n,l=self.n,self.l
        gl=[self.nodes==a for a in range(n)]
        # All Z-axis hub sizes by node, using the symmetry of photon labels within a node.
        hs=[v for v in it.product(*[range(z+1) for z in self.s]) if 0<sum(v)<l]
        # Pair-wise weights use only the optical graph; roles can cross node boundaries.
        for sizes in hs:
            ids=np.concatenate([np.flatnonzero(mask)[:v] for mask,v in zip(gl,sizes)])
            h=len(ids);z=np.zeros(l,bool);z[ids]=1
            cut=z[self.i]^z[self.j]
            if h<2 or h%2:continue
            mm=np.array(list(it.combinations(ids,h//2)),int)
            mm=mm[np.any(mm==ids[0],axis=1)]
            opts=sorted(mm,key=lambda p:(np.max(abs(np.bincount(self.nodes[p],minlength=n)*2-np.asarray(sizes))),tuple(p)))
            if len(opts)>2:opts=[opts[0],opts[-1]]
            for neg in opts:
                s=np.ones(l,int);s[neg]=-1
                lab=labs_for_sign(np.where(z,2,1),np.where(z,s,-1))
                weight=cut*s[self.i]*s[self.j]
                yield weight,lab
    def evaluate_seeds(self,existing=None):
        rng=np.random.default_rng(8235+sum((i+1)*x for i,x in enumerate(self.s)))
        # Reuse QFI tensors across transmission values.
        top=[[] for _ in ETAS]
        ghz=self.ghz_baseline()
        best=ghz.copy()
        if existing:
            for i,r in existing.items():
                if r['F_eff']>best[i]['F_eff']:best[i]=r.copy()
        def add(x,labs,src,ies):
            x=wrap(x); labs=np.asarray(labs,dtype=np.int16); q=self.agg(self.tensor(x),ies)
            vals,axes=self.pick(q,labs)
            for ie,y,lab in zip(ies,vals,axes):
                if y>best[ie]['F_eff']+1e-10:best[ie]=self.rec(y,x,lab,ie,src)
                li=top[ie];li.append((float(y),x,lab));li.sort(key=lambda a:-a[0]);del li[7:]
        # Weak-to-intermediate bipartite candidates. Same unwrapped weight/axes can be reused.
        patterns=list(self.structured(0,rng))
        # Deduplicate phase profiles up to within-node perm by the construction and global inversion.
        for p,lab in patterns:
            for phi in (.10,.29,.52,.81,1.12,1.44):
                add(p*phi,[lab],'balanced_bipartite',range(20))
        # Odd hub split and its unbalanced magnitude, plus edges in the Y group.
        for hub_sizes in it.product(*[range(z+1) for z in self.s]):
            if sum(hub_sizes)!=3:continue
            hs=np.concatenate([np.flatnonzero(self.nodes==a)[:v] for a,v in enumerate(hub_sizes)])
            z=np.isin(np.arange(self.l),hs);cut=z[self.i]^z[self.j]; yy=(~z[self.i])&(~z[self.j])
            for neg in set([int(hs[0]),int(hs[-1])]):
                s=np.ones(self.l,int);s[neg]=-1;sign=s[self.i]*s[self.j]
                lab=labs_for_sign(np.where(z,2,1),np.where(z,s,-1))
                negedge=cut & ((self.i==neg)|(self.j==neg))
                for phi in (.20,.44,.75,1.04):
                    add(phi*(sign*(cut+.14*negedge)-.16*yy),[lab],'odd_hub_seed',range(20))
        # High-transmission structured families and reproducible Sobol full-edge starts.
        hot=np.arange(12,20)
        for a in [.6,1.,1.5,2.,2.45,2.8,3.05]:
            for p in [a,-a]:
                add(np.full(self.d,p),np.tile(np.arange(3)[:,None],(1,self.l)),'uniform_clique',hot)
        for i in range(self.l):
            mask=(self.i==i)|(self.j==i)
            for p,q in [(np.pi,.62),(np.pi,1.),(np.pi,1.48),(2.85,.8),(-2.8,-.72),(2.75,2.4),(np.pi,2.8)]:
                ls=[]
                for a,b in it.product(range(3),repeat=2):
                    ls.append(labs_for_sign(np.where(np.arange(self.l)==i,a,b),np.ones(self.l)))
                    ls.append(labs_for_sign(np.where(np.arange(self.l)==i,a,b),np.where(np.arange(self.l)==i,-1,1)))
                add(np.where(mask,p,q),ls,'two_weight_clique',hot)
        # For N<=3, include previous verified 6-photon circuits padded to larger allocations.
        for old in prior_rows():
            if old['N']!=self.n or old['L']!=6 or self.l<=6 or old['eta'] not in (.15,.35,.5,.65,.75,.8,.85,.9):continue
            for perm in sorted(set(it.permutations(old['allocation']))):
                if any(a>b for a,b in zip(perm,self.s)):continue
                permute=[]
                for a,sz in enumerate(perm):
                    j=next(i for i,x in enumerate(old['allocation']) if x==sz and i not in permute)
                    permute.append(j)
                ids=np.concatenate([np.flatnonzero(self.nodes==permute.index(a))[:s] for a,s in enumerate(old['allocation'])])
                l0=old['L']; W=np.zeros((self.l,self.l));I,J=np.triu_indices(l0,1)
                W[ids[I],ids[J]]=W[ids[J],ids[I]]=old['weights']
                labs=np.full(self.l,2);labs[ids]=[np.flatnonzero(AX==ax)[0] for ax in old['axes']]
                if labs[0]>=3:labs=(labs+3)%6
                add(W[self.i,self.j],[labs],'padded_previous',range(20))
                # Replicate a high-performing role on unused sites.
                for q in range(2):
                    ext=ids.copy(); unused=np.setdiff1d(np.arange(self.l),ids)
                    donor=np.array([int(rng.choice(ids)) for _ in unused])
                    lookup=np.arange(self.l);lookup[unused]=donor
                    add(W[lookup[self.i],lookup[self.j]],[labs[lookup]],'replicated_previous',range(20))
        signed=rng.integers(0,6,(144,self.l));signed[:,0]%=3
        pa=np.concatenate([signed,np.tile(np.arange(3)[:,None],(1,self.l))])
        sample=qmc.Sobol(self.d,scramble=True,seed=239+sum(self.s)*13+self.n).random_base2(4)
        for j,r in enumerate(sample):
            x=(2*r-1)*(.45,1.2,np.pi)[j%3]
            add(x,pa,'sobol',range(20))
        return best,top,ghz
    def ghz_baseline(self):
        """Exact enumeration of all block partitions and signed local axes."""
        # Photon-label permutations within a node never change the set of
        # block signed node populations. Enumerate unsigned count multisets first.
        n=self.n;l=self.l
        @lru_cache(None)
        def comps(remaining,lo):
            if not any(remaining):return ((),)
            cand=[]
            for b in it.product(*[range(v+1) for v in remaining]):
                if not any(b) or b<lo:continue
                rem=tuple(x-y for x,y in zip(remaining,b))
                for tt in comps(rem,b):cand.append((b,)+tt)
            return tuple(cand)
        best=[None]*len(ETAS); vmax=np.full(len(ETAS),-1.)
        all_parts=comps(self.s,(0,)*n)
        for pp in all_parts:
            nn=list(pp);ms=np.array(nn).sum(axis=1)
            # Global orientation of each rank-one block does not affect Q.
            signs=[]
            for b in nn:
                vs=[v for v in it.product(*[range(-k,k+1,2) for k in b])
                    if next((x for x in v if x!=0),0)>=0]
                signs.append(vs)
            vs=np.array(list(it.product(*signs)),float)
            Q=np.einsum('eb,sbi,sbj->esij',ETAS[:,None]**ms,vs,vs)
            z=score(Q); js=z.argmax(axis=1); zz=z[np.arange(20),js]
            for ie in np.flatnonzero(zz>vmax+1e-11):
                vmax[ie]=zz[ie]; signed=vs[js[ie]]
                a=np.array([np.flatnonzero(self.nodes==i)[0] for i in range(n)])
                w=np.zeros((l,l));axes=np.full(l,2);sg=np.ones(l,int)
                for bb,v in zip(nn,signed):
                    ids=[]
                    for j,c in enumerate(bb):
                        ids.extend(range(a[j],a[j]+c))
                        sg[a[j]:a[j]+(c-int(v[j]))//2]=-1
                        a[j]+=c
                    for j in ids[1:]:w[ids[0],j]=w[j,ids[0]]=np.pi;axes[j]=0
                best[ie]=self.rec(zz[ie],w[self.i,self.j],labs_for_sign(axes,sg),ie,'GHZ')
        return best
    def improve_axes(self,q,lab,exact=False):
        if exact:
            return self.exact_axes(q)
        unsigned=lab%3
        signs=np.array(list(it.product([0,3],repeat=self.l-1)),int)
        signs=np.column_stack([np.zeros(len(signs),int),signs])
        pool=[unsigned]
        for j in range(self.l):
            for a in range(3):
                if a==unsigned[j]:continue
                u=unsigned.copy();u[j]=a;pool.append(u)
        labs=(np.array(pool)[:,None,:]+signs).reshape(-1,self.l)
        z,l=self.pick(q[None],labs)
        return float(z[0]),l[0]
    def exact_axes(self,q):
        # 6**L / 2 configurations, streamed by unsigned axis groups.
        axes=np.array(list(it.product(range(3),repeat=self.l)),dtype=np.int16)
        s=np.array(list(it.product([0,3],repeat=self.l-1)),dtype=np.int16)
        s=np.column_stack([np.zeros(len(s),dtype=np.int16),s])
        best=-1.; lab=None
        for beg in range(0,len(axes),192):
            labs=(axes[beg:beg+192,None,:]+s).reshape(-1,self.l)
            z,u=self.pick(q[None],labs)
            if z[0]>best:best,lab=float(z[0]),u[0]
        return best,lab

def run(job):
    s,steps,out=job;m=Model(s);path=Path(out)/('a_'+'_'.join(map(str,s))+'.json')
    if path.exists():return str(path)
    old={int(round(r['eta']*20))-1:r for r in prior_rows() if tuple(r['allocation'])==s}
    if old:
        best=[old[i].copy() for i in range(20)];ghz=m.ghz_baseline()
    else:
        best,top,ghz=m.evaluate_seeds()
        def update(ie,x,lab,y,src):
            if y>best[ie]['F_eff']+5e-11:best[ie]=m.rec(y,x,lab,ie,src)
        for ie in range(19):
            # Refine the strongest distinct starts in every condition.
            li=top[ie]
            if not li:continue
            starts=li[:1]
            if m.l<8 or ie in [7,9,11,13,14,15,16,17]:
                for z,x,a in li[1:]:
                    if np.linalg.norm(wrap(x-starts[0][1]))>.22:
                        starts.append((z,x,a));break
            for z,x,a in starts:
                x,y=m.fit(x,a,ie,steps);update(ie,x,a,y,'weight_refined')
            r=best[ie]
            if r['source']=='GHZ':continue
            x=r['weights'];a=np.array([np.flatnonzero(AX==v)[0] for v in r['axes']])
            z,lab=m.improve_axes(m.agg(m.tensor(x),[ie])[0],a)
            if z>r['F_eff']+5e-9:
                update(ie,x,lab,z,'axes_refined')
                x,y=m.fit(x,lab,ie,steps);update(ie,x,lab,y,'alternating')
        # Neighboring transmission values seed branches missed by grid starts.
        prev=list(best)
        for ie in range(1,19):
            options=[]
            for j in (ie-1,ie+1):
                r=prev[j];x=np.array(r['weights'])
                a=np.array([np.flatnonzero(AX==v)[0] for v in r['axes']])
                z=float(score(m.fixed(m.agg(m.tensor(x),[ie])[0],a)[None])[0])
                if z>0 and z>=best[ie]['F_eff']*.985: options.append((z,x,a))
            if not options:continue
            z,x,a=max(options,key=lambda u:u[0])
            if z>best[ie]['F_eff']*.9998 and np.linalg.norm(wrap(x-best[ie]['weights']))>.15:
                x,y=m.fit(x,a,ie,steps);update(ie,x,a,y,'continuation')
    for r,g in zip(best,ghz):
        r['GHZ_R']=g['R']
        r['GHZ_F_eff']=g['F_eff']
    atomic(path,{'allocation':s,'calls':m.calls,'best':best,'GHZ':ghz})
    return str(path)
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--max-l',type=int,default=8);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--steps',type=int,default=26);p.add_argument('--output',type=Path,default=D/'runs')
    p.add_argument('--min-l',type=int,default=2)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    al=[s for l in range(a.max_l,a.min_l-1,-1) for n in range(2,min(5,l)+1) for s in allocs(l,n)]
    st=time.monotonic()
    with ProcessPoolExecutor(a.workers) as pool:
        futures=[pool.submit(run,(s,a.steps,str(a.output))) for s in al]
        for f in as_completed(futures):
            pa=Path(f.result());r=json.loads(pa.read_text());print(pa.stem,r['calls'],round(time.monotonic()-st,1),flush=True)
    print('done',len(al),time.monotonic()-st,flush=True)
if __name__=='__main__':main()
