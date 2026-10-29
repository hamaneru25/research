"""Reproducible bounded numerical search. Best found, NOT exhaustive.

Family 1: systematic balanced rank-one sign seeds plus general valid sign seeds.
Family 2: all hub/leaf-group sizes, three independent edge-type weights.
Family 3: unrestricted individual edge weights, structured multistart + SPSA.
Family 4: analytic full-N GHZ baseline. No setting mixtures or ancillas.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,gc,itertools,json,math,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from scipy.stats import qmc
from engine import Kernel,ghz
from dense_engine import qfi_tensor,layout

ROOT=Path(__file__).resolve().parent
ETAS=np.arange(1,21)/20

def write(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

def expand(sizes,weights):
    groups=np.repeat(np.arange(len(sizes)),sizes);pairs=list(itertools.combinations_with_replacement(range(len(sizes)),2));lookup=dict(zip(pairs,weights))
    return np.array([lookup[tuple(sorted((groups[i],groups[j])))]for i,j in itertools.combinations(range(sum(sizes)),2)])

def best_axes(tensor,n,axes):
    signs=np.array([(1,)+s for s in itertools.product((1,-1),repeat=n-1)])
    targets=signs/n;gather=3*np.arange(n)+np.array(axes)
    out=[]
    for eta,T in zip(ETAS,tensor):
        q=T[gather[:,:,None],gather[:,None,:]]
        vals,vec=np.linalg.eigh(q)
        coord=np.einsum('aji,si->asj',vec.swapaxes(-1,-2),targets)
        active=vals>np.maximum(1e-12,np.max(np.abs(vals),axis=-1,keepdims=True)*1e-10)
        valid=np.sum(np.where(active[:,None,:],0,coord**2),axis=-1)<1e-14
        cost=np.divide(coord**2,vals[:,None,:],out=np.zeros_like(coord),where=active[:,None,:]).sum(-1)
        score=np.divide(1,n*cost,out=np.zeros_like(cost),where=valid&(cost>1e-14))
        ai,si=np.unravel_index(np.argmax(score),score.shape)
        out.append({'eta':float(eta),'R':float(score[ai,si]),'axes':[('+'if s>0 else '-')+'XYZ'[a]for a,s in zip(axes[ai],signs[si])],'Qbar':(q[ai]*signs[si,:,None]*signs[si,None,:]).tolist()})
    return out

def catalogue_axes(n,h,rng,count=24,derived=False):
    axes=[tuple([a]*h+[b]*(n-h))for a,b in itertools.product(range(3),repeat=2)]
    if derived:axes.extend([tuple([2]+[1]*(n-1)),tuple([0]*(n-3)+[1,1,2]),tuple([1]*(n-2)+[0,2])])
    for g in [2,3,4]:
        labels=np.minimum(g-1,np.arange(n)*g//n)
        for _ in range(5):axes.append(tuple(rng.integers(3,size=g)[labels]))
    while len(set(axes))<count:axes.append(tuple(rng.integers(3,size=n)))
    return np.array(sorted(set(axes)))

def dense_evaluate(n,x,axes):
    T=qfi_tensor(n,x);counts=np.array([s.bit_count()for s in range(1<<n)])
    prob=ETAS[:,None]**counts*(1-ETAS[:,None])**(n-counts)
    avg=np.einsum('es,sij->eij',prob,T);del T
    return best_axes(avg,n,axes)

def group_job(job):
    n,family,sizes,spec,points,outdir=job;started=time.monotonic();kernel=Kernel(sizes);g=len(sizes)
    filename=f'group_N{n}_F{family}_{spec["id"]}.json';path=Path(outdir)/filename
    if path.exists():return filename
    best=[None]*len(ETAS);calls=0
    if family==1:
        params=[[float(phi)]for phi in np.linspace(0,np.pi,points)]
    elif family==2:
        sampler=qmc.Sobol(3,scramble=True,seed=spec['seed'])
        params=(sampler.random_base2(3)*[np.pi,np.pi,2*np.pi]-[0,0,np.pi]).tolist()
        for phi in [.05,.15,.4,.7,1.,1.5,2.5,np.pi]:
            params.extend([[phi,phi,-.2*phi],[phi,min(np.pi,1.25*phi),-.25*phi],[phi,phi,.25*phi]])
    else:
        params=[[float(phi)]for phi in np.linspace(-np.pi,np.pi,points)]
    for p in params:
        if family==1:w=[0 if a==b or spec['sides'][a]==spec['sides'][b] else spec['signs'][a]*spec['signs'][b]*p[0]for a,b in kernel.pairs]
        elif family==2:
            alpha,beta,gamma=p;roles=spec['roles']
            def edge(a,b):
                ra,rb=roles[a],roles[b]
                if ra==rb=='H':return gamma
                if ra=='H' or rb=='H':return alpha if 'P'in(ra,rb)else -beta
                return 0.
            w=[edge(a,b)for a,b in kernel.pairs]
        else:w=[p[0]]*len(kernel.pairs)
        results=kernel.evaluate(w,ETAS);calls+=1
        for e,r in enumerate(results):
            if best[e]is None or r['R']>best[e]['R']+1e-11:
                best[e]=r|{'eta':float(ETAS[e]),'n':n,'family':family,'sizes':sizes,'group_weights':w,'parameters':p,'spec':spec}
    write(path,{'n':n,'family':family,'sizes':sizes,'spec':spec,'calls':calls,'seconds':time.monotonic()-started,'best':best})
    return filename

def dense_job(job):
    n,family,seed,steps,outdir=job;path=Path(outdir)/f'dense_N{n}_F{family}_s{seed}.json'
    if path.exists():return path.name
    rng=np.random.default_rng(seed);pairs=layout(n)[0];started=time.monotonic();h=1+seed%(n-1);m=n-h
    if family==1:
        if h*m%2:h=2;m=n-h
        a=int(rng.integers(m//2+1))if h%2==0 else m//2
        if h%2 and m%2:h=2;m=n-h;a=0
        row_counts=[a]*(h//2)+[m-a]*(h-h//2)if 2*a!=m else[a]*h
        matrix=np.full((h,m),-1)
        for i,c in enumerate(row_counts):matrix[i,rng.choice(m,c,replace=False)]=1
        feature=np.array([matrix[i,j-h]if i<h<=j else 0 for i,j in pairs])
        candidates=[feature*p for p in [.1,.3,.6,.9,1.3,2.,np.pi]]
        metadata={'h':h,'a':a,'sign_matrix':matrix.tolist()}
    else:
        labels=np.minimum(3,np.arange(n)*4//n)
        block=rng.uniform(-np.pi,np.pi,(4,4));block=(block+block.T)/2
        x=np.array([block[labels[i],labels[j]]for i,j in pairs])
        candidates=[x,np.full(len(pairs),.8),np.full(len(pairs),2.6)]
        # Scaled vertex-group seeds from the saved N6/N7 clique patterns.
        # Additional starts use a separate seed namespace, preserving the
        # exact RNG sequence and original starts of the base search.
        original_candidates=list(candidates)
        candidates.append(np.array([-3.132220 if i==0 else -.788634 for i,j in pairs]))
        labels=np.array([0]*(n-3)+[1,1,2]);B=np.array([[.679673,-.951090,3.063340],[-.951090,.661127,3.118100],[3.063340,3.118100,0]])
        candidates.append(np.array([B[labels[i],labels[j]]for i,j in pairs]))
        labels=np.array([0]*(n-2)+[1,2]);B=np.array([[1.222567,-.113949,2.875122],[-.113949,0,-2.756411],[2.875122,-2.756411,0]])
        candidates.append(np.array([B[labels[i],labels[j]]for i,j in pairs]))
        a=(n-2)//2;labels=np.array([0]*a+[1]*(n-2-a)+[2,3]);B=np.array([[-2.564189,2.953700,2.757900,-2.960980],[2.953700,-2.564189,-2.960980,2.757900],[2.757900,-2.960980,0,-2.733211],[-2.960980,2.757900,-2.733211,0]])
        candidates.append(np.array([B[labels[i],labels[j]]for i,j in pairs]))
        metadata={'initial_group_labels':labels.tolist(),'unrestricted_edge_variables':len(pairs)}
        if seed<910000:candidates=original_candidates
    axes=catalogue_axes(n,h,rng,derived=seed>=910000);best=[None]*len(ETAS);calls=0;current=None;current_value=-np.inf
    def evaluate(x):
        nonlocal calls,current,current_value
        rs=dense_evaluate(n,x,axes);calls+=1
        value=float(np.mean([np.log(max(r['R'],1e-15)/eta)for r,eta in zip(rs,ETAS)]))
        for e,r in enumerate(rs):
            if best[e]is None or r['R']>best[e]['R']+1e-11:best[e]=r|{'n':n,'family':family,'weights_rad':np.mod(x,2*np.pi).tolist(),'seed':seed,'metadata':metadata}
        if value>current_value:current=x.copy();current_value=value
        return value
    for x in candidates:evaluate(x)
    if family==3:
        # SPSA perturbs every individual edge: no maximum number of groups.
        x=current.copy()
        for k in range(steps):
            delta=rng.choice([-1.,1.],len(pairs));c=.12/(k+1)**.101
            plus=evaluate(x+c*delta);minus=evaluate(x-c*delta)
            grad=(plus-minus)/(2*c)*delta
            x=np.mod(x+.08/(k+1)**.602*np.clip(grad,-2,2)+np.pi,2*np.pi)-np.pi
            evaluate(x)
    write(path,{'n':n,'family':family,'seed':seed,'calls':calls,'seconds':time.monotonic()-started,'axis_catalogue':axes.tolist(),'best':best})
    return path.name

def jobs_for(n,points,out):
    jobs=[];idx=0
    for h in range(1,n):
        m=n-h
        for hn in range(h//2+1):
            for ln in range(m//2+1):
                if (h-2*hn)*(m-2*ln):continue
                raw=[(h-hn,0,1),(hn,0,-1),(m-ln,1,1),(ln,1,-1)];raw=[r for r in raw if r[0]]
                sizes=[r[0]for r in raw];spec={'id':str(idx),'h':h,'negative_hubs':hn,'negative_leaves':ln,'sides':[r[1]for r in raw],'signs':[r[2]for r in raw]}
                jobs.append((n,1,sizes,spec,points,out));idx+=1
    for h in range(2,n):
        m=n-h
        for positive in range(m+1):
            raw=[(h,'H'),(positive,'P'),(m-positive,'M')];raw=[r for r in raw if r[0]]
            spec={'id':f'{h}_{positive}','h':h,'positive_leaves':positive,'roles':[r[1]for r in raw],'seed':812000+n*100+h*10+positive}
            jobs.append((n,2,[r[0]for r in raw],spec,points,out))
    jobs.append((n,3,[n],{'id':'uniform'},points,out))
    return jobs

def main():
    p=argparse.ArgumentParser();p.add_argument('--sensors',default='8,9,10,11,12');p.add_argument('--workers',type=int,default=4);p.add_argument('--points',type=int,default=17);p.add_argument('--steps',type=int,default=6);p.add_argument('--dense-starts',type=int,default=2);p.add_argument('--output',type=Path,default=ROOT/'results');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    ns=list(map(int,a.sensors.split(',')));settings=vars(a)|{'output':str(a.output),'etas':ETAS.tolist(),'claim':'bounded best-found numerical search, not exhaustive signs or all Clifford axes'}
    write(a.output/'settings.json',settings)
    group_jobs=[j for n in ns for j in jobs_for(n,a.points,str(a.output))]
    with ProcessPoolExecutor(max_workers=a.workers)as pool:
        fs=[pool.submit(group_job,j)for j in group_jobs]
        for i,f in enumerate(as_completed(fs)):print(json.dumps({'stage':'groups','done':i+1,'total':len(fs),'file':f.result()}),flush=True)
    dense_jobs=[(n,f,900000+n*100+f*10+s,a.steps,str(a.output))for n in ns for f in [1,3]for s in range(a.dense_starts)]
    # Limit peak memory of the 12-sensor batched low-rank kernel.
    with ProcessPoolExecutor(max_workers=min(2,a.workers))as pool:
        fs=[pool.submit(dense_job,j)for j in dense_jobs]
        for i,f in enumerate(as_completed(fs)):print(json.dumps({'stage':'dense','done':i+1,'total':len(fs),'file':f.result()}),flush=True)
    records=[]
    for n in ns:
        files=[json.loads(f.read_text())for f in a.output.glob(f'*_N{n}_*.json')]
        for e,eta in enumerate(ETAS):
            baseline={'n':n,'eta':float(eta),'family':4,'R':ghz(n,eta),'state':'full_N_GHZ'}
            by_family=[max((f['best'][e]for f in files if f['family']==family),key=lambda r:r['R'])for family in [1,2,3]]+[baseline]
            records.append({'n':n,'eta':float(eta),'families':by_family,'best':max(by_family,key=lambda r:r['R'])})
    write(a.output/'best.json',records)
    import csv
    with(a.output/'comparison.csv').open('w')as f:
        writer=csv.writer(f);writer.writerow(['n','eta','family1_R','family2_R','family3_R','GHZ_R','best_family','best_R'])
        for r in records:writer.writerow([r['n'],r['eta']]+[x['R']for x in r['families']]+[r['best']['family'],r['best']['R']])
    print(json.dumps({'stage':'finished','rows':len(records)}),flush=True)

if __name__=='__main__':main()
