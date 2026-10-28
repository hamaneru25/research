"""Signed complete-bipartite weighted probes under flagged independent loss.

Family: h hubs and m=N-h leaves, all hub-leaf edges s_i*phi, s_i=+/-1.
In the all-positive gauge the hub effective Pauli is uniform; the physical
axis on negative hubs is conjugated by X. Leaves use a uniform signed Pauli.
This is a symmetry-preserving subset of all fixed local Clifford placements.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import argparse,csv,functools,itertools,json,math,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar

def spin_ops(n):
    z=np.diag((n-2*np.arange(n+1))/2.)
    off=np.sqrt((np.arange(n)+1)*(n-np.arange(n)))/2
    upper=np.diag(off,1)
    return [upper+upper.T,-1j*(upper-upper.T),z]

class Kernel:
    def __init__(self,n,a):
        self.n=n;self.a=a;self.b=n-a;self.branches=[];self.calls=0
        for ka in range(a+1):
            for kb in range(self.b+1):
                if ka+kb==0:continue
                ea=a-ka;eb=self.b-kb
                live=np.array(list(itertools.product(range(ka+1),range(kb+1))))
                lost=np.array(list(itertools.product(range(ea+1),range(eb+1))))
                amp=np.sqrt(np.array([math.comb(ka,x)*math.comb(kb,y) for x,y in live])[:,None]*np.array([math.comb(ea,x)*math.comb(eb,y) for x,y in lost])[None,:]/2.**n)
                features=(live[:,0,None]+lost[None,:,0])*(live[:,1,None]+lost[None,:,1])
                generators=np.array([np.kron(g,np.eye(kb+1)) for g in spin_ops(ka)]+[np.kron(np.eye(ka+1),g) for g in spin_ops(kb)])
                self.branches.append((ka+kb,math.comb(a,ka)*math.comb(self.b,kb),amp,features,generators))

    @functools.lru_cache(maxsize=256)
    def profile(self,phi):
        self.calls+=1
        result=np.zeros((self.n+1,6,6))
        for count,multiplicity,amp,features,g in self.branches:
            matrix=amp*np.exp(1j*phi*features)
            # Thin SVD also handles the full-survival pure state exactly.
            u,s,_=np.linalg.svd(matrix,full_matrices=False)
            vals=s*s;active=vals>1e-13;vals=vals[active];u=u[:,active]
            actions=g@u
            moments=np.einsum('idr,jdr,r->ij',actions.conj(),actions,vals).real
            inside=np.einsum('dr,idt->irt',u.conj(),actions)
            f=vals[:,None]*vals[None,:]/(vals[:,None]+vals[None,:])
            correction=np.einsum('rt,irt,jrt->ij',f,inside,inside.conj()).real
            q=4*moments-8*correction
            q=(q+q.T)/2
            result[count]+=multiplicity*q
        return result

def candidates(n,a,phi):
    """C transforms collective XYZ on canonical parts into selected generators."""
    b=n-a;cs=[];labels=[]
    orientations=[0,1] if a!=b else [0]
    for orient in orientations:
        h=(a,b)[orient]
        for negative in range(h//2+1):
            c=np.cos(negative*phi);s=np.sin(negative*phi)
            rotated=np.array([[c,s,0],[-s,c,0],[0,0,1.]])
            for hubaxis,leafaxis,sign in itertools.product(range(3),range(3),(1,-1)):
                axes=np.zeros((2,6));hub=np.eye(3)[hubaxis];leaf=sign*rotated[leafaxis]
                if orient==0:axes[0,:3]=hub;axes[1,3:]=leaf
                else:axes[0,:3]=leaf;axes[1,3:]=hub
                cs.append(axes)
                labels.append((h,negative,hubaxis,leafaxis,sign))
    return np.array(cs),labels

def effective2(q,w):
    vals,vec=np.linalg.eigh((q+q.swapaxes(-1,-2))/2)
    coords=np.einsum('...ij,i->...j',vec,w)
    tol=np.maximum(1e-12,np.max(np.abs(vals),axis=-1,keepdims=True)*1e-10)
    active=vals>tol
    valid=np.sum(np.where(active,0.,coords**2),axis=-1)<1e-15
    var=np.divide(coords**2,vals,out=np.zeros_like(vals),where=active).sum(-1)
    return np.divide(1.,var,out=np.zeros_like(var),where=valid & (var>1e-15))

class Objective:
    def __init__(self,n,a,etas):
        self.n=n;self.a=a;self.etas=np.array(etas);self.kernel=Kernel(n,a)
        k=np.arange(n+1)
        self.prob=self.etas[:,None]**k*(1-self.etas[:,None])**(n-k)
        self.w=np.array([a,n-a])/n

    def evaluate(self,phi,details=False):
        profile=self.kernel.profile(float(phi))
        tensor=np.einsum('ek,kij->eij',self.prob,profile)
        cs,labels=candidates(self.n,self.a,phi)
        q=np.einsum('cai,eij,cbj->ecab',cs,tensor,cs)
        scores=effective2(q,self.w)/self.n
        chosen=np.argmax(scores,axis=1)
        best=scores[np.arange(len(chosen)),chosen]
        if not details:return best
        return best,[{'hubs':labels[j][0],'negative_hubs':labels[j][1],'hub_axis':'XYZ'[labels[j][2]],'leaf_axis':('+' if labels[j][4]>0 else '-')+'XYZ'[labels[j][3]],'Qbar_group':q[i,j].tolist()} for i,j in enumerate(chosen)]

    def original_K2(self,phi):
        profile=self.kernel.profile(float(phi))
        tensor=np.einsum('ek,kij->eij',self.prob,profile)
        cs,labels=candidates(self.n,self.a,phi)
        index=labels.index((2,1,2,1,-1))
        q=np.einsum('ai,eij,bj->eab',cs[index],tensor,cs[index])
        return effective2(q,self.w)/self.n

def run_task(task):
    n,a,etas,points,peaks,xtol,outdir=task
    started=time.monotonic();obj=Objective(n,a,etas);grid=np.linspace(0,np.pi,points)
    coarse=np.array([obj.evaluate(phi) for phi in grid])
    rows=[];logs=[]
    for e,eta in enumerate(etas):
        values=coarse[:,e];idx=int(np.argmax(values));bestphi=float(grid[idx]);bestr=float(values[idx])
        maxima=[j for j in range(1,points-1) if values[j]>=values[j-1] and values[j]>=values[j+1] and max(values[j]-values[j-1],values[j]-values[j+1])>1e-11]
        maxima=sorted(maxima,key=lambda j:values[j],reverse=True)[:peaks]
        if idx not in (0,points-1) and idx not in maxima:maxima.insert(0,idx)
        for j in maxima:
            opt=minimize_scalar(lambda phi:-obj.evaluate(phi)[e],bounds=(grid[j-1],grid[j+1]),method='bounded',options={'xatol':xtol,'maxiter':80})
            r=-float(opt.fun)
            logs.append({'eta':eta,'grid_peak':int(j),'phi':float(opt.x),'R':r,'success':bool(opt.success),'nfev':int(opt.nfev)})
            if r>bestr:bestr=r;bestphi=float(opt.x)
        score,detail=obj.evaluate(bestphi,True)
        rows.append({'n':n,'eta':eta,'canonical_part_a':a,'phi':bestphi,'R':float(score[e]),**detail[e]})
    if a==2:
        original_coarse=np.array([obj.original_K2(phi) for phi in grid])
        for e,eta in enumerate(etas):
            values=original_coarse[:,e];idx=int(np.argmax(values));bestphi=float(grid[idx]);bestr=float(values[idx])
            maxima=[j for j in range(1,points-1) if values[j]>=values[j-1] and values[j]>=values[j+1] and max(values[j]-values[j-1],values[j]-values[j+1])>1e-11]
            maxima=sorted(maxima,key=lambda j:values[j],reverse=True)[:peaks]
            for j in maxima:
                opt=minimize_scalar(lambda phi:-obj.original_K2(phi)[e],bounds=(grid[j-1],grid[j+1]),method='bounded',options={'xatol':xtol,'maxiter':80})
                if -opt.fun>bestr:bestr=-float(opt.fun);bestphi=float(opt.x)
            rows[e]['original_K2_pm_R']=bestr;rows[e]['original_K2_pm_phi']=bestphi
    result={'n':n,'canonical_part_a':a,'rows':rows,'logs':logs,'calls':obj.kernel.calls,'seconds':time.monotonic()-started,'coarse_grid':grid.tolist(),'coarse_best_R':coarse.tolist()}
    path=Path(outdir)/f'family_N{n}_a{a}.json';path.write_text(json.dumps(result,indent=2)+'\n')
    return {'n':n,'a':a,'seconds':result['seconds'],'calls':result['calls'],'path':str(path)}

def ghz_best(n,eta):
    cost=[0.]+[float('inf')]*n;parts=[[] for _ in range(n+1)]
    for k in range(1,n+1):
        size=min(range(1,k+1),key=lambda s:eta**(-s)+cost[k-s])
        cost[k]=eta**(-size)+cost[k-size];parts[k]=[size]+parts[k-size]
    return n/cost[n],sorted(parts[n])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sensors',default=','.join(str(n) for n in range(8,21)));p.add_argument('--etas',default=','.join(f'{i/20:g}' for i in range(1,21)))
    p.add_argument('--grid-points',type=int,default=65);p.add_argument('--peaks',type=int,default=2);p.add_argument('--xtol',type=float,default=2e-7)
    p.add_argument('--workers',type=int,default=6);p.add_argument('--output',type=Path,default=Path(__file__).with_name('results'));p.add_argument('--resume',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    ns=[int(x) for x in args.sensors.split(',')];etas=[float(x) for x in args.etas.split(',')]
    tasks=[(n,a,etas,args.grid_points,args.peaks,args.xtol,str(args.output)) for n in ns for a in range(1,n//2+1)]
    (args.output/'settings.json').write_text(json.dumps(vars(args)|{'output':str(args.output)},indent=2)+'\n')
    pending=[t for t in tasks if not (args.resume and (args.output/f'family_N{t[0]}_a{t[1]}.json').exists())]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fs=[pool.submit(run_task,t) for t in pending]
        for f in as_completed(fs):print(json.dumps(f.result()),flush=True)
    combined=[]
    for n in ns:
        families=[json.loads((args.output/f'family_N{n}_a{a}.json').read_text()) for a in range(1,n//2+1)]
        for j,eta in enumerate(etas):
            best=max((f['rows'][j] for f in families),key=lambda r:r['R'])
            # K2 is evaluated in both hub orientations and all symmetry-preserving axes.
            k2=families[1]['rows'][j] if n>=4 else None
            ghz,partition=ghz_best(n,eta)
            combined.append(best|{'R_GHZ_partition':ghz,'GHZ_partition':partition,'R_K2_candidate':k2['R'],'K2_candidate_hubs':k2['hubs'],'weighted_over_GHZ_percent':100*(best['R']/ghz-1)})
            combined[-1].update({'R_original_K2_pm':k2['original_K2_pm_R'],'phi_original_K2_pm':k2['original_K2_pm_phi']})
            combined[-1].pop('original_K2_pm_R',None)
            combined[-1].pop('original_K2_pm_phi',None)
    (args.output/'best.json').write_text(json.dumps(combined,indent=2)+'\n')
    with (args.output/'best.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(combined[0]));writer.writeheader();writer.writerows(combined)
    print(json.dumps({'completed_rows':len(combined),'summary':[(n,sorted({min(r['hubs'],n-r['hubs']) for r in combined if r['n']==n})) for n in ns]},indent=2),flush=True)

if __name__=='__main__':main()
