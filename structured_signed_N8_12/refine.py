"""Per-transmission local refinement of the best group-structured candidates."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.optimize import minimize,minimize_scalar
from engine import Kernel
from search import ETAS,ROOT,write

def task(job):
    record,e=job;n=record['n'];family=record['family'];out=ROOT/'results';path=out/f'refined_N{n}_F{family}_e{e}.json'
    if path.exists():return path.name
    started=time.monotonic();sizes=record['sizes'];K=Kernel(sizes);spec=record['spec'];best=[None]*20;calls=0
    def weights(p):
        if family==1:return[0 if a==b or spec['sides'][a]==spec['sides'][b]else spec['signs'][a]*spec['signs'][b]*p[0]for a,b in K.pairs]
        if family==3:return[p[0]]*len(K.pairs)
        roles=spec['roles'];alpha,beta,gamma=p;w=[]
        for a,b in K.pairs:
            ra,rb=roles[a],roles[b]
            w.append(gamma if ra==rb=='H'else(alpha if'P'in(ra,rb)else-beta)if'H'in(ra,rb)else 0.)
        return w
    def f(p):
        nonlocal calls
        p=np.atleast_1d(p);w=weights(p);results=K.evaluate(w,ETAS);calls+=1
        for j,r in enumerate(results):
            if best[j]is None or r['R']>best[j]['R']+1e-12:best[j]=r|{'n':n,'family':family,'eta':float(ETAS[j]),'sizes':sizes,'spec':spec,'group_weights':w,'parameters':p.tolist()}
        return-results[e]['R']
    p=record['parameters'];f(p)
    if family==2:
        opt=minimize(f,p,method='Powell',bounds=[(0,np.pi),(0,np.pi),(-np.pi,np.pi)],options={'maxfev':48,'xtol':2e-5,'ftol':1e-7})
    else:
        lo,hi=(0,np.pi)if family==1 else(-np.pi,np.pi);step=(hi-lo)/16
        a=max(lo,p[0]-step);b=min(hi,p[0]+step)
        opt=minimize_scalar(lambda x:f([x]),bounds=(a,b),method='bounded',options={'maxiter':24,'xatol':2e-6})
    write(path,{'n':n,'family':family,'target_eta':float(ETAS[e]),'sizes':sizes,'spec':spec,'calls':calls,'seconds':time.monotonic()-started,'optimizer_success':bool(opt.success),'optimizer_message':str(opt.message),'best':best})
    return path.name

def main():
    jobs=[]
    for n in range(8,13):
        files=[json.loads(p.read_text())for p in(ROOT/'results').glob(f'group_N{n}_*.json')]
        for family in [1,2,3]:
            subset=[d for d in files if d['family']==family]
            for e in range(20):jobs.append((max((d['best'][e]for d in subset),key=lambda r:r['R']),e))
    with ProcessPoolExecutor(max_workers=4)as pool:
        fs=[pool.submit(task,j)for j in jobs]
        for i,f in enumerate(as_completed(fs)):print(json.dumps({'stage':'refine','done':i+1,'total':len(fs),'file':f.result()}),flush=True)

if __name__=='__main__':main()
