"""Compare all searched allocations; check finalists over all Clifford axes and density matrices."""
from expanded import *
from density_verify import independently
import csv
def sweep(r):
    m=Model(r['allocation']);ie=round(r['eta']*20)-1
    x=np.array(r['weights']);q=m.agg(m.tensor(x),[ie])[0]
    z,lab=m.exact_axes(q)
    gain=z-r['F_eff'];changed=False
    if gain>1e-8:
        old=r
        r=m.rec(z,x,lab,ie,'full_axis_recheck')
        r.update({k:old[k] for k in ('GHZ_R','GHZ_F_eff')})
        x1,y=m.fit(x,lab,ie,32)
        if y>z+1e-8:
            zz,la=m.exact_axes(m.agg(m.tensor(x1),[ie])[0])
            if zz>z+1e-8:
                r=m.rec(zz,x1,la,ie,'full_axis_recheck_and_refine')
                r.update({k:old[k] for k in ('GHZ_R','GHZ_F_eff')})
        changed=True
    r.update(exact_axes_tested=6**m.l//2,axis_recheck_gain=float(gain))
    return r,changed,m.calls
def verify(r):
    Q,z,e=independently(r)
    return dict(r,Qbar=Q,F_eff_verified=z,verification_error=e)
def rk(r):return (r['N'],r['eta'],r['L'],tuple(r['allocation']))
def select(rs,k,tol=1e-8):
    v=max(r[k] for r in rs)
    near=[r for r in rs if v-r[k]<=tol]
    return min(near,key=lambda r:(r['L'],max(r['allocation'])-min(r['allocation']),r['allocation'])),near
def collect(rows):
    out=[]
    for n in range(2,6):
        for t in ETAS:
            rs=[r for r in rows if r['N']==n and r['eta']==t]
            if not rs:continue
            r,rt=select(rs,'R');f,ft=select(rs,'F_eff')
            z=[q for q in rs if q['L']<=6]
            old=max(q['R'] for q in z); oldf=max(q['F_eff'] for q in z)
            ghz=max(q['GHZ_R'] for q in rs);ghzf=max(q['GHZ_F_eff'] for q in rs)
            out.append({'N':n,'eta':float(t),'best_R':r,'best_F':f,'tied_R':[q['allocation'] for q in rt],
                        'tied_F':[q['allocation'] for q in ft],'R_up_to_6':old,'F_up_to_6':oldf,
                        'GHZ_R':ghz,'GHZ_F':ghzf,'R_one_per_node':next(q['R'] for q in rs if q['L']==n)})
    return out
def main():
    files=[json.loads(p.read_text()) for p in sorted((D/'runs').glob('a_*.json'))]
    rows=[r for f in files for r in f['best']];co=collect(rows)
    expected=sum(1 for n in range(2,6) for l in range(n,9) for s in allocs(l,n))*20
    assert len({rk(r) for r in rows})==len(rows)==expected
    # Recheck both resource objectives for every N and eta, plus their 6-photon budget controls.
    finalists={}
    for c in co:
        for r in (c['best_R'],c['best_F']):finalists[rk(r)]=r
        rr=[r for r in rows if r['N']==c['N'] and r['eta']==c['eta'] and r['L']<=6]
        for ob in ('R','F_eff'):
            r,_=select(rr,ob);finalists[rk(r)]=r
    changes=extra=0
    with ProcessPoolExecutor(7) as e:
        fs=[e.submit(sweep,r) for r in finalists.values()]
        for i,f in enumerate(as_completed(fs),1):
            r,c,m=f.result();finalists[rk(r)]=r;changes+=c;extra+=m
            if i%16==0 or i==len(fs):print('axis rechecks',i,len(fs),'improvements',changes,flush=True)
    rr=[finalists.get(rk(r),r) for r in rows]
    print('starting density verification',len(rr),flush=True)
    with ProcessPoolExecutor(8) as e:
        rr=list(e.map(verify,rr,chunksize=2))
    err=max(r['verification_error'] for r in rr)
    if err>1e-7:
        print('worst',max(rr,key=lambda r:r['verification_error']),flush=True)
        raise ValueError(('density check failed',err))
    ghzerr=max(abs(r['F_eff_verified']-r['GHZ_F_eff']) for r in rr if r['source']=='GHZ')
    result={'results':collect(rr),'allocations':rr,'verification_max_F_error':err,
            'GHZ_verification_max_F_error':ghzerr,'condition_count':len(rr),
            'candidate_weight_evaluations':sum(f['calls'] for f in files)+extra,
            'full_axis_rechecks':len(finalists),'improved_by_full_axis_recheck':changes}
    atomic(D/'final.json',result)
    print({k:v for k,v in result.items() if k not in ('results','allocations')},flush=True)
if __name__=='__main__':main()
