"""Recompute saved grouped QFIMs and independently check selected N=8 probes."""
import argparse,json,time
from pathlib import Path
import numpy as np
from search import Objective,candidates,effective2,ghz_best
from verify import dense

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,default=Path(__file__).with_name('results'));p.add_argument('--dense-etas',default='0.2,0.7');args=p.parse_args()
    rows=json.loads((args.results/'best.json').read_text());kernels={};errors=[];qerrors=[];ghzerrors=[];started=time.monotonic()
    for row in rows:
        n=row['n'];a=row['canonical_part_a'];eta=row['eta'];key=(n,a)
        if key not in kernels:kernels[key]=Objective(n,a,[eta])
        obj=kernels[key];k=np.arange(n+1)
        tensor=np.einsum('k,kij->ij',eta**k*(1-eta)**(n-k),obj.kernel.profile(float(row['phi'])))
        cs,labels=candidates(n,a,row['phi']);ha='XYZ'.index(row['hub_axis']);la='XYZ'.index(row['leaf_axis'][1]);ls=1 if row['leaf_axis'][0]=='+' else -1
        index=labels.index((row['hubs'],row['negative_hubs'],ha,la,ls));c=cs[index];q=c@tensor@c.T
        r=float(effective2(q,np.array([a,n-a])/n)/n)
        errors.append(abs(r-row['R']));qerrors.append(float(np.max(abs(q-np.array(row['Qbar_group'])))))
        ghz,parts=ghz_best(n,eta);ghzerrors.append(abs(ghz-row['R_GHZ_partition']))
        assert sum(row['GHZ_partition'])==n
    assert max(errors)<1e-8 and max(qerrors)<1e-8 and max(ghzerrors)<1e-10
    checks=[]
    for eta in [float(e) for e in args.dense_etas.split(',') if e]:
        row=next((r for r in rows if r['n']==8 and abs(r['eta']-eta)<1e-10),None)
        if row is None:continue
        n=8;h=row['hubs'];ha='XYZ'.index(row['hub_axis']);la='XYZ'.index(row['leaf_axis'][1]);ls=1 if row['leaf_axis'][0]=='+' else -1
        q=dense(n,h,row['negative_hubs'],row['phi'],ha,la,ls,eta)
        vals,v=np.linalg.eigh(q);active=vals>1e-10;coords=v.T@(np.ones(n)/n)
        r=1/(n*np.sum(coords[active]**2/vals[active])) if np.linalg.norm(coords[~active])<1e-8 else 0.
        assert abs(r-row['R'])<1e-8
        checks.append({'n':n,'eta':eta,'survivor_masks_checked':256,'R_dense':r,'R_saved':row['R'],'absolute_error':abs(r-row['R'])})
    result={'saved_rows_checked':len(rows),'family_files':len(list(args.results.glob('family_N*_a*.json'))),'max_R_recalculation_error':max(errors),'max_group_QFIM_error':max(qerrors),'max_GHZ_DP_error':max(ghzerrors),'dense_N8_checks':checks,'seconds':time.monotonic()-started,'scope':'Recalculation of saved feasible states, not a certificate of a global optimum.'}
    (args.results/'publication_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    extended=args.results/'extended_validation.json'
    if not extended.exists():
        payload={'dense_N8_checks':checks,'saved_rows_checked':len(rows),'max_R_recalculation_error':max(errors),'note':'Fresh validation. The historical 65-versus-129 grid comparison is not rerun.'}
        extended.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
