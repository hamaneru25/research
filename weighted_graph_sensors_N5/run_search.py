"""Weighted-graph/Clifford numerical search under IID flagged erasure.

All ten K5 edge angles are continuous. Local Clifford axes are enumerated
at EVERY weight evaluation, including every relative sign. No LC reduction
of weighted graphs is assumed. Continuous global optimality is not claimed.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import scipy
from scipy.optimize import minimize,differential_evolution
from engine import (layout,qfi_tensor,best_clifford,fixed_clifford_qfim,
                    WeightObjective,search_catalogue,loss_weights)
ROOT=Path(__file__).resolve().parent


def write_json(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def record(n,weights,score,axes,signs,origin):
    return {'n':n,'weights_rad':np.mod(weights,2*np.pi).tolist(),'score':float(score),
            'axes_indices':np.asarray(axes).tolist(),'signs':np.asarray(signs).tolist(),'origin':origin}


def evaluate(n,weights,probabilities):
    T=qfi_tensor(n,weights);avg=np.einsum('s,sij->ij',probabilities,T)
    value,axes,signs=best_clifford(avg,n)
    return value,axes,signs,T


def old_controls(n,etas):
    base=ROOT.parent/'fixed_clifford_sensors'/'results'
    rows={float(r['eta']):r for r in csv.DictReader((base/'best_by_n_eta.csv').open()) if int(r['n'])==n}
    protocols=json.loads((base/'protocols.json').read_text());pairs=layout(n)[0]
    controls=[]
    for eta in etas:
        if eta not in rows:
            # New eta points: independently exhaust the ordinary-graph class
            # rather than interpolating the previous 20-point results.
            chosen=None
            for item in search_catalogue(n,'lc'):
                weights=[np.pi if list(edge) in item['edges'] else 0. for edge in pairs]
                value,axes,signs,_=evaluate(n,weights,loss_weights(n,[eta])[0])
                if chosen is None or value>chosen['score']+1e-12:
                    chosen=record(n,weights,value,axes,signs,'ordinary_graph_exhaustive_control')
            controls.append(chosen)
            continue
        row=rows[eta];proto=protocols[row['protocol_id']]
        weights=[np.pi if list(edge) in proto['edges'] else 0. for edge in pairs]
        axes=['XYZ'.index(axis[1]) for axis in proto['effective_axes']]
        signs=[1 if axis[0]=='+' else -1 for axis in proto['effective_axes']]
        controls.append(record(n,weights,float(row['information_per_photon']),axes,signs,'previous_exhaustive_Clifford'))
    return controls


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sensors',type=int,default=5,choices=[5])
    p.add_argument('--etas',default=','.join(str(k/20) for k in range(1,21)))
    p.add_argument('--uniform-points',type=int,default=65,help='0..2pi inclusive per unlabeled topology')
    p.add_argument('--uniform-cache',type=Path,help='Reuse saved uniform_scan_seeds.json from an identical scan')
    p.add_argument('--topology-starts',type=int,default=6)
    p.add_argument('--random-starts',type=int,default=64)
    p.add_argument('--perturbed-starts',type=int,default=16)
    p.add_argument('--seed',type=int,default=20261022)
    p.add_argument('--maxiter',type=int,default=400)
    p.add_argument('--gtol',type=float,default=1e-8)
    p.add_argument('--ftol',type=float,default=1e-12)
    p.add_argument('--ridge',type=float,default=1e-10)
    p.add_argument('--de-etas',default='0.5,0.6,0.65,0.7,0.75,0.8')
    p.add_argument('--de-maxiter',type=int,default=100)
    p.add_argument('--de-popsize',type=int,default=6)
    p.add_argument('--improvement-atol',type=float,default=1e-7)
    p.add_argument('--improvement-rtol',type=float,default=1e-7)
    p.add_argument('--output',type=Path,default=ROOT/'results')
    p.add_argument('--overwrite',action='store_true')
    a=p.parse_args();n=a.sensors
    etas=sorted(set(float(e) for e in a.etas.split(',')))
    de_etas=set(float(e) for e in a.de_etas.split(',') if e.strip())
    if not etas or any(not np.isfinite(e) or not 0<e<=1 for e in etas):raise ValueError('eta must be in (0,1]')
    if not de_etas.issubset(etas):raise ValueError('de-etas must be present in etas (or use an empty string)')
    if a.uniform_points<3 or a.uniform_points%2==0:raise ValueError('uniform-points must be odd and >=3 to include pi')
    if a.topology_starts<1 or min(a.random_starts,a.perturbed_starts)<0:raise ValueError('Invalid starts')
    if min(a.maxiter,a.de_maxiter,a.de_popsize)<1:raise ValueError('Invalid iteration count')
    if any(not np.isfinite(x) or x<=0 for x in [a.gtol,a.ftol,a.ridge,a.improvement_atol,a.improvement_rtol]):raise ValueError('Tolerances must be positive')
    out=a.output.resolve()
    if out.exists() and any(out.iterdir()) and not a.overwrite:raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    started=datetime.now(timezone.utc).isoformat();start=time.perf_counter();rng=np.random.default_rng(a.seed)
    probabilities=loss_weights(n,etas);controls=old_controls(n,etas)
    best=[dict(c) for c in controls];coarse=[];graph_rows=[]
    catalogue=search_catalogue(n,'isomorphism');uniform_evaluations=0
    # LC equivalence at pi does not extend to arbitrary edge weights.
    if a.uniform_cache:
        coarse=json.loads(a.uniform_cache.read_text())
        expected={(item['graph_mask'],eta) for item in catalogue for eta in etas}
        if {(r['graph_mask'],r['eta']) for r in coarse}!=expected:
            raise ValueError('Uniform cache does not cover the configured topologies and eta grid')
        for r in coarse:
            e=etas.index(r['eta'])
            if r['score']>best[e]['score']+1e-12:best[e]=dict(r)
            graph_rows.append({'graph_mask':r['graph_mask'],'eta':r['eta'],
                               'uniform_best_found':r['score'],'uniform_phi_rad':r['uniform_phi_rad']})
        print(json.dumps({'stage':'uniform_cache_loaded','records':len(coarse)}),flush=True)
    for gi,item in enumerate(catalogue if not a.uniform_cache else []):
        g=item['graph_mask'];present=np.array([bool(g>>i&1) for i in range(len(layout(n)[0]))])
        local=[None]*len(etas)
        for phi in np.linspace(0,2*np.pi,a.uniform_points):
            weights=phi*present;T=qfi_tensor(n,weights)
            avgs=np.einsum('es,sij->eij',probabilities,T);uniform_evaluations+=1
            for e,eta in enumerate(etas):
                value,axes,signs=best_clifford(avgs[e],n)
                if local[e] is None or value>local[e]['score']+1e-12:
                    local[e]=record(n,weights,value,axes,signs,'uniform_scan')
                    local[e].update({'graph_mask':g,'uniform_phi_rad':float(phi),'eta':eta})
                if value>best[e]['score']+1e-12:best[e]=dict(local[e])
        coarse.extend(local)
        for e,eta in enumerate(etas):
            graph_rows.append({'graph_mask':g,'eta':eta,'uniform_best_found':local[e]['score'],
                               'uniform_phi_rad':local[e]['uniform_phi_rad']})
        print(json.dumps({'stage':'uniform_graph','index':gi+1,'total':len(catalogue),'graph_mask':g,
                          'maximum_gain_so_far':max(b['score']-c['score'] for b,c in zip(best,controls)),
                          'elapsed_seconds':time.perf_counter()-start}),flush=True)
    write_json(out/'uniform_scan_seeds.json',coarse)
    write_csv(out/'uniform_by_graph_eta.csv',graph_rows)
    run_rows=[];previous=None
    for e,eta in enumerate(etas):
        top=sorted([r for r in coarse if r['eta']==eta],key=lambda r:-r['score'])[:a.topology_starts]
        seeds=[('uniform_topology',np.array(r['weights_rad'])) for r in top]
        seeds.append(('Clifford_control',np.array(controls[e]['weights_rad'])))
        if previous is not None:seeds.append(('eta_continuation',previous.copy()))
        for j in range(a.perturbed_starts):
            center=np.array(top[j%len(top)]['weights_rad'])
            seeds.append(('perturbed',center+rng.normal(0,.08 if j%2==0 else .6,len(center))))
        for j in range(a.random_starts):seeds.append(('random',rng.uniform(0,2*np.pi,len(layout(n)[0]))))
        for seed_index,(kind,x0) in enumerate(seeds):
            objective=WeightObjective(n,eta,a.ridge)
            sol=minimize(objective,x0,jac=True,method='L-BFGS-B',
                         options={'maxiter':a.maxiter,'gtol':a.gtol,'ftol':a.ftol,'maxls':40})
            value,axes,signs,_=evaluate(n,sol.x,probabilities[e])
            if value>best[e]['score']+1e-11:best[e]=record(n,sol.x,value,axes,signs,kind)
            run_rows.append({'eta':eta,'method':'L-BFGS-B','seed_index':seed_index,'origin':kind,
                             'score':value,'nfev':sol.nfev,'iterations':sol.nit,'success':bool(sol.success),
                             'message':str(sol.message)})
        if eta in de_etas:
            def value_only(x):
                avg=np.einsum('s,sij->ij',probabilities[e],qfi_tensor(n,x))
                return -best_clifford(avg,n)[0]
            de=differential_evolution(value_only,[(0,2*np.pi)]*len(layout(n)[0]),
                maxiter=a.de_maxiter,popsize=a.de_popsize,seed=int(rng.integers(2**31)),
                polish=False,tol=1e-7,atol=1e-10,x0=np.array(best[e]['weights_rad']))
            polished=minimize(WeightObjective(n,eta,a.ridge),de.x,jac=True,method='L-BFGS-B',
                options={'maxiter':a.maxiter,'gtol':a.gtol,'ftol':a.ftol,'maxls':40})
            value,axes,signs,_=evaluate(n,polished.x,probabilities[e])
            if value>best[e]['score']+1e-11:best[e]=record(n,polished.x,value,axes,signs,'differential_evolution')
            run_rows.append({'eta':eta,'method':'differential_evolution','seed_index':-1,'origin':'population',
                             'score':value,'nfev':de.nfev+polished.nfev,'iterations':de.nit,
                             'success':bool(de.success),'message':str(de.message)})
        previous=np.array(best[e]['weights_rad'])
        best[e]['eta']=eta
        write_json(out/'checkpoint.json',{'completed_eta':eta,'best_by_eta':best})
        print(json.dumps({'stage':'eta_completed','eta':eta,'baseline':controls[e]['score'],'best_found':best[e]['score'],
                          'relative_gain':best[e]['score']/controls[e]['score']-1,
                          'elapsed_seconds':time.perf_counter()-start}),flush=True)
    rows=[];protocols={}
    for eta,c,b in zip(etas,controls,best):
        pid=f'N{n}_eta{eta:g}'
        T=qfi_tensor(n,b['weights_rad']);Q=fixed_clifford_qfim(T,n,b['axes_indices'],b['signs'])
        b['effective_axes']=[('+' if s==1 else '-')+'XYZ'[i] for i,s in zip(b['axes_indices'],b['signs'])]
        b['edge_order']=[list(pair) for pair in layout(n)[0]]
        b['qfi_by_survivor_mask']=Q.tolist();b['eta']=eta;protocols[pid]=b
        delta=b['score']-c['score'];threshold=a.improvement_atol+a.improvement_rtol*abs(c['score'])
        rows.append({'n':n,'eta':eta,'clifford_baseline':c['score'],'best_found':b['score'],
                     'absolute_improvement':delta,'relative_improvement':delta/c['score'],
                     'updated':delta>threshold,'protocol_id':pid,'mean_QCRB_per_trial':1/(n*b['score'])})
    config={k:(v.name if isinstance(v,Path) else v) for k,v in vars(a).items()}
    sources=[ROOT/name for name in ['engine.py','run_search.py','verify.py']]
    summary={'configuration':config,'eta_grid':etas,'edge_order':[list(pair) for pair in layout(n)[0]],
             'unlabeled_graph_topologies':len(catalogue),'uniform_weight_state_evaluations':uniform_evaluations,
             'clifford_choices_per_weight_eta':6**n,'evaluated_relative_signs_per_unsigned_axis':2**(n-1),
             'local_optimization_runs':sum(r['method']=='L-BFGS-B' for r in run_rows),
             'differential_evolution_runs':sum(r['method']=='differential_evolution' for r in run_rows),
             'updated_points':sum(r['updated'] for r in rows),'maximum_relative_improvement':max(r['relative_improvement'] for r in rows),
             'started_utc':started,'finished_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.perf_counter()-start,
             'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
             'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
             'scope':'5 sensors, all ten K5 controlled-phase weights independently variable modulo 2pi, fixed local Clifford, flagged IID erasure, independent phases arithmetic mean, all trials, QFI only.',
             'optimality':'Improved feasible witnesses relative to exhaustive ordinary-graph/Clifford baseline; continuous global optimality NOT proved.',
             'mixed_state_QFI':'General spectral SLD formula; no flat-spectrum stabilizer assumption.',
             'graph_reduction':'Only vertex permutations for uniform scan (34 topologies). No weighted LC reduction.'}
    if a.uniform_cache:
        summary['uniform_cache_sha256']=hashlib.sha256(a.uniform_cache.read_bytes()).hexdigest()
        summary['uniform_cached_states']=len(catalogue)*a.uniform_points
    write_csv(out/'best_by_eta.csv',rows);write_csv(out/'optimization_runs.csv',run_rows)
    write_json(out/'protocols.json',protocols);write_json(out/'summary.json',summary)
    print(json.dumps({'stage':'finished','updated_points':summary['updated_points'],
                      'maximum_relative_improvement':summary['maximum_relative_improvement'],
                      'seconds':summary['elapsed_seconds']}),flush=True)

if __name__=='__main__':main()
