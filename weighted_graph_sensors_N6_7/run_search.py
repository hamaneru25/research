"""N=6,7 weighted-graph search with fixed local Clifford, recorded IID loss.

All N(N-1)/2 weights are independent continuous variables. All 6**N local
Pauli axes are compared at every objective evaluation. No setting mixtures.
Continuous global optimality is not claimed. Parallel workers are processes,
each with one BLAS thread. Per-graph/per-eta results are resumable.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import csv,hashlib,json,platform,time
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import minimize,differential_evolution
from engine import layout,qfi_tensor,best_clifford,fixed_clifford_qfim,WeightObjective,loss_weights

ROOT=Path(__file__).resolve().parent


def write_json(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');temp.replace(path)


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def record(n,weights,value,axes,signs,origin):
    return {'n':n,'weights_rad':np.mod(weights,2*np.pi).tolist(),'score':float(value),
            'axes_indices':np.asarray(axes).tolist(),'signs':np.asarray(signs).tolist(),'origin':origin}


def evaluate(n,x,eta):
    T=qfi_tensor(n,x);avg=np.einsum('s,sij->ij',loss_weights(n,[eta])[0],T)
    value,axes,signs=best_clifford(avg,n)
    return record(n,x,value,axes,signs,'evaluated')


def partitions(n,minimum=1):
    if n==0:yield ()
    for k in range(minimum,n+1):
        for rest in partitions(n-k,k):yield (k,)+rest


def control(n,eta):
    # Every GHZ partition is feasible using disjoint pi-weighted stars.
    # Earlier exhaustive ordinary-graph searches certify this baseline on
    # the published grid; validation rechecks those data independently.
    part=max(partitions(n),key=lambda p:n/sum(eta**(-k) for k in p))
    x=np.zeros(n*(n-1)//2);pairs=layout(n)[0];offset=0
    for k in part:
        for j in range(offset+1,offset+k):x[pairs.index((offset,j))]=np.pi
        offset+=k
    ans=evaluate(n,x,eta);ans['origin']='GHZ_partition_control';ans['partition']=list(part)
    expected=n/sum(eta**(-k) for k in part)
    if abs(ans['score']-expected)>1e-8:raise ArithmeticError('GHZ control mismatch')
    return ans


def graph_worker(job):
    n,gid,edges,anchors,points=job;pairs=layout(n)[0]
    present=np.array([list(e) in edges for e in pairs],float)
    probs=loss_weights(n,anchors);best=[None]*len(anchors)
    for phi in np.linspace(0,np.pi,points):
        x=present*phi;T=qfi_tensor(n,x)
        avgs=np.einsum('es,sij->eij',probs,T)
        for i,eta in enumerate(anchors):
            value,axes,signs=best_clifford(avgs[i],n)
            if best[i] is None or value>best[i]['score']+1e-12:
                best[i]=record(n,x,value,axes,signs,'uniform_topology')
                best[i].update({'eta':eta,'graph_id':gid,'uniform_phi_rad':float(phi)})
    return {'n':n,'graph_id':gid,'candidates':best}


def optimize_worker(job):
    n,eta,top,args=job;start=time.perf_counter();rng=np.random.default_rng(args['seed']+1000*n+round(eta*10000))
    baseline=control(n,eta);best=dict(baseline);m=n*(n-1)//2
    seeds=[('uniform_topology',np.array(r['weights_rad'])) for r in top]
    seeds.append(('GHZ_partition',np.array(baseline['weights_rad'])))
    # Embed the N5 best-found input with independent additional sensors.
    old=json.loads((ROOT.parent/'weighted_graph_sensors_N5/results/protocols.json').read_text())
    key=min(old,key=lambda k:abs(old[k]['eta']-eta));proto=old[key]
    x=np.zeros(m)
    for edge,weight in zip(proto['edge_order'],proto['weights_rad']):x[layout(n)[0].index(tuple(edge))]=weight
    seeds.append(('embedded_N5',x))
    # Opposite signed hub groups generalize the observed N5 K2,3 witness.
    for hubs in range(1,n//2+1):
        for phi in [eta,2/3,1.,2.35]:
            x=np.zeros(m)
            for j,(u,v) in enumerate(layout(n)[0]):
                if u<hubs<=v:x[j]=phi*(1 if u<(hubs+1)//2 else -1)
            seeds.append((f'signed_K{hubs}_{n-hubs}',x))
    for j in range(args['perturbed_starts']):
        center=np.array(top[j%len(top)]['weights_rad'])
        seeds.append(('perturbed',center+rng.normal(0,.08 if j%2==0 else .6,m)))
    for _ in range(args['random_starts']):seeds.append(('random',rng.uniform(0,2*np.pi,m)))
    runs=[]
    options={'maxiter':args['maxiter'],'gtol':args['gtol'],'ftol':args['ftol'],'maxls':40}
    for si,(kind,x0) in enumerate(seeds):
        initial=evaluate(n,x0,eta)
        if initial['score']>best['score']+1e-11:best={**initial,'origin':kind+'_initial'}
        obj=WeightObjective(n,eta,args['ridge'])
        sol=minimize(obj,x0,jac=True,method='L-BFGS-B',options=options)
        candidate=evaluate(n,sol.x,eta)
        if candidate['score']>best['score']+1e-11:best={**candidate,'origin':kind}
        runs.append({'n':n,'eta':eta,'method':'L-BFGS-B','seed_index':si,'origin':kind,
                     'score':candidate['score'],'nfev':sol.nfev,'iterations':sol.nit,
                     'success':bool(sol.success),'message':str(sol.message)})
    if eta in args['de_etas']:
        def f(x):return -evaluate(n,x,eta)['score']
        de=differential_evolution(f,[(0,2*np.pi)]*m,maxiter=args['de_maxiter'],popsize=args['de_popsize'],
            seed=int(rng.integers(2**31)),polish=False,tol=1e-7,atol=1e-10,x0=best['weights_rad'])
        sol=minimize(WeightObjective(n,eta,args['ridge']),de.x,jac=True,method='L-BFGS-B',options=options)
        for candidate in [evaluate(n,de.x,eta),evaluate(n,sol.x,eta)]:
            if candidate['score']>best['score']+1e-11:best={**candidate,'origin':'differential_evolution'}
        runs.append({'n':n,'eta':eta,'method':'differential_evolution','seed_index':-1,'origin':'population',
                     'score':-de.fun,'nfev':de.nfev+sol.nfev,'iterations':de.nit,
                     'success':bool(de.success),'message':str(de.message)})
    best['eta']=eta
    return {'n':n,'eta':eta,'baseline':baseline,'best':best,'runs':runs,'elapsed_seconds':time.perf_counter()-start}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sensors',default='6,7')
    p.add_argument('--etas',default=','.join(str(k/20) for k in range(1,21)))
    p.add_argument('--coarse-etas',default='0.2,0.5,0.7,0.85')
    p.add_argument('--uniform-points',type=int,default=33,help='0..pi inclusive; complex conjugation covers pi..2pi')
    p.add_argument('--topology-starts',type=int,default=6)
    p.add_argument('--random-starts',type=int,default=64)
    p.add_argument('--perturbed-starts',type=int,default=16)
    p.add_argument('--seed',type=int,default=20261023)
    p.add_argument('--maxiter',type=int,default=400)
    p.add_argument('--gtol',type=float,default=1e-8)
    p.add_argument('--ftol',type=float,default=1e-12)
    p.add_argument('--ridge',type=float,default=1e-10)
    p.add_argument('--de-etas',default='0.5,0.6,0.65,0.7,0.75,0.8')
    p.add_argument('--de-maxiter',type=int,default=100)
    p.add_argument('--de-popsize',type=int,default=6)
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--continuation-passes',type=int,default=2,help='Alternating forward/backward eta warm starts after multistart')
    p.add_argument('--output',type=Path,default=ROOT/'results')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();start=time.perf_counter();started=datetime.now(timezone.utc).isoformat()
    ns=sorted(set(map(int,a.sensors.split(','))));etas=sorted(set(map(float,a.etas.split(','))))
    anchors=sorted(set(map(float,a.coarse_etas.split(','))));de_etas=sorted(set(float(x) for x in a.de_etas.split(',') if x.strip()))
    if not ns or any(n not in [6,7] for n in ns):raise ValueError('Supported sensors: 6,7')
    if not etas or not anchors or any(not np.isfinite(x) or not 0<x<=1 for x in etas+anchors):raise ValueError('eta must be in (0,1]')
    if not set(de_etas)<=set(etas):raise ValueError('DE eta points must be on the final grid')
    if min(a.workers,a.maxiter,a.uniform_points,a.topology_starts,a.de_maxiter,a.de_popsize)<1:raise ValueError('Positive counts required')
    if min(a.random_starts,a.perturbed_starts,a.continuation_passes)<0:raise ValueError('Nonnegative counts required')
    if any(not np.isfinite(x) or x<=0 for x in [a.gtol,a.ftol,a.ridge]):raise ValueError('Positive tolerances required')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items() if k not in ['resume','output']}
    config.update({'sensors':ns,'etas':etas,'coarse_etas':anchors,'de_etas':de_etas})
    config_path=out/'configuration.json'
    if config_path.exists():
        if not a.resume:raise FileExistsError('Existing results: use --resume or another --output')
        if json.loads(config_path.read_text())!=config:raise ValueError('Resume configuration mismatch')
    else:write_json(config_path,config)
    cache=out/'cache';cache.mkdir(exist_ok=True)
    atlas=json.loads((ROOT/'data/topologies.json').read_text())['graphs'];coarse=[];jobs=[]
    for n in ns:
        for gid,edges in enumerate(atlas[str(n)]):
            path=cache/f'coarse_N{n}_g{gid}.json'
            if path.exists():coarse.append(json.loads(path.read_text()))
            else:jobs.append((n,gid,edges,anchors,a.uniform_points))
    cached_coarse_graphs=len(coarse)
    print(json.dumps({'stage':'coarse_start','cached':len(coarse),'pending_graphs':len(jobs),'workers':a.workers}),flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(graph_worker,j):j for j in jobs}
        for future in as_completed(futures):
            result=future.result();coarse.append(result)
            write_json(cache/f"coarse_N{result['n']}_g{result['graph_id']}.json",result)
            if len(coarse)%25==0 or len(coarse)==sum(len(atlas[str(n)]) for n in ns):
                print(json.dumps({'stage':'coarse_progress','graphs_completed':len(coarse),'elapsed_seconds':time.perf_counter()-start}),flush=True)
    coarse.sort(key=lambda r:(r['n'],r['graph_id']))
    write_json(out/'uniform_scan_seeds.json',coarse)
    jobs=[];results=[];args={**vars(a),'de_etas':de_etas}
    for n in ns:
        for eta in etas:
            path=cache/f'optimized_N{n}_eta{eta:g}.json'
            if path.exists():results.append(json.loads(path.read_text()));continue
            nearest=sorted(anchors,key=lambda e:abs(e-eta))[:2]
            tops=[]
            for anchor in nearest:
                candidates=[r for g in coarse if g['n']==n for r in g['candidates'] if r['eta']==anchor]
                tops.extend(sorted(candidates,key=lambda r:-r['score'])[:a.topology_starts])
            # Re-score the union of topologies at the actual eta, then choose
            # the top distinct seed weights; no performance interpolation.
            unique={tuple(np.round(r['weights_rad'],10)):r for r in tops}
            actual=[{**evaluate(n,r['weights_rad'],eta),'origin':'coarse_rescored'} for r in unique.values()]
            top=sorted(actual,key=lambda r:-r['score'])[:a.topology_starts]
            jobs.append((n,eta,top,args))
    print(json.dumps({'stage':'optimization_start','cached':len(results),'pending_cases':len(jobs)}),flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(optimize_worker,j):j for j in jobs}
        for future in as_completed(futures):
            result=future.result();results.append(result)
            write_json(cache/f"optimized_N{result['n']}_eta{result['eta']:g}.json",result)
            print(json.dumps({'stage':'eta_completed','n':result['n'],'eta':result['eta'],
                'baseline':result['baseline']['score'],'best_found':result['best']['score'],
                'gain':result['best']['score']/result['baseline']['score']-1,'completed':len(results),
                'elapsed_seconds':time.perf_counter()-start}),flush=True)
    results.sort(key=lambda r:(r['n'],r['eta']))
    # Recompute cached incumbents with the current unregularized kernel.
    # This also makes resumptions insensitive to small score-roundoff changes.
    for r in results:
        candidate=evaluate(r['n'],r['best']['weights_rad'],r['eta'])
        candidate.update({'eta':r['eta'],'origin':r['best']['origin']})
        baseline=control(r['n'],r['eta'])
        r['baseline']=baseline
        r['best']=candidate if candidate['score']>=baseline['score'] else {**baseline,'eta':r['eta']}
    # Serial eta continuation is cheap and preserves the multistart winner.
    for sweep in range(a.continuation_passes):
        for n in ns:
            selected=[r for r in results if r['n']==n]
            if sweep%2:selected=selected[::-1]
            for previous,current in zip(selected,selected[1:]):
                eta=current['eta'];obj=WeightObjective(n,eta,a.ridge)
                sol=minimize(obj,previous['best']['weights_rad'],jac=True,method='L-BFGS-B',
                    options={'maxiter':a.maxiter,'gtol':a.gtol,'ftol':a.ftol,'maxls':40})
                candidate=evaluate(n,sol.x,eta)
                if candidate['score']>current['best']['score']+1e-11:
                    current['best']={**candidate,'eta':eta,'origin':f'continuation_pass_{sweep+1}'}
                current['runs'].append({'n':n,'eta':eta,'method':'L-BFGS-B','seed_index':-1,
                    'origin':f'continuation_pass_{sweep+1}','score':candidate['score'],'nfev':sol.nfev,
                    'iterations':sol.nit,'success':bool(sol.success),'message':str(sol.message)})
        print(json.dumps({'stage':'continuation_completed','pass':sweep+1,'elapsed_seconds':time.perf_counter()-start}),flush=True)
    rows=[];protocols={};profiles={};run_rows=[]
    for r in results:
        n,eta,b,c=r['n'],r['eta'],r['best'],r['baseline'];pid=f'N{n}_eta{eta:g}'
        T=qfi_tensor(n,b['weights_rad']);Q=fixed_clifford_qfim(T,n,b['axes_indices'],b['signs'])
        counts=np.array([s.bit_count() for s in range(1<<n)])
        b['effective_axes']=[('+' if s==1 else '-')+'XYZ'[i] for i,s in zip(b['axes_indices'],b['signs'])]
        b['edge_order']=[list(e) for e in layout(n)[0]];b['eta']=eta
        b['qfim_by_survivor_count']=[Q[counts==k].sum(axis=0).tolist() for k in range(n+1)]
        b['average_QFIM']=np.einsum('s,sij->ij',loss_weights(n,[eta])[0],Q).tolist()
        protocols[pid]=b;profiles[pid]=Q;run_rows.extend(r['runs'])
        delta=b['score']-c['score']
        rows.append({'n':n,'eta':eta,'clifford_baseline':c['score'],'best_found':b['score'],
            'absolute_improvement':delta,'relative_improvement':delta/c['score'],
            'updated':delta>1e-7+1e-7*c['score'],'protocol_id':pid,'mean_QCRB_per_trial':1/(n*b['score'])})
    write_csv(out/'best_by_n_eta.csv',rows);write_csv(out/'optimization_runs.csv',run_rows)
    write_json(out/'protocols.json',protocols);np.savez_compressed(out/'qfim_profiles.npz',**profiles)
    sources=[ROOT/name for name in ['engine.py','run_search.py','verify.py']]
    summary={'configuration':config,'started_utc':started,'finished_utc':datetime.now(timezone.utc).isoformat(),
        'elapsed_seconds_this_invocation':time.perf_counter()-start,
        'elapsed_seconds_since_first_configuration':time.time()-config_path.stat().st_mtime,
        'first_configuration_utc':datetime.fromtimestamp(config_path.stat().st_mtime,timezone.utc).isoformat(),
        'cached_coarse_graphs_at_start':cached_coarse_graphs,
        'coarse_graphs':{str(n):len(atlas[str(n)]) for n in ns},
        'coarse_state_evaluations':sum(len(atlas[str(n)]) for n in ns)*a.uniform_points,
        'local_optimization_runs':sum(r['method']=='L-BFGS-B' for r in run_rows),
        'differential_evolution_runs':sum(r['method']=='differential_evolution' for r in run_rows),
        'updated_points':sum(r['updated'] for r in rows),'maximum_relative_improvement':max(r['relative_improvement'] for r in rows),
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources if p.exists()},
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
        'scope':'Independent edge weights; fixed local Clifford before IID flagged erasure and independent phase encoding; arithmetic mean; all trials; QFI bound only.',
        'optimality':'Best found by continuous heuristic search; global weighted optimum NOT certified.',
        'coarse_search':'All unlabeled graph shapes, uniform weight at four anchor survival probabilities by default; 0..pi covers conjugate 2pi-phi inputs with transformed Clifford axes.',
        'graph_reduction':'Only vertex permutations in coarse graph atlas. No weighted LC equivalence or stabilizer assumption.',
        'clifford_candidates':{str(n):{'physical_axes':6**n,'relative_sign_representatives':6**n//2} for n in ns}}
    write_json(out/'summary.json',summary)
    print(json.dumps({'stage':'finished','updated_points':summary['updated_points'],'maximum_gain':summary['maximum_relative_improvement'],'seconds':summary['elapsed_seconds_this_invocation']}),flush=True)


if __name__=='__main__':main()
