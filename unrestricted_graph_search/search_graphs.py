"""Topology-unrestricted graph pool search; exact small-system enumeration.

Exhaustive: every graph with N+M<=7, plus the known universal M=N optimum.
Large systems: seeded graph families, arbitrary random graphs and edge edits.
All loss masks are exact; high-M central settings are sampled and documented.
"""
from pathlib import Path
from itertools import combinations
from collections import Counter
import argparse,csv,gzip,json,time
import numpy as np
import networkx as nx
from stabilizer_engine import graph_profile,common_phase_upper,optimize_profile,pure_profile,coefficients,qmatrix
from clique_reference import information

SEED=20261020


def sensor_families(n):
    yield 'empty',[]
    yield 'clique',list(combinations(range(n),2))
    yield 'star',[(0,i) for i in range(1,n)]
    yield 'path',[(i,i+1) for i in range(n-1)]
    yield 'cycle',[(i,(i+1)%n) for i in range(n)]
    yield 'balanced_bipartite',[(i,j) for i in range(n//2) for j in range(n//2,n)]
    for size in range(2,n):
        edges=[]
        for a in range(0,n,size):edges+=list(combinations(range(a,min(a+size,n)),2))
        yield f'clique_blocks_{size}',edges


def candidate_graphs(n,m,rng,random_count):
    for name,es in sensor_families(n):
        yield name+'_leaves',es+[(i,n+i) for i in range(m)]
        if m:
            yield name+'_shared_checks',es+[(i,n+j) for j in range(m) for i in range(n) if (i+j)%(m+1)!=0]
    if m:
        yield 'central_star_with_partial_buffers',[(n,i) for i in range(m-1,n)]+[(n,n+1+i) for i in range(m-1)]+[(n+1+i,i) for i in range(m-1)]
        yield 'alternating_central_path',[(n+i,n+i+1) for i in range(m-1)]+[(i,n+(i%m)) for i in range(n)]
    all_edges=list(combinations(range(n+m),2))
    for k in range(random_count):
        density=(.12,.22,.35,.5,.7)[k%5]
        edges=[e for e in all_edges if rng.random()<density]
        yield f'random_{k}',edges
    # Edge mutations have no restrictions on sensor/central connectivity.
    bases=[list(combinations(range(n),2))+[(i,n+i) for i in range(m)],
           [(i,i+1) for i in range(n+m-1)]]
    for k in range(random_count):
        edges=set(bases[k%2])
        for _ in range(1+k%5):
            e=all_edges[int(rng.integers(len(all_edges)))];edges.symmetric_difference_update([e])
        yield f'edge_mutation_{k}',list(edges)


def build_pool(n,out,random_count,basis_limit):
    rng=np.random.default_rng(SEED+n);atlas=nx.graph_atlas_g()
    profiles=[];ids={};owners={m:{} for m in range(n)};counts=Counter();seen={m:set() for m in range(n)}
    started=time.monotonic()
    def add(m,edges,source,exact):
        edges=tuple(sorted(tuple(sorted(e)) for e in edges))
        counts['generated_'+str(m)]+=1
        if edges in seen[m]:return
        seen[m].add(edges)
        profile,audit=graph_profile(n,m,edges,None if exact or m<=5 else basis_limit,seed=SEED)
        if profile not in ids:ids[profile]=len(profiles);profiles.append(profile)
        pid=ids[profile]
        record=dict(m=m,profile_id=pid,edges=edges,source=source,**audit)
        old=owners[m].get(pid)
        if old is None or (not old['all_xy_settings'] and audit['all_xy_settings']) or (old['all_xy_settings']==audit['all_xy_settings'] and len(edges)<len(old['edges'])):
            owners[m][pid]=record
        counts['evaluated_'+str(m)]+=1
    # Every unlabeled simple graph, and every sensor/central role assignment.
    for m in range(n):
        total=n+m
        if total<=7:
            for g in atlas:
                if len(g)!=total:continue
                for sensors in combinations(range(total),n):
                    order=list(sensors)+[i for i in range(total) if i not in sensors]
                    remap={v:i for i,v in enumerate(order)}
                    add(m,[(remap[a],remap[b]) for a,b in g.edges()],'exhaustive_atlas_roles',True)
        for source,edges in candidate_graphs(n,m,rng,random_count):add(m,edges,source,False)
        print(json.dumps(dict(stage='catalog',n=n,m=m,graphs=counts['evaluated_'+str(m)],
                              profiles=len(profiles),elapsed=round(time.monotonic()-started,1))),flush=True)
    payload=dict(n=n,profiles=profiles,owners={str(m):list(v.values()) for m,v in owners.items()},
                 counts=dict(counts),seed=SEED,basis_limit=basis_limit,random_count=random_count)
    path=out/f'catalog_N{n}.json.gz';temporary=path.with_suffix('.gz.tmp')
    with gzip.open(temporary,'wt') as f:json.dump(payload,f)
    temporary.replace(path)
    return payload


def evaluate_pool(data,out,etas):
    n=data['n'];profiles=[tuple(tuple(cs) for cs in p) for p in data['profiles']]
    owners={int(m):v for m,v in data['owners'].items()}
    # Upper polynomial for each tested profile under its available X/Y settings.
    upper_coeff=[]
    for profile in profiles:
        totals=np.zeros(n+1)
        for s,cs in enumerate(profile,1):totals[s.bit_count()]+=max(c.bit_count() for c in cs)
        upper_coeff.append(totals)
    upper_coeff=np.array(upper_coeff)
    results=[];by_m=[];solves=0;max_gap=0.;cache={}
    for eta in etas:
        coeff=np.array([0.]+[eta**k*(1-eta)**(n-k) for k in range(1,n+1)])
        upper=upper_coeff@coeff
        baseline_values=[information(n,m,eta)['per_photon'] for m in range(n+1)]
        global_best=max(baseline_values);best=dict(m=int(np.argmax(baseline_values)),source='complete_sensor_graph_reference',
                            edges=list(combinations(range(n),2))+[(i,n+i) for i in range(int(np.argmax(baseline_values)))],
                            per_photon=global_best,feff=global_best*(n+int(np.argmax(baseline_values))),profile_id=None)
        local=[]
        for m in range(n):
            val=baseline_values[m];record=dict(m=m,source='complete_sensor_graph_reference',
                      per_photon=val,feff=val*(n+m),profile_id=None,
                      edges=list(combinations(range(n),2))+[(i,n+i) for i in range(m)])
            independent=n*eta/(n+m)
            if independent>val:
                val=independent
                record.update(source='independent_sensors_with_leaves',per_photon=val,feff=n*eta,
                              edges=[(i,n+i) for i in range(m)])
            if val>global_best:global_best=val;best=record.copy()
            ranked=sorted(owners[m],key=lambda r:upper[r['profile_id']],reverse=True)
            pruned=0;remaining_upper=val
            universal=(n*eta+n*(n-1)*eta**2)/(n+m)
            for candidate in ranked:
                pid=candidate['profile_id']
                bound=min(universal,upper[pid]/(n+m))
                if bound<=max(val,global_best)*(1+2e-7):
                    pruned+=1;remaining_upper=max(remaining_upper,bound);continue
                key=(pid,float(eta))
                if key not in cache:
                    witness=optimize_profile(n,profiles[pid],float(eta))
                    if witness['relative_gap']>2e-5:raise ArithmeticError(('witness_gap',n,m,eta,witness['relative_gap']))
                    cache[key]=witness;solves+=1;max_gap=max(max_gap,witness['relative_gap'])
                witness=cache[key];score=witness['feff']/(n+m)
                if score>val*(1+1e-8):
                    val=score;record=dict(**candidate,per_photon=score,feff=witness['feff'],witness=witness)
                    if val>global_best*(1+1e-8):global_best=val;best=record.copy()
            record.update(n=n,eta=float(eta),pruned_profiles=pruned,
                          per_auxiliary_count_upper=float(max(val,remaining_upper)),per_auxiliary_count_exact=bool(remaining_upper<=val*(1+2e-7)))
            local.append(record)
            if val>global_best*(1+1e-8):global_best=val;best=record.copy()
        local.append(dict(n=n,m=n,eta=float(eta),per_photon=baseline_values[n],feff=baseline_values[n]*2*n,
                          source='universal_upper_bound_attained',profile_id=None,
                          edges=list(combinations(range(n),2))+[(i,n+i) for i in range(n)]))
        best.update(n=n,eta=float(eta),clique_reference=max(baseline_values),
                    independent_per_photon=float(eta),gain_over_clique_reference=global_best/max(baseline_values),
                    gain_over_best_previous=max(global_best,float(eta))/max(max(baseline_values),float(eta)),
                    best_including_independent=max(global_best,float(eta)),
                    best_method_including_independent='independent' if eta>global_best else 'graph',
                    all_graphs_globally_searched=(n==4))
        results.append(best);by_m.extend(local)
        print(json.dumps(dict(stage='optimization',n=n,eta=round(float(eta),4),m=best['m'],
                              score=round(global_best,8),gain=round(best['gain_over_clique_reference'],5),solves=solves)),flush=True)
    payload=dict(n=n,graph_counts=data['counts'],profile_count=len(profiles),solves=solves,max_witness_gap=max_gap,
                 optimal=results,by_auxiliary_count=by_m)
    (out/f'results_N{n}.json').write_text(json.dumps(payload,indent=2))
    with (out/f'comparison_N{n}.csv').open('w',newline='') as f:
        fields=['n','m','eta','feff','per_photon','source','profile_id'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(by_m)
    return payload


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--sensors',default='4,5,6,7,8,9,10')
    ap.add_argument('--etas',default=','.join(str(e/20) for e in range(1,21)))
    ap.add_argument('--random-count',type=int,default=12);ap.add_argument('--basis-limit',type=int,default=32)
    ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'results')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    all_results=[]
    for n in map(int,args.sensors.split(',')):
        path=args.output/f'catalog_N{n}.json.gz'
        if path.exists():
            with gzip.open(path,'rt') as f:data=json.load(f)
        else:data=build_pool(n,args.output,args.random_count,args.basis_limit)
        all_results.append(evaluate_pool(data,args.output,list(map(float,args.etas.split(',')))))
        pure_profile.cache_clear()
    (args.output/'summary.json').write_text(json.dumps(dict(seed=SEED,sensors=[r['n'] for r in all_results],
          random_count=args.random_count,basis_limit=args.basis_limit,
          results=[{k:v for k,v in r.items() if k!='by_auxiliary_count'} for r in all_results]),indent=2))


if __name__=='__main__':main()
