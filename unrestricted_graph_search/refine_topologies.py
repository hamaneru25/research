"""Extend the unrestricted pool using structures discovered by small search."""
from pathlib import Path
from itertools import combinations
import gzip,json
import numpy as np
from search_graphs import SEED,evaluate_pool
from stabilizer_engine import graph_profile,pure_profile


def integer_partitions(n,smallest=1):
    if n==0:
        yield ();return
    for k in range(smallest,n+1):
        for tail in integer_partitions(n-k,k):yield (k,)+tail


def grouped_stars(n,m):
    for parts in integer_partitions(n):
        hubs=[];internal=[];offset=0
        for k in parts:
            hubs.append(offset);internal.extend((offset,offset+j) for j in range(1,k));offset+=k
        g=len(hubs)
        if m==1:
            yield 'grouped_stars_one_center_'+str(parts),internal+[(h,n) for h in hubs]
        if m==g-1:
            yield 'grouped_stars_chain_'+str(parts),internal+[(hubs[j],n+j) for j in range(m)]+[(hubs[j+1],n+j) for j in range(m)]
            yield 'grouped_stars_radial_'+str(parts),internal+[(hubs[0],n+j) for j in range(m)]+[(hubs[j+1],n+j) for j in range(m)]
        if m>1:
            core=[(n+j,n+j+1) for j in range(m-1)]
            yield 'grouped_stars_central_chain_'+str(parts),internal+core+[(h,n+(j%m)) for j,h in enumerate(hubs)]


def refine(n,out,basis_limit=32):
    path=out/f'catalog_N{n}.json.gz'
    with gzip.open(path,'rt') as f:data=json.load(f)
    profiles=[tuple(tuple(cs) for cs in p) for p in data['profiles']];ids={p:i for i,p in enumerate(profiles)}
    owners={int(m):{r['profile_id']:r for r in rs} for m,rs in data['owners'].items()}
    rng=np.random.default_rng(SEED+1000+n)
    for m in range(1,n):
        if n+m<=7:continue # already exhausted all graphs, so no new topology.
        seen={tuple(tuple(e) for e in r['edges']) for r in owners[m].values()}
        candidates=list(grouped_stars(n,m))
        # Arbitrary edge edits around the discovered tree family.
        base=candidates[:]
        all_edges=list(combinations(range(n+m),2))
        for j in range(min(40,len(base))):
            source,edges=base[int(rng.integers(len(base)))];edges=set(edges)
            for _ in range(1+j%3):edges.symmetric_difference_update([all_edges[int(rng.integers(len(all_edges)))]])
            candidates.append(('mutated_discovered_tree_'+str(j),list(edges)))
        count=0
        for source,edges in candidates:
            edges=tuple(sorted(tuple(sorted(e)) for e in edges))
            if edges in seen:continue
            seen.add(edges);count+=1
            profile,audit=graph_profile(n,m,edges,None if m<=5 else basis_limit,seed=SEED)
            if profile not in ids:ids[profile]=len(profiles);profiles.append(profile)
            pid=ids[profile];old=owners[m].get(pid)
            r=dict(m=m,profile_id=pid,edges=edges,source=source,**audit)
            if old is None or (old['all_xy_settings']==audit['all_xy_settings'] and len(edges)<len(old['edges'])):owners[m][pid]=r
        data['counts']['refinement_evaluations_'+str(m)]=count
        print(json.dumps(dict(stage='refine',n=n,m=m,graphs=count,profiles=len(profiles))),flush=True)
    data.update(profiles=profiles,owners={str(m):list(v.values()) for m,v in owners.items()},refinement_seed=SEED+1000+n)
    temporary=path.with_suffix('.gz.tmp')
    with gzip.open(temporary,'wt') as f:json.dump(data,f)
    temporary.replace(path)
    return evaluate_pool(data,out,[i/100 for i in range(1,101)])


def main():
    out=Path(__file__).resolve().parent/'results'
    result=[]
    for n in range(4,11):
        result.append(refine(n,out));pure_profile.cache_clear()
    (out/'summary.json').write_text(json.dumps(dict(seed=SEED,refined=True,eta_step=.01,
        results=[{k:v for k,v in r.items() if k!='by_auxiliary_count'} for r in result]),indent=2))


if __name__=='__main__':main()
