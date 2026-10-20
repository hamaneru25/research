"""All one-edge changes plus seeded two-edge changes around discovered winners."""
from pathlib import Path
from itertools import combinations
import gzip,json
import numpy as np
from search_graphs import SEED,evaluate_pool
from stabilizer_engine import graph_profile,pure_profile


def improve(n,out):
    with gzip.open(out/f'catalog_N{n}.json.gz','rt') as f:data=json.load(f)
    previous=json.load((out/f'results_N{n}.json').open())
    profiles=[tuple(tuple(cs) for cs in p) for p in data['profiles']];ids={p:i for i,p in enumerate(profiles)}
    owners={int(m):{r['profile_id']:r for r in rs} for m,rs in data['owners'].items()}
    seeds={}
    for r in previous['optimal']:
        m=r['m']
        if r['gain_over_best_previous']>1.000001 and m<n and n+m>7:
            seeds[(m,tuple(tuple(e) for e in r['edges']))]=r
    rng=np.random.default_rng(SEED+2000+n);count=0
    for (m,edges),r in seeds.items():
        known={tuple(tuple(e) for e in o['edges']) for o in owners[m].values()}
        options=list(combinations(range(n+m),2));candidates=[]
        for edge in options:
            e=set(edges);e.symmetric_difference_update([edge]);candidates.append(sorted(e))
        for _ in range(60):
            e=set(edges)
            for ix in rng.choice(len(options),2,replace=False):e.symmetric_difference_update([options[int(ix)]])
            candidates.append(sorted(e))
        for edges_new in candidates:
            key=tuple(edges_new)
            if key in known:continue
            known.add(key);count+=1
            profile,audit=graph_profile(n,m,key,None,seed=SEED)
            if profile not in ids:ids[profile]=len(profiles);profiles.append(profile)
            pid=ids[profile]
            old=owners[m].get(pid)
            if old is None or len(key)<len(old['edges']):
                owners[m][pid]=dict(m=m,profile_id=pid,edges=key,source='local_edge_search',**audit)
    data.update(profiles=profiles,owners={str(m):list(v.values()) for m,v in owners.items()})
    data['counts']['local_edge_evaluations']=count
    path=out/f'catalog_N{n}.json.gz';temporary=path.with_suffix('.gz.tmp')
    with gzip.open(temporary,'wt') as f:json.dump(data,f)
    temporary.replace(path)
    print(json.dumps(dict(stage='local_search',n=n,graphs=count,seeds=len(seeds))),flush=True)
    result=evaluate_pool(data,out,[i/100 for i in range(1,101)])
    pure_profile.cache_clear()
    return result


def main():
    out=Path(__file__).resolve().parent/'results'
    for n in range(7,11):improve(n,out)


if __name__=='__main__':main()
