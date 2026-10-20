"""Role-aware local-Clifford catalogue and symmetry-reduced QFI upper bounds."""
from pathlib import Path
from itertools import combinations
from collections import deque,Counter,defaultdict
from functools import lru_cache
import json,gzip,itertools
import numpy as np
import networkx as nx
import pynauty
from algebra import sensor_stabilizers,pure_profile,undominated,blocks_of,qmatrix
ROOT=Path(__file__).resolve().parent

def graph_rows(graph):
    t=len(graph);adj=[0]*t
    for a,b in graph.edges():adj[a]|=1<<b;adj[b]|=1<<a
    return tuple((1<<i)|(adj[i]<<t) for i in range(t))


def incidence(graph,sensors=None):
    t=len(graph);group=[0]
    for row in graph_rows(graph):group += [v^row for v in group]
    adjacency={i:[] for i in range(4*t+len(group)-1)}
    def edge(a,b):adjacency[a].append(b);adjacency[b].append(a)
    for i in range(t):
        for a in range(3):edge(i,t+3*i+a)
    for j,row in enumerate(group[1:],4*t):
        for i in range(t):
            a=((row>>i)&1)+2*((row>>(t+i))&1)
            if a:edge(j,t+3*i+a-1)
    colors=[set(range(t)),set(range(t,4*t)),set(range(4*t,len(adjacency)))]
    if sensors is not None:
        sensors=set(sensors);colors=[sensors,set(range(t))-sensors]+colors[1:]
    return pynauty.Graph(len(adjacency),False,adjacency,[s for s in colors if s])


def automorphisms(graph,sensors=None):
    return pynauty.autgrp(incidence(graph,sensors))[0]


def image_mask(mask,perm):
    out=0
    while mask:
        bit=mask&-mask;out|=1<<perm[bit.bit_length()-1];mask^=bit
    return out


def mask_orbits(t,generators,allowed=None):
    """Orbit representative -> (image mask, vertex transporter) list."""
    unseen=set(range(1<<t) if allowed is None else allowed);result=[]
    generators=[tuple(g[:t]) for g in generators if tuple(g[:t])!=tuple(range(t))]
    while unseen:
        seed=min(unseen);unseen.remove(seed)
        queue=deque([(seed,tuple(range(t)))]);orbit=[]
        while queue:
            mask,transport=queue.popleft();orbit.append((mask,transport))
            for gen in generators:
                target=image_mask(mask,gen)
                if target in unseen:
                    unseen.remove(target)
                    queue.append((target,tuple(gen[transport[i]] for i in range(t))))
        result.append(orbit)
    return result


def role_representatives(graph):
    t=len(graph);gens=automorphisms(graph)
    return mask_orbits(t,gens)


def relabel_roles(graph,mask):
    t=len(graph);order=[i for i in range(t) if mask>>i&1]+[i for i in range(t) if not mask>>i&1]
    inverse={v:i for i,v in enumerate(order)}
    return nx.relabel_nodes(graph,inverse,copy=True)


def transform_code(n,code,perm):
    out=0
    while code:
        bit=code&-code;i,j=divmod(bit.bit_length()-1,n)
        out|=1<<(perm[i]*n+perm[j]);code^=bit
    return out


def basis_orbits(n,m,generators):
    """Basis transporters include local-axis permutations from LC symmetries."""
    t=n+m;unseen=set(range(3**m));orbits=[]
    # The incidence axes 0,1,2 denote X,Z,Y, respectively.
    actions=[]
    for gen in generators:
        a=[]
        for i in range(m):
            a.append([divmod(gen[t+3*(n+i)+axis]-t,3) for axis in range(3)])
        actions.append((tuple(gen[:n]),a))
    while unseen:
        seed=min(unseen);unseen.remove(seed)
        queue=deque([(seed,tuple(range(n)))]);orbit=[]
        while queue:
            basis,transport=queue.popleft();orbit.append((basis,transport))
            digits=[(basis//3**i)%3 for i in range(m)]
            for perm,action in actions:
                target=0
                for i,d in enumerate(digits):
                    qubit,axis=action[i][d];target+=axis*3**(qubit-n)
                if target in unseen:
                    unseen.remove(target)
                    queue.append((target,tuple(perm[transport[i]] for i in range(n))))
        orbits.append(orbit)
    return orbits


def reduced_profile(n,m,edges):
    """Only measurement/loss orbit representatives are gathered and boundd.

    pure_profile computes all losses in one fast zeta transform. Measurement
    representatives are evaluated once; equivariant transport restores every
    allowed setting. Only loss-orbit representatives enter the convex problem.
    """
    graph=nx.Graph();graph.add_nodes_from(range(n+m));graph.add_edges_from(edges)
    generators=automorphisms(graph,range(n))
    sensor_gens=[tuple(g[:n]) for g in generators]
    losses=mask_orbits(n,sensor_gens,range(1,1<<n))
    bases=basis_orbits(n,m,generators)
    candidates=[set() for _ in losses];pure_states={}
    for orbit in bases:
        basis=orbit[0][0];axes=tuple('XZY'[(basis//3**i)%3] for i in range(m))
        rows=sensor_stabilizers(n,m,edges,axes)
        if rows not in pure_states:pure_states[rows]=pure_profile(n,rows)
        profile=pure_states[rows]
        for _,perm in orbit:
            inverse=[0]*n
            for i,j in enumerate(perm):inverse[j]=i
            for k,loss in enumerate(losses):
                source=image_mask(loss[0][0],inverse)
                candidates[k].add(transform_code(n,profile[source-1],perm))
    codes=[undominated(n,c) for c in candidates]
    # Reconstruct the full labeled profile for hashing and independent checks.
    full=[None]*((1<<n)-1)
    for orbit,cs in zip(losses,codes):
        for mask,perm in orbit:full[mask-1]=tuple(sorted(transform_code(n,c,perm) for c in cs))
    return dict(n=n,m=m,profile=tuple(full),loss_orbits=losses,codes=codes,
                sensor_generators=sensor_gens,raw_measurements=3**m,
                measurement_orbits=len(bases),raw_losses=(1<<n)-1,
                distinct_sensor_states=len(pure_states),loss_orbit_count=len(losses))


def connected_graphs(vertices):
    return [nx.from_graph6_bytes(line) for line in
            (ROOT/'data'/f'vncorbits{vertices}.g6').read_bytes().splitlines() if line]

def role_catalog(max_n,max_m):
    path=ROOT/'results'/f'role_catalog_{max_n}_{max_m}.json.gz'
    if path.exists():
        with gzip.open(path,'rt') as f:return json.load(f)
    result=[]
    for total in range(1,max_n+max_m+1):
        for index,graph in enumerate(connected_graphs(total)):
            for orbit in role_representatives(graph):
                mask=orbit[0][0];n=mask.bit_count();m=total-n
                if n and n<=max_n and m<=max_m:result.append([total,index,mask,n,m])
        print(json.dumps(dict(stage='role_catalog',vertices=total,types=len(result))),flush=True)
    with gzip.open(path,'wt') as f:json.dump(result,f)
    return result


@lru_cache(None)
def source_graph(total,index):return connected_graphs(total)[index]


def candidates(n,m,catalog):
    groups=defaultdict(list)
    for record in catalog:
        if record[3]<=n and record[4]<=m:groups[record[3],record[4]].append(record)
    shapes=sorted(groups)
    def shape_partitions(a,b,start,chosen):
        if a==b==0:yield chosen;return
        for k in range(start,len(shapes)):
            x,y=shapes[k]
            if x<=a and y<=b:yield from shape_partitions(a-x,b-y,k,chosen+[shapes[k]])
    for partition in shape_partitions(n,m,0,[]):
        counts=Counter(partition)
        choices=[itertools.combinations_with_replacement(groups[shape],count) for shape,count in sorted(counts.items())]
        for combo in itertools.product(*choices):
            pieces=[record for chunk in combo for record in chunk]
            graph=nx.Graph();sensors=[];offset=0
            for total,index,mask,ns,ma in pieces:
                g=source_graph(total,index);graph.add_nodes_from(range(offset,offset+total))
                graph.add_edges_from((a+offset,b+offset) for a,b in g.edges())
                sensors += [offset+i for i in range(total) if mask>>i&1];offset+=total
            graph=relabel_roles(graph,sum(1<<i for i in sensors))
            yield sorted(tuple(sorted(e)) for e in graph.edges())
