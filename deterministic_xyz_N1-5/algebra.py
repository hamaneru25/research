"""Exact unsigned stabilizer QFI support functions; no mixture optimization."""
from functools import lru_cache
import numpy as np

def rref(rows):
    pivots={}
    for v in rows:
        while v:
            p=v.bit_length()-1
            if p in pivots:v^=pivots[p]
            else:
                for q in list(pivots):
                    if (pivots[q]>>p)&1:pivots[q]^=v
                pivots[p]=v
                break
    return tuple(pivots[p] for p in sorted(pivots,reverse=True))


def measure_remove(rows, vertex, axis, total):
    if axis not in 'XYZ':raise ValueError(axis)
    p=(0 if axis=='Z' else 1<<vertex)|(0 if axis=='X' else 1<<(total+vertex))
    x=p&((1<<total)-1);z=p>>total
    anti=[i for i,g in enumerate(rows) if (((g&((1<<total)-1))&z).bit_count()+((g>>total)&x).bit_count())%2]
    rows=list(rows)
    if anti:
        pivot=rows[anti[0]]
        for j in anti[1:]:rows[j]^=pivot
        rows[anti[0]]=p
    # Trace out the now measured qubit by retaining the subgroup with I here.
    local=(1<<vertex)|(1<<(total+vertex))
    ix=next(i for i,g in enumerate(rows) if g&local)
    pivot=rows[ix]
    for j in range(len(rows)):
        if j!=ix and rows[j]&local:rows[j]^=pivot
    rows.pop(ix)
    return tuple(rows)


def sensor_stabilizers(n,m,edges,bases):
    total=n+m;adj=[0]*total
    for i,j in edges:adj[i]|=1<<j;adj[j]|=1<<i
    rows=tuple((1<<i)|(adj[i]<<total) for i in range(total))
    for j,b in enumerate(bases):rows=measure_remove(rows,n+j,b,total)
    mask=(1<<n)-1
    return rref([(v&mask)|(((v>>total)&mask)<<n) for v in rows])


@lru_cache(None)
def masks(n):
    diag=[];corr=[]
    for s in range(1<<n):
        vs=[i for i in range(n) if (s>>i)&1]
        diag.append(sum(1<<(i*n+i) for i in vs))
        corr.append(sum(1<<(i*n+j) for i in vs for j in vs if i!=j))
    return diag,corr


@lru_cache(maxsize=10000)
def pure_profile(n,rows):
    """Return maximal positive-correlation QFIM for every sensor survivor set.

    For a flat-spectrum stabilizer state, a local Pauli has QFI 1 exactly when
    it anticommutes with a retained stabilizer. Two active local Paulis have
    covariance magnitude 1 exactly when their product is in the stabilizer.
    The nontrivial pair-correlation components are disjoint GHZ blocks (Bell
    pairs can have several equivalent axis choices). Their signs are local.
    """
    size=1<<n;mask=size-1
    group=np.array([0],dtype=np.uint64)
    for g in rows:group=np.concatenate((group,group^np.uint64(g)))
    x=group&np.uint64(mask);z=group>>np.uint64(n)
    support=(x|z).astype(np.int64)
    types=np.zeros((size,n),dtype=np.uint8)
    for i in range(n):
        a=(((x>>np.uint64(i))&1)+2*((z>>np.uint64(i))&1)).astype(np.int64)
        bit=np.array([0,1,2,4],dtype=np.uint8)[a]
        np.bitwise_or.at(types[:,i],support,bit)
    # OR-zeta transform: collect all stabilizers supported within survivors S.
    for i in range(n):
        a=types.reshape(-1,2,1<<i,n)
        a[:,1]|=a[:,0]
    parent=list(range(n));axis=[0]*n
    def root(i):
        while parent[i]!=i:i=parent[i]
        return i
    for v,s in zip(group.tolist(),support.tolist()):
        if s.bit_count()!=2:continue
        i=(s&-s).bit_length()-1;j=(s^(1<<i)).bit_length()-1
        parent[root(i)]=root(j)
        for k in (i,j):
            a=((v>>k)&1)+2*((v>>(n+k))&1)
            if not axis[k]:axis[k]=(0,1,2,4)[a]
    components={}
    for i in range(n):components.setdefault(root(i),0);components[root(i)]|=1<<i
    components=[s for s in components.values() if s.bit_count()>1]
    diagonal,corr=masks(n)
    active=np.sum((types!=0).astype(np.uint16)*(1<<np.arange(n,dtype=np.uint16)),axis=1)
    out=[]
    for s in range(1,size):
        code=diagonal[int(active[s])]
        for component in components:
            block=s&component
            if block.bit_count()<2:continue
            i=(block&-block).bit_length()-1
            if int(types[s,i])&~axis[i]:code|=corr[block]
        out.append(code)
    return tuple(out)


@lru_cache(maxsize=200000)
def blocks_of(n,code):
    active=[i for i in range(n) if (code>>(i*n+i))&1]
    unseen=set(active);blocks=[]
    while unseen:
        i=min(unseen)
        block=[j for j in unseen if (code>>(i*n+j))&1]
        unseen.difference_update(block)
        blocks.append(sum(1<<j for j in block))
    return tuple(blocks)


@lru_cache(maxsize=200000)
def qmatrix(n,code):
    return np.array([(code>>i)&1 for i in range(n*n)],dtype=float).reshape(n,n)


def undominated(n,codes):
    codes=set(codes)
    if len(codes)<2:return tuple(codes)
    # Positive d support functions: every block contained in a larger block.
    keep=[]
    order=sorted(codes,key=lambda c:(c.bit_count(),c),reverse=True)
    for c in order:
        cb=blocks_of(n,c)
        if any(all(any(a&b==a for b in blocks_of(n,k)) for a in cb) for k in keep):continue
        keep.append(c)
    return tuple(sorted(keep))
