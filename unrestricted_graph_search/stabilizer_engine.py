"""Exact Pauli-QFIM profiles for arbitrary graph states with heralded erasure.

Unsigned stabilizers suffice: measurement outcomes change stabilizer signs,
which can be absorbed in the allowed outcome-dependent local Pauli axes.
Central measurements are X/Y, chosen after the loss pattern is known.
"""
from functools import lru_cache
from itertools import product
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
    p=(1<<vertex)|((1<<(total+vertex)) if axis=='Y' else 0)
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


def measurement_bases(m,limit=None,seed=0):
    if limit is None or (1<<m)<=limit:return list(product('XY',repeat=m))
    rng=np.random.default_rng(seed)
    choices={0,(1<<m)-1}
    # Include a single X in Y and a single Y in X before random samples.
    for i in range(m):choices.add(1<<i);choices.add(((1<<m)-1)^(1<<i))
    while len(choices)<limit:choices.add(int(rng.integers(1<<m)))
    return [tuple('Y' if (v>>i)&1 else 'X' for i in range(m)) for v in sorted(choices)]


def graph_profile(n,m,edges,limit=None,seed=0):
    bases=measurement_bases(m,limit,seed)
    seen={};profiles=[]
    for basis in bases:
        rows=sensor_stabilizers(n,m,edges,basis)
        if rows not in seen:
            p=pure_profile(n,rows);seen[rows]=p;profiles.append(p)
    profile=tuple(undominated(n,vals) for vals in zip(*profiles))
    return profile,dict(measurement_settings=len(bases),distinct_unsigned_sensor_states=len(seen),
                        all_xy_settings=(len(bases)==1<<m))


def coefficients(n,eta):
    return np.array([eta**(s.bit_count()-1)*(1-eta)**(n-s.bit_count()) for s in range(1,1<<n)])


def common_phase_upper(n,profile,eta):
    p=coefficients(n,eta)
    scores=np.array([max(c.bit_count() for c in cs) for cs in profile])
    return float(eta*(p@scores))


def optimize_profile(n,profile,eta,tolerance=2e-6):
    """Convex dual plus a feasible QFIM witness, with a reported bound gap.

    F_eff*=min_{d>=0,sum d=N} sum_S p_S max_Q d'Qd. Restricting d>=0 is
    exact here: arbitrary local signs make the support function depend on |d|.
    At zero coordinates of d, independent recorded sign flips recover a
    primal QFIM with Fd constant; their means are also saved in the witness.
    """
    import cvxpy as cp
    from graph_sensing_core import effective_information
    if eta==0:return dict(feff=0.,upper=0.,relative_gap=0.,qfim=np.zeros((n,n)).tolist())
    p=coefficients(n,eta);base=np.zeros((n,n));amb=[]
    for weight,cs in zip(p,profile):
        if weight==0:continue
        if len(cs)==1:base+=weight*qmatrix(n,cs[0])
        else:amb.append((weight,cs))
    if not amb:
        return optimize_fixed_matrix(base,eta)
    d=cp.Variable(n);constraints=[d>=0,cp.sum(d)==n]
    objective=cp.quad_form(d,cp.psd_wrap(base));records=[]
    for weight,cs in amb:
        t=cp.Variable();entries=[]
        for c in cs:
            con=cp.quad_form(d,cp.psd_wrap(qmatrix(n,c)))<=t
            constraints.append(con);entries.append((c,con))
        objective+=weight*t;records.append((weight,entries))
    problem=cp.Problem(cp.Minimize(objective),constraints)
    problem.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_gap_rel=1e-9,tol_feas=1e-9,max_iter=300)
    if problem.status not in ('optimal','optimal_inaccurate'):raise RuntimeError(problem.status)
    direction=np.maximum(d.value,0);direction*=n/direction.sum()
    upper=float(eta*sum(w*max(direction@qmatrix(n,c)@direction for c in cs) for w,cs in zip(p,profile)))
    f=base.copy();mix=[]
    for weight,entries in records:
        weights=np.maximum([float(np.asarray(con.dual_value).item()) for c,con in entries],0)
        weights/=weights.sum()
        for (c,con),a in zip(entries,weights):f+=weight*a*qmatrix(n,c)
        mix.append(weights.tolist())
    # Independent local sign flips on dual-zero sensors repair KKT slack.
    fd=f@direction;target=float(direction@f@direction/n)
    flips=np.ones(n)
    for i in range(n):
        if direction[i]<1e-4 and fd[i]>target:flips[i]=target/fd[i]
    adjusted=flips[:,None]*f*flips[None,:]
    np.fill_diagonal(adjusted,np.diag(f))
    feasible=eta*effective_information(adjusted)
    gap=max(0.,upper-feasible)/max(upper,1e-15)
    return dict(feff=float(feasible),upper=upper,relative_gap=float(gap),status=problem.status,
                qfim=(eta*adjusted).tolist(),dual_direction=direction.tolist(),
                independent_sign_means=flips.tolist(),ambiguous_loss_groups=len(amb),
                mixture_weights=mix)


def optimize_fixed_matrix(f,eta):
    """Exact c-optimal recorded sign randomization of one positive QFIM."""
    from scipy.optimize import minimize
    from graph_sensing_core import effective_information
    n=len(f);scale=max(1.,float(np.linalg.norm(f,ord=2)))
    if np.max(np.ptp(f.sum(axis=1)))<1e-11:
        direction=np.ones(n)
    else:
        result=minimize(lambda d:float(d@f@d/scale),np.ones(n),jac=lambda d:2*f@d/scale,
                        bounds=[(0,None)]*n,constraints=[dict(type='eq',fun=lambda d:d.sum()-n,jac=lambda d:np.ones(n))],
                        method='SLSQP',options={'ftol':1e-13,'maxiter':300})
        direction=np.maximum(result.x,0);direction*=n/direction.sum()
    upper=float(eta*(direction@f@direction));fd=f@direction
    target=float(direction@f@direction/n);flips=np.ones(n)
    for i in range(n):
        if direction[i]<1e-5 and fd[i]>target:flips[i]=target/fd[i]
    adjusted=flips[:,None]*f*flips[None,:];np.fill_diagonal(adjusted,np.diag(f))
    value=eta*effective_information(adjusted)
    return dict(feff=float(value),upper=upper,relative_gap=max(0.,upper-value)/max(upper,1e-15),
                status='quadratic_program',qfim=(eta*adjusted).tolist(),dual_direction=direction.tolist(),
                independent_sign_means=flips.tolist(),ambiguous_loss_groups=0,mixture_weights=[])
