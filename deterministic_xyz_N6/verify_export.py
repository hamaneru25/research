"""Independently realize and verify ALL exported best-found deterministic policies.

This runs after search.py. Graph states, partial traces, measurement outcomes,
physical Clifford gates, and SLD QFIMs are computed as dense complex matrices.
"""
from itertools import product,combinations
from fractions import Fraction as F
from pathlib import Path
from datetime import datetime,timezone
import csv,gzip,hashlib,json
import numpy as np
from search import N,ETAS,ROOT,endpoint_edges,bound_fraction,information_batch,graph_key
from algebra import qmatrix,sensor_stabilizers,pure_profile
from record_io import read_graph_records
from dense_reference import I,X,Y,Z,H,tensor,graph_state,conditional_sensor_states,local_generators,qufim,effective_information


def exact_information(matrix):
    """Solve F x=w over rational numbers, including singular identifiable F."""
    a=[list(row)+[F(1,N)] for row in matrix];pivots=[];r=0
    for col in range(N):
        j=next((i for i in range(r,N) if a[i][col]),None)
        if j is None:continue
        a[r],a[j]=a[j],a[r];d=a[r][col];a[r]=[v/d for v in a[r]]
        for i in range(N):
            if i!=r and a[i][col]:
                v=a[i][col];a[i]=[x-v*y for x,y in zip(a[i],a[r])]
        pivots.append(col);r+=1
    if any(not any(row[:N]) and row[N] for row in a):return F(0)
    x=[F(0)]*N
    for i,col in enumerate(pivots):x[col]=a[i][N]
    variance=sum(x)/N
    return 1/variance if variance else F(0)


def exact_average(codes,eta):
    matrix=[[F(0) for _ in range(N)] for _ in range(N)]
    for mask,code in enumerate(codes,1):
        k=mask.bit_count();p=eta**k*(1-eta)**(N-k)
        for i in range(N):
            for j in range(N):
                if code>>(i*N+j)&1:matrix[i][j]+=p
    return matrix


def reference_policy(family):
    m=0 if family in ['independent','bare_clique'] else (5 if family=='endpoint' else 6)
    edges=[] if family=='independent' else list(combinations(range(N),2)) if family=='bare_clique' else endpoint_edges()
    codes=[];bases=[]
    for mask in range(1,1<<N):
        live=[i for i in range(N) if mask>>i&1];k=len(live)
        if family in ['endpoint','endpoint_extra'] or family=='bare_clique' and k==N:
            code=sum(1<<(i*N+j) for i in live for j in live)
        elif family=='independent' or family=='bare_clique' and k>=2:code=sum(1<<(i*N+i) for i in live)
        else:code=0
        if m:
            r=sum(mask>>i&1 for i in range(5))
            if mask>>5&1:axis='X' if (5-r)%2 else 'Y';basis=''.join('Z' if mask>>i&1 else axis for i in range(5))
            else:axis='X' if r%2 else 'Y';basis=''.join(axis if mask>>i&1 else 'Z' for i in range(5))
            if m==6:basis+='Z'
        else:basis=''
        codes.append(code);bases.append(basis)
    return dict(m=m,edges=edges,loss_codes=codes,auxiliary_bases=bases,family=family)


def gate(axis,sign):
    phase=np.diag([1,1j])
    return {('Z',1):('I',I),('Z',-1):('X',X),('X',1):('H',H),('X',-1):('HZ',H@Z),
            ('Y',1):('HSdg',H@phase.conj().T),('Y',-1):('HS',H@phase)}[axis,int(sign)]


def find_axes(rho,target):
    k=len(target);axes=list(product('XYZ',repeat=k))
    all_generators=[tensor([{'X':X,'Y':Y,'Z':Z}[axis]/2 if j==i else I for j in range(k)]) for i in range(k) for axis in 'XYZ']
    full=qufim(rho,all_generators)
    ix=np.array([[3*i+'XYZ'.index(a) for i,a in enumerate(row)] for row in axes])
    matrices=full[ix[:,:,None],ix[:,None,:]]
    matching=np.where(np.max(np.abs(np.abs(matrices)-target),axis=(1,2))<1e-8)[0]
    if not len(matching):raise ArithmeticError('No physical local-axis realization of retained block code')
    return axes[int(matching[0])]


def signs_for(raw,target):
    k=len(raw);signs=np.zeros(k,dtype=int)
    for seed in range(k):
        if signs[seed]:continue
        signs[seed]=1;queue=[seed]
        while queue:
            i=queue.pop()
            for j in range(k):
                if i!=j and target[i,j] and not signs[j]:
                    signs[j]=signs[i]*(1 if raw[i,j]>0 else -1);queue.append(j)
    assert np.max(np.abs(raw*np.outer(signs,signs)-target))<1e-8
    return signs


def main():
    summary=json.loads((ROOT/'results/summary.json').read_text());assert summary['status']=='search_completed'
    assert summary['exact_graph_counts']=={'0':26,'1':115,'2':964,'3':10946}
    records=read_graph_records(ROOT);by_id={r['id']:r for r in records}
    provenance=json.loads((ROOT/'results/provenance.json').read_text())
    assert len(summary['local_runs'])==7*provenance['command_parameters']['restarts']
    from collections import Counter
    for run in summary['local_runs']:
        target=ETAS.index(F(run['eta']));terminal=by_id[run['terminal_graph']]
        assert run['termination_reason']=='no_improving_one_edge_neighbor'
        assert abs(terminal['deterministic_values'][target]-run['terminal_value'])<1e-10
        assert len(run['neighbor_ids'])==45
        edges=set(map(tuple,terminal['edges']))
        expected=Counter(graph_key(4,edges^{e}) for e in combinations(range(10),2))
        actual=Counter(graph_key(4,by_id[i]['edges']) for i in run['neighbor_ids'])
        assert expected==actual
        for ident in run['neighbor_ids']:
            r=by_id[ident];assert target in r['coordinate_optimized_eta_indices']
            assert r['deterministic_values'][target]<=run['terminal_value']+run['tolerance']
    unique={};policy_list=[];loss_cache={};dense_cases=0;maximum_error=0.;average_error=0.
    csv_rows=[]
    # All 70 best-found-by-M entries and the global winners get reproducible policies.
    rows=[r for rs in summary['best_found_by_m'].values() for r in rs]+summary['best_by_eta']
    for row in rows:
        p=reference_policy(row['family']) if row['source']=='reference' else {k:row[k] for k in ['m','edges','loss_codes','auxiliary_bases','family']}
        signature=json.dumps(p,sort_keys=True,separators=(',',':'));ident=hashlib.sha256(signature.encode()).hexdigest()[:16]
        row['policy_id']=ident
        if ident not in unique:
            psi=graph_state(N+p['m'],p['edges']);loss_ids=[]
            graph_signature=json.dumps([p['m'],p['edges']],sort_keys=True)
            for mask,(code,bases) in enumerate(zip(p['loss_codes'],p['auxiliary_bases']),1):
                key=hashlib.sha256(json.dumps([graph_signature,mask,code,bases]).encode()).hexdigest()[:20]
                loss_ids.append(key)
                if key in loss_cache:continue
                live=[i for i in range(N) if mask>>i&1];target=qmatrix(N,code)[np.ix_(live,live)]
                branches=conditional_sensor_states(psi,N,p['m'],live,bases)
                axes=find_axes(branches[0][2],target);outcomes=[];probability=F(0)
                for outcomes_sign,pb,rho in branches:
                    raw=qufim(rho,local_generators(len(live),axes));signs=signs_for(raw,target)
                    gates=[gate(a,s) for a,s in zip(axes,signs)];u=tensor([g[1] for g in gates])
                    actual=qufim(u@rho@u.conj().T,local_generators(len(live),tuple('Z'*len(live))))
                    err=float(np.max(np.abs(actual-target)));assert err<1e-8
                    maximum_error=max(maximum_error,err);dense_cases+=1
                    pr=F(pb).limit_denominator(2**p['m']);probability+=pr
                    outcomes.append(dict(auxiliary_outcomes=list(outcomes_sign),probability=str(pr),
                                         sensor_axes=list(axes),axis_signs=signs.tolist(),sensor_cliffords=[g[0] for g in gates]))
                assert probability==1
                loss_cache[key]=dict(survivor_mask=mask,survivors=live,auxiliary_bases=bases,
                                     conditional_qfim=target.astype(int).tolist(),outcomes=outcomes)
            p.update(id=ident,loss_record_ids=loss_ids,empty_survivor_action='Record trial with zero information; measure auxiliaries Z')
            policy_list.append(p);unique[ident]=p
            print(json.dumps(dict(event='policy_verified',id=ident,m=p['m'],dense_outcomes=dense_cases)),flush=True)
        eta=F(row['eta']);policy=unique[ident];matrix=exact_average(policy['loss_codes'],eta)
        value=exact_information(matrix)/(N+policy['m'])
        err=abs(float(value)-row['value']);assert err<1e-8,(row,value)
        average_error=max(average_error,err);row['exact_value']=str(value)
    # Global upper includes every unsearched M=4 graph via U_N/(N+4).
    for row in summary['best_by_eta']:
        eta=F(row['eta']);u=N*eta+N*(N-1)*eta*eta
        upper=max([F(summary['exact_upper_by_m'][str(m)][ETAS.index(eta)]) for m in range(4)]+[u/10,u/11,u/12])
        assert str(upper)==row['global_upper']
        row['global_optimum_certified']=F(row['exact_value'])==upper
        row['absolute_gap']=float(upper-F(row['exact_value']))
        row['relative_gap_to_upper']=float((upper-F(row['exact_value']))/upper)
        csv_rows.append({k:row[k] for k in ['eta','m','family','value','exact_value','policy_id','global_upper','absolute_gap','global_optimum_certified']})
    with gzip.open(ROOT/'results/deterministic_policies.json.gz','wt') as f:json.dump(dict(policies=policy_list,loss_records=loss_cache),f,separators=(',',':'))
    (ROOT/'results/summary.json').write_text(json.dumps(summary,indent=2))
    for filename,data in [('best_by_eta.csv',csv_rows),('best_found_by_m.csv',
            [{k:r[k] for k in ['eta','m','family','value','exact_value','policy_id']} for rs in summary['best_found_by_m'].values() for r in rs])]:
        with (ROOT/'results'/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    fields=['m','start','eta','accepted_moves','neighborhood_checks','terminal_graph','terminal_value',
            'maximum_neighbor_value','tolerance','termination_reason']
    with (ROOT/'results/local_convergence.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(summary['local_runs'])
    for r in records:
        for i,e in enumerate(ETAS):assert r['deterministic_values'][i]<=float(bound_fraction(r['m'],r['upper_coefficients'],e))+1e-8
    sources=json.loads((ROOT/'data/sources.json').read_text())
    for s in sources:assert hashlib.sha256((ROOT/'data'/f"vncorbits{s['vertices']}.g6").read_bytes()).hexdigest()==s['sha256']
    result=dict(verified_at_utc=datetime.now(timezone.utc).isoformat(),unique_policies=len(policy_list),
                verifier_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                record_reader_sha256=hashlib.sha256((ROOT/'record_io.py').read_bytes()).hexdigest(),
                unique_loss_basis_records=len(loss_cache),dense_outcome_cases=dense_cases,max_dense_qfim_error=maximum_error,
                exact_averaged_policy_rows=len(rows),max_achieved_value_error=average_error,
                evaluated_graph_upper_checks=len(records)*10,source_sha256_checks=len(sources),
                converged_graph_walks=len(summary['local_runs']),terminal_neighbors_checked=45*len(summary['local_runs']),
                certified_eta=[r['eta'] for r in summary['best_by_eta'] if r['global_optimum_certified']],
                all_exported_best_policies_dense_verified=True,no_setting_probability_optimization=True,
                no_timing_benchmark=True)
    (ROOT/'results/validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
