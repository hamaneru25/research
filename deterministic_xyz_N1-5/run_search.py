"""No-randomization, XYZ, all-role-graphs certificate for N=1..5.

No setting-selection probabilities or numerical optimization of such
probabilities appear in this program. Exact rational upper bounds certify
the deterministic optima at eta=1/10,...,1. Measurement/Clifford gate
tables are constructed and independently checked by dense-state QFIMs.
"""
from pathlib import Path
from fractions import Fraction
from itertools import combinations
from collections import defaultdict
import csv,gzip,json,hashlib,platform
import numpy as np
from catalogue import ROOT,role_catalog,candidates,reduced_profile
from dense_reference import I,X,Y,Z,H,tensor,graph_state,conditional_sensor_states,local_generators,qufim,effective_information

ETAS=[Fraction(k,10) for k in range(1,11)]
EXPECTED={2:{0:2},3:{0:3,1:5},4:{0:6,1:13,2:54},5:{0:11,1:36,2:200,3:1354}}


def endpoint_edges(n):
    k=n-1
    return [(i,k) for i in range(k)]+[(i,n+i) for i in range(k)]+list(combinations(range(n,2*n-1),2))


def endpoint_bases(n,survivors):
    k=n-1;r=sum(survivors>>i&1 for i in range(k))
    if survivors>>k&1:
        lost=k-r;axis='X' if lost%2 else 'Y'
        return tuple('Z' if survivors>>i&1 else axis for i in range(k))
    axis='X' if r%2 else 'Y'
    return tuple(axis if survivors>>i&1 else 'Z' for i in range(k))


def references(n,eta):
    universal=n*eta+n*(n-1)*eta**2
    rows=[dict(family='independent',m=0,value=eta)]
    if n>1:
        rows += [dict(family='bare_clique',m=0,value=eta*(1-(1-eta)**(n-1))+(n-1)*eta**n),
                 dict(family='endpoint',m=n-1,value=universal/(2*n-1))]
    return rows


def coefficients(n,profile):
    result=[0]*(n+1)
    for survivors,cs in enumerate(profile,1):result[survivors.bit_count()]+=max(c.bit_count() for c in cs)
    return result


def upper(n,m,coeff,eta):
    return sum(a*eta**k*(1-eta)**(n-k) for k,a in enumerate(coeff))/(n+m)


def family_graph(n,family):
    if family=='independent':return 0,[]
    if family=='bare_clique':return 0,list(combinations(range(n),2))
    return n-1,endpoint_edges(n)


def chosen_axes(n,family,survivors):
    live=[i for i in range(n) if survivors>>i&1];k=len(live)
    if family=='independent':return tuple('Z' for _ in live)
    if family=='bare_clique':return tuple(('Y' if k==n else 'X' if k>=2 else 'Z') for _ in live)
    if survivors>>(n-1)&1:return tuple('Z' if i==n-1 else 'X' for i in live)
    return tuple('Y' if k%2 else 'Z' for _ in live)


def target_qfim(n,family,survivors):
    k=survivors.bit_count()
    if family=='independent':return np.eye(k)
    if family=='endpoint' or k==n:return np.ones((k,k))
    return np.eye(k) if k>=2 else np.zeros((k,k))


def clifford(axis,sign):
    phase=np.diag([1,1j])
    table={('Z',1):('I',I),('Z',-1):('X',X),
           ('X',1):('H',H),('X',-1):('HZ',H@Z),
           ('Y',1):('HSdg',H@phase.conj().T),('Y',-1):('HS',H@phase)}
    name,matrix=table[axis,int(sign)]
    assert np.allclose(matrix.conj().T@Z@matrix,sign*{'X':X,'Y':Y,'Z':Z}[axis])
    return name,matrix


def generate_policies():
    policies=[];outcome_cases=0;max_error=0.;average_errors=[]
    for n in range(1,6):
        families=['independent'] if n==1 else ['independent','bare_clique','endpoint']
        for family in families:
            m,edges=family_graph(n,family);psi=graph_state(n+m,edges);loss_records=[]
            for mask in range(1,1<<n):
                live=[i for i in range(n) if mask>>i&1];k=len(live)
                bases=endpoint_bases(n,mask) if family=='endpoint' else ()
                axes=chosen_axes(n,family,mask);target=target_qfim(n,family,mask);outcomes=[];prob_sum=Fraction(0)
                for result,prob,rho in conditional_sensor_states(psi,n,m,live,bases):
                    raw=qufim(rho,local_generators(k,axes));signs=np.ones(k,dtype=int)
                    if np.count_nonzero(target-np.diag(np.diag(target))):
                        signs=np.array([1]+[1 if raw[0,j]>=0 else -1 for j in range(1,k)])
                    gates=[clifford(a,s) for a,s in zip(axes,signs)]
                    unitary=tensor([matrix for name,matrix in gates]);corrected=unitary@rho@unitary.conj().T
                    checked=qufim(corrected,local_generators(k,tuple('Z'*k)))
                    error=float(np.max(np.abs(checked-target)));max_error=max(max_error,error)
                    assert error<1e-8,(n,family,mask,bases,result,error)
                    probability=Fraction(prob).limit_denominator(2**m);prob_sum+=probability
                    outcomes.append(dict(auxiliary_outcomes=list(result),probability=str(probability),
                                         sensor_axes=list(axes),axis_signs=signs.tolist(),
                                         sensor_cliffords=[name for name,matrix in gates]))
                    outcome_cases+=1
                assert prob_sum==1,(n,family,mask,prob_sum)
                loss_records.append(dict(survivor_mask=mask,survivors=live,auxiliary_bases=list(bases),
                                         conditional_qfim=target.astype(int).tolist(),outcomes=outcomes))
            # Empty survivor set is recorded explicitly, with zero information.
            loss_records.insert(0,dict(survivor_mask=0,survivors=[],auxiliary_bases=['Z']*m,
                                       conditional_qfim=[],outcomes=[],action='No sensing photons remain; record trial, information=0'))
            policy=dict(id=f'N{n}_{family}',n=n,m=m,family=family,edges=edges,loss_records=loss_records)
            policies.append(policy)
            for eta_fraction in ETAS:
                eta=float(eta_fraction);f=np.zeros((n,n))
                for loss in loss_records[1:]:
                    live=loss['survivors'];k=len(live);weight=eta**k*(1-eta)**(n-k)
                    f[np.ix_(live,live)]+=weight*np.array(loss['conditional_qfim'])
                actual=effective_information(f)/(n+m)
                expected=float(next(r['value'] for r in references(n,eta_fraction) if r['family']==family))
                error=abs(actual-expected);average_errors.append(error)
                assert error<1e-8,(n,family,eta_fraction,actual,expected)
    with gzip.open(ROOT/'results/policies.json.gz','wt') as f:json.dump(policies,f,separators=(',',':'))
    fields=['n','m','family','survivors','auxiliary_bases','auxiliary_outcomes','outcome_probability','sensor_cliffords']
    for n in range(1,6):
        with (ROOT/'results'/f'policy_N{n}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for policy in policies:
                if policy['n']!=n:continue
                for loss in policy['loss_records']:
                    for outcome in loss['outcomes']:
                        writer.writerow(dict(n=n,m=policy['m'],family=policy['family'],
                                             survivors=' '.join(map(str,loss['survivors'])),
                                             auxiliary_bases=''.join(loss['auxiliary_bases']),
                                             auxiliary_outcomes=' '.join(map(str,outcome['auxiliary_outcomes'])),
                                             outcome_probability=outcome['probability'],
                                             sensor_cliffords=' '.join(outcome['sensor_cliffords'])))
    return dict(policy_count=len(policies),dense_outcome_cases=outcome_cases,max_dense_qfim_error=max_error,
                averaged_qfim_checks=len(average_errors),max_average_information_error=max(average_errors))


def run():
    (ROOT/'results').mkdir(exist_ok=True);(ROOT/'figures').mkdir(exist_ok=True)
    catalog=role_catalog(5,3);records=[];stats=[];answers=[];certificate=[]
    for n in range(1,6):
        nrecords=[]
        for m in range(max(0,n-1)):
            seen_curves=set();count=0;loss_orbits=0;basis_orbits=0
            for index,edges in enumerate(candidates(n,m,catalog)):
                data=reduced_profile(n,m,edges);a=coefficients(n,data['profile'])
                record=dict(n=n,m=m,id=f'N{n}_M{m}_{index}',edges=edges,upper_coefficients=a)
                records.append(record);nrecords.append(record);seen_curves.add(tuple(a));count+=1
                loss_orbits+=data['loss_orbit_count'];basis_orbits+=data['measurement_orbits']
            assert count==EXPECTED[n][m],(n,m,count)
            stats.append(dict(n=n,m=m,graph_representatives=count,distinct_upper_curves=len(seen_curves),
                              raw_nonempty_survivor_sets=(1<<n)-1,raw_measurement_settings=3**m,
                              mean_loss_orbits=loss_orbits/count,mean_measurement_orbits=basis_orbits/count))
        for eta in ETAS:
            refs=references(n,eta);refs.sort(key=lambda r:(-r['value'],r['m'],['independent','bare_clique','endpoint'].index(r['family'])))
            chosen=refs[0];best=chosen['value'];checked_bounds=[]
            for record in nrecords:
                bound=upper(n,record['m'],record['upper_coefficients'],eta)
                assert bound<=best,(n,eta,record['id'],bound,best)
                checked_bounds.append(bound)
            # M=N-1 and M=N are bounded analytically, and attain their bounds.
            endpoint_ms=range(max(0,n-1),n+1)
            universal=n*eta+n*(n-1)*eta**2
            for m in endpoint_ms:
                bound=universal/(n+m);assert bound<=best;checked_bounds.append(bound)
            largest=max(checked_bounds);assert largest==best,(n,eta,largest,best)
            family=chosen['family'];m,edges=family_graph(n,family)
            answers.append(dict(n=n,eta=str(eta),eta_decimal=float(eta),m=m,method=family,
                                per_photon=str(best),per_photon_decimal=float(best),feff=str(best*(n+m)),
                                policy_id=f'N{n}_{family}',edges=edges,
                                known_tied_references=[dict(family=r['family'],m=r['m']) for r in refs if r['value']==best],
                                global_optimum_certified=True))
            certificate.append(dict(n=n,eta=str(eta),feasible_per_photon=str(best),largest_graph_upper=str(largest),
                                    exact_gap='0',enumerated_graphs=len(nrecords),analytical_auxiliary_counts=list(endpoint_ms),
                                    setting_probability_optimization=False))
        print(json.dumps(dict(n=n,enumerated_graphs=len(nrecords),eta_points=10,all_grid_optima_certified=True)),flush=True)
    with gzip.open(ROOT/'results/candidate_bounds.json.gz','wt') as f:json.dump(records,f,separators=(',',':'))
    output=dict(model=dict(sensors=[1,2,3,4,5],auxiliary_measurements='XYZ',eta_grid=[str(e) for e in ETAS],
                           randomized_setting_mixtures=False,loss_before_phase=True,loss_locations_known_before_auxiliary_measurement=True,
                           sensor_local_operations='Outcome-dependent single-qubit Cliffords before phase',objective='F_eff/(N+M)',
                           mean_phase_weights='1/N',all_classical_records_retained=True),
                graph_counts=stats,optimal=answers,certificate=certificate)
    (ROOT/'results/summary.json').write_text(json.dumps(output,indent=2))
    fields=['n','eta','m','method','per_photon','per_photon_decimal','feff','policy_id','global_optimum_certified']
    with (ROOT/'results/best_by_eta.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(answers)
    with (ROOT/'results/graph_counts.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(stats[0]));writer.writeheader();writer.writerows(stats)
    validation=generate_policies()
    validation.update(exact_certificate_rows=len(certificate),total_enumerated_graphs=len(records),
                      all_upper_comparisons_exact_rational=True,setting_probability_solver_used=False,
                      no_timing_benchmark=True,python=platform.python_version())
    (ROOT/'results/validation.json').write_text(json.dumps(validation,indent=2))
    print(json.dumps(validation),flush=True)
    return output


if __name__=='__main__':run()
