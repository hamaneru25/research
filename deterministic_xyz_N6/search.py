"""Six-sensor deterministic XYZ search. No setting-mixture variables or timing probes.

Exact role-preserving LC representatives for M<=3; finite multi-start edge
search for M=4. Every reported lower bound is a concrete deterministic
loss-dependent policy. Upper bounds are never substituted for achievements.
"""
from pathlib import Path
from itertools import combinations,product
from fractions import Fraction
from datetime import datetime,timezone
import argparse,base64,csv,gzip,hashlib,json,platform
import numpy as np
import pynauty
from algebra import sensor_stabilizers,pure_profile,qmatrix
from catalogue import role_catalog,candidates

ROOT=Path(__file__).resolve().parent
N=6
ETAS=[Fraction(i,10) for i in range(1,11)]
MASKS=np.arange(1,1<<N)
SIZES=np.array([int(s).bit_count() for s in MASKS])
WEIGHTS=np.array([[float(e)**k*(1-float(e))**(N-k) for k in SIZES] for e in ETAS])
EXPECTED={0:26,1:115,2:964,3:10946}


def now():return datetime.now(timezone.utc).isoformat()


def information_batch(matrices):
    vals,u=np.linalg.eigh(matrices)
    c=np.einsum('...ij,i->...j',u,np.ones(N)/N)
    positive=vals>1e-10
    variance=np.sum(np.divide(c*c,vals,out=np.zeros_like(vals),where=positive),axis=-1)
    identifiable=np.sum(np.where(positive,0,c*c),axis=-1)<1e-15
    return np.divide(1,variance,out=np.zeros_like(variance),where=identifiable&(variance>1e-15))


def references():
    result=[]
    for eta in ETAS:
        u=N*eta+N*(N-1)*eta**2
        rows=[('independent',0,eta),('bare_clique',0,eta*(1-(1-eta)**(N-1))+(N-1)*eta**N),
              ('endpoint',5,u/11),('endpoint_extra',6,u/12)]
        result.append(rows)
    return result


def endpoint_edges():
    return [(i,5) for i in range(5)]+[(i,6+i) for i in range(5)]+list(combinations(range(6,11),2))


def graph_key(m,edges):
    t=N+m;adj={i:[] for i in range(t)}
    for a,b in edges:adj[a].append(b);adj[b].append(a)
    colors=[set(range(N))]+([set(range(N,t))] if m else [])
    return base64.b64encode(pynauty.certificate(pynauty.Graph(t,False,adj,colors))).decode()


def graph_options(m,edges):
    """Exact scalar support maximum; feasible positive GHZ-block policies.

    An axis policy that deliberately breaks a maximal block is not included
    in this LOWER-bound optimizer. The scalar upper remains valid over ALL
    sensor Cliffords. Each retained code keeps its concrete auxiliary basis.
    """
    opts=[{} for _ in MASKS]
    for bases in product('XYZ',repeat=m):
        profile=pure_profile(N,sensor_stabilizers(N,m,edges,bases))
        for d,code in zip(opts,profile):d.setdefault(code,''.join(bases))
    ordered=[sorted(d,key=lambda c:(-c.bit_count(),c)) for d in opts]
    matrices=[np.array([qmatrix(N,c) for c in cs]) for cs in ordered]
    a=[0]*(N+1)
    for k,cs in zip(SIZES,ordered):a[int(k)]+=cs[0].bit_count()
    return opts,ordered,matrices,a


def bound_fraction(m,coeff,e):
    return sum(a*e**k*(1-e)**(N-k) for k,a in enumerate(coeff))/(N+m)


def optimize_policy(m,ordered,matrices,eta_index,passes=3,starts=2):
    weights=WEIGHTS[eta_index];best_value=-1.;best_indices=None;best_f=None
    rng=np.random.default_rng(61000+eta_index)
    for start in range(starts):
        indices=np.zeros(len(MASKS),dtype=int) if start==0 else np.array([rng.integers(len(a)) for a in matrices])
        f=sum(w*a[j] for w,a,j in zip(weights,matrices,indices))
        for _ in range(passes):
            changed=False
            # Large probability loss records first; all ties deterministic.
            for i in np.argsort(-weights,kind='stable'):
                if weights[i]==0 or len(matrices[i])==1:continue
                trial=f+weights[i]*(matrices[i]-matrices[i][indices[i]])
                values=information_batch(trial);j=int(np.argmax(values))
                previous=float(information_batch(f))
                if values[j]>previous+1e-11:
                    f=trial[j];indices[i]=j;changed=True
            if not changed:break
        value=float(information_batch(f))/(N+m)
        if value>best_value:best_value=value;best_indices=indices.copy();best_f=f.copy()
    return best_value,[ordered[i][j] for i,j in enumerate(best_indices)],best_f


class Search:
    def __init__(self,seed,passes,starts):
        self.rng=np.random.default_rng(seed);self.seed=seed;self.passes=passes;self.starts=starts
        self.records=[];self.cache={};self.best=[];self.exact_upper={m:[Fraction(0)]*10 for m in range(4)}
        self.per_m={m:[None]*10 for m in range(7)};self.local_events=[]
        for i,rows in enumerate(references()):
            family,m,value=max(rows,key=lambda r:(r[2],-r[1]))
            self.best.append(dict(eta=str(ETAS[i]),m=m,family=family,value=float(value),exact_value=str(value),source='reference'))
            for family,m,value in rows:
                old=self.per_m[m][i]
                if old is None or float(value)>old['value']:
                    self.per_m[m][i]=dict(eta=str(ETAS[i]),m=m,family=family,value=float(value),exact_value=str(value),source='reference')
        self.baseline=np.array([b['value'] for b in self.best])

    def evaluate(self,m,edges,source):
        edges=sorted(tuple(sorted(e)) for e in edges);key=(m,graph_key(m,edges))
        if key in self.cache:return self.cache[key]
        opts,ordered,matrices,a=graph_options(m,edges)
        upper_exact=[bound_fraction(m,a,e) for e in ETAS]
        initial=np.array([x[0] for x in matrices]);f=np.einsum('es,sij->eij',WEIGHTS,initial)
        values=information_batch(f)/(N+m)
        codes=[[cs[0] for cs in ordered] for _ in ETAS]
        # Optimize achieved policies only where they can improve the global
        # incumbent; other entries keep a feasible deterministic default.
        optimized=[]
        for i,upper in enumerate(upper_exact):
            if float(upper)>self.best[i]['value']+1e-10:
                value,policy,fi=optimize_policy(m,ordered,matrices,i,self.passes,self.starts)
                if value>=values[i]-1e-10:values[i]=value;codes[i]=policy;f[i]=fi
                optimized.append(i)
        ident=f'N6_M{m}_G{len(self.records):06d}'
        record=dict(id=ident,m=m,edges=edges,source=source,upper_coefficients=a,
                    deterministic_values=values.tolist(),coordinate_optimized_eta_indices=optimized)
        self.records.append(record);self.cache[key]=record
        if source=='exact_lc':
            for i,b in enumerate(upper_exact):self.exact_upper[m][i]=max(self.exact_upper[m][i],b)
        for i,value in enumerate(values):
            item=dict(eta=str(ETAS[i]),m=m,family='searched_graph',value=float(value),source=source,graph_id=ident,
                      edges=edges,loss_codes=codes[i],auxiliary_bases=[opts[j][c] for j,c in enumerate(codes[i])],
                      averaged_qfim=f[i].tolist())
            if self.per_m[m][i] is None or value>self.per_m[m][i]['value']+1e-10:self.per_m[m][i]=item
            if value>self.best[i]['value']+1e-10:
                self.best[i]=item
                print(json.dumps(dict(event='improved',eta=str(ETAS[i]),m=m,value=float(value),graph_id=ident)),flush=True)
        return record

    def seeds(self,m):
        t=N+m;all_edges=list(combinations(range(t),2));rng=self.rng
        seeds=[[],list(combinations(range(N),2))]
        # Deleting surplus auxiliaries from the loss-tolerant endpoint family.
        seeds.append([e for e in endpoint_edges() if max(e)<t])
        seeds.append([(i,5) for i in range(5)]+[(i,N+i) for i in range(m)])
        seeds.append(list(combinations(range(N),2))+[(i,N+i) for i in range(m)]+list(combinations(range(N,t),2)))
        seeds.append([(i,i+1) for i in range(t-1)]+[(0,t-1)])
        # Product endpoint blocks with total auxiliary cost m.
        sensors=list(range(N));remaining=m;aux=N;block=[]
        while sensors:
            size=min(len(sensors),remaining+1);vs=sensors[:size];sensors=sensors[size:]
            if size>1:
                block += [(vs[-1],v) for v in vs[:-1]]+[(v,aux+j) for j,v in enumerate(vs[:-1])]
                block += list(combinations(range(aux,aux+size-1),2));aux+=size-1;remaining-=size-1
        seeds.append(block)
        for density in [.15,.35,.5,.7,.85]:seeds.append([e for e in all_edges if rng.random()<density])
        return seeds

    def local(self,m,restarts,moves):
        pairs=list(combinations(range(N+m),2));seeds=self.seeds(m)
        for start in range(restarts):
            target=1+(start%7) # eta .2 through .8; extreme points use analytic bounds.
            edges=set(tuple(e) for e in seeds[start%len(seeds)])
            current=self.evaluate(m,edges,'local_seed')
            for step in range(moves):
                choices=[]
                for edge in pairs:
                    candidate=edges^{edge};r=self.evaluate(m,candidate,'edge_flip');choices.append(r)
                def score(r):return r['deterministic_values'][target]
                winner=max(choices,key=lambda r:(score(r),-len(r['edges'])))
                improved=score(winner)>score(current)+1e-10
                self.local_events.append(dict(m=m,start=start,step=step,eta=str(ETAS[target]),
                                              current=current['id'],chosen=winner['id'],improved=improved))
                if improved:current=winner;edges=set(map(tuple,winner['edges']))
                else:
                    # A two-edge perturbation, then a new complete neighborhood.
                    for j in self.rng.choice(len(pairs),2,replace=False):edges.symmetric_difference_update([pairs[j]])
                    current=self.evaluate(m,edges,'two_edge_escape')
            self.save('local_checkpoint')
            print(json.dumps(dict(event='restart_complete',m=m,start=start+1,restarts=restarts,unique_graphs=len(self.records))),flush=True)

    def save(self,status):
        rows=[]
        for i,b in enumerate(self.best):
            e=ETAS[i];u=N*e+N*(N-1)*e*e
            unresolved=[(m,u/(N+m)) for m in [4]]
            upper=max([self.exact_upper[m][i] for m in range(4)]+[v for m,v in unresolved]+[u/11,u/12])
            achieved=Fraction(b['exact_value']) if 'exact_value' in b else None
            certified=achieved is not None and upper<=achieved
            rows.append(dict(**b,global_upper=str(upper),global_upper_decimal=float(upper),
                             absolute_gap=max(0,float(upper)-b['value']),global_optimum_certified=certified))
        summary=dict(status=status,n=6,eta_grid=[str(e) for e in ETAS],seed=self.seed,
                     model=dict(auxiliary_measurements='XYZ',randomized_setting_mixtures=False,
                                loss_before_phase=True,loss_locations_known_before_auxiliary_measurement=True,
                                sensor_operations='Outcome-dependent single-qubit Clifford before phase',
                                objective='F_eff/(6+M)',mean_phase_weights='1/6',all_classical_records_retained=True),
                     exact_graph_counts={str(m):sum(r['m']==m and r['source']=='exact_lc' for r in self.records) for m in range(4)},
                     exact_upper_by_m={str(m):[str(v) for v in vs] for m,vs in self.exact_upper.items()},
                     graph_counts={str(m):sum(r['m']==m for r in self.records) for m in range(5)},
                     best_by_eta=rows,best_found_by_m={str(m):vs for m,vs in self.per_m.items()},
                     local_events=self.local_events,
                     limitations=['M=4 graphs not exhaustively enumerated',
                                  'Graph-policy coordinate search is not a complete deterministic-policy optimization',
                                  'Feasible sensor axes use maximal positive GHZ blocks; upper bounds cover all Cliffords',
                                  'No guarantee between the ten eta points'])
        (ROOT/'results/summary.json').write_text(json.dumps(summary,indent=2))
        with gzip.open(ROOT/'results/evaluated_graphs.json.gz','wt') as f:json.dump(self.records,f,separators=(',',':'))
        return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=20261020)
    p.add_argument('--restarts',type=int,default=12);p.add_argument('--moves',type=int,default=3)
    p.add_argument('--policy-passes',type=int,default=3);p.add_argument('--policy-starts',type=int,default=2)
    args=p.parse_args();(ROOT/'results').mkdir(exist_ok=True)
    source_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.glob('*.py')}
    provenance=dict(started_utc=now(),command_parameters=vars(args),source_sha256=source_hashes,
                    python=platform.python_version(),new_n6_search=True,old_performance_results_loaded=False,
                    timing_benchmark=False)
    (ROOT/'results/provenance.json').write_text(json.dumps(provenance,indent=2))
    search=Search(args.seed,args.policy_passes,args.policy_starts);catalog=role_catalog(6,3)
    for m in range(4):
        count=0
        for edges in candidates(N,m,catalog):search.evaluate(m,edges,'exact_lc');count+=1
        assert count==EXPECTED[m],(m,count)
        search.save('exact_low_m_completed')
        print(json.dumps(dict(event='exact_m_complete',m=m,catalogue_count=count,unique_graphs=len(search.records))),flush=True)
    for m in [4]:search.local(m,args.restarts,args.moves)
    summary=search.save('search_completed')
    provenance.update(finished_utc=now(),evaluated_graphs=len(search.records),completed=True)
    (ROOT/'results/provenance.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps(dict(event='completed',graph_counts=summary['graph_counts'],best=[{k:r[k] for k in ['eta','m','value','global_optimum_certified','absolute_gap']} for r in summary['best_by_eta']])),flush=True)


if __name__=='__main__':main()
