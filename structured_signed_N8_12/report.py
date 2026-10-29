"""Aggregate immutable completed search records and render a compact report."""
import json,csv,collections,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from search import ROOT,ETAS
from engine import ghz

def main():
    out=ROOT/'results';files=[]
    for pattern in ['group_N*.json','dense_N*.json','refined_N*.json']:
        files.extend((p,json.loads(p.read_text()))for p in out.glob(pattern))
    records=[]
    for n in range(8,13):
        for e,eta in enumerate(ETAS):
            fs=[]
            for family in [1,2,3]:
                path,d=max(((p,d)for p,d in files if d['n']==n and d['family']==family),key=lambda pd:pd[1]['best'][e]['R'])
                fs.append(d['best'][e]|{'source_file':path.name})
            fs.append({'n':n,'eta':float(eta),'family':4,'R':ghz(n,eta),'state':'full_N_GHZ'})
            best=max(fs,key=lambda r:r['R']);tol=1e-9*max(1,abs(best['R']))
            ties=[r['family']for r in fs if abs(r['R']-best['R'])<=tol]
            if 4 in ties:best=fs[-1]
            records.append({'n':n,'eta':float(eta),'families':fs,'best':best,'tied_families':ties})
    (out/'best.json').write_text(json.dumps(records,indent=2)+'\n')
    with(out/'comparison.csv').open('w')as f:
        w=csv.writer(f);w.writerow(['n','eta','family1_R','family2_R','family3_R','GHZ_R','best_family','best_R'])
        for r in records:w.writerow([r['n'],r['eta']]+[x['R']for x in r['families']]+[r['best']['family'],r['best']['R']])
    coverage=collections.defaultdict(lambda:{'records':0,'profile_evaluations':0,'summed_worker_seconds':0.})
    for p,d in files:
        c=coverage[f"N{d['n']}_F{d['family']}"];c['records']+=1;c['profile_evaluations']+=d['calls'];c['summed_worker_seconds']+=d['seconds']
    summary={'scope':'best-found bounded numerical search; not exhaustive signs, continuous weights, or all local Clifford assignments','eta_points':20,'sensors':list(range(8,13)),'coverage':dict(coverage),'total_profile_evaluations':sum(d['calls']for _,d in files),'completed_records':len(files),'group_refinements':sum(p.name.startswith('refined_')for p,d in files),'sampled_full_edge_variable_counts':{str(n):n*(n-1)//2 for n in range(8,13)},'family_wins':dict(collections.Counter(r['best']['family']for r in records))}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axs=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    labels=['1: signed bipartite','2: hub clique + leaves','3: complete graph','4: full GHZ']
    for ax,n in zip(axs.flat,range(8,13)):
        rows=[r for r in records if r['n']==n]
        for j,label in enumerate(labels):ax.plot(ETAS,[r['families'][j]['R']for r in rows],label=label)
        ax.plot(ETAS,ETAS,':',color='gray',label='independent');ax.set(title=f'N = {n}',xlabel='Survival probability',ylabel='Effective QFI per generated photon R');ax.grid(alpha=.2)
    axs.flat[-1].axis('off');axs.flat[-1].legend(*axs.flat[0].get_legend_handles_labels(),loc='center',frameon=False)
    fig.savefig(out/'comparison.svg');plt.close(fig)
    text=['# N=8–12 structured weighted sensing: completed numerical search','',
          '**Best found within the recorded search budget. No global-optimality or exhaustive-sign claim.**','',
          '[All 100 conditions (CSV)](comparison.csv) · [Weights, axes and QFIMs (JSON)](best.json) · [Coverage](summary.json) · [Validation](validation.json)','',
          '![Comparison](comparison.svg)','',
          '|N|eta|Family 1 R|Family 2 R|Family 3 R|Full GHZ R|Best family|','|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:
        if not any(abs(r['eta']-x)<1e-9 for x in [.2,.5,.8,.9]):continue
        text.append('|'+ '|'.join([str(r['n']),f"{r['eta']:.2f}"]+[f"{x['R']:.9f}"for x in r['families']]+[str(r['best']['family'])])+'|')
    text.extend(['','The full results retain the source record for each selected protocol. All generated photons, including lost photons, are counted. The criterion is an SLD-QFIM bound, not the CFI of a supplied readout. Classical loss records are retained; there is no loss-dependent input or setting mixture.','',f"Total completed profile evaluations: {summary['total_profile_evaluations']}. Completed per-eta local refinements: {summary['group_refinements']}.",'','## Interpretation','',
      'Family 1 includes a systematic permutation-symmetric sign subset and additional valid general sign samples. It is not a complete enumeration of the previously counted sign isomorphism classes. Family 2 compares every hub count and every positive/negative leaf-group size. Its three edge-type weights are sampled and then locally refined; the hub-hub weight may be positive or negative. Family 3 has no maximum number of vertex groups: the dense multistart branch perturbs every edge independently. A uniform-weight clique is additionally scanned and refined. Read the root README for the finite budgets and restricted rotation catalogue.'])
    (out/'report.md').write_text('\n'.join(text)+'\n');print(json.dumps(summary))

if __name__=='__main__':main()
