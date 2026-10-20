"""Compact downloadable results and scientific figures."""
from pathlib import Path
from collections import Counter
import csv,gzip,json
import numpy as np


def main():
    root=Path(__file__).resolve().parent;out=root/'results'
    data=[json.load((out/f'results_N{n}.json').open()) for n in range(4,11)]
    rows=[];summaries=[];graphs={}
    for d in data:
        n=d['n'];peak=max(d['optimal'],key=lambda r:r['gain_over_best_previous'])
        for r in d['by_auxiliary_count']:
            if r['m']==n:
                r['per_auxiliary_count_upper']=r['per_photon'];r['per_auxiliary_count_exact']=True
        intervals=[]
        for r in d['optimal']:
            label='independent' if r['source']=='independent_sensors_with_leaves' and r['m']==0 else str(r['m'])
            if intervals and intervals[-1]['choice']==label:intervals[-1]['last_grid_eta']=r['eta']
            else:intervals.append(dict(choice=label,first_grid_eta=r['eta'],last_grid_eta=r['eta']))
            key=(n,r['m'],tuple(map(tuple,r['edges'])))
            if key not in graphs:graphs[key]=dict(id='graph_'+str(len(graphs)),n=n,m=r['m'],edges=r['edges'],source=r['source'])
            rows.append(dict(n=n,eta=r['eta'],best_auxiliary_count=r['m'],best_graph_id=graphs[key]['id'],
                 per_photon=r['per_photon'],previous_best=max(r['clique_reference'],r['eta']),
                 gain_over_previous=r['gain_over_best_previous'],source=r['source'],
                 global_topology_optimum_certified=(n==4)))
        count=d['graph_counts'];evaluations=sum(v for k,v in count.items() if k.startswith('evaluated_') or k.startswith('refinement_evaluations_') or k=='local_edge_evaluations')
        summaries.append(dict(n=n,evaluations_including_revisits=evaluations,profile_count=d['profile_count'],
             convex_optimizations=d['solves'],max_witness_gap=d['max_witness_gap'],
             peak_gain=peak['gain_over_best_previous'],peak_eta=peak['eta'],peak_m=peak['m'],
             peak_graph_edges=peak['edges'],grid_intervals=intervals,
             exhaustive_topology_ranges=[m for m in range(n) if n+m<=7],
             m_equals_n_optimum_known=True,global_optimum_certified=(n==4)))
        # Store detailed witnesses in small compressed files for transfer.
        b=json.dumps(d,separators=(',',':')).encode()
        (out/f'results_N{n}.json.gz').write_bytes(gzip.compress(b,compresslevel=9,mtime=0))
        fields=['n','m','eta','feff','per_photon','per_auxiliary_count_upper','per_auxiliary_count_exact','source','profile_id']
        with (out/f'comparison_N{n}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(d['by_auxiliary_count'])
    with (out/'best_by_eta.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'best_graphs.json').write_text(json.dumps(list(graphs.values()),indent=2))
    compact=dict(seed=20261020,eta_grid='0.01,0.02,...,1.00',
                 total_evaluations_including_revisits=sum(r['evaluations_including_revisits'] for r in summaries),
                 results=summaries)
    (out/'search_summary.json').write_text(json.dumps(compact,indent=2))
    figures(out,data,graphs)
    print(json.dumps(compact))


def figures(out,data,graphs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import networkx as nx
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,4,figsize=(15,8))
    for d,ax in zip(data,axes.flat):
        eta=np.array([r['eta'] for r in d['optimal']]);new=np.array([r['per_photon'] for r in d['optimal']]);old=np.array([max(r['clique_reference'],r['eta']) for r in d['optimal']])
        ax.plot(eta,old,'--',color='#777',lw=2,label='Previous best')
        ax.plot(eta,new,color='#176b9b',lw=2,label='Topology search')
        ax.set_title('N='+str(d['n']));ax.set_xlabel('Transmittance eta');ax.set_ylabel('Effective QFI / generated photon');ax.grid(alpha=.2);ax.set_xlim(0,1);ax.set_ylim(bottom=0)
    axes.flat[-1].axis('off');axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(),loc='center',frameon=False)
    fig.suptitle('Graph topology search: same ideal model and photon budget',fontsize=16)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for d in data:
        eta=[r['eta'] for r in d['optimal']]
        axes[0].plot(eta,[r['gain_over_best_previous'] for r in d['optimal']],label='N='+str(d['n']),lw=2)
        axes[1].step(eta,[r['m'] for r in d['optimal']],where='mid',label='N='+str(d['n']),lw=1.8)
    axes[0].set_ylabel('Gain over previous best (including independent)');axes[0].axhline(1,color='#777',ls='--');axes[0].legend(frameon=False,ncol=2)
    axes[1].set_ylabel('Best auxiliary count in searched pool');axes[1].set_yticks(range(11))
    for ax in axes:ax.set_xlim(0,1);ax.set_xlabel('Transmittance eta');ax.grid(alpha=.2)
    fig.suptitle('Additional gain from graph topology optimization',fontsize=15);fig.tight_layout();fig.savefig(out/'gain_and_auxiliary_count.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for n,m,ax in ((6,1,axes[0]),(9,2,axes[1])):
        d=data[n-4];r=max([r for r in d['optimal'] if r['m']==m],key=lambda r:r['gain_over_best_previous'])
        g=nx.Graph();g.add_nodes_from(range(n+m));g.add_edges_from(r['edges'])
        pos=nx.spring_layout(g,seed=6)
        nx.draw_networkx_edges(g,pos,ax=ax,width=2,edge_color='#707a86')
        nx.draw_networkx_nodes(g,pos,nodelist=list(range(n)),node_color='#b9dce8',edgecolors='#23566b',node_size=750,ax=ax)
        nx.draw_networkx_nodes(g,pos,nodelist=list(range(n,n+m)),node_shape='s',node_color='#f5d19d',edgecolors='#9e6724',node_size=750,ax=ax)
        labels={i:'S'+str(i+1) for i in range(n)};labels.update({n+i:'C'+str(i+1) for i in range(m)})
        nx.draw_networkx_labels(g,pos,labels=labels,font_size=11,ax=ax)
        ax.set_title(f'N={n}, M={m}: example winning tree');ax.axis('off')
    fig.tight_layout();fig.savefig(out/'example_graphs.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
