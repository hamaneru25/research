"""Heatmaps of best-found topology and gain over the optimal GHZ partition.

No search or physical revalidation is repeated. The GHZ benchmark is optimized
over every integer partition by exact dynamic programming at each N/eta.
"""
import csv,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm,TwoSlopeNorm
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parent

def ghz_partition(n,eta):
    costs=[0.]+[float('inf')]*n;blocks=[[]for _ in range(n+1)]
    for k in range(1,n+1):
        s=min(range(1,k+1),key=lambda s:eta**(-s)+costs[k-s])
        costs[k]=eta**(-s)+costs[k-s];blocks[k]=sorted(blocks[k-s]+[s])
    return n/costs[n],blocks[n]

def topology(r):
    b=r['best'];n=r['n'];family=b['family']
    if family==4:return 'GHZ','GHZ'
    if family==1:
        h=b.get('spec',{}).get('h',b.get('metadata',{}).get('h'))
        a,c=sorted([h,n-h]);return 'B',f'B {a}|{c}'
    if family==2:
        h=b['spec']['h'];m=n-h
        gamma=b['parameters'][2]
        if abs(gamma)<1e-8:
            a,c=sorted([h,m]);return 'B',f'B {a}|{c}'
        if m==1:return 'C2','C2'
        return 'H',f'H {h}|{m}'
    if 'sizes'in b and len(b['sizes'])==1:return 'C1','C1'
    return 'C*','C*'

def axes_style(ax,ns,etas):
    ax.set_xticks(range(len(etas)),[f'{x:.2f}'for x in etas],fontsize=9)
    ax.set_yticks(range(len(ns)),ns,fontsize=11)
    ax.set_xlabel('Survival probability '+r'$\eta$',fontsize=12)
    ax.set_ylabel('Number of sensors N',fontsize=12)
    ax.set_xticks(np.arange(-.5,len(etas),1),minor=True);ax.set_yticks(np.arange(-.5,len(ns),1),minor=True)
    ax.grid(which='minor',color='white',linewidth=1.4);ax.tick_params(which='minor',bottom=False,left=False)
    for spine in ax.spines.values():spine.set_visible(False)

def main():
    out=ROOT/'results';records=json.loads((out/'best.json').read_text())
    ns=sorted({r['n']for r in records});etas=sorted({r['eta']for r in records})
    assert len(records)==len(ns)*len(etas)==100
    index={(r['n'],r['eta']):r for r in records}
    kinds=['B','H','C1','C2','C*','GHZ'];colors=['#4477AA','#EE9944','#228833','#66AA88','#AA4499','#8866AA']
    desc=['B a|b: signed complete bipartite K(a,b)','H h|m: hub clique + independent leaves',
          'C1: complete graph, common weight','C2: complete graph, 1 + (N-1) vertex groups',
          'C*: complete graph, independent edge weights','GHZ: full-N GHZ (ties included)']
    category=np.zeros((len(ns),len(etas)));gain=np.zeros_like(category);labels={};rows=[]
    for i,n in enumerate(ns):
        for j,eta in enumerate(etas):
            r=index[n,eta];kind,label=topology(r);category[i,j]=kinds.index(kind);labels[i,j]=label
            ref,partition=ghz_partition(n,eta);value=r['best']['R'];g=100*(value/ref-1)
            if abs(g)<1e-8:g=0.
            gain[i,j]=g
            rows.append({'n':n,'eta':eta,'topology':kind,'cell_label':label,'source_family':r['best']['family'],'R_best_found':value,'R_GHZ_partition':ref,'GHZ_partition':partition,'gain_percent':g})
    with(out/'heatmap_data.csv').open('w')as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    (out/'heatmap_data.json').write_text(json.dumps(rows,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(17,5.4));fig.subplots_adjust(left=.065,right=.99,top=.83,bottom=.29)
    ax.imshow(category,cmap=ListedColormap(colors),norm=BoundaryNorm(np.arange(-.5,len(kinds)),len(kinds)),aspect='auto',origin='lower',interpolation='nearest')
    axes_style(ax,ns,etas)
    for (i,j),label in labels.items():ax.text(j,i,label,ha='center',va='center',fontsize=9,color='white',fontweight='bold')
    fig.suptitle('Best-found graph topology: N = 8–12',fontsize=18,fontweight='bold',y=.97)
    fig.text(.065,.87,'Finite numerical search; 100 evaluated conditions. Cell boundaries are not optimized transition boundaries.',fontsize=10,color='#444444')
    present=set(row['topology']for row in rows)
    handles=[Patch(facecolor=c,label=d)for k,c,d in zip(kinds,colors,desc)if k in present]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.015),ncol=2,frameon=False,fontsize=10)
    for ext in ['svg','png']:fig.savefig(out/f'topology_heatmap.{ext}',dpi=160)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(17,5.2));fig.subplots_adjust(left=.065,right=.915,top=.80,bottom=.19)
    vmax=max(5,5*math.ceil(float(gain.max())/5));vmin=min(0,5*math.floor(float(gain.min())/5))
    kwargs={'cmap':'YlGnBu','vmin':0,'vmax':vmax}if vmin==0 else{'cmap':'RdBu_r','norm':TwoSlopeNorm(vmin=vmin,vcenter=0,vmax=vmax)}
    im=ax.imshow(gain,aspect='auto',origin='lower',interpolation='nearest',**kwargs)
    axes_style(ax,ns,etas)
    for i in range(len(ns)):
        for j in range(len(etas)):
            rgba=im.cmap(im.norm(gain[i,j]));luma=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
            ax.text(j,i,f'{gain[i,j]:.1f}',ha='center',va='center',fontsize=9,color='white'if luma<.49 else'#17212B')
    cax=fig.add_axes([.93,.19,.014,.61]);fig.colorbar(im,cax=cax,label='Gain over optimal GHZ partition (%)')
    fig.suptitle('Gain over the optimal GHZ partition',fontsize=18,fontweight='bold',y=.97)
    fig.text(.065,.86,r'$100\,[R_{\mathrm{best\ found}}/R_{\mathrm{GHZ\ partition}}-1]$; all integer partitions are compared.',fontsize=12)
    fig.text(.065,.045,'Same independent unknown phases, flagged loss and generated-photon normalization. Values are percentages; no new search was run.',fontsize=10,color='#444444')
    for ext in ['svg','png']:fig.savefig(out/f'ghz_partition_gain_heatmap.{ext}',dpi=160)
    plt.close(fig)
    print(json.dumps({'cells':len(rows),'min_gain_percent':float(gain.min()),'max_gain_percent':float(gain.max()),'best_gain_cell':max(rows,key=lambda r:r['gain_percent']),'topologies':{k:sum(r['topology']==k for r in rows)for k in kinds}}))

if __name__=='__main__':main()
