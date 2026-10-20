"""Optimize retained leaves on a complete sensor graph, including photon cost.

Independent unknown phases; target is their mean. Classical records of loss,
central outcomes and randomized settings are retained. Phase encoding follows
loss and local Clifford preparation. See README.md for scope and derivation.
"""
from pathlib import Path
import argparse
import csv
import gzip
import io
import json
import numpy as np
from scipy.optimize import brentq, minimize_scalar


def information(n, m, eta):
    if not (n >= 3 and 0 <= m <= n and 0 <= eta <= 1):
        raise ValueError('Need N>=3, 0<=M<=N and 0<=eta<=1')
    if eta == 0:
        return dict(feff=0., per_photon=0., cross=0., cross_max=0., mode='zero')
    q = n-m
    if q == 0:
        feff = n*eta*(1+(n-1)*eta)
        return dict(feff=feff, per_photon=feff/(2*n), cross=0., cross_max=0., mode='all_protected')
    # Normalize the QFIM by eta for stability near zero transmission.
    du = 1. if q == 1 else (-np.expm1((n-1)*np.log1p(-eta)) if eta < 1 else 1.)
    if m == 0:
        feff = n*eta*(du+(n-1)*eta**(n-1))
        return dict(feff=float(feff), per_photon=float(feff/n), cross=0., cross_max=0., mode='no_aux')
    ah = 1+(m-1)*eta**(q+1)
    au = du+(q-1)*eta**(q-1)
    cmax = eta**q
    limit = min(ah/m, au/q)
    cross = min(cmax, limit)
    if cmax >= limit:
        feff = n*n*eta*limit
        mode = 'sign_randomization' if cross < cmax else 'all_plus'
    else:
        det = ah*au-m*q*cross**2
        denom = m*au+q*ah-2*m*q*cross
        feff = n*n*eta*det/denom
        mode = 'all_plus'
    return dict(feff=float(feff), per_photon=float(feff/(n+m)),
                cross=float(eta*cross), cross_max=float(eta*cmax), mode=mode)


def matrix(n, m, eta):
    info = information(n,m,eta)
    if eta == 0:
        return np.zeros((n,n))
    q = n-m
    if q == 0:
        return eta*(1-eta)*np.eye(n)+eta**2*np.ones((n,n))
    du = eta if q == 1 else eta*(-np.expm1((n-1)*np.log1p(-eta)) if eta < 1 else 1.)
    f = np.zeros((n,n))
    f[:m,:m] = eta**(q+2)
    f[m:,m:] = eta**q
    f[:m,m:] = info['cross']
    f[m:,:m] = info['cross']
    np.fill_diagonal(f, [eta]*m+[du]*q)
    return f


def scores(n, eta):
    return np.array([information(n,m,float(eta))['per_photon'] for m in range(n+1)])


def best(n, eta):
    values = scores(n,eta)
    vmax = float(values.max())
    tol = 1e-11*max(vmax,1e-300)
    ties = np.flatnonzero(vmax-values <= tol).tolist()
    return dict(best_aux=ties[0], tied_aux=ties, best_per_photon=vmax,
                best_with_independent='independent' if eta > vmax+tol else str(ties[0]),
                best_including_independent=max(float(eta),vmax))


def optimal_intervals(n, include_independent=False):
    """Locate every pairwise crossing, then form the upper envelope.

    Roots are bracketed on 20,001 points and refined with Brent's method.
    Repeated/irrelevant roots are removed when adjacent winners are merged.
    Values are normalized by eta; eta=0 is reported separately as a full tie.
    """
    xs = np.linspace(1e-8,1.,20001)
    curves = np.array([scores(n,e)/e for e in xs]).T
    labels = list(range(n+1))
    if include_independent:
        curves = np.vstack((curves,np.ones(len(xs))))
        labels.append('independent')
    def value(k,e):
        return 1. if labels[k] == 'independent' else information(n,labels[k],e)['per_photon']/e
    roots = []
    for i in range(len(labels)):
        for j in range(i+1,len(labels)):
            delta = curves[i]-curves[j]
            for k in np.flatnonzero(delta[:-1]*delta[1:] < 0):
                root = brentq(lambda e:value(i,e)-value(j,e),xs[k],xs[k+1],xtol=1e-14)
                if 1e-9 < root < 1-1e-9:
                    roots.append(root)
    ends = [0.]
    for r in sorted(roots):
        if r-ends[-1] > 1e-9:ends.append(float(r))
    ends.append(1.)
    intervals = []
    for a,b in zip(ends[:-1],ends[1:]):
        e=(a+b)/2
        win=labels[int(np.argmax([value(k,e) for k in range(len(labels))]))]
        if intervals and intervals[-1]['winner'] == win:
            intervals[-1]['eta_max']=b
        else:
            intervals.append(dict(eta_min=a,eta_max=b,winner=win))
    transitions=[]
    for left,right in zip(intervals[:-1],intervals[1:]):
        transitions.append(dict(eta=left['eta_max'], tied_winners=[left['winner'],right['winner']]))
    return dict(intervals=intervals,transitions=transitions)


def make_figures(out, ns):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    eta=np.linspace(0,1,1001)
    cmap=plt.get_cmap('viridis',11)
    fig,axes=plt.subplots(2,4,figsize=(17,8.3),sharex=True)
    for n,ax in zip(ns,axes.flat):
        vals=np.array([scores(n,e) for e in eta])
        for m in range(n+1):
            ax.plot(eta,vals[:,m],color=cmap(m),lw=1.3,label=f'M={m}')
        ax.plot(eta,vals.max(axis=1),color='#111111',lw=2.3,label='Optimal envelope')
        ax.plot(eta,eta,'--',color='#d14c42',lw=1.8,label='Independent')
        ax.set_title(f'N={n} sensors');ax.set_xlim(0,1);ax.set_ylim(bottom=0);ax.grid(alpha=.18)
        ax.set_xlabel('Transmittance eta');ax.set_ylabel('Effective QFI / generated photon')
    axes.flat[-1].axis('off')
    axes.flat[-1].legend(*axes.flat[-2].get_legend_handles_labels(),loc='center',ncol=2,frameon=False)
    fig.suptitle('Complete sensor graph + M retained leaves: all M=0,...,N',fontsize=17)
    fig.tight_layout(rect=(0,0,1,.95))
    for ext in ('png','svg'):fig.savefig(out/f'comparison.{ext}',dpi=170)
    plt.close(fig)
    colors=[cmap(m) for m in range(11)]
    fig,ax=plt.subplots(figsize=(12,4.8))
    data=np.array([[best(n,float(e))['best_aux'] for e in eta[1:]] for n in ns])
    im=ax.imshow(data,origin='lower',aspect='auto',extent=(0,1,ns[0]-.5,ns[-1]+.5),
                 cmap=ListedColormap(colors),norm=BoundaryNorm(np.arange(-.5,11.5),11),interpolation='nearest')
    ax.set_yticks(ns);ax.set_xlabel('Transmittance eta');ax.set_ylabel('Sensor count N')
    ax.set_title('Optimal auxiliary count M (fixed graph family; photon cost included)')
    fig.colorbar(im,ax=ax,ticks=range(11),label='Optimal M')
    fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(out/f'optimal_auxiliary_count.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    for n in ns:
        grid=eta[1:]
        vals=np.array([scores(n,e) for e in grid])
        base=np.maximum(grid,vals[:,0])
        axes[0].plot(grid,vals.max(axis=1)/base,lw=2,label=f'N={n}')
        wins=np.argmax(vals,axis=1)
        axes[1].step(grid,wins,where='mid',lw=1.8,label=f'N={n}')
    axes[0].axhline(1,color='#777',ls='--');axes[0].set_ylabel('Gain over max(M=0, independent)')
    axes[1].set_ylabel('Optimal auxiliary count M');axes[1].set_yticks(range(11))
    for ax in axes:ax.set_xlabel('Transmittance eta');ax.set_xlim(0,1);ax.grid(alpha=.2)
    axes[0].legend(frameon=False,ncol=2)
    fig.suptitle('Photon-normalized optimum and resource choice',fontsize=15)
    fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(out/f'gain_and_optimum.{ext}',dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'results')
    parser.add_argument('--grid-step',type=float,default=.001)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    ns=list(range(4,11));grid=np.linspace(0,1,round(1/args.grid_step)+1)
    rows=[];optimal=[];summary=[]
    for n in ns:
        for e in grid:
            e=float(e);opt=best(n,e);base=max(e,information(n,0,e)['per_photon'])
            optimal.append(dict(sensors=n,eta=e,**{k:v for k,v in opt.items() if k!='tied_aux'},
                                tied_aux=';'.join(map(str,opt['tied_aux'])),
                                gain_over_best_no_aux=opt['best_per_photon']/base if base else 1.))
            for m in range(n+1):
                info=information(n,m,e)
                rows.append(dict(sensors=n,auxiliaries=m,generated_photons=n+m,eta=e,**info))
        fixed=optimal_intervals(n);with_ind=optimal_intervals(n,True)
        xs=np.linspace(1e-6,1,20001)
        def gain(e):
            values=scores(n,e)
            return float(values.max()/max(e,values[0]))
        gs=np.array([gain(e) for e in xs]);k=int(np.argmax(gs))
        peak=minimize_scalar(lambda e:-gain(e),bounds=(xs[max(0,k-1)],xs[min(len(xs)-1,k+1)]),method='bounded',options={'xatol':1e-13})
        summary.append(dict(sensors=n,fixed_family=fixed,including_independent=with_ind,
                            peak_gain_over_best_no_aux=gain(float(peak.x)),peak_eta=float(peak.x),
                            peak_aux=best(n,float(peak.x))['best_aux'],
                            samples=[dict(eta=e,**best(n,e),values=scores(n,e).tolist()) for e in (.1,.2,.3,.4,.5,.6,.7,.8,.9,1.)]))
        print(json.dumps({k:v for k,v in summary[-1].items() if k!='samples'}),flush=True)
    for filename,data in [('comparison.csv',rows),('optimal_by_eta.csv',optimal)]:
        with (out/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    # Per-sensor gzip files keep downloadable data small.
    for n in ns:
        buf=io.StringIO(newline='')
        writer=csv.DictWriter(buf,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(r for r in rows if r['sensors']==n)
        (out/f'comparison_N{n}.csv.gz').write_bytes(gzip.compress(buf.getvalue().encode(),compresslevel=9,mtime=0))
    (out/'summary.json').write_text(json.dumps(dict(grid_step=float(grid[1]-grid[0]),sensor_counts=ns,
          graph_conditions=sum(n+1 for n in ns),curve_rows=len(rows),optimal_rows=len(optimal),results=summary),indent=2))
    make_figures(out,ns)


if __name__=='__main__':main()
