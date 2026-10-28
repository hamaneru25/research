"""Plot absolute effective QFI, with full-range and high-survival panels.

Read saved CSVs only; no optimization, interpolation or smoothing is performed.
R is effective QFI per generated photon, and F_eff=N*R is per trial.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parent


def read_results(out):
    path = out / 'best_by_eta.csv'
    if not path.exists():
        path = out / 'best_by_n_eta.csv'
    with path.open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    return {n: sorted((r for r in rows if int(r['n']) == n), key=lambda r: float(r['eta']))
            for n in sorted({int(r['n']) for r in rows})}


def draw(datasets, target, total=False, zoom=(.5, .9), preview=None):
    ns = sorted(datasets)
    with matplotlib.rc_context({'svg.hashsalt': 'weighted-absolute-information', 'font.size': 10}):
        fig, axs = plt.subplots(2, len(ns), figsize=(5*len(ns), 8.3), squeeze=False)
        for col, n in enumerate(ns):
            rows = datasets[n]
            eta = np.array([float(r['eta']) for r in rows])
            base = np.array([float(r['clifford_baseline']) for r in rows])
            best = np.array([float(r['best_found']) for r in rows])
            assert len(set(eta)) == len(eta)
            assert np.all(np.isfinite(best)) and np.all(best >= base-1e-9)
            factor = n if total else 1
            for row in range(2):
                ax = axs[row, col]
                ax.plot(eta, factor*eta, ':', color='#6b7280', lw=1.8, zorder=1)
                ax.plot(eta, factor*base, 'o--', color='#2769a3', lw=1.6,
                        markersize=5, markerfacecolor='white', zorder=2)
                ax.plot(eta, factor*best, 's-', color='#d76422', lw=1.8,
                        markersize=3.8, zorder=3)
                ax.set_title(f'{n} sensors' + (' | high-survival detail' if row else ' | full range'))
                ax.set_xlabel('Single-photon survival probability, eta')
                ax.set_ylabel('Effective QFI per trial, F_eff' if total else 'Effective QFI per generated photon, R')
                ax.grid(alpha=.22)
                if row:
                    ax.set_xlim(*zoom)
                    selected = (eta >= zoom[0]) & (eta <= zoom[1])
                    if not selected.any():
                        raise ValueError('zoom interval contains no saved eta')
                    ax.set_ylim(0, 1.08*max((factor*best[selected]).max(), (factor*base[selected]).max()))
                else:
                    ax.set_xlim(0, 1)
                    ax.set_ylim(bottom=0)
        fig.suptitle('Weighted graph sensing: absolute information\n'
                     + ('F_eff = N R = 1 / (w^T Qbar^+ w)' if total else 'R = F_eff / N = 1 / (N w^T Qbar^+ w)'),
                     fontsize=14, y=.985)
        legend = [Line2D([], [], color='#6b7280', ls=':', label='Independent sensors'),
                  Line2D([], [], color='#2769a3', ls='--', marker='o', markerfacecolor='white', label='Ordinary graph + Clifford optimum'),
                  Line2D([], [], color='#d76422', marker='s', label='Weighted graph + Clifford best found')]
        fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, .915), ncol=1 if len(ns)==1 else 3,
                   fontsize=9, frameon=False)
        fig.text(.5, .015, 'Saved eta grid only; lines guide the eye. Fixed inputs, all loss trials; QFI bound, not measured CFI.',
                 ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .035, 1, .81 if len(ns)==1 else .86))
        fig.savefig(target, metadata={'Date': None})
        if preview:
            fig.savefig(preview, dpi=140)
        plt.close(fig)


def make_information(results, zoom=(.5, .9), preview=None, combined=None):
    if not 0 <= zoom[0] < zoom[1] <= 1:
        raise ValueError('zoom must be within [0,1], in increasing order')
    all_data = {}
    for out in map(Path, results):
        data = read_results(out)
        if set(all_data) & set(data):
            raise ValueError('duplicate sensor count across input folders')
        all_data.update(data)
        draw(data, out/'information.svg', zoom=zoom)
        draw(data, out/'information_total.svg', total=True, zoom=zoom)
    if combined:
        draw(all_data, combined, zoom=zoom, preview=preview)
    elif preview:
        draw(all_data, Path(results[-1])/'information_combined.svg', zoom=zoom, preview=preview)
    return all_data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, nargs='+', default=[ROOT/'results', ROOT.parent/'weighted_graph_sensors_N6_7/results'])
    p.add_argument('--zoom', type=float, nargs=2, default=(.5, .9), metavar=('LOW', 'HIGH'))
    p.add_argument('--combined', type=Path, help='Optional combined SVG across all supplied results')
    p.add_argument('--preview', type=Path, help='Optional combined PNG')
    a = p.parse_args()
    make_information(a.results, tuple(a.zoom), a.preview, a.combined)
    print('Absolute-information figures written from saved results.')


if __name__ == '__main__':
    main()
