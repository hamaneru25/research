"""Plot the actual signed-edge hub count, keeping topology figures separate."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np


def plot_hub_counts(results):
    results = Path(results)
    rows = json.loads((results / 'best.json').read_text())
    sensors = sorted({r['n'] for r in rows})
    etas = sorted({r['eta'] for r in rows})
    lookup = {(r['n'], r['eta']): r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError('Duplicate sensor-count / transmission records')
    hubs = np.array([[lookup[n, eta]['hubs'] for eta in etas] for n in sensors])
    for n in sensors:
        for eta in etas:
            r = lookup[n, eta]
            if not (1 <= r['hubs'] < n and min(r['hubs'], n-r['hubs']) == r['canonical_part_a']):
                raise ValueError(f'Inconsistent hub count: N={n}, eta={eta}')
    maximum = int(hubs.max())
    values = np.unique(hubs)
    palette = plt.get_cmap('tab10') if maximum <= 10 else plt.get_cmap('turbo', maximum)
    cmap = ListedColormap([palette(int(h)-1) for h in values])
    norm = BoundaryNorm(np.arange(-.5, len(values)+.5), len(values))
    categories = np.searchsorted(values, hubs)
    fig, ax = plt.subplots(figsize=(13, max(4, .36*len(sensors)+1.6)), layout='constrained')
    im = ax.imshow(categories, cmap=cmap, norm=norm, aspect='auto', origin='upper')
    colorbar = fig.colorbar(im, ax=ax, ticks=range(len(values)), pad=.015)
    colorbar.set_ticklabels([str(h) for h in values])
    colorbar.set_label('Actual hub count h')
    ax.set_xticks(range(len(etas)), [f'{eta:.2f}' for eta in etas], fontsize=8)
    ax.set_yticks(range(len(sensors)), [str(n) for n in sensors])
    ax.set(xlabel='Survival probability eta', ylabel='Number of sensors N',
           title='Best-found signed complete-bipartite probe: actual hub count h')
    for i in range(len(sensors)):
        for j in range(len(etas)):
            rgb = cmap(norm(categories[i, j]))[:3]
            luminance = np.dot(rgb, [.2126, .7152, .0722])
            ax.text(j, i, str(hubs[i, j]), ha='center', va='center', fontsize=8,
                    color='white' if luminance < .48 else '#172436')
    fig.supxlabel('Color and cell labels use best.json: hubs, not min(h, N-h).  h=1: star/GHZ branch.', fontsize=9)
    for suffix in ('svg', 'png'):
        fig.savefig(results / f'hub_counts.{suffix}', dpi=160)
    plt.close(fig)
    return hubs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path(__file__).with_name('results'))
    args = parser.parse_args()
    counts = plot_hub_counts(args.results)
    print(json.dumps({'rows': int(counts.size), 'hub_counts': np.unique(counts).tolist()}))
