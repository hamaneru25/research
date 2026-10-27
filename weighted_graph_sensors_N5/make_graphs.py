"""Draw the saved N5 weighted protocols without rerunning optimization.

Match the N6/N7 circular layout, phase colors, widths and display threshold.
All thresholds affect illustrations only; protocols.json is never modified.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
GATES = {'+X': 'H', '-X': 'HZ', '+Y': 'HSdg', '-Y': 'HS', '+Z': 'I', '-Z': 'X'}


def graph_description(proto, tol):
    """Classify only the graph visible after principal-angle thresholding."""
    n = proto['n']
    edges = []
    for pair, angle in zip(proto['edge_order'], proto['weights_rad']):
        principal = float((angle + np.pi) % (2 * np.pi) - np.pi)
        if abs(principal) > tol:
            edges.append((pair, principal))
    adjacent = [[] for _ in range(n)]
    for (i, j), _ in edges:
        adjacent[i].append(j)
        adjacent[j].append(i)
    color = {}
    bipartite = True
    for root in range(n):
        if root in color:
            continue
        color[root] = 0
        todo = [root]
        while todo:
            u = todo.pop()
            for v in adjacent[u]:
                if v not in color:
                    color[v] = 1 - color[u]
                    todo.append(v)
                elif color[v] == color[u]:
                    bipartite = False
    sides = [sum(c == b for c in color.values()) for b in [0, 1]]
    if not edges:
        family = 'independent'
    elif len(edges) == n * (n - 1) // 2:
        family = f'K{n}'
    elif bipartite and len(edges) == sides[0] * sides[1]:
        family = 'K' + ','.join(map(str, sorted(sides)))
    else:
        family = f'{len(edges)} edges'
    return edges, family, sorted(map(len, adjacent), reverse=True)


def make_graphs(out, threshold=1e-4, preview=None):
    out = Path(out)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError('graph threshold must be finite and nonnegative')
    with (out / 'best_by_eta.csv').open(encoding='utf-8', newline='') as f:
        rows = sorted(csv.DictReader(f), key=lambda r: float(r['eta']))
    protocols = json.loads((out / 'protocols.json').read_text(encoding='utf-8'))
    if not rows:
        raise ValueError('no saved protocols to draw')
    cols = 4
    nr = (len(rows) + cols - 1) // cols
    fig, axs = plt.subplots(nr, cols, figsize=(12, 3.1 * nr), squeeze=False)
    descriptions = {}
    for ax, row in zip(axs.flat, rows):
        proto = protocols[row['protocol_id']]
        n = proto['n']
        if n != 5:
            raise ValueError('this renderer expects five sensors')
        edges, family, degrees = graph_description(proto, threshold)
        descriptions[row['protocol_id']] = {
            'family_for_plot': family, 'plotted_edges': len(edges),
            'degree_sequence': degrees, 'plot_threshold_rad': threshold,
            'signed_edges_rad': [{'edge': e, 'weight': w} for e, w in edges],
        }
        angles = np.pi / 2 + 2 * np.pi * np.arange(n) / n
        xy = np.column_stack([np.cos(angles), np.sin(angles)])
        for (i, j), weight in edges:
            ax.plot(xy[[i, j], 0], xy[[i, j], 1],
                    color='#2376b7' if weight > 0 else '#d14b42',
                    linewidth=.6 + 2 * abs(weight) / np.pi, alpha=.75, zorder=1)
        ax.scatter(xy[:, 0], xy[:, 1], s=330, color='#f5f5f5',
                   edgecolor='#202b3a', zorder=2)
        for i, (x, y) in enumerate(xy):
            ax.text(x, y, str(i), ha='center', va='center', fontsize=10, zorder=3)
        ax.set_title(f"eta={float(row['eta']):.2f}  R={float(row['best_found']):.6f}\n{family}",
                     fontsize=10)
        gates = [GATES[axis] for axis in proto['effective_axes']]
        ax.text(.5, -.02, ', '.join(gates), transform=ax.transAxes,
                ha='center', fontsize=7)
        ax.set_aspect('equal')
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.axis('off')
    for ax in list(axs.flat)[len(rows):]:
        ax.axis('off')
    fig.suptitle('5 sensors: best-found weighted graphs; blue +phase, red -phase\n'
                 f'Line width follows |phase|; omit edges below {threshold:g} rad for display only',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .95))
    # Omit the SVG timestamp so repeated rendering has no date-only changes.
    fig.savefig(out / 'graphs_N5.svg', metadata={'Date': None})
    if preview:
        fig.savefig(preview, dpi=100)
    plt.close(fig)
    (out / 'graph_structures.json').write_text(
        json.dumps(descriptions, indent=2) + '\n', encoding='utf-8')
    return descriptions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=ROOT / 'results')
    p.add_argument('--graph-threshold', type=float, default=1e-4,
                   help='Display only; original weights and scores are unchanged')
    p.add_argument('--preview', type=Path, help='Optional PNG for visual inspection')
    a = p.parse_args()
    make_graphs(a.results, a.graph_threshold, a.preview)
    print(a.results / 'graphs_N5.svg')


if __name__ == '__main__':
    main()
