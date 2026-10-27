"""Draw each saved optimal graph once, merging exact graph isomorphisms.

Only graphs selected in best_by_n_eta.csv are included. Vertex permutations
are merged; distinct graphs are not merged merely because they are LC-equivalent.
Gate variants and every saved eta remain available in the index and JSON.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def adjacency(n, edges):
    a = [set() for _ in range(n)]
    for i, j in edges:
        assert 0 <= i < n and 0 <= j < n and i != j
        a[i].add(j)
        a[j].add(i)
    return a


def isomorphism(n, edges, representative):
    """Return an exact original-to-representative vertex map, or None."""
    a, b = adjacency(n, edges), adjacency(n, representative)
    if sorted(map(len, a)) != sorted(map(len, b)):
        return None
    order = sorted(range(n), key=lambda i: (-len(a[i]), i))
    mapping, used = {}, set()

    def visit(k):
        if k == n:
            return dict(mapping)
        i = order[k]
        for j in range(n):
            if j in used or len(a[i]) != len(b[j]):
                continue
            if any((u in a[i]) != (v in b[j]) for u, v in mapping.items()):
                continue
            mapping[i] = j
            used.add(j)
            result = visit(k + 1)
            if result is not None:
                return result
            del mapping[i]
            used.remove(j)
        return None

    return visit(0)


def components(n, edges):
    a = adjacency(n, edges)
    unseen, result = set(range(n)), []
    while unseen:
        todo = [min(unseen)]
        unseen.remove(todo[0])
        group = []
        while todo:
            i = todo.pop()
            group.append(i)
            for j in sorted(a[i] & unseen):
                unseen.remove(j)
                todo.append(j)
        result.append(sorted(group))
    return result


def eta_text(etas):
    values = sorted(set(etas))
    if len(values) > 2 and all(abs(b-a-.05) < 1e-9 for a, b in zip(values, values[1:])):
        return f'{values[0]:.2f}-{values[-1]:.2f} (step .05)'
    return ', '.join(f'{x:.2f}' for x in values)


def collect(out):
    with (out/'best_by_n_eta.csv').open(encoding='utf-8', newline='') as f:
        rows = sorted(csv.DictReader(f), key=lambda r: (int(r['n']), float(r['eta'])))
    protocols = json.loads((out/'protocols.json').read_text(encoding='utf-8'))
    groups = []
    for row in rows:
        key = row['protocol_id']
        p = protocols[key]
        n = int(row['n'])
        assert p['n'] == n
        for g in groups:
            mapping = isomorphism(n, p['edges'], g['edges']) if g['n'] == n else None
            if mapping is not None:
                break
        else:
            g = {'structure_id': f'N{n}_S{1+sum(x["n"] == n for x in groups):02d}',
                 'n': n, 'representative_protocol_id': key, 'edges': p['edges'],
                 'sensor_gates': p['sensor_gates'], 'effective_axes': p['effective_axes'],
                 'components': components(n, p['edges']), 'variants': [], 'eta_values': []}
            groups.append(g)
            mapping = dict(enumerate(range(n)))
        g['eta_values'].append(float(row['eta']))
        variant = next((v for v in g['variants'] if v['protocol_id'] == key), None)
        if variant is None:
            gates, axes = [None]*n, [None]*n
            for i, j in mapping.items():
                gates[j], axes[j] = p['sensor_gates'][i], p['effective_axes'][i]
            variant = {'protocol_id': key, 'original_to_diagram': [mapping[i] for i in range(n)],
                       'gates_in_diagram_order': gates, 'axes_in_diagram_order': axes,
                       'eta_values': [], 'R_by_eta': {}}
            g['variants'].append(variant)
        eta = float(row['eta'])
        variant['eta_values'].append(eta)
        variant['R_by_eta'][row['eta']] = float(row['information_per_photon'])
    return groups, len(rows)


def positions(g):
    comps = g['components']
    k = len(comps)
    cols = 1 if k == 1 else (2 if k == 4 and max(map(len, comps)) > 1 else min(k, 4))
    rows = math.ceil(k/cols)
    xy = {}
    a = adjacency(g['n'], g['edges'])
    for index, comp in enumerate(comps):
        cx = (index % cols + .5)/cols
        cy = .86-(index//cols+.5)*.70/rows
        # Leave a full node diameter between components in adjacent rows.
        radius = min(.31, .39/cols, .70/(2*rows)-.065)
        if len(comp) == 1:
            xy[comp[0]] = (cx, cy)
            continue
        hubs = [i for i in comp if len(a[i]) == len(comp)-1]
        if hubs and len(comp) >= 3:
            hub = min(hubs)
            xy[hub] = (cx, cy)
            leaves = [i for i in comp if i != hub]
        else:
            leaves = comp
        for j, i in enumerate(leaves):
            angle = math.pi/2+2*math.pi*j/len(leaves)
            xy[i] = (cx+radius*math.cos(angle), cy+radius*math.sin(angle))
    return xy


def graph_report_lines():
    return ['', '## 重複を除いた採用グラフ', '',
            '![採用グラフの構造一覧](graphs_unique.svg)', '',
            '[各構造の透過率・Clifford操作・元の構成ID](graphs.md)', '',
            '同じ頂点数で番号交換により一致する構造は1枚に集約。異なるLC同値グラフは別構造として扱う。',
            '青い辺は全てCZ（重みπ）。下のゲート列は図の頂点番号順、積は右から作用。',
            'etaは保存済み評価点での採用範囲であり、連続区間の切替境界や全同率解の一覧ではない。', '']


def make_graphs(out, columns=3, preview=None):
    out = Path(out)
    if columns < 1:
        raise ValueError('columns must be positive')
    groups, count = collect(out)
    if not groups:
        raise ValueError('no selected graphs to render')
    nr = math.ceil(len(groups)/columns)
    with matplotlib.rc_context({'svg.hashsalt': 'fixed-clifford-unique-graphs'}):
        fig, axs = plt.subplots(nr, columns, figsize=(4.2*columns, 3.6*nr), squeeze=False)
        for ax, g in zip(axs.flat, groups):
            xy = positions(g)
            for i, j in g['edges']:
                ax.plot([xy[i][0], xy[j][0]], [xy[i][1], xy[j][1]], color='#2376b7', lw=2, zorder=1)
            ax.scatter([xy[i][0] for i in range(g['n'])], [xy[i][1] for i in range(g['n'])],
                       s=360, c='#f5f5f5', edgecolors='#202b3a', zorder=2)
            for i, (x, y) in xy.items():
                ax.text(x, y, str(i), ha='center', va='center', fontsize=10, zorder=3)
            sizes = '+'.join(map(str, sorted(map(len, g['components']))))
            ax.set_title(f"{g['structure_id']}  |  component sizes: {sizes}\neta: {eta_text(g['eta_values'])}", fontsize=10)
            ax.text(.5, .035, 'C[0..]: '+', '.join(g['sensor_gates']), ha='center', fontsize=8)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.axis('off')
        for ax in list(axs.flat)[len(groups):]:
            ax.axis('off')
        fig.suptitle('Fixed Clifford: saved optimal graph structures, one panel per topology\n'
                     'All edges are CZ (phase pi); gate products act right to left; eta lists saved grid points', fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .96))
        fig.savefig(out/'graphs_unique.svg', metadata={'Date': None})
        if preview:
            fig.savefig(preview, dpi=110)
        plt.close(fig)
    index = {'deduplication': 'exact graph isomorphism at the same vertex count; not LC equivalence',
             'selected_grid_rows': count, 'unique_structures': len(groups), 'structures': groups}
    (out/'graph_structures.json').write_text(json.dumps(index, indent=2)+'\n', encoding='utf-8')
    lines = ['# 固定Clifford：重複を除いた採用グラフ', '',
             f'保存済み{count}評価点の採用構成を、{len(groups)}種類の接続構造に集約した。', '',
             '![構造一覧](graphs_unique.svg)', '',
             '全ての頂点はセンサ。辺はCZ（重みπ）。図の下のゲートは頂点番号順で、積は右から作用する。',
             '同じ頂点数で番号交換により一致する構造をまとめた。Clifford操作が異なる場合も図は重複させず、下表とJSONに全変種を残す。',
             '異なる頂点数・接続構造を、GHZへのLC同値性だけでまとめることはしない。',
             '表示はbest_by_n_eta.csvに保存された代表解のみ。同率最適解の網羅や評価点間の最適性を意味しない。', '',
             '|構造ID|センサ数|連結成分のサイズ|採用したeta（保存評価点）|図の頂点番号順のゲート|元の構成ID|',
             '|---|---:|---|---|---|---|']
    for g in groups:
        sizes = '+'.join(map(str, sorted(map(len, g['components']))))
        for v in g['variants']:
            lines.append(f"|{g['structure_id']}|{g['n']}|{sizes}|{eta_text(v['eta_values'])}|`{', '.join(v['gates_in_diagram_order'])}`|`{v['protocol_id']}`|")
    lines += ['', '[構造・辺・番号対応・正確な性能値のJSON](graph_structures.json)',
              '／ [元の全評価点の数値](best_by_n_eta.csv) ／ [元の構成データ](protocols.json)', '']
    (out/'graphs.md').write_text('\n'.join(lines), encoding='utf-8')
    return index


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=ROOT/'results')
    p.add_argument('--columns', type=int, default=3, help='Number of diagram columns')
    p.add_argument('--preview', type=Path, help='Optional PNG for visual inspection')
    args = p.parse_args()
    index = make_graphs(args.results, args.columns, args.preview)
    print(f"{args.results}: {index['selected_grid_rows']} grid rows -> {index['unique_structures']} diagrams")


if __name__ == '__main__':
    main()
