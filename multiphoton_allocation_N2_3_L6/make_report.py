"""Create a portable report and simple physical circuits from saved results."""
import argparse
import csv
import html
import json
from fractions import Fraction
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from run_search import ROOT, GATES, write_csv


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def simple_ghz(protocol):
    g = protocol['GHZ_partition_witness']
    l = protocol['total_photons']
    edges, axes = [], ['+Z']*l
    for block in g['photon_blocks']:
        center = block[0]
        for i in block:
            sign = '+' if g['signs'][i] > 0 else '-'
            axes[i] = sign + ('Z' if i == center else 'X')
            if i != center:
                edges.append([center, i])
    return {'nodes': protocol['nodes'], 'total_photons': l, 'eta': protocol['eta'],
            'photons_per_node': g['allocation'], 'photon_to_node': g['photon_to_node'],
            'edges': edges, 'effective_axes': axes, 'clifford_gates': [GATES[a] for a in axes],
            'photon_blocks': g['photon_blocks'], 'signed_node_counts': g['signed_node_counts'],
            'R': protocol['R'], 'R_exact': protocol['R_exact']}


def compressed_ranges(rows, fields):
    result = []
    for row in rows:
        if result and all(row.get(k) == result[-1].get(k) for k in fields):
            result[-1]['eta_last'] = row['eta']
        else:
            result.append({k: row[k] for k in fields} | {'eta_first': row['eta'], 'eta_last': row['eta']})
    return result


def html_table(rows, fields):
    return '<table><thead><tr>' + ''.join('<th>'+html.escape(k)+'</th>' for k in fields) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in fields) + '</tr>' for r in rows) + '</tbody></table>'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=ROOT/'results')
    args = p.parse_args(); out = args.results
    budget = read_csv(out/'best_under_budget.csv')
    total = read_csv(out/'best_total_QFI_under_budget.csv')
    allocation = read_csv(out/'best_by_allocation_eta.csv')
    protocols = json.loads((out/'protocols.json').read_text())
    summary = json.loads((out/'summary.json').read_text())
    validation = json.loads((out/'validation.json').read_text())
    simple = {k: simple_ghz(v) for k,v in protocols.items()}
    (out/'simple_GHZ_graphs.json').write_text('{\n' + ',\n'.join(
        '  '+json.dumps(k)+': '+json.dumps(v, ensure_ascii=False, separators=(',', ':'))
        for k,v in simple.items()) + '\n}\n')
    fields = ['N','L','GHZ_allocation','best_GHZ_partition','all_tied_L','all_tied_allocations']
    ranges = compressed_ranges(budget, fields)
    write_csv(out/'optimal_efficiency_grid_ranges.csv', ranges)
    tranges = compressed_ranges(total, ['N','L','GHZ_allocation','best_GHZ_partition'])
    write_csv(out/'optimal_total_QFI_grid_ranges.csv', tranges)
    nodes = summary['configuration']['nodes']
    fig, axs = plt.subplots(1, len(nodes), figsize=(6*len(nodes),4.3), squeeze=False)
    for ax,n in zip(axs[0], nodes):
        r = [r for r in budget if int(r['N']) == n]
        etas = [float(r['eta']) for r in r]
        ax.plot(etas, [float(v['R']) for v in r], 'o-', label='Best R, total photons <= limit')
        a = [r for r in total if int(r['N']) == n]
        ax.plot(etas, [float(v['R']) for v in a], 's--', label='R of best per-trial QFI input')
        ax.plot(etas, etas, ':', color='black', label='Independent photons')
        ax.set(xlabel='Photon survival probability eta', ylabel='Effective QFI / generated photon R', title=f'{n} unknown phase nodes')
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out/'performance.svg'); fig.savefig(out/'performance.png', dpi=170); plt.close(fig)
    # Show one physical graph per distinct total-QFI optimum, not only GHZ sizes.
    selected = [simple[next(r['protocol'] for r in total if r['N'] == t['N'] and r['eta'] == t['eta_first'])] for t in tranges]
    columns = 4; count = len(selected)
    fig, axs = plt.subplots((count+columns-1)//columns, columns, figsize=(13, 2.6*((count+columns-1)//columns)), squeeze=False)
    colors = ['#0072b2','#d55e00','#009e73','#cc79a7','#f0e442','#56b4e9']
    for ax, g in zip(axs.flat, selected):
        coords = {}
        for b, block in enumerate(g['photon_blocks']):
            cx = 3.6*b
            if len(block) == 1:
                coords[block[0]] = (cx,0)
            else:
                coords[block[0]] = (cx,0)
                for j,i in enumerate(block[1:]):
                    angle = 2*np.pi*j/(len(block)-1)
                    coords[i] = (cx+1.1*np.cos(angle), 1.1*np.sin(angle))
        for i,j in g['edges']:
            ax.plot([coords[i][0],coords[j][0]],[coords[i][1],coords[j][1]], color='#777', zorder=1)
        for i,(x,y) in coords.items():
            ax.scatter(x,y,s=250,color=colors[g['photon_to_node'][i]],zorder=2)
            ax.text(x,y,str(i),ha='center',va='center',color='white',fontsize=8,zorder=3)
        ax.set_title(f"N={g['nodes']}, eta={g['eta']}, allocation={g['photons_per_node']}\nGHZ blocks="+'+'.join(map(str,map(len,g['photon_blocks']))), fontsize=9)
        ax.set_xlim(-1, max(x for x,y in coords.values())+1)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal'); ax.axis('off')
    for ax in list(axs.flat)[count:]: ax.axis('off')
    fig.suptitle('Simple optimal graph circuits: node 0 blue, node 1 orange, node 2 green\nCZ edges; H on each star leaf; see JSON for gates and photon-to-node mapping',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.91)); fig.savefig(out/'optimal_total_QFI_graphs.png',dpi=180); plt.close(fig)
    statistics = [{k:s[k] for k in ['N','L','node_set_partitions','unique_node_QFIM_polynomials','ghz_unique_polynomials']} for s in summary['search_statistics']]
    limit = summary['configuration']['max_total_generated_photons']
    grid = summary['configuration']['eta_grid']
    title = f"{'・'.join(map(str, nodes))}ノード、総光子数{limit}個以下：全配分の最適化"
    improves = any(Fraction(r['R_exact']) > max(Fraction(a['R_exact']) for a in allocation
                   if a['N'] == r['N'] and a['eta'] == r['eta'] and len(set(a['allocation'].split('+'))) == 1)
                   for r in budget)
    unequal_ties = any(len(set(a.split('+'))) > 1 for r in budget for a in r['all_tied_allocations'].split(';'))
    conclusion = ('不均等配分で最高値が更新された点があった。' if improves else '不均等配分による最高値の更新はなかった。')
    tie_text = ('同率の最適配分に不均等配分を含む。' if unequal_ties else '同率の最適配分にも不均等配分はなかった。')
    doc = '''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>光子数上限・全配分の最適化</title>
<style>body{font:16px/1.7 system-ui,sans-serif;max-width:1150px;margin:35px auto;padding:0 20px;color:#182232}table{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}th,td{border:1px solid #cdd5df;padding:6px;text-align:left}th{background:#eef3f7}svg{max-width:100%;height:auto}code{background:#eef3f7;padding:2px 4px}h1,h2{line-height:1.3}.scroll{overflow:auto}</style>
'''
    doc += '<h1>'+title+'</h1>'
    doc += f'<p>各ノードの光子数は探索変数。全ての通常CZグラフ、固定局所クリフォード、正の整数配分、N≤L≤{limit}を探索した。生存確率η={grid[0]}〜{grid[-1]}の{len(grid)}点。目標は独立な未知位相の<strong>単純平均</strong>。各ノード内の全光子に同じ位相を付与する。</p>'
    doc += '<p><strong>'+conclusion+'</strong>'+tie_text+'1回当たり総情報量についても全配分を探索し、結果を別表に示す。GHZ分割が通常グラフ＋固定Cliffordクラスの上界を達成した。</p>'
    doc += '<p>光子当たり効率は R=1/[L wᵀ Q̄⁺ w]、w=(1/N,…,1/N)。1回当たり有効QFIは F_eff=L R。光子数で割るかどうかで最適構成が異なる場合がある。</p>'
    doc += (out/'performance.svg').read_text()
    doc += '<h2>光子当たり効率の最適構成</h2><p>同率では最小Lを代表に選択。範囲は隣接する計算点をまとめたもの。</p><div class="scroll">'+html_table(ranges, ['N','eta_first','eta_last']+fields[1:])+'</div>'
    doc += '<h2>1回当たり総有効QFIの最適構成</h2><div class="scroll">'+html_table(tranges,['N','eta_first','eta_last','L','GHZ_allocation','best_GHZ_partition'])+'</div>'
    doc += '<h2>各点の最適値</h2><div class="scroll">'+html_table(budget,['N','eta','L','GHZ_allocation','best_GHZ_partition','R','gain_over_independent_budget'])+'</div>'
    doc += '<h2>探索規模</h2>'+html_table(statistics,list(statistics[0]))
    doc += '<h2>検証と保証範囲</h2><p>全物理ブロックの損失多項式をGHZで上から抑える整数係数の検査と、全配分の符号付きGHZ候補を有理数で完全評価した。全'+str(len(allocation))+'配分×η点で上界と実際のグラフ達成値が一致。検証合格：'+str(validation['all_passed'])+'。一般混合状態の密度行列・部分トレース・SLD式から独立再計算し、全保存グラフの全損失マスクを確認した。</p><p>保証は通常グラフ＋固定局所Clifford・共通η・今回の離散グリッドに対するもの。任意状態、重み付き辺、任意回転、損失適応入力、設定混合は含まない。制限付き局所測定のCFIは最適化していない。ノードに光子ゼロでは未知位相の単純平均を識別できないため、正の配分のみ探索した。</p><p>配分ごとの全数値：best_by_allocation_eta.csv。グラフと回転：protocols.json、簡単な等価達成回路：simple_GHZ_graphs.json。検算：validation.json。再実行方法：README.md。</p></html>'
    (out/'report.html').write_text(doc, encoding='utf-8')
    print(json.dumps({'report': str(out/'report.html'), 'simple_GHZ_circuits': len(simple), 'efficiency_ranges': len(ranges)}))


if __name__ == '__main__':
    main()
