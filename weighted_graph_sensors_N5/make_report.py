"""Write comparison, a readable physical protocol table, and static plots."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from fixed_clifford_sensors.engine import GATES
from make_graphs import make_graphs
from make_information import make_information


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=ROOT/'results')
    p.add_argument('--preview',type=Path,help='Optional PNG preview for visual inspection')
    p.add_argument('--graph-threshold',type=float,default=1e-4,help='Graph display only; saved state is not rounded')
    a=p.parse_args();out=a.results
    make_graphs(out,a.graph_threshold)
    rows=list(csv.DictReader((out/'best_by_eta.csv').open()))
    s=json.loads((out/'summary.json').read_text());v=json.loads((out/'validation.json').read_text())
    protocols=json.loads((out/'protocols.json').read_text())
    runs=list(csv.DictReader((out/'optimization_runs.csv').open()))
    simple=json.loads((out/'simple_witness.json').read_text())
    wide=[]
    for r in rows:
        proto=protocols[r['protocol_id']]
        row={'eta':r['eta'],'R':r['best_found'],'clifford_baseline':r['clifford_baseline']}
        for pair,weight in zip(proto['edge_order'],proto['weights_rad']):row[f'phi_{pair[0]}{pair[1]}_rad']=weight
        for i,axis in enumerate(proto['effective_axes']):
            row[f'axis_{i}']=axis;row[f'gate_{i}']=GATES[axis]
        wide.append(row)
    with (out/'best_protocols.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(wide[0]));writer.writeheader();writer.writerows(wide)
    eta=np.array([float(r['eta']) for r in rows]);base=np.array([float(r['clifford_baseline']) for r in rows])
    best=np.array([float(r['best_found']) for r in rows]);gain=(best/base-1)*100
    fig,axs=plt.subplots(2,1,figsize=(7.4,6.4),sharex=True,gridspec_kw={'height_ratios':[1.4,1]})
    axs[0].plot(eta,base,'o--',label='Ordinary graph + Clifford optimum',markersize=4)
    axs[0].plot(eta,best,'s-',label='Weighted graph + Clifford best found',markersize=4)
    axs[0].set_ylabel('Effective QFI per generated photon, R');axs[0].legend(fontsize=8)
    axs[1].plot(eta,gain,'o-',color='#1b8b6d');axs[1].set_ylabel('QFI gain over baseline (%)')
    axs[1].set_xlabel('Single-sensor survival probability, eta');axs[1].set_ylim(bottom=-.5)
    for ax in axs:ax.grid(alpha=.25);ax.set_xlim(0,1.02)
    fig.suptitle('5 sensors: independent phases, flagged erasure, fixed input')
    fig.tight_layout();fig.savefig(out/'performance.svg')
    if a.preview:fig.savefig(a.preview,dpi=140)
    plt.close(fig)
    maximum=max(rows,key=lambda r:float(r['relative_improvement']))
    unchanged=', '.join(format(float(r['eta']),'.2f') for r in rows if r['updated']!='True')
    local=[r for r in runs if r['method']=='L-BFGS-B'];de=[r for r in runs if r['method']=='differential_evolution']
    text=['# 5センサ・重み付きグラフ＋固定Clifford：結果','',
          '**重みを変えると、従来の通常グラフ＋Clifford最適値を上回る構成が見つかった。**','',
          f"{len(rows)}評価点のうち{s['updated_points']}点で更新。最大の有効QFI増加はeta={maximum['eta']}で{float(maximum['relative_improvement'])*100:.4f}%。",
          '局所回転はCliffordのまま。全試行の平均位相QCRBから評価し、損失した試行も除外していない。',
          '**ここでの値は最良発見値であり、連続重みの大域最適値を証明したものではない。**','',
          '![performance](performance.svg)','',
          '## 既存最適値との比較','',
          'R=1/[5 w^T Q(eta)^+ w]。w=ones(5)/5。Rが大きいほど平均位相の分散下限が小さい。',
          '既存値は全通常グラフ・全局所Cliffordを調べた値で、最良GHZ分割と一致する。','',
          '|eta|既存Clifford最適値R|重み可変の最良発見R|QFI増加|',
          '|---:|---:|---:|---:|']
    for r in rows:text.append(f"|{float(r['eta']):.2f}|{float(r['clifford_baseline']):.9f}|{float(r['best_found']):.9f}|{float(r['relative_improvement'])*100:.4f}%|")
    text+=['',f"QFI増加率は分散や標準偏差の改善率とは異なる。eta={maximum['eta']}ではRが約{float(maximum['best_found'])/float(maximum['clifford_baseline']):.5f}倍になり、",
           f"対応する分散下限は約{(1-float(maximum['clifford_baseline'])/float(maximum['best_found']))*100:.2f}%低下する。",'',
           '## 再現しやすい改善例','',
           'eta=0.50、phi=2/3 rad。頂点をA={0,3}、B={1,2,4}に分けたK2,3の6辺だけを使う。','',
           '|辺|制御位相の重み|','|---|---:|',
           '|0–1、0–2、0–4|+2/3 rad|',
           '|1–3、2–3、3–4|−2/3 rad|',
           '|その他|0|','',
           '負の重みは2π−2/3 radと同じゲート。局所Cliffordはセンサ0〜4の順に',
           '`[I, HS, HS, X, HS]`（右から作用）。有効位相軸は`[+Z, -Y, -Y, -Z, -Y]`。','',
           f"この簡単な構成だけでR={simple['R']:.12f}となり、既存最適値0.5を超える。",
           f"平均位相の1試行あたりQCRBは{simple['mean_QCRB_per_trial']:.12f}（既存0.4）。",
           '数値最適化の微小な辺重みを正確な0に置き換え、非ゼロ重みを±2/3へ丸めた、別途検算済みの構成。','',
           '```bash','python simple_witness.py --eta 0.5 --phi 0.6666666666666666','```','',
           '各etaの数値最良構成はbest_protocols.csvに10辺の重みと5個の局所ゲートを保存した。',
           '読み出し測定基底の最適化をした結果ではない。','',
           '## 各透過率のグラフ構造','',
           '![5センサの重み付きグラフ一覧](graphs_N5.svg)','',
           '各パネルは保存済みの最良発見構成。節点番号は0〜4、全てセンサ。',
           '青は正、赤は負の主値角（−π以上π未満）。辺の太さは角度の絶対値に対応。',
           '下のゲート列は頂点番号順の局所Cliffordで、ゲート積は右から作用する。',
           f'図だけは絶対値{a.graph_threshold:g} rad以下の辺を省略。元の重み・スコアは変更していない。',
           '構造名は描画閾値に基づく分類であり、厳密な零重みや対称性の証明ではない。','',
           '[描画した辺と構造の一覧](graph_structures.json) ／ [全辺の正確な重みとゲート](best_protocols.csv)','',
           '## 探索方法','',
           f"- 共通重みの粗い探索：全{s['unlabeled_graph_topologies']}同型代表×{s['configuration']['uniform_points']}重み×{len(rows)}生存確率。π以外にLC削減は使わない。",
           '- 個別重み：全10辺の位相を独立に変え、辺なしも含む。',
           '- 各重み評価で6^5通りの局所Clifford有効軸を調べる。同時全符号反転のみ同一視。',
           f"- 連続局所最適化：{s['local_optimization_runs']:,}初期値。成功終了は{sum(r['success']=='True' for r in local)}/{len(local)}回。",
           f"- 独立の集団探索：{len(de)}回。成功終了は{sum(r['success']=='True' for r in de)}/{len(de)}回。",
           f"- 集団探索は{s['configuration']['de_maxiter']}世代上限に達した候補も局所最適化して評価。大域探索の収束は主張しない。",
           f"- 再開後の実行時間：{s['elapsed_seconds']:.2f}秒。保存済みの粗い探索を再利用したため、その計算時間は含まない。",
           '- 出力のsummary.jsonとoptimization_runs.csvに設定・環境・停止状況・ソースハッシュを保存。','',
           '## 混合状態の独立検算','',
           '一般の重みでは損失後のスペクトルは平坦でないため、一般の混合状態SLD公式で計算した。',
           'CPゲートとCliffordを物理的に作用させた状態を部分トレースし、密度行列微分から独立にQFIMを再計算。','',
           f"- ランダム入力{v['kernel_physical_protocols']}構成×32損失パターン。最大誤差{v['maximum_kernel_QFIM_error']:.3g}。",
           f"- 保存した全{v['saved_winners_checked']}構成×32損失パターン。最大QFIM誤差{v['maximum_saved_QFIM_error']:.3g}。",
           f"- 保存スコアの最大誤差{v['maximum_saved_score_error']:.3g}。",
           f"- 不均一で非ゼロの独立位相でも確認。最大誤差{v['maximum_nonzero_phase_QFIM_error']:.3g}。",
           f"- 解析勾配の数値微分照合：最大誤差{v['maximum_weight_gradient_error']:.3g}。",
           f"- 6^5={v['literal_signed_Clifford_assignments_checked']:,}通りの直接列挙とClifford削減が一致。",
           f"- 既存全探索結果との基準値照合：{v['earlier_baseline_points_matched']}点。",'',
           '## 結論の範囲','',
           '通常のグラフ状態＋局所Cliffordの最適性は、重み付きグラフ状態まで広げると成立しない。',
           '一方、新しく発見した構成が重み付きグラフ全体で最適であるかは未証明。',
           f"eta={unchanged or '（該当なし）'}では今回の探索で改善候補が見つからなかった。",
           'これは未評価の中間透過率、任意局所回転を同時に許した場合、または具体的な局所測定CFIに関する結論ではない。','']
    make_information([out])
    text += ['', '## 情報量そのものの比較（追加図）', '',
             '既存の改善率の図に加え、光子当たりの有効QFIと1試行当たりの有効QFIを示す。',
             '上段は全範囲、下段は透過率0.50〜0.90。保存済み評価点を線で結んだもので、追加最適化や平滑化は行っていない。', '',
             '![光子当たり有効QFI R](information.svg)', '',
             '[1試行当たりの有効QFI F_eff = N R](information_total.svg)', '']
    (out/'report.md').write_text('\n'.join(text),encoding='utf-8')
    print(out/'report.md')

if __name__=='__main__':main()


