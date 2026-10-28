"""Human-readable report, exact protocol tables and scientific SVG figures."""
import argparse,csv,gzip,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from fixed_clifford_sensors.engine import GATES
from weighted_graph_sensors_N5.make_information import make_information


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def graph_description(proto,tol):
    n=proto['n'];edges=[]
    for pair,angle in zip(proto['edge_order'],proto['weights_rad']):
        a=float((angle+np.pi)%(2*np.pi)-np.pi)
        if abs(a)>tol:edges.append((pair,a))
    adjacent=[[] for _ in range(n)]
    for (i,j),_ in edges:adjacent[i].append(j);adjacent[j].append(i)
    color={};bipartite=True
    for root in range(n):
        if root in color:continue
        color[root]=0;todo=[root]
        while todo:
            u=todo.pop()
            for v in adjacent[u]:
                if v not in color:color[v]=1-color[u];todo.append(v)
                elif color[v]==color[u]:bipartite=False
    sides=[sum(c==b for c in color.values()) for b in [0,1]]
    if not edges:family='independent'
    elif len(edges)==n*(n-1)//2:family=f'K{n}'
    elif bipartite and len(edges)==sides[0]*sides[1]:family='K'+','.join(map(str,sorted(sides)))
    else:family=f'{len(edges)} edges'
    return edges,family,sorted(map(len,adjacent),reverse=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=ROOT/'results')
    p.add_argument('--graph-threshold',type=float,default=1e-4,help='Illustration only; saved state is not rounded')
    p.add_argument('--preview',type=Path,help='Optional comparison PNG')
    a=p.parse_args();out=a.results
    rows=list(csv.DictReader((out/'best_by_n_eta.csv').open()))
    protocols=json.loads((out/'protocols.json').read_text())
    s=json.loads((out/'summary.json').read_text());v=json.loads((out/'validation.json').read_text())
    # The full coarse table is repetitive; retain exact JSON in a compact
    # deterministic gzip artifact rather than publishing a multi-MB text file.
    raw=(out/'uniform_scan_seeds.json').read_bytes()
    (out/'uniform_scan_seeds.json.gz').write_bytes(gzip.compress(raw,mtime=0))
    runs=list(csv.DictReader((out/'optimization_runs.csv').open()))
    ns=sorted({int(r['n']) for r in rows});descriptions={};wide_by_n={}
    for n in ns:
        wide=[];selected=[r for r in rows if int(r['n'])==n]
        for r in selected:
            proto=protocols[r['protocol_id']];edges,family,degrees=graph_description(proto,a.graph_threshold)
            descriptions[r['protocol_id']]={'family_for_plot':family,'plotted_edges':len(edges),'degree_sequence':degrees,
                'plot_threshold_rad':a.graph_threshold,'signed_edges_rad':[{'edge':e,'weight':w} for e,w in edges]}
            row={'n':n,'eta':r['eta'],'R':r['best_found'],'clifford_baseline':r['clifford_baseline'],
                 'relative_improvement':r['relative_improvement'],'graph_for_plot':family}
            for (i,j),weight in zip(proto['edge_order'],proto['weights_rad']):
                row[f'phi_{i}{j}_rad']=weight
                row[f'principal_phi_{i}{j}_rad']=float((weight+np.pi)%(2*np.pi)-np.pi)
            for i,axis in enumerate(proto['effective_axes']):row[f'axis_{i}']=axis;row[f'gate_{i}']=GATES[axis]
            wide.append(row)
        write_csv(out/f'best_protocols_N{n}.csv',wide);wide_by_n[n]=wide
        cols=4;nr=(len(selected)+cols-1)//cols
        fig,axs=plt.subplots(nr,cols,figsize=(12,3.1*nr),squeeze=False)
        for ax,r in zip(axs.flat,selected):
            proto=protocols[r['protocol_id']];edges,family,_=graph_description(proto,a.graph_threshold)
            angles=np.pi/2+2*np.pi*np.arange(n)/n
            xy=np.column_stack([np.cos(angles),np.sin(angles)])
            for (i,j),weight in edges:
                ax.plot(xy[[i,j],0],xy[[i,j],1],color='#2376b7' if weight>0 else '#d14b42',
                    linewidth=.6+2*abs(weight)/np.pi,alpha=.75,zorder=1)
            ax.scatter(xy[:,0],xy[:,1],s=330,color='#f5f5f5',edgecolor='#202b3a',zorder=2)
            for i,(x,y) in enumerate(xy):ax.text(x,y,str(i),ha='center',va='center',fontsize=10,zorder=3)
            ax.set_title(f"eta={float(r['eta']):.2f}  R={float(r['best_found']):.6f}\n{family}",fontsize=10)
            gates=[GATES[axis] for axis in proto['effective_axes']]
            ax.text(.5,-.02,', '.join(gates),transform=ax.transAxes,ha='center',fontsize=7)
            ax.set_aspect('equal');ax.set_xlim(-1.3,1.3);ax.set_ylim(-1.3,1.3);ax.axis('off')
        for ax in list(axs.flat)[len(selected):]:ax.axis('off')
        fig.suptitle(f'{n} sensors: best-found weighted graphs; blue +phase, red -phase\n'
            f'Line width follows |phase|; omit edges below {a.graph_threshold:g} rad for display only',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.95));fig.savefig(out/f'graphs_N{n}.svg')
        if a.preview:fig.savefig(a.preview.with_name(f'{a.preview.stem}_N{n}.png'),dpi=100)
        plt.close(fig)
    (out/'graph_structures.json').write_text(json.dumps(descriptions,indent=2)+'\n')
    fig,axs=plt.subplots(2,len(ns),figsize=(6.3*len(ns),7),squeeze=False,sharex=True)
    for col,n in enumerate(ns):
        rs=[r for r in rows if int(r['n'])==n]
        eta=np.array([float(r['eta']) for r in rs]);base=np.array([float(r['clifford_baseline']) for r in rs]);best=np.array([float(r['best_found']) for r in rs])
        axs[0,col].plot(eta,base,'o--',label='Ordinary graph + Clifford optimum',markersize=4)
        axs[0,col].plot(eta,best,'s-',label='Weighted graph + Clifford best found',markersize=4)
        axs[0,col].plot(eta,eta,':',label='Independent sensors',color='grey')
        axs[0,col].set_title(f'{n} sensors');axs[0,col].set_ylabel('Effective information per input sensor, R')
        axs[0,col].legend(fontsize=8)
        axs[1,col].plot(eta,100*(best/base-1),'o-',color='#13876b');axs[1,col].set_ylabel('Gain over ordinary-graph optimum (%)')
        axs[1,col].set_xlabel('Single-sensor survival probability, eta');axs[1,col].set_ylim(bottom=-.5)
    for ax in axs.flat:ax.grid(alpha=.25);ax.set_xlim(0,1.02)
    fig.suptitle('Fixed input, independent unknown phases, flagged IID erasure; QFI bound')
    fig.tight_layout();fig.savefig(out/'performance.svg')
    if a.preview:fig.savefig(a.preview,dpi=130)
    plt.close(fig)
    local=[r for r in runs if r['method']=='L-BFGS-B'];de=[r for r in runs if r['method']=='differential_evolution']
    text=['# 6・7センサ：重み付きグラフ＋固定Cliffordの探索結果','',
        '**各辺の重みを独立に最適化し、従来の通常グラフ＋固定Cliffordの全探索値と比較した。**','',
        '連続重みはヒューリスティック探索であり、保存値は最良発見値。大域最適性は未証明。',
        '入力、損失モデル、独立未知位相の平均、全試行の扱いは5センサ探索と同じ。',
        'Cliffordの確率混合、補助、適応的入力は使っていない。','',
        '![comparison](performance.svg)','',
        '|センサ数|改善した点数|最大QFI増加率|そのときのeta|従来R|最良発見R|',
        '|---:|---:|---:|---:|---:|---:|']
    for n in ns:
        rs=[r for r in rows if int(r['n'])==n];r=max(rs,key=lambda r:float(r['relative_improvement']))
        text.append(f"|{n}|{sum(r['updated']=='True' for r in rs)}/{len(rs)}|{100*float(r['relative_improvement']):.4f}%|{float(r['eta']):.2f}|{float(r['clifford_baseline']):.9f}|{float(r['best_found']):.9f}|")
    text+=['','QFI増加率は分散・標準偏差の改善率とは異なる。Rがa倍なら分散下限は1/a倍。',
        'R=1/[N w^T Q(eta)^+ w]、w=ones(N)/N。nu試行の平均位相分散下限は1/(nu N R)。','',
        '## 生存確率ごとの比較','',
        '|N|eta|通常グラフ最適R|重み可変の最良発見R|QFI増加率|描画上の構造|',
        '|---:|---:|---:|---:|---:|---|']
    for r in rows:
        d=descriptions[r['protocol_id']]
        text.append(f"|{r['n']}|{float(r['eta']):.2f}|{float(r['clifford_baseline']):.9f}|{float(r['best_found']):.9f}|{100*float(r['relative_improvement']):.4f}%|{d['family_for_plot']}|")
    text+=['','## 各点のグラフ・重み・回転','',
        f'図だけは絶対値{a.graph_threshold:g} rad未満の辺を省略。計算と保存構成は丸めていない。',
        '図の構造名はこの描画閾値に基づく分類であり、厳密な零重みや対称性の証明ではない。',
        '青は正、赤は負の主値角。太さは絶対値に対応。節点は全てセンサ。',
        '各図の下は頂点番号順の局所ゲート。ゲート積は右から作用。','']
    for n in ns:
        text += [f'### {n}センサ','',f'![graphs N{n}](graphs_N{n}.svg)','',
            f'[各辺の正確な重みと各センサの回転：best_protocols_N{n}.csv](best_protocols_N{n}.csv)','']
    text+=['## 探索範囲・停止条件','',
        f"- 粗い探索：全同型代表N6={s['coarse_graphs'].get('6',0)}、N7={s['coarse_graphs'].get('7',0)}。非連結・孤立頂点を含む。",
        f"- 共通重み[0,pi]の{s['configuration']['uniform_points']}点、生存確率{s['configuration']['coarse_etas']}。共役側の重複を省略。",
        '- 全形状の共通重みを調べたが、各形状の全独立重みを全探索したわけではない。',
        '- 各最終etaでは近い粗い点の上位候補を再評価し、全15／21辺の角度を独立に最適化。',
        '- 全局所Cliffordの比較：6^6=46,656、6^7=279,936。全符号反転のみ同一視。',
        '- 初期値：粗い上位候補、GHZ分割、5センサ構成の埋込み、符号付き二部グラフ、摂動、全辺ランダム。',
        f"- 通常・継続の局所最適化：{len(local):,}回。成功終了{sum(r['success']=='True' for r in local):,}回。これに加え各集団探索の候補を局所最適化。",
        f"- 集団探索：{len(de)}回。成功終了{sum(r['success']=='True' for r in de)}回。世代上限に達した探索も局所最適化して比較。",
        '- 最適化中のみ正則化。保存スコアは正則化なし。元の基準構成も候補として保持。',
        '- 通常グラフの比較基準は全GHZ分割を列挙し、従来の通常グラフ＋Clifford全探索結果とも照合。',
        f"- 最終の起動からの経過時間：{s['elapsed_seconds_this_invocation']:.2f}秒。並列ワーカー数{s['configuration']['workers']}。",
        f"- 初回の設定保存から最終集計までの経過時間：{s['elapsed_seconds_since_first_configuration']:.2f}秒。中断・再開・同時実行を含み、単独実行のベンチマークではない。",
        '- 実行設定はconfiguration.json、各試行の停止理由・評価回数はoptimization_runs.csv。','',
        '## 独立検算','',
        '探索は一般混合状態SLD公式の低ランク表現を使用。スタビライザー状態とは仮定しない。',
        '検算はCPとCliffordを物理的に作用させ、全損失で密度行列微分から別実装のSLD公式を計算。','',
        f"- 保存した{v['saved_winners_checked']}構成、合計{v['saved_loss_masks_checked']}生存マスクを確認。",
        f"- 保存QFIMの最大誤差：{v['maximum_saved_QFIM_error']:.3g}。",
        f"- 保存Rの最大誤差：{v['maximum_saved_score_error']:.3g}。",
        f"- 生存数集計・平均QFIMの最大誤差：{v['maximum_aggregate_QFIM_error']:.3g}。",
        f"- 非ゼロの独立位相{v['nonzero_independent_phase_checks']}構成：最大誤差{v['maximum_nonzero_phase_QFIM_error']:.3g}。",
        f"- ランダム入力等と元の全密度行列カーネルの誤差：{v['maximum_reference_tensor_error']:.3g}。",
        f"- 解析勾配と有限差分：最大誤差{v['maximum_gradient_error']:.3g}。",
        f"- 5センサの全{v['N5_saved_winners_checked']}保存最良値を再現：最大誤差{v['maximum_N5_regression_score_error']:.3g}。",
        f"- 従来の6・7センサ全探索基準値と{v['earlier_baseline_points_matched']}点を照合：最大誤差{v['maximum_earlier_baseline_error']:.3g}。",'',
        '## 結論の範囲','',
        '改善は、同じモデルで通常グラフ＋固定Cliffordの最良値を上回る実行可能な重み付き構成があることを示す。',
        '改善のない点も、未探索の重みで改善できないとは証明していない。',
        '重み付きグラフ全体の最適性、任意局所回転との同時最適化、局所読み出しの達成可能性、',
        '生成成功率・ゲート誤差を含めた実験上の優位性は別の問題。','']
    make_information([out])
    text += ['', '## 情報量そのものの比較（追加図）', '',
             '既存の改善率の図に加え、光子当たりの有効QFIと1試行当たりの有効QFIを示す。',
             '上段は全範囲、下段は透過率0.50〜0.90。保存済み評価点を線で結んだもので、追加最適化や平滑化は行っていない。', '',
             '![光子当たり有効QFI R](information.svg)', '',
             '[1試行当たりの有効QFI F_eff = N R](information_total.svg)', '']
    (out/'report.md').write_text('\n'.join(text),encoding='utf-8')
    print(out/'report.md')


if __name__=='__main__':main()

