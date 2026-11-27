"""Render a self-contained Japanese report for the completed search."""
from expanded import D
from pathlib import Path
import csv, io, base64, zipfile, json, html
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def format_a(r):
    return '('+', '.join(map(str,r['allocation']))+')'
def make():
    d=json.loads((D/'final.json').read_text())
    rs=d['results'];allr=d['allocations']
    key=lambda r:(r['N'],r['eta'])
    fig,axs=plt.subplots(2,2,figsize=(11,6.7),layout='constrained')
    for n,ax in zip(range(2,6),axs.flat):
        aa=sorted((r for r in rs if r['N']==n),key=key);t=[r['eta'] for r in aa]
        for y,lab,color,style in [([r['best_R']['R'] for r in aa],'Weighted, L ≤ 8','#126d66','-'),
            ([r['R_up_to_6'] for r in aa],'Weighted, L ≤ 6','#7a4b9a','--'),
            ([r['GHZ_R'] for r in aa],'Best GHZ partition, L ≤ 8','#a85d21',':')]:
            ax.plot(t,y,style,marker='o',ms=2.9,lw=1.7,color=color,label=lab)
        ax.set(xlabel='Transmission',ylabel='R per generated photon',title=f'{n} nodes');ax.grid(alpha=.2)
        ax.legend(frameon=False,fontsize=8)
    buf=io.BytesIO();fig.savefig(buf,format='png',dpi=150);plt.close(fig)
    fig2,axs=plt.subplots(1,2,figsize=(11,3.3),layout='constrained')
    fig_zoom,ax_zoom=plt.subplots(figsize=(8,4.2),layout='constrained')
    for n,c in zip(range(2,6),['#126d66','#325b9a','#97551e','#843e75']):
        aa=sorted((r for r in rs if r['N']==n),key=key)
        t=np.array([r['eta'] for r in aa])
        gain_R=np.array([100*(r['best_R']['R']/r['R_one_per_node']-1) for r in aa])
        series=dict(label=f'{n} nodes',marker='o',ms=3,color=c)
        axs[0].plot(t,gain_R,**series)
        axs[1].plot(t,[100*(r['best_F']['F_eff']/r['F_up_to_6']-1) for r in aa],**series)
        zoom=t<=0.80
        ax_zoom.plot(t[zoom],gain_R[zoom],**series)
    panels=[('Per photon: R','Gain vs one photon per node (%)'),
            ('Per trial: effective QFI','Gain vs L ≤ 6 (%)'),
            ('Per photon: R, transmission up to 0.80','Gain vs one photon per node (%)')]
    for ax,(title,baseline) in zip([*axs,ax_zoom],panels):
        ax.set(xlabel='Transmission',ylabel=baseline,title=title)
        ax.grid(alpha=.22)
        ax.legend(loc='upper left',frameon=False,fontsize=8)
    ax_zoom.set(xlim=(.035,.815),ylim=(-1.5,50),xticks=[.05,.20,.40,.60,.80],yticks=range(0,51,10))
    buf2=io.BytesIO();fig2.savefig(buf2,format='png',dpi=150);plt.close(fig2)
    zoom_png=io.BytesIO()
    fig_zoom.savefig(zoom_png,format='png',dpi=150)
    plt.close(fig_zoom)
    csvrows=[];table=[];high=[]
    for r in rs:
        b,f=r['best_R'],r['best_F']; gain=100*(b['R']/r['R_up_to_6']-1)
        rr=dict(N=r['N'],eta=r['eta'],R8=b['R'],R8_allocation=format_a(b),R8_ties=r['tied_R'],
            F8=f['F_eff'],F8_allocation=format_a(f),F8_ties=r['tied_F'],R6=r['R_up_to_6'],F6=r['F_up_to_6'],
            best_GHZ_partition_R=r['GHZ_R'],best_GHZ_partition_F=r['GHZ_F'],R_one_per_node=r['R_one_per_node'],
            R_gain_percent=gain,F_gain_percent=100*(f['F_eff']/r['F_up_to_6']-1))
        csvrows.append(rr)
        cols=[r['N'],f"{r['eta']:.2f}",format_a(b),f"{b['R']:.7f}",f"{r['R_up_to_6']:.7f}",f"{gain:+.3f}%",
              f"{r['GHZ_R']:.7f}",format_a(f),f"{f['F_eff']:.7f}"]
        table.append('<tr>'+''.join(f'<td>{x}</td>' for x in cols)+'</tr>')
        if r['eta']==.5:high.append('<tr>'+''.join(f'<td>{x}</td>' for x in cols[:8])+'</tr>')
    def b64(s):return base64.b64encode(s).decode()
    def csvbytes(rows,cols=None):
        f=io.StringIO();wr=csv.DictWriter(f,fieldnames=cols or list(rows[0]),extrasaction='ignore');wr.writeheader();wr.writerows(rows);return f.getvalue().encode()
    src=io.BytesIO()
    with zipfile.ZipFile(src,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in ['kernel.py','prior_solver.py','prior_verified.json','expanded.py','density_verify.py','finish.py','report.py','final.json']:
            z.write(D/p,p)
        z.writestr('README.txt','Requires Python with NumPy, SciPy, Matplotlib.\npython expanded.py\npython finish.py\npython report.py\n')
    links=[
        ('80条件の比較 CSV','expanded_budget_comparison.csv','text/csv',csvbytes(csvrows)),
        ('全配分の CSV','expanded_by_allocation.csv','text/csv',csvbytes(allr,['N','L','allocation','eta','R','F_eff','GHZ_R','GHZ_F_eff','axes','weights','source','Qbar','verification_error'])),
        ('再計算コード・全回路 JSON','expanded_weighted_allocation.zip','application/zip',src.getvalue())]
    footer=''.join(f'<a download="{n}" href="data:{m};base64,{b64(x)}">{s}</a>' for s,n,m,x in links)
    css='*{box-sizing:border-box}body{margin:0;background:#eef4f5;color:#162f38;font:16px/1.65 system-ui,sans-serif}main{max-width:1140px;margin:auto;padding:24px}header,section{background:#fff;border:1px solid #d8e4e8;border-radius:12px;padding:23px;margin-bottom:15px}h1,h2{margin:0 0 10px;line-height:1.3}p{margin:0 0 8px}img{width:100%;height:auto}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;white-space:nowrap;font-variant-numeric:tabular-nums}td,th{padding:7px 9px;border-bottom:1px solid #dce7ea;text-align:right}th{background:#e6f0f0}a{color:#166378;margin-right:16px;display:inline-block}'
    dest=D/'expanded_weighted_allocation_N2_5_L8.html'
    dest.write_text(f'''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>2〜5ノード・総8光子の配分最適化</title><style>{css}</style><main>
<header><h1>重み付きグラフの光子配分</h1><p>2・3・4・5ノード、総8光子以下、透過率 η = 0.05〜1.00（0.05刻み）。生成光子あたりの効率 R と、1試行あたりの有効QFIを比較。</p></header>
<section><h2>η=0.50 の比較</h2><div class="scroll"><table><thead><tr><th>ノード</th><th>η</th><th>Rの配分</th><th>R（上限8）</th><th>R（上限6）</th><th>R改善</th><th>GHZ分割R</th><th>Fの配分</th></tr></thead><tbody>{''.join(high)}</tbody></table></div></section>
<section><h2>透過率ごとの性能</h2><img alt="ノード数別の光子効率 R" src="data:image/png;base64,{b64(buf.getvalue())}"><img alt="左は各ノード1光子を基準にしたRの改善率、右は総6光子以下を基準にした有効QFIの改善率" src="data:image/png;base64,{b64(buf2.getvalue())}"><img alt="透過率0.80までのRの改善率" src="data:image/png;base64,{b64(zoom_png.getvalue())}"></section>
<section><h2>探索と検算</h2><ul><li>全配分×透過率の {d['condition_count']} 条件。ノードの交換だけを同一視し、不均等配分も含む。2・3ノードの総6光子以下は既存の検算済み最良回路を再利用。</li>
<li>全光子ペアに独立な連続CP重みを許す。配分ごとに二部、多部に相当する群内辺、完全グラフ、ランダムな重み、GHZ分割を初期値に使い、固定局所軸で最適化。光子が失われる位置に応じて軸は変更しない。</li>
<li>重み評価 {d['candidate_weight_evaluations']:,} 回（再利用分を除く）。最良R、最良F、および6光子以下の対照として選んだ計 {d['full_axis_rechecks']} 回路について、固定した重みで全クリフォード軸を照合し、{d['improved_by_full_axis_recheck']} 回路を更新。</li>
<li>各配分の保存回路を、対称性で集約せず、全損失パターンの密度行列を使って検算。Fの最大差は {d['verification_max_F_error']:.2e}。</li>
<li>GHZ対照は、ノードへの光子配分、GHZブロックへの分割、位相符号を全列挙。ここでは総8光子以下の通常CZグラフ全体に対する上界とは主張しない。</li>
<li>評価は損失パターンの平均QFIMから求めるノード位相の算術平均。有効QFI F = 1 / (wᵀ Q⁺ w)、w = 1/N、R = F/L。特定測定での達成精度ではない。連続重みの大域最適性および最終候補以外の軸全探索は保証しない。</li></ul></section>
<section><h2>全80条件の最良発見値</h2><p>同率候補からは総光子数の少ない配分を代表に表示。全同率配分はCSVに記録。</p><div class="scroll"><table><thead><tr><th>ノード</th><th>η</th><th>Rの配分</th><th>R（上限8）</th><th>R（上限6）</th><th>R改善</th><th>GHZ分割R</th><th>Fの配分</th><th>F（上限8）</th></tr></thead><tbody>{''.join(table)}</tbody></table></div></section><footer>{footer}</footer></main></html>''',encoding='utf-8')
    print(dest,dest.stat().st_size)
if __name__=='__main__':make()
