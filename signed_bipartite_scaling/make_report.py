"""Build static scientific figures and a self-contained downloadable report."""
import argparse,base64,csv,html,io,json,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm,ListedColormap

def table(head,rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in r)+'</tr>' for r in rows)+'</tbody></table>'

def intervals(rows,key):
    groups=[]
    for r in rows:
        value=key(r)
        if groups and groups[-1]['value']==value:groups[-1]['end']=r['eta']
        else:groups.append({'start':r['eta'],'end':r['eta'],'value':value})
    return groups

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,default=Path(__file__).with_name('results'));args=p.parse_args()
    out=args.results;base=Path(__file__).parent
    rows=json.loads((out/'best.json').read_text());ns=sorted({r['n'] for r in rows});etas=sorted({r['eta'] for r in rows})
    byn={n:sorted([r for r in rows if r['n']==n],key=lambda r:r['eta']) for n in ns}
    summary=[]
    for n,rs in byn.items():
        peak=max(rs,key=lambda r:r['weighted_over_GHZ_percent'])
        summary.append({'n':n,'topology_intervals':intervals(rs,lambda r:r['canonical_part_a']),'K2_best_etas':[r['eta'] for r in rs if r['canonical_part_a']==2],'strict_GHZ_wins':sum(r['R']>r['R_GHZ_partition']+1e-9 for r in rs),'GHZ_ties':sum(abs(r['R']-r['R_GHZ_partition'])<=1e-9 for r in rs),'GHZ_losses':sum(r['R']<r['R_GHZ_partition']-1e-9 for r in rs),'max_gain_percent':peak['weighted_over_GHZ_percent'],'max_gain_eta':peak['eta']})
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    selected=[n for n in [8,12,16,20] if n in byn]
    fig,axs=plt.subplots(2,2,figsize=(12,8),squeeze=False)
    series=[('R','Best signed bipartite family','#d76422','-'),('R_original_K2_pm','Original K2,+/-phi probe','#2974ad','--'),('R_GHZ_partition','Optimized GHZ partition','#7654a6','-.'),('eta','Independent','#777777',':')]
    for ax,n in zip(axs.flat,selected):
        for key,label,color,style in series:ax.plot(etas,[r[key] for r in byn[n]],style,color=color,label=label,lw=1.8,marker='o',markersize=2.5)
        ax.set(title=f'N = {n}',xlabel='Survival probability eta',ylabel='Information per generated photon R')
        ax.grid(alpha=.22);ax.legend(fontsize=7.5);ax.set_xlim(.025,1.025)
    fig.suptitle('Mean of independent phases; fixed inputs; all loss records included')
    fig.tight_layout();fig.savefig(out/'information.svg');fig.savefig(out/'information.png',dpi=160);plt.close(fig)
    a=np.array([[r['canonical_part_a'] for r in byn[n]] for n in ns])
    gain=np.array([[r['weighted_over_GHZ_percent'] for r in byn[n]] for n in ns])
    cmap=ListedColormap(plt.get_cmap('tab10').colors);norm=BoundaryNorm(np.arange(.5,11.5),10)
    fig,axs=plt.subplots(2,1,figsize=(13,9),layout='constrained')
    im=axs[0].imshow(a,cmap=cmap,norm=norm,aspect='auto',origin='upper')
    cbar=fig.colorbar(im,ax=axs[0],ticks=range(1,11),pad=.015);cbar.set_label('Smaller part size a')
    im=axs[1].imshow(np.maximum(0,gain),cmap='YlOrRd',aspect='auto',origin='upper',vmin=0)
    fig.colorbar(im,ax=axs[1],pad=.015).set_label('Gain over best GHZ partition (%)')
    for index,ax in enumerate(axs):
        ax.set_xticks(range(len(etas)),[f'{x:.2f}' for x in etas],fontsize=8)
        ax.set_yticks(range(len(ns)),[str(n) for n in ns]);ax.set(xlabel='Survival probability eta',ylabel='Number of sensors N')
        for i in range(len(ns)):
            for j in range(len(etas)):
                value=str(a[i,j]) if index==0 else f'{max(0,gain[i,j]):.1f}'
                color='white' if index==1 and gain[i,j]>26 else '#172436'
                ax.text(j,i,value,ha='center',va='center',fontsize=7,color=color)
    axs[0].set_title('Best-found topology K(a, N-a): 1 = star/GHZ, 2 = K(2, N-2)')
    axs[1].set_title('Relative information gain; GHZ partition optimized separately at every point')
    fig.savefig(out/'topology_and_gain.svg');fig.savefig(out/'topology_and_gain.png',dpi=160);plt.close(fig)
    extended=json.loads((out/'extended_validation.json').read_text())
    validation=json.loads((base/'validation.json').read_text())
    summaryrows=[]
    for s in summary:
        n=s['n'];rs=byn[n];pick=lambda eta:next(r for r in rs if r['eta']==eta)
        def topology(r):return f"K({r['canonical_part_a']},{n-r['canonical_part_a']})"
        summaryrows.append([n,topology(pick(.2)),topology(pick(.5)),topology(pick(.8)),f"{s['max_gain_percent']:.3f}%",s['strict_GHZ_wins'],s['GHZ_ties']])
    parts=['<!doctype html><html lang="ja"><meta charset="utf-8"><title>8〜20センサ：符号付き二部グラフの重み最適化</title><style>body{font-family:system-ui,sans-serif;max-width:1240px;margin:36px auto;padding:0 20px;line-height:1.8;color:#172436}table{border-collapse:collapse;width:100%;font-size:13px;margin:18px 0}th,td{border:1px solid #d7dfeb;padding:6px;text-align:right}th{background:#eef3f8}svg{width:100%;height:auto}pre{white-space:pre-wrap;font-size:13px}details{margin:16px 0}a{color:#1763a4}</style>',
    '<h1>8〜20センサ：符号付き完全二部グラフの重み最適化</h1>',
    '<p>8〜20センサの全13サイズ、透過率0.05〜1.00の全20点、合計260点。結論は、探索した状態族の中でも K₂,ₙ₋₂ が常に最良ではなく、低透過率では4個以上のハブが有利になること。中〜高透過率ではハブ2個、高透過率では星型の通常GHZ状態が選ばれた。各サイズで最大改善率は透過率0.50にあり、GHZ最良分割に対してN=8で28.092%、N=20で42.797%。</p>',
    '<p><strong>探索範囲：</strong>K(h,N−h) の全二部サイズ、各ハブの辺の符号が一様な ±φ、負のハブ数q、両部をハブとする選択、部内対称性を保つ固定局所クリフォードを比較。すべての一般グラフ、独立な全辺重み、部内で異なる全クリフォード配置を探索した結果ではない。重みの連続最適化も数値的な最良発見値であり、大域最適性の証明ではない。</p>',
    '<h2>代表的な構造とGHZ比較</h2>',table(['N','η=0.20の構造','η=0.50の構造','η=0.80の構造','最大GHZ改善率','GHZより高い点数','同値の点数'],summaryrows),
    '<h2>全260点の構造と改善率</h2>','<svg'+(out/'topology_and_gain.svg').read_text().split('<svg',1)[1],
    '<p>上図は同じグラフ K(a,N−a) を a≤N−a で表記する。実際にどちらの部をハブにしたか、正負の個数、軸は下表とJSONに記録している。高透過率で a=1 の星型は重みπの通常GHZ状態。改善率の絶対値1e−9以下は同値と判定し、描画では丸め誤差の負値を0にしている。</p>',
    '<h2>情報量の曲線</h2>','<svg'+(out/'information.svg').read_text().split('<svg',1)[1],
    '<p>オレンジは今回の状態族の最良値、青は元の「ハブ2個、片方+φ・片方−φ、軸+Z/−Z、葉−Y」だけを重み最適化した値、紫は全整数GHZ分割の最良値、灰色は独立センサ。入力設定の確率混合は用いない。</p>',
    '<h2>目的関数と損失の扱い</h2><p>各光子を独立な未知位相θᵢのセンサとし、g=Σᵢθᵢ/Nを推定する。入力の重み・クリフォードは損失前に固定。生存位置は記録するが、入力をその記録に応じて変えない。全損失パターンの混合状態SLD-QFIMを先に平均して Q̄=ΣₛpₛQₛ とする。w=(1,…,1)/N に対し R=1/[N wᵀQ̄⁺w]。wがQ̄の像にないときR=0。分母は損失光子を含む生成N光子で、事後選別による再正規化は行わない。</p>',
    '<p>GHZ分割はブロックサイズkⱼに対して R=N/[Σⱼη⁻ᵏʲ]。サイズの合計Nという条件の全整数分割を動的計画法で最適化した。単独光子のブロックも許すため、独立センサもGHZ対照に含まれる。</p>',
    '<h2>探索方法と計算の削減</h2><p>φは0〜πの129点を走査し、各二部サイズ・透過率について上位4個までの局所ピーク区間を有界1次元最適化で精密化した。各φで許す符号数と固定軸をすべて比較する。重みごとに最良軸が切り替わるため、目的関数は滑らかとは限らない。</p>',
    '<p>符号付き状態を局所ユニタリで数学的に全正符号の二部グラフへ変換し、各部の置換対称性を用いる。残存光子数k,lの縮約状態を、各部のディッケ基底で厳密に表すと次元は(k+1)(l+1)。部分トレースを表す振幅行列を薄い特異値分解にかけ、一般混合状態のQFI公式を評価する。純粋状態の分散公式を混合状態に流用していない。追加の物理的な非クリフォード回転を許したわけでもない。</p>',
    '<p>各部の集合的XYZ生成子の6×6QFIテンソルから、選んだ2部の軸について2×2QFIM B を作る。平均後の置換対称性により、部内の位相差方向は全体平均方向と分離するため、u=(h/N,(N−h)/N) として R=1/[N uᵀB⁺u] が元の独立N位相問題と一致する。全位相が同じという仮定ではない。</p>',
    '<h2>検証</h2><p>5〜7センサの一般の軸・符号例、保存済みN=5の最良構成、星型GHZ端点を、通常の全計算基底密度行列・位相微分から独立照合した。さらにN=8のη=0.20と0.70の採用構成を全256損失パターンで照合。GHZ分割計算は以前の8センサ全探索結果とも一致した。</p><pre>'+html.escape(json.dumps(extended,indent=2))+'</pre>',
    '<p>65点・上位2ピークの初回探索と129点・上位4ピークの再探索は、N=8〜12の全二部サイズ・全透過率で丸め誤差内に一致した。これらは数値整合性の確認であり、連続重みの大域最適性の証明ではない。</p>',
    '<h2>全構成の表</h2><p>hは実際に符号を割り当てたハブ数、qは負のハブ数。正のハブはh−q個。ハブ軸は正のハブの軸であり、負のハブではXによる共役（X→X、Y→−Y、Z→−Z）を用いる。葉軸は全葉で同じ。</p>']
    for n,rs in byn.items():
        parts.append(f'<details><summary>{n}センサ・全20点</summary>')
        parts.append(table(['η','構造','h','q','φ [rad]','ハブ軸','葉軸','R','元のK₂ R','GHZ最良 R','GHZ分割','改善率'],[[f"{r['eta']:.2f}",f"K({r['canonical_part_a']},{n-r['canonical_part_a']})",r['hubs'],r['negative_hubs'],f"{r['phi']:.9f}",r['hub_axis'],r['leaf_axis'],f"{r['R']:.9f}",f"{r['R_original_K2_pm']:.9f}",f"{r['R_GHZ_partition']:.9f}",'+'.join(map(str,r['GHZ_partition'])),f"{r['weighted_over_GHZ_percent']:+.4f}%"] for r in rs]))
        parts.append('</details>')
    parts.append('<h2>コードとデータ</h2><p>一式ZIPには全88二部サイズの探索記録、最良260構成、図、CSV、JSON、検証結果、実行可能コードとREADMEを含む。探索条件はコマンド引数で変更できる。</p><ul>')
    archive=out/'signed_bipartite_N8_20_code_and_results.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name in ['search.py','verify.py','validate_results.py','make_report.py','README.md','METHODS.md','requirements.txt','validation.json']:z.write(base/name,name)
        for file in sorted(out.glob('*')):
            if file.is_file() and file.suffix in {'.json','.csv','.svg','.png'}:z.write(file,'results/'+file.name)
    for file in [archive,out/'best.csv',out/'best.json',base/'README.md']:
        encoded=base64.b64encode(file.read_bytes()).decode()
        parts.append(f'<li><a download="{file.name}" href="data:application/octet-stream;base64,{encoded}">{file.name}</a></li>')
    parts.append('</ul></html>')
    report=out/'signed_bipartite_report.html';report.write_text(''.join(parts),encoding='utf-8')
    print(json.dumps({'rows':len(rows),'families':extended['family_files'],'report':str(report),'report_bytes':report.stat().st_size,'summary':summary},indent=2))

if __name__=='__main__':main()
