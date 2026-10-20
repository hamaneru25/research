"""Create Japanese notes, scientific figures, and an artifact hash manifest."""
from fractions import Fraction
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from search import ROOT,ETAS,references
from record_io import pack_graph_records


def main():
    pack_graph_records(ROOT)
    s=json.loads((ROOT/'results/summary.json').read_text());v=json.loads((ROOT/'results/validation.json').read_text())
    uv=json.loads((ROOT/'results/upper_validation.json').read_text());rows=s['best_by_eta']
    m4=s['graph_counts']['4'];total=sum(s['graph_counts'].values());runs=s['local_runs']
    x=[float(Fraction(r['eta'])) for r in rows];y=[float(Fraction(r['exact_value'])) for r in rows]
    baseline=[float(max(r[2] for r in ref)) for ref in references()]
    upper=[float(Fraction(r['global_upper'])) for r in rows]
    fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    ax[0].plot(x,baseline,'--',label='Best of three reference families',color='#999999')
    ax[0].plot(x,y,'o-',label='Best achieved in this search',color='#236fa1')
    ax[0].plot(x,upper,'s:',label='Global upper bound',color='#bf7928')
    ax[0].set_ylabel('Effective QFI / generated photon');ax[0].legend(fontsize=8)
    ax[1].scatter(x,[r['m'] for r in rows],s=65,color='#236fa1')
    certified=[i for i,r in enumerate(rows) if r['global_optimum_certified']]
    ax[1].scatter(np.array(x)[certified],[rows[i]['m'] for i in certified],s=140,facecolors='none',edgecolors='#267c42',label='Global optimum certified')
    ax[1].set_ylabel('Auxiliary count of best found policy');ax[1].set_yticks(range(7));ax[1].legend(fontsize=8)
    for a in ax:a.set_xlabel('Sensor transmission eta');a.grid(alpha=.25)
    (ROOT/'figures').mkdir(exist_ok=True);fig.savefig(ROOT/'figures/performance.png',dpi=170);plt.close(fig)
    one_aux=s['best_found_by_m']['1'][7];g=nx.Graph();g.add_nodes_from(range(7));g.add_edges_from(one_aux['edges'])
    pos={6:(0,1.5),4:(-1,0.5),5:(1,0.5),0:(-1.6,-.7),3:(-.4,-.7),1:(.4,-.7),2:(1.6,-.7)}
    fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
    nx.draw_networkx_edges(g,pos,ax=ax,width=2,edge_color='#666666')
    nx.draw_networkx_nodes(g,pos,nodelist=list(range(6)),node_size=1050,node_color='#3384b5',ax=ax)
    nx.draw_networkx_nodes(g,pos,nodelist=[6],node_size=1100,node_shape='s',node_color='#e6a13a',ax=ax)
    nx.draw_networkx_labels(g,pos,labels={i:f'S{i}' for i in range(6)}|{6:'A0'},font_color='white',font_size=12,ax=ax)
    ax.set_title('N=6, M=1: feasible improvement at eta=0.8');ax.axis('off')
    fig.savefig(ROOT/'figures/eta08_graph.png',dpi=170);plt.close(fig)
    steps=np.array([[next(r['accepted_moves'] for r in runs if r['start']==seed and r['eta']==str(eta)) for eta in ETAS[1:8]] for seed in range(len(runs)//7)])
    fig,ax=plt.subplots(figsize=(7,5),layout='constrained')
    im=ax.imshow(steps,cmap='Blues',aspect='auto',vmin=0)
    for i in range(steps.shape[0]):
        for j in range(steps.shape[1]):ax.text(j,i,str(steps[i,j]),ha='center',va='center',color='white' if steps[i,j]>steps.max()/2 else 'black')
    ax.set_xticks(range(7),[f'{float(e):.1f}' for e in ETAS[1:8]])
    ax.set_yticks(range(steps.shape[0]),[str(i+1) for i in range(steps.shape[0])])
    ax.set_xlabel('Sensor transmission eta');ax.set_ylabel('Initial graph number');ax.set_title('Accepted graph updates before no improving neighbor remained')
    fig.colorbar(im,ax=ax,label='Accepted one-edge updates')
    fig.savefig(ROOT/'figures/convergence.png',dpi=170);plt.close(fig)
    table=['|透過率|最良の補助数|達成値／光子|全候補への上界|最適性の保証|','|---:|---:|---:|---:|---|']
    for r in rows:
        table.append(f"|{float(Fraction(r['eta'])):.1f}|{r['m']}|{float(Fraction(r['exact_value'])):.6f}|{float(Fraction(r['global_upper'])):.6f}|{'保証あり' if r['global_optimum_certified'] else '未保証'}|")
    improvement=100*(float(Fraction(one_aux['exact_value']))/baseline[7]-1)
    text=r'''# 6センサ限定：確定的XYZ測定と上界付きグラフ探索

対象は**6センサのみ**。補助M=0〜6、透過率eta=0.1,…,1.0の10点。
設定の確率混合・選択確率の最適化は使わない。7センサ以上は計算していない。

今回の更新では、旧「グラフ更新3回」の打ち切りを撤廃。STARTCOUNT初期グラフ×7透過率のWALKCOUNT経路を、1辺変更で改善がなくなるまで実行した。
測定方針の座標更新も固定3巡を撤廃し、1巡して改善がなくなるまで続ける。

## 結果

TABLE

評価量は平均位相推定の有効QFIを全生成光子数で割ったF_eff/(6+M)。大きいほどよい。
**0.1・0.9・1.0の3点は大域最適値を保証。その他7点は今回見つかった最良値であり、最適性は未保証。**
透過率の間を結ぶ線は表示用で、連続区間の保証はない。

![達成値・上界・補助数](figures/performance.png)

補助数別の最良発見値はbest_found_by_m.csv、全体の最良値は上表に記録した。
eta=0.8では**補助1個**の構成が、補助5個の2.181818…からONEAUXVALUEへIMPROVEMENT%改善することを確認した。
「補助0個かN−1個しか有利にならない」という一般則は成立しない。
ただし、この補助1個構成が全グラフで最適だと証明したわけではない。

![透過率0.8で改善を示す補助1個の構成](figures/eta08_graph.png)

センサ頂点は0〜5、補助頂点は6。
辺は(0,4),(1,5),(2,5),(3,4),(4,6),(5,6)。
補助A0がセンサS4・S5につながり、S4にS0・S3、S5にS1・S2がつながる木構造。
この構成も消失位置に応じて補助の基底を切り替え、結果に応じて局所Cliffordを行う。
具体的方針IDは`ONEAUXID`。CSVから方針を参照できる。

## どこまで探索したか

|補助数M|評価したグラフ|網羅性|
|---:|---:|---|
|0|26|役割付きLC代表を全列挙|
|1|115|役割付きLC代表を全列挙|
|2|964|役割付きLC代表を全列挙|
|3|10,946|役割付きLC代表を全列挙|
|4|M4COUNT|改善停止まで局所探索。通常の役割付きグラフ同型で重複除去|
|5|解析的構成|センサ数固定の上限U=6eta+30eta²を達成|
|6|解析的構成|M=5構成に独立補助を1個追加。情報は同じでコスト増|

最終実行で合計TOTALCOUNTグラフを評価。M=0〜3は12,051代表。
非連結も含み、補助だけの連結成分は、除去して補助数を減らす方が資源当たりで有利なので比較から除外した。
**グラフの全列挙と、各グラフの全確定的測定方針の最適化は別。** M=0〜3でも各M固定の最適達成値をすべて保証したわけではない。

M=4のM4COUNT候補には通常の同型を超えたLC重複が残る可能性があり、LC代表の網羅率としては扱えない。
全232,691役割付きLC代表に対して換算上限でもCOVERAGE%の範囲であり、未探索部分が残る。
best_found_by_m.csvは各補助数の「最良発見値」であって、全てが最適値とは限らない。

## 探索アルゴリズム

1. 独立、補助なし完全グラフ、補助5個の構成を、実現可能な比較基準にする。
2. M=0〜3は公開LC分類に役割を付け、役割を保つLC・番号交換をまとめた全代表を評価する。
3. 各グラフの全3^M補助XYZ基底を扱う。安定化子から63個の非空生存集合に対する上界を求める。
4. M=0〜3は上界が全体の既知最良値を上回る場合に、確定的な損失別方針を座標更新する。M=4の局所探索では、全体の最良値に届かない候補も探索対象etaで方針を更新してから比較する。
5. M=4はSTARTCOUNT初期構造それぞれをeta=0.2〜0.8の7点で探索し、計WALKCOUNT本の経路を実行。毎回全45種類の1辺反転を調べ、最大の改善候補へ移動する。改善量が1e-10以下になったら終了する。グラフ更新回数に上限を設けない。
6. 初期構造は独立、完全グラフ、補助削減構成、星、サイクル、ブロック積、密度を変えた乱数グラフ。乱数seed=20261020。

グラフ探索中の乱数は候補生成用であり、実験設定の確率混合を導入しない。
測定方針の座標探索は、損失条件ごとに実行可能なQFIMを1つ選び直して、平均QFIMの有効情報量を改善する。
既定では2初期方針それぞれについて、全損失条件を1巡して改善がなくなるまで続ける。改善判定はF_effで1e-11。方針の巡回数にも上限を設けない。局所停止であり、方針の大域最適性は未保証。

## 改善停止の確認

以前の3段階での打ち切りを撤廃した。改善しないときの2辺ランダム反転は行わず、その出発点・etaの探索を終了する。
各経路の終端で45近傍すべてを評価済みであることと、同じ測定方針評価器で改善する近傍がないことを検証する。
計WALKCOUNT経路、終端近傍NEIGHBORCOUNT件を照合。受理した更新回数の最小はMINMOVES、最大はMAXMOVES。
この収束は、採用した方針候補・数値許容誤差のもとでの1辺変更に関する局所停止であり、全グラフ最適性ではない。

![初期値・透過率ごとの更新回数](figures/convergence.png)

results/local_convergence.csvに停止理由・更新回数・終端値・最良近傍値、summary.jsonに45近傍の全IDを保存した。
test_convergence.pyでは5回の改善を要する人工的な評価関数を使い、旧3回制限を超えて終了条件まで進めることを検査する。

達成値の探索には、各補助基底で実現できる正のGHZ型最大相関ブロックのQFIMを使う。
ブロックを意図的に壊す軸・負の相関を含む全方針は総当たりしていない。
一方、以下の上界は全センサ局所Cliffordを含む。
再利用は安定化子状態ごとの計算キャッシュ。M=4の局所探索に完全なLC正規化や、未評価の損失条件だけを使う途中打ち切りは実装していない。

## 上界と保証範囲

各グラフGについて

\[
B_G(\eta)=\frac{1}{6+M}\sum_{S\ne\varnothing}\eta^{|S|}(1-\eta)^{6-|S|}
\max_{b,C}\mathbf1^TQ_G(S,b,C)\mathbf1
\]

を上界とする。F_eff≤1^T F 1を使っており、一般にB_Gが達成可能とは限らない。
maxの対象は全XYZ補助基底と全センサ局所Clifford。整数の相関支持数を係数として、有理数で評価する。

M=0〜3は全代表のB_Gの最大を取る。未探索M=4についてはU/(6+4)、M=5,6はU/(6+M)を使う。
これらの最大が表の「全候補への上界」。探索済み候補だけの上界ではなく、未探索M=4も含む。
達成値も、方針が固定された後に平均QFIMを有理数で構築し、連立方程式を解いて厳密に計算した。
上下が一致した3点だけを最適性保証ありとする。

eta=0.2〜0.7では真の最適値は今回達成値の1〜1.10倍の範囲にある。
eta=0.8ではETAEIGHTLOWER以上、ETAEIGHTUPPER以下。中間の7点の保証を完成させるには、主にM=4の上界を狭める必要がある。
同率になる全グラフ・全補助数も未列挙。

## モデル

- 単純無向CZグラフ状態。各センサのみ独立確率etaで生存し、補助は損失なし。
- 損失は未知位相の付与前。消費せずに損失位置を把握できる理想条件。
- 順序は損失把握→補助のXYZ測定→結果依存のセンサ局所Clifford→未知位相付与。
- 各地点の位相は独立に未知。U_i=exp[-i phi_i (I-Z_i)/2]、目標は全6地点の平均。
- w=ones/6、F_eff=1/(w^T F^+ w)。識別不能なら0。
- 全損失・補助結果を記録し、QFIMを先に確率平均。全センサ損失もゼロ情報の試行として含める。
- 生成光子数6+Mで規格化。辺数、生成成功率、補助損失、操作誤差のコストは含めない。
- QFIMの評価。位相付与後の具体的読み出し測定・CFIの最適化ではない。

## 独立検証と操作表

全出力の最良方針（補助数別70行＋全体10行、重複を除いてPOLICIES方針）を密度行列で検証。
部分トレースでセンサ損失を再現し、非ゼロ補助測定結果CASESケースで、物理的Cliffordゲート後のQFIMを確認した。
最大誤差ERROR。平均QFIMと達成値も有理数で再計算した。
さらにN=6の任意候補UPPERCASES結果で全3^k局所軸・2^(k-1)相対符号を直接列挙し、安定化子の支持上界と比較した。
最大誤差UPPERERROR。

results/deterministic_policies.json.gzに、方針、損失条件ごとの基底、各補助結果の確率、センサCliffordを保存。
頂点0〜5がセンサ、6〜6+M−1が補助。センサゲートはsurvivors順、補助結果は補助番号順。
outcome probabilityはBorn確率で、最適化した設定選択確率ではない。
ゲート積は右から作用し、S=diag(1,i)、Sdg=S†。

|C†ZC|ゲートC|
|---|---|
|+Z|I|
|−Z|X|
|+X|H|
|−X|HZ|
|+Y|HSdg|
|−Y|HS|

全センサ損失時は情報ゼロとして記録し、補助はZ測定。補助結果の詳細表は非空生存集合について保存する。

## 再実行

```bash
cd deterministic_xyz_N6
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python search.py
python verify_export.py
python verify_upper.py
python make_report.py
```

Windowsは仮想環境の有効化方法を変更する。
search.py --restarts 24で初期グラフを増やせる。13個目以降は追加の乱数グラフを生成する。更新回数の指定はなく、改善停止まで実行する。実行結果はresultsを上書きする。
操作表を取り出す例：

```bash
python export_policy.py ONEAUXID --output policy_eta08.json
```

search.pyの最終実行は今回の6センサ用の新規計算で、過去の性能ファイルは読み込んでいない。
既存の安定化子計算コードと公開分類入力は再利用した。
開始・終了UTC、実行パラメータ、開始時ソースSHA-256はresults/provenance.json、最終成果物ハッシュはresults/artifact_manifest.json。
探索・検証ログも同梱。時間計測用ベンチマークは実行していない。

## 主なファイル

- results/best_by_eta.csv：透過率別の最良達成値、方針ID、全候補上界、保証の有無。
- results/best_found_by_m.csv：各補助数の最良発見値。各M固定の最適性は未保証。
- results/summary.json：方針、平均QFIM、探索数、保証範囲。
- results/evaluated_graphs.part*.json.gz：評価したTOTALCOUNTグラフ、辺、上界の整数係数、達成値。800候補ずつ分割して保存。検証コードは分割版も直接読み込める。
- results/deterministic_policies.json.gz：全出力の測定・Clifford操作表。
- results/validation.json、upper_validation.json：独立検証。
- search.py、algebra.py、catalogue.py：探索・安定化子・役割付き分類。
- dense_reference.py、verify_export.py、verify_upper.py：密度行列による検証。

## 分類入力の出典

[Lars Eirik Danielsen: Database of Self-Dual Quantum Codes](https://www.codetables.de/larsed/vncorbits/)

L. E. Danielsen and M. G. Parker, “On the classification of all self-dual additive codes over GF(4) of length up to 12,”
Journal of Combinatorial Theory, Series A 113(7), 1351–1367 (2006).
[arXiv:math/0504522](https://arxiv.org/abs/math/0504522)

入力は1〜9頂点の連結LC代表。非連結とセンサ・補助の役割はコードで構成する。
'''
    replacements={'TABLE':'\n'.join(table),'IMPROVEMENT':f'{improvement:.2f}',
                'ONEAUXID':one_aux['policy_id'],'ONEAUXVALUE':f"{float(Fraction(one_aux['exact_value'])):.9f}",
                'M4COUNT':f'{m4:,}','TOTALCOUNT':f'{total:,}','COVERAGE':f'{100*m4/232691:.2f}',
                'WALKCOUNT':str(len(runs)),'STARTCOUNT':str(len(runs)//7),'NEIGHBORCOUNT':str(45*len(runs)),
                'ETAEIGHTLOWER':f'{y[7]:.9f}','ETAEIGHTUPPER':f'{upper[7]:.9f}',
                'MINMOVES':str(min(r['accepted_moves'] for r in runs)),'MAXMOVES':str(max(r['accepted_moves'] for r in runs)),
                'POLICIES':str(v['unique_policies']),'CASES':str(v['dense_outcome_cases']),
                'UPPERCASES':str(uv['dense_axis_sign_outcomes']),'UPPERERROR':f"{uv['max_error']:.2e}",
                'ERROR':f"{v['max_dense_qfim_error']:.2e}"}
    for a in sorted(replacements,key=len,reverse=True):text=text.replace(a,replacements[a])
    (ROOT/'README.md').write_text(text)
    provenance=json.loads((ROOT/'results/provenance.json').read_text())
    final_sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}
    core=['search.py','algebra.py','catalogue.py']
    assert all(provenance['source_sha256'][name]==final_sources[name] for name in core)
    provenance.update(publication_source_sha256=final_sources,search_core_unchanged_during_run=True)
    (ROOT/'results/provenance.json').write_text(json.dumps(provenance,indent=2))
    files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.name.startswith(('role_catalog_','initial_search')) and p.name not in ['artifact_manifest.json','evaluated_graphs.json.gz']]
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    (ROOT/'results/artifact_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Wrote README, figures, and manifest.')


if __name__=='__main__':main()
