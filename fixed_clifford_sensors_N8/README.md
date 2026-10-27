# 8センサ：グラフと固定局所Cliffordの全探索

6・7センサと同じモデル・同じスタビライザーQFIMカーネルを8センサへ拡張する。
補助なし、各地点の独立な未知位相の算術平均、共通生存確率eta、全試行を評価する。
**結果はQFIMから得る精度限界であり、具体的な読み出し測定のCFI最適化ではない。**

- [8センサの結果と6・7センサとの比較](results/report.md)
- [8センサの最良値CSV](results/best_by_n_eta.csv)
- [採用したグラフ・局所ゲート・QFIM](results/protocols.json)
- [独立検算](results/validation.json)
- [以前の6・7センサの結果](../fixed_clifford_sensors_N6_7/results/report.md)

## 公開結果

182代表で、孤立センサの感度ゼロ軸を除いた1,010,389個の符号なし軸配置を、
グラフの配置対称性で417,105個に削減した。全代表を評価し、
生存数集計QFIMの異なる多項式は13,109種類だった。
eta=0.05,0.10,...,1の全20評価点で、最良値はGHZブロック直積の最良値と有理数で一致した。

|eta|同じ最適値を達成するGHZブロック分割|独立センサに対する有効情報量比|
|---:|---|---:|
|0.05〜0.50（0.05刻み）|1+1+1+1+1+1+1+1|1|
|0.55・0.60・0.65|2+2+2+2|それぞれ1.1・1.2・1.3|
|0.70|2+3+3|1.451851852|
|0.75|4+4|1.6875|
|0.80|4+4|2.048|
|0.85・0.90・0.95・1|8|それぞれ2.564616706・3.8263752・5.586698369・8|

数字はGHZのサイズ。1は独立センサ、2はベル対。
同率最適はあり得る。倍率は有効情報量の比であり、標準偏差の改善倍率ではない。
主探索の経過時間はsummary.jsonに保存する。ホスト上の他処理の影響も含み、
専用ベンチマークによる速度保証ではない。
公開実行は起動が重複した時間帯があり、記録された約449.6秒は処理の競合を含む。
6・7センサの単独実行との速度比を計算するためのベンチマーク値ではない。

## 再実行

Python 3.10以降。researchリポジトリ全体を取得して実行する。
`../fixed_clifford_sensors/engine.py` のスタビライザー計算を再利用する。
分類データは同梱し、実行時のダウンロードは不要。

```bash
cd fixed_clifford_sensors_N8
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python run_search.py --output my_results
python verify.py --results my_results
python make_report.py --results my_results
```

保存済みのresultsを置換するときは `--overwrite` を明示する。
公開実行は `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` を設定し、小さい行列での過剰な並列化を避ける。
既定の生存確率はeta=0.05,0.10,...,1の20点。連続区間の切替境界を決定した結果ではない。

## 手動実行時に変更できるパラメータ

|引数|既定値|内容|
|---|---|---|
|`--min-sensors`|8|最小センサ数|
|`--max-sensors`|8|最大センサ数。対応範囲1〜8|
|`--eta-min`|0.05|生存確率の始点|
|`--eta-max`|1|生存確率の終点|
|`--eta-points`|20|両端を含む点数。1なら始点のみ|
|`--etas`|未指定|カンマ区切りで指定。グリッド指定より優先|
|`--axis-reduction`|symmetry|`none`でグラフ自己同型による配置削減を無効化|
|`--tie-tol`|1e-10|有理数で再比較する近接候補幅|
|`--output`|results|保存先|
|`--overwrite`|なし|既存出力を置換|
|`--log-every-graphs`|1|何グラフごとに進捗を表示するか。0で非表示|

```bash
# 任意の生存確率。全損失と無損失の極限も指定できる
python run_search.py --etas 0,0.2,0.5,0.65,0.8,1 --output selected_points

# etaの評価点を増やす
python run_search.py --eta-min 0.025 --eta-points 40 --output fine_grid

# グラフ配置対称性を使わずに照合
python run_search.py --axis-reduction none --output no_symmetry
python verify.py --results my_results --reference-results no_symmetry

# 6・7センサの再計算
python run_search.py --min-sensors 6 --max-sensors 7 --output rerun_6_7
python verify.py --results rerun_6_7 --reference-results ../fixed_clifford_sensors_N6_7/results
```

`verify.py --seed` はランダムな符号付き混合状態の独立検算の乱数seed。
既定20261021。全符号照合を省略する `--skip-sign-enumeration` も利用でき、
省略した場合はvalidation.jsonのsign_checksが空になる。公開結果では省略しない。
`make_report.py --comparison-results` で、比較する既存6・7センサ結果の場所を変更できる。
比較表で存在しない評価点は「—」と表示する。

## モデルと評価量

1. 非連結グラフと孤立頂点を含む、全ての単純グラフ状態を候補とする。
2. 各センサに固定局所Cliffordを作用させる。入力は損失パターンに適応しない。
3. 各センサを独立に確率1−etaで部分トレースする。
4. 生存センサにexp(−i theta_i Z_i/2)で独立な未知位相を付与する。
5. 最終的に損失位置の記録を利用する。全損失試行も含む。

Q(eta)は各損失パターンのQFIMの確率平均。w=ones/Nとして、
平均位相の1試行あたりQCRBはC_Q=w^T Q^+ w。
wがQの像にない場合は識別不能とし、情報量を0とする。
最大化するのは生成光子あたり有効情報R=1/(N C_Q)。
独立センサはR=eta、全NセンサGHZはR=N eta^N。
nu試行後の分散下限は1/(nu N R)。

## 全探索と削減

8頂点の連結LC代表101個に、小さい連結成分の全多重集合を加え、
非連結を含む全182代表を生成する。LC変換は局所Cliffordに吸収し、
共通eta・等重み平均のため番号交換もまとめる。

各Cliffordは有効位相軸C_i† Z_i C_i∈{±X,±Y,±Z}に置き換える。
単純な符号付き列挙は182×6^8=305,690,112配置。
純粋スタビライザーのQFIMが符号付きrank-one相関ブロックに分解される性質を使い、
平均位相を識別できる符号を先にそろえる。孤立センサの感度ゼロ軸は除く。
グラフ自己同型で軸配置の軌道を取り、個数と被覆をBurnsideの補題でも照合する。

損失後のスタビライザー状態は、射影Piとランクrを使ってrho=Pi/rと書ける。
混合状態の固有値によるQFIM公式は、局所生成子P_i/2に対して

```text
Q_ij = Re Tr[Pi P_i (I-Pi) P_j] / r
```

となる。保持スタビライザーと反交換する軸に対角感度があり、
感度のある2軸の交差成分は符号付き二体Pauli期待値で求まる。
混合状態に純粋状態の4Covをそのまま適用していない。

各グラフで3^8軸×2^8損失の整数QFIMを一時的にまとめて計算する。
主配列は107,495,424 byteで、全グラフ分を同時に保持しない。
その後、生存数ごとのA_k=sum_{|S|=k}Q_Sだけに集約する。
相関ブロックの大きさとA_k係数の多重集合が一致すれば、全etaで同じ平均CRBなので、
情報多項式として重複をまとめる。

GHZ直積の候補だけに探索を制限したり、GHZ上界で候補を打ち切ったりしない。
全軌道代表を評価し、整数分割の全列挙によるGHZ直積の最良値とも別に照合する。
各相関ブロックの活動度がeta^(ブロックサイズ)以下であることも整数係数で確認する。

## 独立検算と精度

- 全代表・全符号なし軸について、損失なしQFIMをグラフ状態ベクトルから計算する。
- 全305,690,112符号付き配置で、平均の識別可能性を固有空間への射影から判定し、符号削減と照合する。
- 全符号なし軸と全損失で、相関ブロック外に新しい相関が発生しないことを確認する。
- 各代表のランダムな符号付き入力で、実際のCliffordゲート・部分トレース・混合状態の固有値公式を用いてQFIMを照合する。
- 保存した全採用構成の全256損失パターンを密度行列で検算する。
- 不均一で非ゼロの局所位相でもQFIMが一致することを確認する。
- 保存した最良値を有理数で再計算する。全情報多項式は通常の擬似逆行列による評価とも照合する。

有理数比較の前に浮動小数点で近接候補を選別するため、区間演算による誤差保証ではない。
極端に小さいetaでは必要に応じて高精度計算へ拡張する。
任意角度の局所回転、入力設定の確率混合、補助、生成失敗・ゲート誤差・辺のコスト、
センサ間で異なる透過率、具体的な読み出し測定の最適化は対象外。

## 出典と保存物

分類データ：L. E. Danielsen and M. G. Parker,
“On the classification of all self-dual additive codes over GF(4) of length up to 12,”
J. Combin. Theory Ser. A 113, 1351–1367 (2006).
https://www.codetables.de/larsed/vncorbits/

`data/sources.json` にURL・連結クラス数・SHA-256を保存する。
`results/summary.json` に探索数・計算時間・環境・実行時ソースのハッシュ、
`manifest.json` に公開ファイルのSHA-256を保存する。
`results/best_by_graph.csv` は各代表での最良値、
`results/protocols.json` は採用グラフ・有効軸・物理ゲート・A_kを収録する。
センサ番号は0始まり、テンソル順ではセンサ0が左端。
ゲート積は右から作用する。例えばHSdgはS†の後にH。

## 採用グラフの図

[重複を除いた構造一覧と透過率・ゲートの対応表](results/graphs.md) ／ [グラフ図](results/graphs_unique.svg)

同じ頂点数で番号交換により一致する構造は1枚に集約する。全ての辺はCZ（重みπ）。
各図の下は頂点番号順の局所Cliffordで、ゲート積は右から作用する。
図は保存済み評価点の採用代表解を示し、全同率解や評価点間の最適性を意味しない。
詳細な辺・ゲート変種・元の番号との対応は[graph_structures.json](results/graph_structures.json)に保存する。

図だけを再生成する場合（researchリポジトリのルートから実行、最適化は再実行しない）：

```bash
python fixed_clifford_sensors/make_graphs.py --results fixed_clifford_sensors_N8/results --columns 3
```

`make_report.py`でも同じ図と対応表を再生成する。描画列数は上の`--columns`で変更できる。


