# 6・7センサ：グラフと固定Cliffordの効率的全探索

補助量子ビットなし、各地点の独立な未知位相の算術平均を測る。
グラフと固定局所Cliffordを最適化し、独立損失の全試行を含むQFIMで評価する。
既存のN=1〜5の結果は変更せず、このフォルダにN=6,7の結果を追加する。

**結果はQFIによる性能上限。具体的な読み出し測定・CFIの最適化ではない。**

- [結果・比較図](results/report.md)
- [最良値のCSV](results/best_by_n_eta.csv)
- [採用したグラフ・ゲート・集計QFIM](results/protocols.json)
- [検証結果](results/validation.json)

## 実行方法

Python 3.10以降。researchリポジトリ全体を取得して実行する。
検証済みのQFIカーネル `../fixed_clifford_sensors/engine.py` を再利用する。
分類データは同梱済みで、実行時のダウンロードは不要。

```bash
cd fixed_clifford_sensors_N6_7
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt

python run_search.py --output my_results
python verify.py --results my_results
python make_report.py --results my_results
```

既定は6・7センサ、eta=0.05,0.10,...,1の20点。
既定出力先resultsは公開結果が入っているため、再実行には別の出力先を指定するか、
同じ出力先を置換することを明示して `--overwrite` を使う。

## 変更できるパラメータ

|引数|既定値|内容|
|---|---|---|
|`--min-sensors`|6|最小センサ数|
|`--max-sensors`|7|最大センサ数。対応範囲1〜7|
|`--eta-min`|0.05|生存確率の始点。含む|
|`--eta-max`|1|生存確率の終点。含む|
|`--eta-points`|20|両端を含む点数。1なら始点のみ|
|`--etas`|未指定|カンマ区切りで生存確率を指定。上のグリッド指定より優先|
|`--axis-reduction`|symmetry|`symmetry`は頂点交換で配置をまとめる。`none`はまとめず照合する|
|`--output`|results|出力先|
|`--overwrite`|なし|同名出力ファイルを置換|
|`--tie-tol`|1e-10|有理数で再比較する近接候補幅。max(1,数値最良値)を掛ける|
|`--log-every-graphs`|1|何グラフごとに進捗を出すか。0は無効|

```bash
# 7センサだけ、40点
python run_search.py --min-sensors 7 --max-sensors 7 --eta-min 0.025 --eta-points 40 --output n7_40points

# 指定点だけ。0と1も指定できる
python run_search.py --etas 0,0.2,0.5,0.65,0.8,1 --output selected_points

# 配置対称性を使わない照合計算
python run_search.py --axis-reduction none --output no_symmetry
python verify.py --results my_results --reference-results no_symmetry

# 以前の1〜5センサ計算との照合
python run_search.py --min-sensors 1 --max-sensors 5 --output backward_check
python verify.py --results my_results --reference-results no_symmetry --small-results backward_check
```

verify.pyは標準では全符号照合も行う。保存構成の検算だけを素早く行う場合は
`--skip-sign-enumeration` を指定でき、その省略はvalidation.jsonに空のsign_checksとして残る。
verify.pyとmake_report.pyには `--results` だけ渡せば、元のグリッドを読み取る。
混合状態の追加検算に使う乱数seedはverify.pyの `--seed`（既定20261021）で変更できる。

公開時は小さな行列計算での過剰な並列化を避けるため、OPENBLAS_NUM_THREADS=1、
OMP_NUM_THREADS=1で実行した。これらは実行環境の調整で、探索範囲には影響しない。

## モデルと比較基準

1. 全センサ間のCZグラフ状態を生成する。非連結グラフ、孤立センサも含む。
2. 各センサに固定した局所Cliffordを作用させる。
3. 各センサが独立に確率etaで生存する。損失は部分トレース。
4. 生存センサにexp(-i theta_i Z_i/2)を作用させる。
5. 最終的に生存位置を古典記録として利用する。全損失試行も情報ゼロとして含める。

入力グラフと回転は、同じetaでの全試行で固定。損失パターンを見て入力を変えない。
補助、設定の確率混合、任意角度回転、異なる地点間の透過率差は含めない。
位相は試行間で同じ未知ベクトルとして、局所的な推定誤差を評価する。
この消失モデルでは位相付与と損失の順序を交換できる。

Qは各生存集合のQFIMを確率平均した行列。w=ones/Nとし、
平均位相の1試行当たりQCRBはw^T Q^+ w。wがQの像にない場合は識別不能。
光子当たり有効情報R=1/[N w^T Q^+ w]を最大化する。
独立センサはR=eta、GHZ_NはR=N eta^N。
試行数nuでの平均位相の分散下限は1/(nu N R)。

## 削減の内容

### 1. LC代表と非連結グラフ

同梱するDanielsen分類の連結LC代表と、それらの成分の多重集合を使う。
全Cliffordを許すのでLC変換の違いは入力回転に吸収でき、共通eta・等重み平均では
センサの番号交換も性能を変えない。全代表数はN=6で26、N=7で59。
代表グラフの辺数が最少とは限らない。辺の生成コストは目的関数に含めていない。

### 2. スタビライザーから有効な符号を決める

局所Cliffordは有効な位相軸C_i† Z C_i=±X,±Y,±Zで表す。
まず符号なしX,Y,Zを選び、純粋な損失なしQFIMを求める。
純粋なスタビライザー状態の局所Pauli方向では、Q_iiは0または1、
Q_ijは±P_iP_jがスタビライザーに属する場合に±1、それ以外は0。

感度がある頂点間の非ゼロ二体相関は推移的なので、QFIMは符号付きrank-oneブロックに分かれる。
各ブロックで必要な符号をそろえると、Qはブロックごとの全要素1の行列になる。
ブロック内の相対符号を変えると、正の等重み平均がQの像から外れ識別不能になる。
一方、各ブロックの全符号を同時反転してもQFIMは変わらない。
失われた量子ビットをトレースしても、新しい二体Pauli相関は作られないため、
このブロック全符号反転による等価性は全損失パターンで保持される。

また、位相に依存しない損失チャネルで情報は増えないため、損失なしで平均が識別不能な
符号配置が損失を加えることで有用になることもない。
この議論は、固定入力の現在のモデルについてのもの。異なる設定の確率混合には拡張しない。

孤立した|+>センサの有効X軸は感度ゼロなので除き、Y,Zを残す。
これでN=6の1,213,056符号付き配置は14,761配置、N=7の16,516,224配置は101,693配置になる。

### 3. 軸配置の対称性

各代表グラフの通常の自己同型群で、軸配置の軌道代表を取る。
N=6は4,928配置、N=7は38,344配置。軌道数をBurnsideの補題でも独立に照合する。
この段階で、局所Cliffordによる全スタビライザー自己同型群までは求めていない。

### 4. スタビライザーによるQFIMと、同じ情報の配置の重複除去

部分トレース後は平坦スペクトルのスタビライザー混合状態になる。
局所Pauli方向の感度は、保持スタビライザーと反交換するものがある場合に1。
両方向に感度がある場合の非対角成分は、二体Pauliの符号付き期待値で求まる。
密度行列の固有値分解で探索する必要はない。

整数QFIカーネルはグラフごとに全符号なし軸と全損失を一時バッチ計算し、
軌道代表だけを符号付け・集計・評価する。N=7の一時配列は約14 MB。
全グラフの損失配列を保存し続けることはしない。

共通etaではA_k=sum_{|S|=k} Q_Sだけを保存すればよい。
Q(eta)=sum_k eta^k (1-eta)^(N-k) A_k。組合せ係数はA_kに含まれている。
同じ生存数集計QFIMを、相関ブロックのサイズと係数列の多重集合として正規化する。
これが一致する配置は、どのetaでも平均位相のQCRBが同じなので1代表だけ評価する。
この意味での同一情報は状態そのもののLC同値性を意味しない。

Q=sum_b a_b J_{n_b}のとき、w=ones/Nに対する光子当たり有効情報は
R=N/[sum_b 1/a_b]。a_b=0のブロックがあれば識別不能でR=0。
この式は、全独立情報多項式について通常のQFIM固有値分解とも照合している。

### 5. GHZブロック上界による照合

純粋入力の相関ブロックのQFIはrank one。
部分トレース後のQFIMは入力QFIM以下なので、そのブロックの一員を失ったとき、
残存成分だけに情報を残すことはできない。残存行列の像が元のrank-one方向から外れるためである。
したがって、サイズkのブロックの活動度a_bはeta^k以下。
この上界は独立なGHZ_kブロックで達成される。

コードではA_mのブロック係数が0以上でcomb(N-k,m-k)以下であることを、全候補について整数で確認する。
この係数上界から任意の0<=eta<=1でa_b<=eta^kが従う。
全整数分割を列挙したGHZ直積の値R=N/[sum_b eta^(-n_b)]も別途計算する。
**この上界で候補を打ち切らず、全軌道代表を評価してから最良値の一致を検査した。**
位相軸が任意角度の場合や、入力を混合する場合の一般的な最適性を主張するものではない。

## 検証

- 全LC代表の全符号なし軸について、純粋QFIMを状態ベクトルから独立に計算する。
- N=6の1,213,056、N=7の16,516,224の全符号付き配置について、平均の識別可能性を
  固有空間への射影から判定し、符号削減で残すクラスと一致することを照合する。
- 全符号なし配置について、全損失で新しい二体相関が発生しないことを検査し、
  残す符号の別代表でも同じ全損失QFIMになることを照合する。
- 配置の対称性を使わない14,761+101,693配置の探索と、全40点の有理数最良値を比較する。
- 以前のN=1〜5の全符号列挙による100点の最良値と照合する。
- 保存した全採用構成は、実際のCliffordゲートを作用させた状態を作り、全損失の
  密度行列QFIMを再計算する。不均一な非ゼロ位相でも検算する。
- 保存値は有理数で再計算。通常の固有値評価とブロック式も独立照合する。

浮動小数点で近接候補を選別してから有理数で比較する。閾値は--tie-tolで変更可能。
厳密な区間演算による浮動小数点誤差の証明ではない。極端に小さいetaの設定では
double精度の限界にも注意し、必要なら有理数計算・高精度計算へ拡張する。

## 保存物

- results/best_by_n_eta.csv：最良値と有理数、独立・GHZ比較、同率GHZブロック分割。
- results/best_by_graph.csv：各LC代表での最良値。
- results/protocols.json：採用構成の辺、物理ゲート、有効軸、A_k、相関ブロック。
- results/summary.json：パラメータ、全件数、時間、ソースハッシュ、環境のバージョン。
- results/validation.json：全符号・密度行列・削減なし比較の検証結果。
- results/report.md、performance.svg：概要と比較図。
- manifest.json：公開ファイルのSHA-256。manifest自身は対象外。

ゲート積は右から作用し、HSdgは先にS†、次にH。センサ番号は0始まり。
QFIMの行列添字もセンサ番号順。密度行列のテンソル順ではセンサ0を左端に置く。

## 分類データ

data/sources.jsonに各入力のURL、件数、SHA-256を記録している。

L. E. Danielsen and M. G. Parker, “On the classification of all self-dual additive codes
over GF(4) of length up to 12,” J. Combin. Theory Ser. A 113, 1351–1367 (2006).
https://www.codetables.de/larsed/vncorbits/

N. Shettell and D. Markham, “Graph States as a Resource for Quantum Metrology,”
Physical Review Letters 124, 110502 (2020). https://arxiv.org/abs/1908.05047

本実装は、共通の単一位相の感度ではなく、独立な局所位相の平均のQCRBを評価する。

## 採用グラフの図

[重複を除いた構造一覧と透過率・ゲートの対応表](results/graphs.md) ／ [グラフ図](results/graphs_unique.svg)

同じ頂点数で番号交換により一致する構造は1枚に集約する。全ての辺はCZ（重みπ）。
各図の下は頂点番号順の局所Cliffordで、ゲート積は右から作用する。
図は保存済み評価点の採用代表解を示し、全同率解や評価点間の最適性を意味しない。
詳細な辺・ゲート変種・元の番号との対応は[graph_structures.json](results/graph_structures.json)に保存する。

図だけを再生成する場合（researchリポジトリのルートから実行、最適化は再実行しない）：

```bash
python fixed_clifford_sensors/make_graphs.py --results fixed_clifford_sensors_N6_7/results --columns 3
```

`make_report.py`でも同じ図と対応表を再生成する。描画列数は上の`--columns`で変更できる。


