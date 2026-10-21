# 補助なし・最大5センサ：グラフと固定局所Cliffordの全探索

グラフの辺と、位相付与前のセンサ局所Cliffordを最適化する。
各センサの位相は独立に未知で、目標は全地点の算術平均。
損失別に入力回転を変えず、補助測定・設定の確率混合は使わない。

結果は [results/report.md](results/report.md)、数値表は
[results/best_by_n_eta.csv](results/best_by_n_eta.csv)。
**QFIに基づく性能上限の探索であり、具体的な局所読み出し測定のCFI探索ではない。**

## 実行

Python 3.10以降。Windowsでも利用でき、外部のグラフ分類DBやネットワーク接続は不要。

```bash
cd fixed_clifford_sensors
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python run_search.py --output my_results
python verify.py --results my_results
python make_report.py --results my_results
```

既存の空でない出力先は誤上書きを防ぐため停止する。
同じ出力先で再計算する場合は `--overwrite` を明示する。
公開結果の実行コマンドは `python run_search.py`。既定の出力先はこのフォルダ内のresults。

## 後から変更できるパラメータ

|引数|既定値|意味|
|---|---|---|
|`--min-sensors`|1|最小センサ数|
|`--max-sensors`|5|最大センサ数。実装上1〜5をサポート|
|`--eta-min`|0.05|生存確率グリッドの始点。含む|
|`--eta-max`|1|生存確率グリッドの終点。含む|
|`--eta-points`|20|始点・終点を含む点数。1なら始点のみ|
|`--etas`|未指定|カンマ区切りの生存確率を直接指定。グリッド引数より優先|
|`--graph-reduction`|lc|`lc`、`isomorphism`、`labeled`から選択|
|`--output`|スクリプト隣のresults|結果の出力先|
|`--overwrite`|なし|同名の出力ファイルを置換|
|`--eig-rtol`|1e-10|QFIMの固有値の相対閾値|
|`--range-tol`|1e-8|平均位相の識別可能性の判定閾値|
|`--tie-tol`|1e-10|有理数で再比較する数値近接候補の幅。max(1,最良値)を掛ける|
|`--log-every`|10000|候補数ごとの進捗表示。0で無効|

```bash
# 5センサだけ、eta=0.025から1まで40点
python run_search.py --min-sensors 5 --max-sensors 5 --eta-min 0.025 --eta-points 40 --output n5_40points

# 調べたい点を直接指定（0と1も利用可能）
python run_search.py --etas 0,0.2,0.5,0.65,0.8,1 --output selected_points

# 削減の影響を自分で検証：小さいNで番号付き全グラフを走査
python run_search.py --max-sensors 3 --graph-reduction labeled --output all_labeled_n3
python verify.py --results my_results --unreduced-results all_labeled_n3
```

出力先を変えても、verify.pyとmake_report.pyはそのsummary.jsonの設定を読む。
透過率や点数をほかのファイルで再指定する必要はない。
極端に小さいetaでは相対固有値閾値により小さな固有値を失う場合がある。
保存された最良構成は有理数で照合するため、数値計算と不一致なら実行を停止する。

## 物理モデル

1. NセンサのCZグラフ状態を生成する。非連結・孤立頂点も含む。
2. 各センサへ固定した局所Clifford C_iを作用させる。
3. 各センサが独立に確率etaで生存する。失われた量子ビットは部分トレースする。
4. 生存センサにU_i=exp(-i theta_i Z_i/2)を作用させる。
5. 全損失試行を含め、最終的に生存位置を古典記録として保持する。

入力回転はグラフ生成後、損失・位相付与より前に固定する。
Z/2と(I-Z)/2は、位相パラメータ全体の符号と全体位相の違いしかなく、今回のQFIMは同じ。
補助なし・独立な消失チャネルなので、位相付与と消失の順序を交換しても
生存状態と損失記録のモデルは同じになる。

生存集合SごとのN×N QFIM Q_Sを計算し、失われた地点の行・列はゼロに埋める。
Q(eta)=sum_S eta^|S| (1-eta)^(N-|S|) Q_S を**先に加算**する。
w=ones/Nとして、平均位相のQCRBはw^T Q^+ w。
wがQの像にない場合は識別不能なので、QCRBを無限大、有効情報をゼロとする。
評価値R=1/[N w^T Q^+ w]は、生成光子1個当たりの有効QFI。
独立センサのR=eta、GHZのR=N eta^N。
試行回数nuを含む分散下限は1/(nu N R)。

各試行で同じ未知位相ベクトルを測る局所推定問題。
位相が試行間で変動するモデル、任意角度回転、センサ間で異なる透過率や重みは含まない。
全センサ損失は確率に含め、情報ゼロとして数える。

## 全探索と削減

LC代表は外部DBから取得せず、そのNの番号付き全グラフを列挙し、
番号交換と局所補完の閉包で生成する。1〜5センサの代表数は1,2,3,6,11。
局所回転も探索するので、LC同値なグラフに対する任意の局所Cliffordは代表上でも再現できる。
センサ交換は共通eta・等重みの平均・QFI評価に対して不変。
辺数や生成成功率はコストに入れていないため、代表が最少辺の物理実装とは限らない。

QFIについてはC_i^dagger Z C_iが±X,±Y,±Zのどれかだけに依存する。
24種類のCliffordを6種類の有効軸にまとめる。軸間の相対符号は残す。
どの生存集合でも同じ軸を使い、負相関・識別不能状態も全候補として評価する。
正相関GHZブロックだけを仮定した候補制限はしていない。

各部分トレース後の状態は平坦スペクトルのスタビライザー混合状態になる。
局所Pauli/2方向のQFIは、保持スタビライザーとの反交換により0または1。
両方向がactiveのときの非対角成分は、対応する二体Pauliの符号付き期待値。
これで全生存集合のQFIMを整数として計算し、全集合で同一のQFIMを与える候補だけ重複除去する。

全候補は94,038。最適性は指定点における固定Cliffordクラス内の全探索の数値結果。
固有値と識別可能性の判定は浮動小数点を使うため、厳密な区間演算の証明ではない。
近接候補と保存する最良値は有理数で再比較・再計算する。
全同率回路は列挙せず、代表1構成と、近接候補中で有理数値が一致したQFIプロファイル数を保存する。
単一点で最良の構成が、その間の連続区間でも最良とは主張しない。

## 保存ファイルと検証

- best_by_n_eta.csv：各N・etaの最良値、有理数値、独立・GHZ比較、構成ID。
- best_by_lc_class.csv：各グラフ代表での最良値。削減方法を変更した場合も同じファイル名。
- protocols.json：採用した構成の辺、物理ゲート、符号付き有効軸、全生存集合のQFIM。
- summary.json：全パラメータ、件数、グラフ分類、実行日時、実行時間、環境、ソースSHA-256。
- validation.json：独立した密度行列検算、保存値の有理数照合、テスト結果。
- report.md / performance.svg：概要と比較図。

ゲートは右から作用する。HSdgは先にS†、次にH。頂点番号は0始まり。
QFIMの配列添字sは、生存センサiがビットiに対応するマスク。
例えばs=5ならセンサ0,2が生存する。密度行列のテンソル順はセンサ0が左端。

verify.pyでは、3センサまで全グラフ・全符号なし軸の密度行列QFIMと比較し、
4・5センサの全LC代表について固定seedの符号付き回転を物理ゲートで検算する。
3センサまでは番号付き全グラフの符号付きQFIプロファイル集合も、削減後と完全照合する。
さらに独立センサ、GHZ、負相関・特異行列の識別可能性を検証する。
保存された全採用構成の全生存集合を密度行列で再計算する。

公開時には4センサまで、番号付き全グラフ・全符号付き軸の84,750候補も別に実行した。
透過率0,0.05,0.5,0.55,0.65,0.8,1で計算し、主計算と重なる24点の有理数最良値が一致した。
これは以下で再実行できる（unreduced_checkは検証用の別出力先）。

```bash
python run_search.py --max-sensors 4 --etas 0,0.05,0.5,0.55,0.65,0.8,1 --graph-reduction labeled --output unreduced_check
python verify.py --results results --unreduced-results unreduced_check
```

## 参考

N. Shettell and D. Markham, “Graph States as a Resource for Quantum Metrology,”
Physical Review Letters 124, 110502 (2020), https://arxiv.org/abs/1908.05047

この実装は、共通の単一位相ではなく独立な局所位相の平均を評価する。
