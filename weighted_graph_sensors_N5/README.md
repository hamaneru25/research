# 5センサ：重み付きグラフ状態＋固定局所Clifford

各辺の制御位相をπから任意角に広げ、損失下の平均位相推定を数値探索する。
局所回転はCliffordに限定する。全試行で同じ状態・回転を使い、補助や設定混合は使わない。

- [結果レポート](results/report.md)
- [既存最適値との数値比較](results/best_by_eta.csv)
- [各辺の重み・局所ゲートの表](results/best_protocols.csv)
- [全損失パターンのQFIM](results/protocols.json)
- [独立検算](results/validation.json)
- [簡単に再現できる6辺の改善例](results/simple_witness.json)

**既存の通常グラフ＋Cliffordの全探索最適値を上回る実行可能な構成が見つかった。**
新しい連続重みの大域最適性を証明した結果ではない。
具体的な読み出し測定のCFI、生成成功率、ゲート実装コストは評価していない。

## 状態と物理モデル

各センサを|+>で初期化し、各辺へ
`CP(phi_ab)=diag(1,1,1,exp(i phi_ab))`をかける。
5頂点の全10組のphi_abを独立に変数とし、角度は2πを法とする。
phi=0は辺なし、phi=πはCZ。負の角度も2πから引いた正の角度で表せる。

その後、センサiに固定Clifford C_iをかけ、確率etaの独立な生存、
生存センサへの独立未知位相 `exp(-i theta_i Z_i/2)`、最終検出を行う。
失われたセンサは部分トレースする。損失位置は古典記録として保持し、
全損失を含む全試行を数える。損失を見て入力状態・回転を変更しない。

`Q(eta)=sum_S eta^|S| (1-eta)^(5-|S|) Q_S`を先に加算する。
`w=ones(5)/5`に対して`C_Q=w^T Q(eta)^+ w`、`R=1/(5 C_Q)`を評価。
平均方向がQの像に含まれなければR=0。nu試行の分散下限は`1/(nu*5*R)`。
比較対象は既存の通常グラフ＋Cliffordの全探索結果で、最適なGHZ分割と一致する。

## 混合状態QFIとCliffordの列挙

一般の重みでは、損失後の状態は平坦スペクトルのスタビライザー混合状態ではない。
ここでは各部分トレース状態rhoの固有値・固有ベクトルから一般の混合状態SLD-QFIを計算する。
純粋状態の4Covを使うのは、全5センサが生存した純粋状態の枝だけ。

Pauli/2生成子の全15方向についてQFIテンソルを計算する。
各C_iは有効軸 `C_i^dagger Z C_i in {+X,-X,+Y,-Y,+Z,-Z}`で表せる。
**重みの各評価で6^5通りを有限列挙して最良Cliffordを選ぶ。**
計算上は3^5の符号なし軸ごとに一度対角化し、2^4の相対符号を調べる。
全軸の同時符号反転だけはRを変えないためまとめる。通常グラフ用の符号整列削減は使わない。

混合状態のSLD恒等式を用いた辺の重みの解析勾配で連続最適化する。
最適化中のみ1e-10の対角正則化を用い、保存する物理スコアからは除く。
最良候補の検算では、CPゲートとCliffordを実際に作用させてから部分トレースし、
密度行列の位相微分から別実装のSLD-QFIMを計算して照合する。

## 探索範囲と手順

1. 5頂点の全34同型代表（非連結・孤立頂点を含む）を使い、各グラフの全辺に
   共通角phiを0〜2πの65点で与えて粗く探索する。**weighted graphにLC削減は使わない。**
2. 上位6トポロジー、既存Clifford最良構成、摂動16点、一様ランダム64点を初期値にする。
3. K5の10角度を個別に最適化する。重み0も許すので辺の追加・削除を含む。
4. 一つ前のetaの最良解も初期値に使う。
5. eta=0.5,0.6,0.65,0.7,0.75,0.8では独立のDifferential Evolutionも実行する。

グラフの連続パラメータ領域は全重み付きグラフを含むが、最適化はヒューリスティック。
有限回の数値探索で新しい大域最適値や全透過率区間の最適性は主張しない。

## 再実行

Python 3.10以降。共有の代表生成と既存結果を使うのでresearchリポジトリをフォルダ構成ごと取得する。

```bash
cd weighted_graph_sensors_N5
python -m pip install -r requirements.txt
python run_search.py --output my_results
python verify.py --results my_results
python simple_witness.py --output my_results/simple_witness.json
python make_report.py --results my_results
```

公開実行は粗い探索の完了後に一度プロセスが止まったため、保存済みの粗い探索を再利用し、
局所最適化を最初から実行した。再開後の実行条件は以下。

```bash
python run_search.py --uniform-cache data/uniform_scan_seeds.json --output my_results
```

粗い探索を再計算する場合は`--uniform-cache`を省略する。
summary.jsonの時間は再開後の重み個別最適化に対応し、先行した粗い探索の時間は含まない。
同梱の粗い探索キャッシュには34グラフ×20透過率の最良発見値と初期値を保存する。
既存の空でない出力先は保護する。別名を使うか、再実行に`--overwrite`を明示する。

## 手動実行で変更するパラメータ

|引数|既定値|意味|
|---|---:|---|
|`--etas`|0.05,0.10,…,1|生存確率。(0,1]内のカンマ区切り|
|`--uniform-points`|65|共通重みの粗い探索点数。πを含む奇数|
|`--uniform-cache`|なし|同一eta・同一粗い探索の保存済み初期値を再利用|
|`--topology-starts`|6|粗い探索で上位のトポロジー数|
|`--random-starts`|64|各etaのランダム初期値数|
|`--perturbed-starts`|16|上位トポロジーを摂動する初期値数|
|`--seed`|20261022|乱数seed|
|`--maxiter`|400|各局所最適化の最大反復数|
|`--gtol`|1e-8|勾配停止閾値|
|`--ftol`|1e-12|目的関数の停止閾値|
|`--ridge`|1e-10|最適化中のみの対角正則化|
|`--de-etas`|0.5,0.6,0.65,0.7,0.75,0.8|集団探索を追加するeta。空文字列で無効|
|`--de-maxiter`|100|DEの最大世代数|
|`--de-popsize`|6|DEの個体数倍率。10変数なので60個体|
|`--improvement-atol`|1e-7|既存最適値の更新判定の絶対許容誤差|
|`--improvement-rtol`|1e-7|同じく相対許容誤差|
|`--output`|results|出力先|
|`--overwrite`|なし|既存出力の再利用を許可|

```bash
# 評価点を変える。新しいetaでは通常グラフ＋Cliffordの基準値も改めて全探索する
python run_search.py --etas 0.33,0.5,0.77 --de-etas 0.5 --output selected_points

# 初期値を増やして再探索
python run_search.py --random-starts 256 --seed 123 --output more_starts

# 簡単な改善例だけ再計算
python simple_witness.py --eta 0.5 --phi 0.6666666666666666
```

`verify.py --random-checks`、`--seed`で独立検算用のランダム構成数・seedを変更できる。
小行列計算のため、公開実行ではBLAS・OMPのスレッド数を1に設定した。

## 保存される構成

protocols.jsonの`weights_rad`は`edge_order`の順。番号は0始まり。
`axes_indices`の0,1,2はX,Y,Zで、`signs`を掛けたものが有効位相軸。
物理Cliffordの代表は以下。ゲート積は右から作用する。

|有効軸|ゲートC（C†ZC=有効軸）|
|---|---|
|+X|H|
|-X|HZ|
|+Y|HSdg|
|-Y|HS|
|+Z|I|
|-Z|X|

`best_protocols.csv`には各etaの10辺の重みと5センサのゲートを同時に保存する。
全32生存パターンのQFIMでは、マスクのビットiがセンサiの生存を表す。
密度行列のテンソル順ではセンサ0が左端。
