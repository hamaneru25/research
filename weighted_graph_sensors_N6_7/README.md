# 6・7センサ：重み付きグラフ状態と固定局所Clifford

5センサの重み最適化を、6・7センサへ拡張する。
全15辺／21辺の制御位相を独立に変え、各重み評価で全局所Clifford有効軸を比較する。
各試行の入力・回転は固定し、Cliffordの確率混合や補助量子ビットは使わない。

- [結果と探索範囲](results/report.md)
- [従来値との比較](results/best_by_n_eta.csv)
- [N=6の各辺の重み・各センサの回転](results/best_protocols_N6.csv)
- [N=7の各辺の重み・各センサの回転](results/best_protocols_N7.csv)
- [N=6のグラフ図一覧](results/graphs_N6.svg)
- [N=7のグラフ図一覧](results/graphs_N7.svg)
- [構成と集計QFIM](results/protocols.json)
- [独立検算](results/validation.json)

**保存値は連続最適化での最良発見値。重み付きグラフ全体の大域最適性は未証明。**
評価はQFIによる分散下限であり、具体的な読み出し測定のCFIではない。

## モデル

1. N個のセンサを|+>で初期化。
2. 各組(i,j)にCP(phi_ij)=diag(1,1,1,exp(i phi_ij))を作用。
3. 各センサに固定局所Clifford C_iを作用。
4. 各センサが独立に確率etaで生存。損失は部分トレース。
5. 生存センサにexp(-i theta_i Z_i/2)を作用。
6. 生存位置は最終的な古典記録として利用。全損失を含む全試行を数える。

未知位相ベクトルは試行間で共通とし、算術平均を局所推定する。
Q(eta)=sum_S eta^|S|(1-eta)^(N-|S|) Q_S、w=ones(N)/Nとする。
平均位相のQCRBはC=w^T Q^+ w、有効情報R=1/(N C)。
wがQの像に含まれない場合はR=0。nu試行の分散下限は1/(nu N R)。
独立センサはR=eta、GHZ_NはR=N eta^N。
同じetaでは全試行に同じ重み・Cliffordを使い、損失位置を見て入力を変えない。

## 5センサ探索からの変更

- 生存確率の最終評価点は同じ20点：0.05,0.10,...,1.00。
- 全辺の重みを独立に最適化する領域、評価指標、固定Cliffordの扱いは同じ。
- 粗い初期値探索は全同型代表156形状／1,044形状を使用。**LC代表ではない。**
- 粗い探索だけは計算量を抑えるためeta=0.20,0.50,0.70,0.85の4点を使用。
  各最終etaで近い2点の上位候補を改めて評価し、上位6候補を初期値にする。
  性能値の補間はしない。最終20点では各点固有の目的関数を最適化する。
- 共通重みは[0,pi]の33点。全重みの符号反転は複素共役であり、
  有効Y軸の符号反転でQFIを保存するため、[pi,2pi]の共役候補は重複。
  この削減は辺が全て同じ重みの粗い探索に用い、独立重みの探索は全角度を許す。
- 低ランクQFIカーネルで密度行列の大きな零固有空間の対角化を省く。
- 5センサの最良構成＋独立センサ、符号付き二部グラフも初期値に追加。
- 全点の多初期値探索後、生存確率の昇順・降順に継続最適化を行う。
- プロセス並列化と途中保存を使用。保存キャッシュから再開可能。

粗い探索で全形状を扱っても、形状ごとの独立辺重みを全て調べたことにはならない。
局所最適化は完全グラフの全辺を変数に持つため、初期形状から辺を追加・削除できる。
重み0は辺なし、piは通常のCZ、負の角度は2piを法として等価。

## 混合状態QFIの高速化

損失後の状態は一般の混合状態であり、スタビライザーの平坦スペクトルを仮定しない。
生存k個、損失N-k個の場合、部分トレース状態のランクは高々2^min(k,N-k)。
振幅行列の薄い特異値分解で非零固有空間を求め、一般のSLD公式を評価する。

Pauli演算子P_i（生成子はP_i/2）に対し、

`Q_ij = Re Tr(rho P_i P_j) - sum_ab [2 lambda_a lambda_b/(lambda_a+lambda_b)] Re[(P_i)_ab (P_j)_ba]`。

第1項は純粋入力でのPauli積の期待値から求め、第2項は小さい支持空間だけで計算する。
これは一般の混合状態公式の変形であり、純粋状態の4Covを混合状態に流用していない。
分母の数値カットオフは1e-14。相殺誤差を抑えるため絶対値1e-13未満のQFI成分を0にする。

重みの解析勾配もSLDから計算する。SLDの支持空間と零固有空間の間の項を含める。
目的関数は最良Cliffordの切替点で滑らかでないため、L-BFGS-Bはヒューリスティック。
最適化中のみ対角正則化1e-10を使用し、最終スコアは正則化なしで評価する。

## Cliffordの探索

有効軸C_i^dagger Z C_iは±X,±Y,±Zの6種類。
6センサで6^6=46,656、7センサで6^7=279,936通り。
全軸の同時符号反転だけをまとめ、実評価は23,328／139,968候補。
各符号なし配置の疑似逆行列から全符号の二次形式を一括計算する。
特異行列では平均方向の識別可能性も固有空間で検査する。
非零固有値の条件数が1e6を超える場合は、相殺誤差を避けるため元の安定な固有座標評価へ戻す。
通常グラフ用の符号整列・スタビライザー削減は使わない。

回転の代表（右から作用）：+X=H、-X=HZ、+Y=HSdg、-Y=HS、+Z=I、-Z=X。
これは状態生成後・損失と未知位相の前にセンサへ作用する操作。

## 実行

researchリポジトリ全体を取得し、Python 3.10以降で実行する。
共通の評価関数と5センサの既存結果を使うため、このフォルダだけでは実行しない。
形状データはNetworkX Graph Atlasの同型代表を同梱し、実行時のNetworkXやネット接続は不要。

```bash
python -m pip install -r weighted_graph_sensors_N6_7/requirements.txt
python weighted_graph_sensors_N6_7/run_search.py --output my_results
python weighted_graph_sensors_N6_7/verify.py --results my_results
python weighted_graph_sensors_N6_7/make_report.py --results my_results
```

既定出力先resultsには公開結果がある。新しい出力先を指定する。
中断後に同じ設定で再開する場合は--resumeを付ける。

```bash
# 7センサのみ、3点。DE対象は最終etaグリッドの部分集合にする
python weighted_graph_sensors_N6_7/run_search.py --sensors 7 --etas 0.5,0.7,0.9 --de-etas 0.5,0.7 --output selected

# 同一設定・同一出力先から再開
python weighted_graph_sensors_N6_7/run_search.py --output my_results --resume
```

|引数|既定値|内容|
|---|---|---|
|--sensors|6,7|センサ数のカンマ区切り|
|--etas|0.05,...,1|最終評価する生存確率|
|--coarse-etas|0.2,0.5,0.7,0.85|粗い共通重み探索の生存確率|
|--uniform-points|33|0〜piの共通重み点数|
|--topology-starts|6|粗い探索から使う初期値数|
|--random-starts|64|各n・etaの全辺ランダム初期値数|
|--perturbed-starts|16|上位の粗い初期値を摂動する回数|
|--seed|20261023|乱数seed。ケースごとのseedはセンサ数とetaから再現|
|--maxiter|400|局所最適化の最大反復|
|--gtol|1e-8|勾配の停止基準|
|--ftol|1e-12|目的関数の停止基準|
|--ridge|1e-10|最適化中だけの正則化|
|--de-etas|0.5,0.6,0.65,0.7,0.75,0.8|集団探索を追加する点。空文字列で省略|
|--de-maxiter|100|集団探索の世代上限|
|--de-popsize|6|変数数に対する集団数倍率|
|--workers|6|同時プロセス数。各BLASスレッドは1|
|--continuation-passes|2|全点探索後の昇順・降順継続最適化回数|
|--output|results|出力フォルダ|
|--resume|なし|同じ設定の途中保存から再開|

verify.pyでは--random-checks、--gradient-checks、--seedを変更できる。
--kernel-onlyは保存構成の完成前に、ランダム入力・5センサ既存結果との照合だけを行う。

## 保存内容と検算

best_protocols_N6.csv／N7.csv：全辺の角度、各センサの有効軸と物理ゲート、改善率。
graphs_N6.svg／N7.svg：各etaの採用グラフ。描画の閾値は計算上の状態を変更しない。
protocols.json：全構成、Q(eta)、生存数ごとのQFI合計。
qfim_profiles.npz：各構成の全生存マスクQFIM。NumPyで読み出せる。
optimization_runs.csv：全最適化の停止理由、評価回数、最終値。
同CSVのdifferential_evolution行のscoreは集団探索本体の最終値、nfevは仕上げの局所最適化も含む。
仕上げの候補も最良構成の選択では比較している。
uniform_scan_seeds.json.gz：全形状・粗いetaの最良共通重み（gzip圧縮JSON）。
cache/：再開用ファイル。実行中に生成し、公開リポジトリには含めない。

最終全構成でCPとCliffordを実際に作用させ、全損失の密度行列と位相微分から
別実装の一般SLD公式でQFIMを再計算する。非ゼロ独立位相、解析勾配、
5センサの全保存最良構成、6・7センサの通常グラフ既存値とも比較する。
閾値以内の一致が得られた結果だけを公開する。

ゲート誤差、状態生成成功確率、重み設定誤差、具体的な読み出し測定、
Cliffordの確率混合、損失位置に応じた入力適応は評価していない。
