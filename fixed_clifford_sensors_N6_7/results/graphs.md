# 固定Clifford：重複を除いた採用グラフ

保存済み40評価点の採用構成を、9種類の接続構造に集約した。

![構造一覧](graphs_unique.svg)

全ての頂点はセンサ。辺はCZ（重みπ）。図の下のゲートは頂点番号順で、積は右から作用する。
同じ頂点数で番号交換により一致する構造をまとめた。Clifford操作が異なる場合も図は重複させず、下表とJSONに全変種を残す。
異なる頂点数・接続構造を、GHZへのLC同値性だけでまとめることはしない。
表示はbest_by_n_eta.csvに保存された代表解のみ。同率最適解の網羅や評価点間の最適性を意味しない。

|構造ID|センサ数|連結成分のサイズ|採用したeta（保存評価点）|図の頂点番号順のゲート|元の構成ID|
|---|---:|---|---|---|---|
|N6_S01|6|1+1+1+1+1+1|0.05-0.50 (step .05)|`HSdg, HSdg, HSdg, HSdg, HSdg, HSdg`|`N6_g0_pY_pY_pY_pY_pY_pY`|
|N6_S02|6|2+2+2|0.55-0.65 (step .05)|`H, I, H, I, H, I`|`N6_g11_pX_pZ_pX_pZ_pX_pZ`|
|N6_S03|6|3+3|0.70, 0.75|`I, H, H, I, H, H`|`N6_g14_pZ_pX_pX_pZ_pX_pX`|
|N6_S04|6|6|0.80-1.00 (step .05)|`H, H, H, H, H, I`|`N6_g15_pX_pX_pX_pX_pX_pZ`|
|N7_S01|7|1+1+1+1+1+1+1|0.05-0.50 (step .05)|`HSdg, HSdg, HSdg, HSdg, HSdg, HSdg, HSdg`|`N7_g0_pY_pY_pY_pY_pY_pY_pY`|
|N7_S02|7|1+2+2+2|0.55, 0.60|`HSdg, H, I, H, I, H, I`|`N7_g11_pY_pX_pZ_pX_pZ_pX_pZ`|
|N7_S03|7|2+2+3|0.65, 0.70|`H, I, H, I, I, H, H`|`N7_g26_pX_pZ_pX_pZ_pZ_pX_pX`|
|N7_S04|7|3+4|0.75, 0.80|`I, H, H, H, H, H, I`|`N7_g31_pZ_pX_pX_pX_pX_pX_pZ`|
|N7_S05|7|7|0.85-1.00 (step .05)|`H, H, H, H, H, H, I`|`N7_g33_pX_pX_pX_pX_pX_pX_pZ`|

[構造・辺・番号対応・正確な性能値のJSON](graph_structures.json)
／ [元の全評価点の数値](best_by_n_eta.csv) ／ [元の構成データ](protocols.json)
