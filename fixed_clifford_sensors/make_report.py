"""Create a Japanese results report and an SVG plot from an existing run."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from make_graphs import make_graphs, graph_report_lines


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path(__file__).resolve().parent/"results")
    args = p.parse_args()
    results = args.results.resolve()
    rows = list(csv.DictReader((results/"best_by_n_eta.csv").open()))
    protocols = json.loads((results/"protocols.json").read_text())
    summary = json.loads((results/"summary.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for n in sorted({int(r["n"]) for r in rows}):
        selected = [r for r in rows if int(r["n"]) == n]
        eta = [float(r["eta"]) for r in selected]
        axes[0].plot(eta, [float(r["information_per_photon"]) for r in selected], marker=".", label=f"N={n}")
        axes[1].plot(eta, [float(r["gain_over_independent"]) if r["gain_over_independent"] else float("nan") for r in selected], marker=".", label=f"N={n}")
    etas = sorted({float(r["eta"]) for r in rows})
    axes[0].plot(etas, etas, "k--", linewidth=1.3, label="Independent sensors")
    axes[1].axhline(1, color="black", linestyle="--", linewidth=1.3)
    axes[0].set_ylabel("Effective QFI / generated photon")
    axes[1].set_ylabel("Gain over independent sensors")
    axes[0].set_title("Fixed graph + local Clifford: exhaustive search")
    axes[1].set_title("QFI bound; readout not yet optimized")
    for ax in axes:
        ax.set_xlabel("Sensor survival probability eta")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.savefig(results/"performance.svg")
    plt.close(fig)
    lines = ["# 固定Clifford・補助なし：探索結果", "",
             "センサ数1〜5、独立な局所位相の算術平均。全損失試行を含めたQFIMを評価する。",
             "これは読み出し測定を最適化したCFIの実測可能性能ではなく、QFIに基づく上限評価である。", "",
             "![performance](performance.svg)", "", "## 探索件数", "",
             "|N|グラフ代表|回転込み候補|異なるQFIプロファイル|", "|---:|---:|---:|---:|"]
    for s in summary["search_statistics"]:
        lines.append(f"|{s['n']}|{s['graph_representatives']}|{s['evaluated_candidates']:,}|{s['unique_qfi_profiles']}|")
    lines += ["", "重複プロファイルは、全生存集合に対する符号付きQFIMが一致する場合だけまとめた。",
              "固定したグラフと回転を全試行で使う。損失パターン別の入力最適化、設定の確率混合は行っていない。", "",
              "## 透過率ごとの最良値", "",
              "各値は光子当たり有効QFI。独立センサはetaと一致する。", "",
              "|eta|"+"|".join(f"N={n}" for n in sorted({int(r['n']) for r in rows}))+"|",
              "|---:|"+"---:|"*len({r['n'] for r in rows})]
    for eta in summary["configuration"]["eta_grid"]:
        selected = [r for r in rows if r["eta"] == eta]
        lines.append("|"+eta+"|"+"|".join(f"{float(r['information_per_photon']):.8f}" for r in selected)+"|")
    lines += ["", "## 各Nの採用構成が変わる評価点", "",
              "同率の場合は代表1構成を示す。区間の厳密な切替境界や全同率回路の列挙ではない。",
              "番号は0始まり。HSdgは先にS†、次にHを作用させる。", "",
              "|N|その構成を最初に採用したeta|辺|センサ0から順のゲート|", "|---:|---:|---|---|"]
    for n in sorted({int(r["n"]) for r in rows}):
        previous = None
        for row in rows:
            if int(row["n"]) != n:
                continue
            protocol = protocols[row["protocol_id"]]
            if previous != row["protocol_id"]:
                lines.append(f"|{n}|{row['eta']}|`{protocol['edges']}`|`{protocol['sensor_gates']}`|")
                previous = row["protocol_id"]
    lines += ["", "## 範囲と限界", "",
              "- 全探索の範囲は、全単純無向グラフと固定局所Clifford。任意角度の局所回転は含まない。",
              "- 最適性は指定された透過率点と数値許容誤差の範囲。近接候補は有理数で比較した。",
              "- 指標は平均位相のQCRB。平均が識別不能な場合は有効情報ゼロ。",
              "- 辺の生成コスト、操作誤差、検出器不完全性は含めない。",
              "- センサの読み出し測定基底や、局所測定による達成可能性は未探索。", ""]
    make_graphs(results)
    lines += graph_report_lines()
    (results/"report.md").write_text("\n".join(lines), encoding="utf-8")
    print(results/"report.md")


if __name__ == "__main__":
    main()

