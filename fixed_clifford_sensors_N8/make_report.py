"""Generate figures and Japanese report for an existing efficient sensing run."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core import ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=ROOT/"results")
    p.add_argument("--comparison-results", type=Path,
                   default=ROOT.parent/"fixed_clifford_sensors_N6_7/results",
                   help="Existing 6/7-sensor results to include in the comparison")
    args = p.parse_args()
    out = args.results.resolve()
    rows = list(csv.DictReader((out/"best_by_n_eta.csv").open()))
    summary = json.loads((out/"summary.json").read_text())
    protocols = json.loads((out/"protocols.json").read_text())
    comparison_rows = []
    comparison_path = args.comparison_results/"best_by_n_eta.csv"
    if comparison_path.exists():
        own_sensors = {int(r["n"]) for r in rows}
        comparison_rows = [r for r in csv.DictReader(comparison_path.open())
                           if int(r["n"]) not in own_sensors
                           and r["eta"] in summary["configuration"]["eta_grid"]]
    display_rows = sorted(rows+comparison_rows, key=lambda r: (int(r["n"]), float(r["eta"])))
    sensors = sorted({int(r["n"]) for r in display_rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for n in sensors:
        selected = [r for r in display_rows if int(r["n"]) == n]
        eta = [float(r["eta"]) for r in selected]
        line, = axes[0].plot(eta, [float(r["information_per_photon"]) for r in selected], marker=".", label=f"N={n} optimized")
        axes[0].plot(eta, [float(r["ghz_per_photon"]) for r in selected], color=line.get_color(), linestyle=":", label=f"N={n} GHZ")
        axes[1].plot(eta, [float(r["gain_over_independent"]) if r["gain_over_independent"] else float("nan") for r in selected], marker=".", label=f"N={n}")
    etas = sorted({float(r["eta"]) for r in rows})
    axes[0].plot(etas, etas, "k--", label="Independent sensors")
    axes[1].axhline(1, color="black", linestyle="--")
    axes[0].set_title("Fixed Clifford: exhaustive reduced search")
    axes[1].set_title("QFI bound; local readout not optimized")
    axes[0].set_ylabel("Effective QFI / generated photon")
    axes[1].set_ylabel("Gain over independent sensors")
    for ax in axes:
        ax.set_xlabel("Sensor survival probability eta")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.savefig(out/"performance.svg")
    plt.close(fig)
    run_sensors = sorted({int(r["n"]) for r in rows})
    title = "・".join(map(str, run_sensors))
    lines = [f"# {title}センサ：固定Cliffordの効率的全探索", "",
             "補助なし、独立な未知位相の算術平均、共通透過率、全損失試行を利用。",
             "結果はQFIMに基づく性能上限であり、実際の局所読み出し測定のCFI最適化ではない。", "",
             *( ["比較図・比較表の6・7センサ値は、既存の保存結果を使用している。", ""] if comparison_rows else [] ),
             "![performance](performance.svg)", "", "## 全探索件数", "",
             "|N|LC代表|符号付き単純列挙|対称性なし・符号決定後|配置対称性で削減後|情報多項式の種類|", "|---:|---:|---:|---:|---:|---:|"]
    for s in summary["search_statistics"]:
        lines.append(f"|{s['n']}|{s['lc_representatives']}|{s['unreduced_signed_axes']:,}|{s['valid_unsigned_axes']:,}|{s['evaluated_axis_candidates']:,}|{s['unique_QFIM_polynomials']:,}|")
    lines += ["", f"この環境での主探索時間は{summary['elapsed_seconds']:.3f}秒（検証・図作成を除く）。他のPCでの実行時間は保証しない。",
              "配置はすべて評価し、上界による候補の打切りやGHZ状態だけへの候補制限は行っていない。", "",
              "## 透過率ごとの最良値", "",
              "倍率は独立センサに対する有効QFI比。分散下限の倍率はその逆数。",
              "構成欄は、同じ最適値を達成するGHZブロックのサイズ。1は独立センサ、2はベル対。", ""]
    header = "|eta|"+"|".join(f"N={n} 情報/光子|倍率|構成" for n in sensors)+"|"
    lines += [header, "|---:|"+"---:|---:|---|"*len(sensors)]
    for eta in summary["configuration"]["eta_grid"]:
        selected = [r for r in display_rows if r["eta"] == eta]
        text = "|"+eta+"|"
        by_sensor = {int(r["n"]): r for r in selected}
        for n in sensors:
            if n not in by_sensor:
                text += "—|—|—|"
                continue
            r = by_sensor[n]
            gain = f"{float(r['gain_over_independent']):.6f}" if r["gain_over_independent"] else "—"
            text += f"{float(r['information_per_photon']):.8f}|{gain}|{r['best_GHZ_partition']}|"
        lines.append(text)
    lines += ["", "全評価点で、探索最良値はGHZブロック直積の最良値と有理数で一致した。",
              "この探索範囲では、それを上回るグラフ状態は見つからなかった。",
              "ブロック活動度の係数上界も全候補で検査した。切替境界の連続的な探索は行っていない。", "",
              "## 保存した操作表", "",
              "同じ構成が連続する場合は最初の評価点だけを示す。ゲートはセンサ0から順。積は右から作用。", "",
              "|N|最初のeta|辺|ゲート|", "|---:|---:|---|---|"]
    for n in sensors:
        previous = None
        for row in rows:
            if int(row["n"]) != n or row["protocol_id"] == previous:
                continue
            previous = row["protocol_id"]
            protocol = protocols[previous]
            lines.append(f"|{n}|{row['eta']}|`{protocol['edges']}`|`{protocol['sensor_gates']}`|")
    validation_path = out/"validation.json"
    if validation_path.exists():
        v = json.loads(validation_path.read_text())
        lines += ["", "## 検証", "",
                  f"- 密度行列で検証した採用構成：{v['saved_protocols_dense_checked']}。",
                  f"- 損失パターンの密度行列検算：{v['survivor_patterns_dense_checked']}。",
                  f"- 各損失パターンQFIMの最大差：{v['maximum_dense_QFIM_error']:.3e}。",
                  f"- 生存数集計QFIMの最大差：{v['maximum_dense_aggregate_error']:.3e}。",
                  f"- 不均一・非ゼロの位相での最大差：{v['unequal_nonzero_phase_vector_error']:.3e}。",
                  f"- 全符号を照合した配置数：{sum(s['signed_assignments_checked'] for s in v['sign_checks'].values()):,}。",
                  f"- 有理数で確認した保存値：{v['rational_grid_rows_checked']}点。"]
        if "graph_symmetry_reduction_comparison" in v:
            lines.append(f"- 配置対称性を使わない探索との一致：{len(v['graph_symmetry_reduction_comparison']['matched_rows'])}点。")
        if "previous_N1_to_5_comparison" in v:
            lines.append(f"- 以前のN=1〜5の全符号列挙との最良値一致：{len(v['previous_N1_to_5_comparison']['matched_rows'])}点。")
    lines += ["", "## 適用範囲", "",
              "- 固定Clifford入力のみ。任意角度回転、設定の確率混合、補助、損失別の入力変更は含めない。",
              "- 古典基準は、同じ生成光子数・損失確率の独立センサ。",
              "- 辺の生成コスト・操作誤差・検出器不完全性は含めない。",
              "- 具体的な読み出し測定と、そのCFIによる達成可能性は未探索。",
              "- 有理数比較の前に浮動小数点の近接候補選別を使う。任意角度までの大域最適性ではない。", ""]
    (out/"report.md").write_text("\n".join(lines), encoding="utf-8")
    print(out/"report.md")


if __name__ == "__main__":
    main()
