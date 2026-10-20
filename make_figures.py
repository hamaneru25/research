"""Generate exact topology and performance plots from the saved optimization."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

out = Path("results")
plt.rcParams.update({"font.size": 13, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 180})

fig, ax = plt.subplots(figsize=(8, 4.8))
positions = {"S1": (-1, .5), "S2": (1, .5), "S3": (0, 1.6),
             "C1": (0, -.6), "C2": (2.15, .5), "C3": (-2.15, .5)}
edges = [("S1", "S3"), ("S1", "C1"), ("S1", "C3"),
         ("S2", "S3"), ("S2", "C1"), ("S2", "C2")]
for a, b in edges:
    xa, ya = positions[a]; xb, yb = positions[b]
    ax.plot([xa, xb], [ya, yb], color="#697586", lw=2.8, zorder=1)
for label, (x, y) in positions.items():
    color = "#f2ab57" if label.startswith("S") else "#276c85"
    ax.scatter([x], [y], s=1500, color=color, edgecolor="white", linewidth=2, zorder=2)
    ax.text(x, y, label, va="center", ha="center", fontsize=17, fontweight="bold",
            color="#132b38" if label.startswith("S") else "white", zorder=3)
ax.set_xlim(-2.8, 2.8); ax.set_ylim(-1.3, 2.15); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("An optimal six-qubit graph", fontsize=20, pad=10)
legend = [Line2D([], [], marker="o", color="none", markerfacecolor="#f2ab57", markersize=12,
                 label="S: sensor qubits (transmitted)"),
          Line2D([], [], marker="o", color="none", markerfacecolor="#276c85", markersize=12,
                 label="C: central qubits (retained)")]
ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, -.05), frameon=False, fontsize=12)
fig.tight_layout(); fig.savefig(out/"optimal_graph.png", bbox_inches="tight")
fig.savefig(out/"optimal_graph.svg", bbox_inches="tight"); plt.close(fig)

with open(out/"optimal_summary.csv") as f:
    rows = list(csv.DictReader(f))
eta = np.array([float(r["eta"]) for r in rows])
score = np.array([float(r["best_feff"]) for r in rows])
t = np.linspace(0, 1, 301)
opt = 3*t + 6*t*t
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
axes[0].plot(t, opt, color="#236b8e", lw=2.4, label=r"Exact optimum: $3\eta+6\eta^2$")
axes[0].scatter(eta, score, color="#d37b2f", s=24, zorder=4, label="SDP results")
axes[0].plot(t, 3*t, "--", color="#74828d", lw=1.8, label="3 independent sensor probes")
axes[0].set_ylabel(r"Effective QFI per trial, $F_{\rm eff}$")
axes[0].set_title("Per trial"); axes[0].legend(fontsize=10)
axes[1].plot(t, opt/6, color="#236b8e", lw=2.4, label="Optimal graph / 6 photons")
axes[1].plot(t, t, "--", color="#74828d", lw=1.8, label="Independent probes / 3 photons")
axes[1].axvline(.5, ls=":", color="#c6cdd2", lw=1.3)
axes[1].set_ylabel("Effective QFI per generated photon")
axes[1].set_title("Equal total generated-photon budget")
axes[1].legend(fontsize=10)
for ax in axes:
    ax.set_xlabel(r"Sensor transmission probability $\eta$")
    ax.set_xlim(0, 1); ax.set_ylim(bottom=0); ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(out/"performance.png", bbox_inches="tight")
fig.savefig(out/"performance.svg", bbox_inches="tight"); plt.close(fig)
