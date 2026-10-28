"""Figures for the Omega study (tau = 0). Needs results/summary.json from study.py.
  results/omega_contours.png : Omega over the weight triangle for TQQQ/SHNY/HYG and TQQQ/UGL/HYG
  results/omega_frontier.png : efficient frontiers, x = omega risk (expected shortfall below 0), y = expected 1-year return
Run: python omega/plots.py
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.tri import Triangulation
sys.path.insert(0, str(Path(__file__).resolve().parent))
from study import MON, TRIOS, UNIV, scenarios

ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT / "results"
RES = json.loads((OUT / "summary.json").read_text())
BLUE, ORANGE, AQUA, INK, INK2, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
pct = lambda v, _=None: f"{v:.0%}"


def contours(S):
    step = 0.01
    P = np.array([(s, g) for s in np.arange(0, 1 + 1e-9, step) for g in np.arange(0, 1 - s + 1e-9, step)])
    W = np.c_[P[:, 0], P[:, 1], 1 - P.sum(1)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.9), constrained_layout=True)
    levels = np.arange(1.0, 7.51, 0.5)
    for ax, (name, cols) in zip(axes, TRIOS.items()):
        X = S[cols].values @ W.T
        om = np.maximum(X, 0).mean(0) / np.maximum(-X, 0).mean(0)
        tri = Triangulation(P[:, 1], P[:, 0])
        cf = ax.tricontourf(tri, np.clip(om, levels[0], levels[-1] - 1e-9), levels=levels, cmap=SEQ)
        cs = ax.tricontour(tri, om, levels=[2, 3, 4, 5, 6, 7], colors=SURF, linewidths=0.9)
        ax.clabel(cs, fmt=lambda v: f"{v:g}", fontsize=7, colors=INK)
        best = RES["full"][f"max-Omega, {name}"]; bw = best["w"]
        bs, bg = bw.get(cols[0], 0), bw.get(cols[1], 0)
        ax.plot(bg, bs, marker="*", ms=15, color=ORANGE, mec=SURF, mew=1.5, zorder=5)
        ax.annotate(f"max Ω {best['omega']:.2f}\n{bs:.0%} {cols[0]} / {bg:.0%} {cols[1]} / {1 - bs - bg:.0%} HYG", (bg, bs),
                    xytext=(14, 8), textcoords="offset points", fontsize=8, bbox=dict(boxstyle="round,pad=0.25", fc=SURF, ec=GRID))
        ref = "30 TQQQ/30 UGL/40 HYG" if "UGL" in name else "50 TQQQ/10 SHNY/40 HYG"
        rw = RES["full"][ref]["w"]; rs, rg = rw.get(cols[0], 0), rw.get(cols[1], 0)
        ax.plot(rg, rs, marker="o", ms=7, color=SURF, mec=INK, mew=1.5, zorder=5)
        ax.annotate(f"{ref.replace(' ', ' ')} (Ω {RES['full'][ref]['omega']:.2f})", (rg, rs), xytext=(10, -14), textcoords="offset points", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.25", fc=SURF, ec=GRID))
        ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xlabel=f"{cols[1]} weight  (HYG = the rest)", ylabel=f"{cols[0]} weight")
        ax.xaxis.set_major_formatter(pct); ax.yaxis.set_major_formatter(pct)
        ax.set_title(f"{name}, rebalanced yearly", loc="left", fontsize=10)
    fig.colorbar(cf, ax=axes, shrink=0.8, label="Omega(τ = 0)  =  expected gain above 0 ÷ expected loss below 0")
    fig.suptitle("Omega ratio contours, 1-year horizon (block bootstrap of 1986–2026 joint monthly returns)", x=0.01, ha="left", fontsize=11)
    fig.savefig(OUT / "omega_contours.png", dpi=170); plt.close(fig)


def frontier_plot():
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    F = RES["frontier"]
    xmax = max(p["omega_risk"] for f in F.values() for p in f) * 1.12
    ymax = max(p["mean"] for f in F.values() for p in f) * 1.08
    for om in (1.5, 2, 3, 4, 6):                               # iso-Omega rays: mean = (Omega - 1) * omega_risk  (tau = 0)
        x_end = min(xmax, ymax / (om - 1))
        ax.plot([0, x_end], [0, (om - 1) * x_end], color=GRID, lw=1, zorder=0)
        ax.text(x_end, (om - 1) * x_end, f"Ω={om:g} ", color=MUTED, fontsize=7, va="top", ha="right" if x_end < xmax else "left")
    for (name, pts), col in zip(F.items(), (BLUE, ORANGE, AQUA)):
        ax.plot([p["omega_risk"] for p in pts], [p["mean"] for p in pts], color=col, lw=2, label=f"Efficient frontier: {name}")
        key = "max-Omega, all 10 funds" if name == "all 10 funds" else f"max-Omega, {name}"
        b = RES["full"][key]
        ax.plot(b["omega_risk"], b["mean"], marker="*", ms=14, color=col, mec=SURF, mew=1.5, zorder=6)
    b = RES["full"]["max-Omega, all 10 funds"]
    ax.plot([0, b["omega_risk"]], [0, b["mean"]], color=BLUE, lw=1, ls=(0, (4, 3)))
    ax.annotate("max Ω, all funds: " + ", ".join(f"{v:.0%} {k}" for k, v in sorted(b["w"].items(), key=lambda kv: -kv[1])) + f"  (Ω {b['omega']:.2f})",
                (b["omega_risk"], b["mean"]), xytext=(30, -38), textcoords="offset points", fontsize=8, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8), bbox=dict(boxstyle="round,pad=0.25", fc=SURF, ec=GRID))
    off = []
    for a, s in RES["assets"].items():
        if s["omega_risk"] > xmax or s["mean"] < 0:
            off.append(f"{a}: mean {s['mean']:+.0%}, omega risk {s['omega_risk']:.0%}"); continue
        ax.plot(s["omega_risk"], s["mean"], marker="o", ms=5, color=INK2, zorder=5)
        ax.annotate(a, (s["omega_risk"], s["mean"]), xytext=(4, -11) if a in ("QQQ", "HYG") else (4, 3), textcoords="offset points", fontsize=8, color=INK2)
    r = RES["full"]["30 TQQQ/30 UGL/40 HYG"]
    ax.plot(r["omega_risk"], r["mean"], marker="o", ms=7, color=SURF, mec=INK, mew=1.5, zorder=6)
    ax.annotate("30/30/40 TQQQ/UGL/HYG", (r["omega_risk"], r["mean"]), xytext=(-150, 6), textcoords="offset points", fontsize=8)
    ax.set(xlim=(0, xmax), ylim=(min(0, min(s["mean"] for s in RES["assets"].values()) * 1.1), ymax),
           xlabel="Omega risk: expected 1-year loss below 0  (E[max(0 − R, 0)])", ylabel="Expected 1-year return (annualized, arithmetic mean)")
    ax.xaxis.set_major_formatter(pct); ax.yaxis.set_major_formatter(pct)
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    if off:
        ax.text(0.99, 0.30, "Off chart: " + "; ".join(off), transform=ax.transAxes, ha="right", fontsize=8, color=INK2)
    ax.text(0.99, 0.25, "Mean of 1-year returns; for leveraged funds the long-run compounded return is far lower\n(TQQQ: 50% mean vs about 15%/yr compounded 1986–2026).",
            transform=ax.transAxes, ha="right", fontsize=7.5, color=MUTED)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("Omega-risk efficient frontiers (τ = 0), long-only, rebalanced yearly. Stars = max-Omega portfolios", loc="left", fontsize=10)
    fig.savefig(OUT / "omega_frontier.png", dpi=170); plt.close(fig)


if __name__ == "__main__":
    contours(scenarios(MON, 0))
    frontier_plot()
    print("saved", OUT / "omega_contours.png", OUT / "omega_frontier.png")
