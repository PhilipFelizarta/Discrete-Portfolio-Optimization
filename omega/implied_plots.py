"""Figures for the option-implied study. Needs results/implied_summary.json from implied_study.py.
  results/implied_marginals.png : CDFs of 15-month returns per fund: risk-neutral (Q), real-world tilt (P), history
  results/implied_frontier.png  : omega risk vs annualised return; arithmetic-mean and log-growth frontiers, three measures
  results/implied_contours.png  : Omega contours over the TQQQ/UGL/HYG triangle under Q and P
  results/walk_forward.png      : 10-year-trailing walk-forward refits vs fixed mixes, 1997-2026
Run: python omega/implied_plots.py
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.tri import Triangulation
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pricedin_marginals as pm
from implied_study import marginals, joint, UNIV, TRIOS
from study import MON

ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT / "results"
RES = json.loads((OUT / "implied_summary.json").read_text()); T = RES["T"]
BLUE, ORANGE, AQUA, YELLOW, VIOLET, PINK, INK, INK2, MUTED, GRID, SURF = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#e87ba4", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb")
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
pct = lambda v, _=None: f"{v:.0%}"
MEAS = [("hist15", "History (15-month block bootstrap, 1986–2026)", INK2), ("Q", "Options, risk-neutral (Q)", BLUE),
        ("P", "Options + risk premium (P)", ORANGE)]


def marginal_plot(marg, H):
    key = ["TQQQ", "UPRO", "UGL", "SHNY", "HYG", "QQQ", "SPY", "GLD"]
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.8), constrained_layout=True)
    for ax, a in zip(axes.flat, key):
        m = marg[a]; o = np.argsort(m["x"]); x = m["x"][o] - 1
        lo, hi = np.percentile(H[a], 0.5), np.percentile(H[a], 99.5)
        lo = min(lo, np.interp(0.01, np.cumsum(m["Q"][o]), x)); hi = max(hi, np.interp(0.99, np.cumsum(m["P"][o]), x))
        hs = np.sort(H[a].values)
        ax.plot(hs, np.arange(1, len(hs) + 1) / len(hs), color=INK2, lw=1.6, label="History (bootstrap)")
        ax.plot(x, np.cumsum(m["Q"][o]), color=BLUE, lw=1.8, label="Options, Q")
        ax.plot(x, np.cumsum(m["P"][o]), color=ORANGE, lw=1.8, ls=(0, (5, 2)), label="Options, P")
        ax.axvline(0, color=GRID, lw=0.8)
        mq = RES["marginals"][a]
        ax.set_title(f"{a}  ({mq['source'].split(' (')[0]})\nmean/yr: hist {RES['hist15']['assets'][a]['mean_ann']:+.0%}, "
                     f"Q {mq['Q_mean_ann']:+.0%}, P {mq['P_mean_ann']:+.0%}", loc="left", fontsize=8)
        ax.set(xlim=(max(lo, -1), hi), ylim=(0, 1)); ax.xaxis.set_major_formatter(pct); ax.yaxis.set_major_formatter(pct)
        ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    axes[0, 0].legend(frameon=False, fontsize=7.5, loc="lower right")
    for ax in axes[:, 0]:
        ax.set_ylabel("P(15-month return ≤ x)")
    fig.suptitle(f"Marginal distributions of total return to Jan 21, 2028 ({T:.2f} years): option-implied vs history",
                 x=0.01, ha="left", fontsize=11)
    fig.savefig(OUT / "implied_marginals.png", dpi=160); plt.close(fig)


def frontier_plot():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), constrained_layout=True, sharey=True)
    ymax = max(max(p["mean_ann"] for p in RES[k]["frontier"]["all 10 funds"]) for k, _, _ in MEAS) * 1.08
    xmax = max(max(p["omega_risk"] for p in RES[k]["frontier"]["all 10 funds"]) for k, _, _ in MEAS) * 1.05
    ymin = min(-0.05, min(min(p["growth_ann"] for p in RES[k]["growth_frontier"]["all 10 funds"]) for k, _, _ in MEAS) - 0.02)
    offs = {"hist15": ((70, -34), (-14, 22)), "Q": ((18, 26), (-10, -30)), "P": ((18, 26), (-10, -30))}
    for ax, (k, title, _) in zip(axes, MEAS):
        R = RES[k]; o_om, o_g = offs[k]
        for name, col in (("all 10 funds", BLUE), ("TQQQ/UGL/HYG", ORANGE)):
            fa, fg = R["frontier"][name], R["growth_frontier"][name]
            ax.plot([p["omega_risk"] for p in fa], [p["mean_ann"] for p in fa], color=col, lw=2, label=f"{name}: arithmetic mean")
            ax.plot([p["omega_risk"] for p in fg], [p["growth_ann"] for p in fg], color=col, lw=2, ls=(0, (4, 2)), label=f"{name}: compounded growth")
        for nm, col, mk in (("max-Omega, all 10 funds", BLUE, "*"), ("max-Omega, TQQQ/UGL/HYG", ORANGE, "*")):
            b = R["portfolios"][nm]
            ax.plot(b["omega_risk"], b["mean_ann"], marker=mk, ms=14, color=col, mec=SURF, mew=1.2, zorder=6)
        b = R["portfolios"]["max-Omega, all 10 funds"]
        ax.annotate("max Ω: " + ", ".join(f"{v:.0%} {a}" for a, v in sorted(b["w"].items(), key=lambda kv: -kv[1]) if v >= 0.02) + f"\n(Ω {b['omega']:.2f})",
                    (b["omega_risk"], b["mean_ann"]), xytext=o_om, textcoords="offset points", fontsize=7.5,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8), bbox=dict(boxstyle="round,pad=0.25", fc=SURF, ec=GRID))
        g = R["max_growth"]
        ax.plot(g["omega_risk"], g["growth_ann"], marker="D", ms=7, color=BLUE, mec=SURF, zorder=6)
        ax.annotate("max growth: " + ", ".join(f"{v:.0%} {a}" for a, v in sorted(g["w"].items(), key=lambda kv: -kv[1]) if v >= 0.02),
                    (g["omega_risk"], g["growth_ann"]), xytext=o_g, textcoords="offset points", ha="right", fontsize=7.5,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8), bbox=dict(boxstyle="round,pad=0.25", fc=SURF, ec=GRID))
        for a in ("TQQQ", "UGL", "HYG", "SPY", "QQQ", "SHNY"):
            s = R["assets"][a]
            if s["omega_risk"] <= xmax:
                ax.plot(s["omega_risk"], s["mean_ann"], "o", ms=4.5, color=INK2, zorder=5)
                ax.plot(s["omega_risk"], s["growth_ann"], "o", ms=4.5, mfc=SURF, mec=INK2, zorder=5)
                ax.plot([s["omega_risk"]] * 2, [s["growth_ann"], s["mean_ann"]], color=GRID, lw=0.8, zorder=1)
                ax.annotate(a, (s["omega_risk"], s["mean_ann"]), xytext=(4, 2), textcoords="offset points", fontsize=7.5, color=INK2)
        r = R["portfolios"]["30 TQQQ/30 UGL/40 HYG"]
        ax.plot(r["omega_risk"], r["mean_ann"], "o", ms=7, color=SURF, mec=INK, mew=1.4, zorder=6)
        ax.plot(r["omega_risk"], r["growth_ann"], "s", ms=6, color=SURF, mec=INK, mew=1.4, zorder=6)
        ax.annotate("30/30/40", (r["omega_risk"], r["growth_ann"]), xytext=(5, -10), textcoords="offset points", fontsize=7.5)
        ax.axhline(0, color="#c3c2b7", lw=1); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.set(xlim=(0, xmax), ylim=(ymin, ymax), title=title, xlabel="Omega risk: expected loss below 0 over the 15-month horizon")
        ax.title.set_fontsize(10); ax.title.set_ha("left"); ax.title.set_x(0)
        ax.xaxis.set_major_formatter(pct); ax.yaxis.set_major_formatter(pct)
    axes[0].set_ylabel("Annualised return  (solid: arithmetic mean · dashed: compounded growth, exp(E log)/yr)")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    axes[2].text(0.99, 0.02, "Dots: single funds (filled = mean, hollow = compounded growth).\nStars: max-Omega (τ = 0). Diamond: growth-optimal.",
                 transform=axes[2].transAxes, ha="right", fontsize=7.5, color=MUTED)
    fig.suptitle("Omega-risk efficient frontiers to Jan 2028, long-only, held to the horizon: history vs option-implied marginals (historical copula)",
                 x=0.01, ha="left", fontsize=11)
    fig.savefig(OUT / "implied_frontier.png", dpi=160); plt.close(fig)


def contour_plot(sets):
    cols = TRIOS["TQQQ/UGL/HYG"]; step = 0.01
    P = np.array([(s, g) for s in np.arange(0, 1 + 1e-9, step) for g in np.arange(0, 1 - s + 1e-9, step)])
    W = np.c_[P[:, 0], P[:, 1], 1 - P.sum(1)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.9), constrained_layout=True)
    oms = {k: (lambda X: np.maximum(X, 0).mean(0) / np.maximum(-X, 0).mean(0))(S[cols].values @ W.T) for k, S in sets.items()}
    vmax = np.percentile(np.concatenate(list(oms.values())), 99.5)
    levels = np.linspace(1, max(vmax, 2), 15)
    tri = Triangulation(P[:, 1], P[:, 0])
    for ax, (k, title, _) in zip(axes, MEAS):
        om = oms[k]
        cf = ax.tricontourf(tri, np.clip(om, levels[0], levels[-1] - 1e-9), levels=levels, cmap=SEQ)
        cs = ax.tricontour(tri, om, levels=[l for l in (1.5, 2, 3, 4, 5, 6, 8) if l < levels[-1]], colors=SURF, linewidths=0.9)
        ax.clabel(cs, fmt=lambda v: f"{v:g}", fontsize=7, colors=INK)
        b = RES[k]["portfolios"]["max-Omega, TQQQ/UGL/HYG"]; bw = b["w"]; bs, bg = bw.get("TQQQ", 0), bw.get("UGL", 0)
        ax.plot(bg, bs, marker="*", ms=15, color=ORANGE, mec=SURF, mew=1.5, zorder=5)
        ax.annotate(f"max Ω {b['omega']:.2f}\n{bs:.0%} TQQQ / {bg:.0%} UGL / {1 - bs - bg:.0%} HYG", (bg, bs), xytext=(14, 8),
                    textcoords="offset points", fontsize=8, bbox=dict(boxstyle="round,pad=0.25", fc=SURF, ec=GRID))
        r = RES[k]["portfolios"]["30 TQQQ/30 UGL/40 HYG"]
        ax.plot(0.3, 0.3, marker="o", ms=7, color=SURF, mec=INK, mew=1.5, zorder=5)
        ax.annotate(f"30/30/40 (Ω {r['omega']:.2f})", (0.3, 0.3), xytext=(10, -14), textcoords="offset points", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.25", fc=SURF, ec=GRID))
        ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xlabel="UGL weight  (HYG = the rest)", ylabel="TQQQ weight")
        ax.xaxis.set_major_formatter(pct); ax.yaxis.set_major_formatter(pct)
        ax.set_title(title, loc="left", fontsize=9.5)
    fig.colorbar(cf, ax=axes, shrink=0.8, label="Omega(τ = 0) over the 15-month horizon")
    fig.suptitle("Omega contours, TQQQ/UGL/HYG held to Jan 2028", x=0.01, ha="left", fontsize=11)
    fig.savefig(OUT / "implied_contours.png", dpi=160); plt.close(fig)


def walk_forward_plot():
    wf = RES["walk_forward"]; years = [wf["start"] - 1] + [r["year"] for r in wf["years"]]
    fig, ax = plt.subplots(figsize=(10, 5.4), constrained_layout=True)
    style = {"max-Omega, all 10 funds": (BLUE, "-"), "max-Omega, TQQQ/UGL/HYG": (ORANGE, "-"), "max-growth, all 10 funds": (VIOLET, "-"),
             "30 TQQQ/30 UGL/40 HYG": (AQUA, "-"), "100 SPY": (INK2, (0, (4, 2))), "100 QQQ": (MUTED, (0, (1, 1.5))), "100 TQQQ": (PINK, (0, (4, 2)))}
    for k, p in wf["paths"].items():
        col, ls = style[k]; s = wf["summary"][k]
        ax.plot(np.array(years) + 1, p, color=col, ls=ls, lw=2 if "Omega" in k or "30" in k else 1.4,
                label=f"{k}: {s['cagr']:+.1%}/yr, worst year {s['worst_year']:+.0%}")
    ax.set_yscale("log"); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.set(xlabel="End of year", ylabel="Growth of $1 (log scale)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title(f"Walk-forward, {wf['start']}–2026: each January re-solve on the trailing {wf['train_years']} years, hold 12 months "
                 "(no look-ahead). Fixed mixes rebalanced yearly", loc="left", fontsize=9.5)
    fig.savefig(OUT / "walk_forward.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    marg, _, _, _ = marginals()
    SQ, H = joint(marg, "Q"); SP, _ = joint(marg, "P")
    marginal_plot(marg, H)
    frontier_plot()
    contour_plot({"hist15": H, "Q": SQ, "P": SP})
    walk_forward_plot()
    print("saved implied_marginals.png, implied_frontier.png, implied_contours.png, walk_forward.png")
