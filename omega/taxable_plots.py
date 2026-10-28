"""Figure for the taxable study (needs results/taxable_summary.json from taxable.py).
  results/taxable_key.png : (a) after-tax compounded growth of the key portfolios under three views,
                            (b) rebalancing frequency, tax-free vs taxable
Run: python omega/taxable_plots.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / "results"
R = json.loads((OUT / "taxable_summary.json").read_text())
BLUE, ORANGE, AQUA, INK, INK2, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
pct = lambda v, _=None: f"{v:.0%}"
VIEWS = [("hist_path", "Actual history, 1987–2026", BLUE), ("hist_mc", "History + copula, 20-yr scenarios", ORANGE),
         ("options_mc", "Options market + copula, 20-yr scenarios", AQUA)]
TAX, IRA = "taxable, yearly + harvest", "IRA, yearly"


def label(name, w):
    mix = " / ".join(f"{v:.0%} {a}" for a, v in w.items())
    return f"{name}\n{mix}" if not name.startswith("100") else name


fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.2), constrained_layout=True, gridspec_kw={"width_ratios": [1.45, 1]})
names = list(R["portfolios"])
for i, n in enumerate(names):
    y0 = len(names) - 1 - i; p = R["portfolios"][n]
    for j, (vk, vl, col) in enumerate(VIEWS):
        y = y0 + (1 - j) * 0.22; t = p["views"][vk][TAX]["growth"]; ira = p["views"][vk][IRA]["growth"]
        a1.plot([t, ira], [y, y], color=col, lw=1.2, alpha=0.6)
        a1.plot(ira, y, "o", ms=7, mfc=SURF, mec=col, mew=1.6, zorder=4)
        a1.plot(t, y, "o", ms=8, color=col, mec=SURF, mew=1.5, zorder=5, label=vl if i == 0 else None)
        a1.text(min(t, ira) - 0.006, y, f"{t:+.1%}", ha="right", va="center", fontsize=7.5, color=INK2)
    pl = p["views"]["options_mc"][TAX]["p_loss"]
    a1.text(0.255, y0, f"{pl:.0%}", ha="center", va="center", fontsize=8.5, color=INK)
a1.text(0.255, len(names) - 0.45, "P(lose money\nover 20 yrs),\noptions view", ha="center", va="bottom", fontsize=7.5, color=INK2)
a1.set_yticks(range(len(names))[::-1]); a1.set_yticklabels([label(n, R["portfolios"][n]["w"]) for n in names], fontsize=8)
a1.axvline(0, color="#c3c2b7", lw=1); a1.grid(axis="x", color=GRID, lw=0.6); a1.set_axisbelow(True)
a1.set_xlim(-0.06, 0.28); a1.set_ylim(-0.6, len(names) + 0.35); a1.xaxis.set_major_formatter(pct)
a1.set_xlabel("After-tax compounded growth per year (filled = taxable, top federal bracket, yearly rebalance + loss harvesting; hollow = IRA)")
a1.legend(frameon=False, fontsize=8, loc="upper left", ncol=1, bbox_to_anchor=(0.0, 1.0))
a1.set_title("(a) Key portfolios: after-tax growth in a taxable account", loc="left", fontsize=10)

freq = ["monthly", "quarterly", "yearly", "every 2 years", "never"]; x = np.arange(len(freq))
for n, col in (("30 TQQQ/30 UGL/40 HYG (green)", BLUE), ("growth-max, 50/50 history + options", ORANGE), ("growth-max, options", AQUA)):
    F = R["rebalance_frequency"][n]
    ira = [F[f]["hist_mc"]["IRA"] for f in freq]; tx = [F[f]["hist_mc"]["top"] for f in freq]
    a2.plot(x, ira, color=col, lw=2, ls=(0, (4, 2)), marker="o", ms=6, mfc=SURF, mec=col)
    a2.plot(x, tx, color=col, lw=2, marker="o", ms=7, mec=SURF, label=n.replace(" (green)", " (green)"))
    a2.annotate("IRA", (x[-1], ira[-1]), xytext=(6, 0), textcoords="offset points", fontsize=7.5, color=INK2, va="center")
    a2.annotate("taxable", (x[-1], tx[-1]), xytext=(6, 0), textcoords="offset points", fontsize=7.5, color=INK2, va="center")
a2.set_xticks(x); a2.set_xticklabels(freq); a2.set_xlim(-0.3, len(freq) - 0.4)
a2.yaxis.set_major_formatter(pct); a2.grid(axis="y", color=GRID, lw=0.6); a2.set_axisbelow(True)
a2.set_ylabel("Compounded growth per year (history + copula, 20-yr scenarios)")
a2.legend(frameon=False, fontsize=8, loc="lower center")
a2.set_title("(b) How often to rebalance (dashed = IRA, solid = taxable)", loc="left", fontsize=10)
fig.suptitle("Taxable accounts: sales taxed at the long-term rate when held ≥ 1 year, otherwise as ordinary income; "
             "HYG income taxed yearly as ordinary income; UGL marked to market (60/40)", x=0.01, ha="left", fontsize=10.5)
fig.savefig(OUT / "taxable_key.png", dpi=160); plt.close(fig)
print("saved", OUT / "taxable_key.png")
