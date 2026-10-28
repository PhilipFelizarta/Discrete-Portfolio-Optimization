"""Diagnostics of the joint distribution used by study.py: marginals of each fund and the (empirical) copula.
Writes results/copula_tables.json and results/copula_marginals.png, results/copula_pairs.png.
Run: python omega/copula_report.py
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as st
sys.path.insert(0, str(Path(__file__).resolve().parent))
from study import MON, UNIV, scenarios
from omega_lp import rolling_annual

OUT = Path(__file__).resolve().parent.parent / "results"
KEY = ["TQQQ", "UGL", "SHNY", "HYG", "SPY", "GLD"]
BLUE, ORANGE, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": "#c3c2b7", "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def lower_tail(u, v, q=0.10):
    """Empirical lower-tail dependence at level q: P(U < q | V < q). Independence gives q."""
    return float(((u < q) & (v < q)).mean() / q)


def pseudo(x):
    return st.rankdata(x) / (len(x) + 1)


def gaussian_twin(S, n, seed=3):
    """Same marginals and Spearman matrix, but a Gaussian copula: shows what a Gaussian-copula model would imply."""
    rho_s = st.spearmanr(S).statistic
    rho = 2 * np.sin(np.pi * rho_s / 6)                                   # Spearman -> Pearson for a Gaussian copula
    w, V = np.linalg.eigh(rho); rho = V @ np.diag(np.maximum(w, 1e-8)) @ V.T
    z = np.random.default_rng(seed).multivariate_normal(np.zeros(len(rho)), rho, n)
    return st.norm.cdf(z)


if __name__ == "__main__":
    S = scenarios(MON, 0)[UNIV]
    Hh = pd.DataFrame(rolling_annual(MON[UNIV].values), columns=UNIV)
    out = {"monthly": {}, "annual_bootstrap": {}, "annual_history": {}, "spearman_annual": {}, "spearman_monthly": {}, "kendall_annual": {},
           "lower_tail_q10": {}, "lower_tail_q10_gaussian": {}, "crash_months": {}}
    for a in KEY:
        m = MON[a]
        out["monthly"][a] = {"mean": m.mean(), "vol_ann": m.std() * np.sqrt(12), "skew": float(st.skew(m)), "ex_kurt": float(st.kurtosis(m)),
                             "worst": m.min(), "best": m.max(), "worst_month": m.idxmin()}
        for k, X in (("annual_bootstrap", S[a]), ("annual_history", Hh[a])):
            out[k][a] = {"mean": X.mean(), "median": X.median(), "sd": X.std(), "p5": X.quantile(.05), "p95": X.quantile(.95),
                         "p_loss": (X < 0).mean(), "skew": float(st.skew(X))}
    U = np.column_stack([pseudo(S[a].values) for a in KEY])
    G = gaussian_twin(S[KEY].values, len(S))
    for i, a in enumerate(KEY):
        for j, b in enumerate(KEY):
            if j <= i: continue
            p = f"{a}-{b}"
            out["spearman_annual"][p] = float(st.spearmanr(S[a], S[b]).statistic)
            out["spearman_monthly"][p] = float(st.spearmanr(MON[a], MON[b]).statistic)
            out["kendall_annual"][p] = float(st.kendalltau(S[a].values[:5000], S[b].values[:5000]).statistic)
            out["lower_tail_q10"][p] = lower_tail(U[:, i], U[:, j])
            out["lower_tail_q10_gaussian"][p] = lower_tail(G[:, i], G[:, j])
    worst = MON["TQQQ"].nsmallest(int(len(MON) * 0.1)).index                # TQQQ's worst 10% of months
    for a in KEY[1:]:
        out["crash_months"][a] = {"mean_in_tqqq_worst_decile": MON.loc[worst, a].mean(), "mean_other_months": MON.drop(worst)[a].mean(),
                                  "pearson_in_worst_decile": float(MON.loc[worst, ["TQQQ", a]].corr().iloc[0, 1]),
                                  "pearson_all": float(MON[["TQQQ", a]].corr().iloc[0, 1])}
    js = json.loads(json.dumps(out, default=float))
    (OUT / "copula_tables.json").write_text(json.dumps(js, indent=1))

    # marginals: 1-year return distributions, bootstrap vs historical windows
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), constrained_layout=True)
    for ax, a in zip(axes.flat, KEY):
        lo, hi = np.percentile(S[a], [0.5, 99.5]); bins = np.linspace(lo, hi, 50)
        ax.hist(S[a], bins=bins, density=True, color=BLUE, alpha=0.85, label="Bootstrap (20,000)")
        ax.hist(Hh[a], bins=bins, density=True, histtype="step", color=ORANGE, lw=1.6, label="Historical 12-month windows")
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(f"{a}: mean {S[a].mean():+.0%}, median {S[a].median():+.0%}, P(loss) {(S[a] < 0).mean():.0%}", loc="left", fontsize=8.5)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}"); ax.set_yticks([])
    axes[0, 0].legend(frameon=False, fontsize=7.5)
    fig.suptitle("Marginal distributions of 1-year returns", x=0.01, ha="left", fontsize=11)
    fig.savefig(OUT / "copula_marginals.png", dpi=160); plt.close(fig)

    # copula: pseudo-observations (ranks) for the key pairs; lower-left corner = joint bad years
    pairs = [("TQQQ", "UGL"), ("TQQQ", "HYG"), ("UGL", "HYG"), ("TQQQ", "SHNY")]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), constrained_layout=True)
    idx = np.random.default_rng(0).choice(len(S), 3000, replace=False)
    for ax, (a, b) in zip(axes, pairs):
        ua, ub = pseudo(S[a].values)[idx], pseudo(S[b].values)[idx]
        ax.scatter(ua, ub, s=2, color=BLUE, alpha=0.35, linewidths=0)
        ax.add_patch(plt.Rectangle((0, 0), 0.1, 0.1, fill=False, ec=ORANGE, lw=1.5))
        p = f"{a}-{b}" if f"{a}-{b}" in js["spearman_annual"] else f"{b}-{a}"
        ax.set_title(f"{a} vs {b}\nSpearman {js['spearman_annual'][p]:+.2f}, λ_L(10%) {js['lower_tail_q10'][p]:.2f}", fontsize=8.5, loc="left")
        ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xlabel=f"rank of {a} 1-yr return", ylabel=f"rank of {b}")
    fig.suptitle("Empirical copula of 1-year returns (rank scatter). Box = both in their worst 10%; independence would put 1% of points there",
                 x=0.01, ha="left", fontsize=10)
    fig.savefig(OUT / "copula_pairs.png", dpi=160); plt.close(fig)
    for k in ("spearman_annual", "lower_tail_q10", "lower_tail_q10_gaussian"):
        print(k, {p: round(v, 2) for p, v in js[k].items() if "TQQQ" in p or "UGL-HYG" == p})
    print("crash", {a: {kk: round(vv, 3) for kk, vv in v.items()} for a, v in js["crash_months"].items()})
    for a in KEY:
        print(a, {k: round(v, 3) if isinstance(v, float) else v for k, v in js["monthly"][a].items()})
        print("   boot", {k: round(v, 3) for k, v in js["annual_bootstrap"][a].items()}, "\n   hist", {k: round(v, 3) for k, v in js["annual_history"][a].items()})
