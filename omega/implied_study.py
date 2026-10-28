"""Omega study with OPTION-IMPLIED marginals, plus the two follow-up checks (compounded growth, walk-forward).

1. Marginals: one per fund, from its Jan-2028 option chain (CBOE delayed quotes, data/chains/) with the PricedIn
   formulation solved as a QP (pricedin_marginals.fit_extended). SHNY has no listed options, so it is mapped from
   GLD's implied distribution. Two measures:
     Q  risk-neutral, as priced: every fund's expected total return = T-bill (less the leveraged funds' decay);
     P  real-world: Esscher tilt adding a risk premium, equity 5%/yr, high yield 2%/yr, gold 0 (L x for L-times funds).
2. Copula: historical ranks. 20,000 joint 15-month scenarios from the same block bootstrap as study.py (horizon =
   option expiry, 1.23 years). Each fund's rank in a scenario is mapped through its implied inverse CDF.
3. Max-Omega (tau = 0) and two frontiers: omega risk vs (a) arithmetic mean, (b) expected log growth (realised).
4. Walk-forward on history: refit max-Omega each January on the trailing 10 years, hold 12 months, 1997-2026.
Writes results/implied_summary.json.  Run: python omega/implied_study.py   (needs cvxpy; set PRICEDIN_ALGO_PATH to
also benchmark PricedIn's own SLSQP solver).
"""
import json, sys
from datetime import date
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats as st
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pricedin_marginals as pm
from scipy.optimize import minimize
from omega_lp import max_omega, min_lpm, frontier, lpm1, omega, bootstrap_annual
from study import MON, UNIV, TRIOS, REFS, history, as_dict

ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT / "results"
EXPIRY, VALUATION = "280121", date(2026, 10, 27)
PREMIUM = {"QQQ": .05, "SPY": .05, "GLD": 0.0, "HYG": .02}
CHAINED = ["QQQ", "SPY", "GLD", "HYG", "TQQQ", "QLD", "UPRO", "SSO", "UGL"]


def marginals():
    """Per fund: support x (gross total return over the horizon) and probabilities under Q and P."""
    fits, out, bench = {}, {}, {}
    for s in CHAINED:
        c = pm.chain_inputs(s, EXPIRY, VALUATION)
        f = pm.fit_extended(c); fits[s] = (c, f)
        qp = pm.fit_qp(c); al = pm.compare_with_algo_layer(c)
        bench[s] = {"n_strikes": len(qp["k"]), "qp_seconds": qp["seconds"], "qp_objective": qp["objective"],
                    "asis_mass_at_0": float(qp["pmf"][0]), "asis_mass_at_top": float(qp["pmf"][-1]),
                    "asis_mean_over_F": float(qp["pmf"] @ qp["k"] / c["F"]),
                    "extended_seconds": f["seconds"], "extended_rmse_half_spreads": f["rmse_half_spreads"],
                    "extended_share_within_spread": f["share_within_spread"], "n_otm_quotes": f["n_quotes"],
                    "strike_range_over_spot": f["strike_range"], "forward_over_spot": c["F"] / c["spot"], "parity_carry_q": c["q"],
                    "holder_yield_d": c["d"]}
        if al:
            bench[s].update({"slsqp_seconds": al["seconds"], "slsqp_converged": al["converged"], "slsqp_objective": al.get("objective")})
    T = fits["QQQ"][0]["T"]
    for s in CHAINED:
        c, f = fits[s]; x = pm.gross_tr(c, f); prem = PREMIUM.get(s)
        if prem is None:
            u, L = pm.LEV[s][:2]; prem = L * PREMIUM[u]
        pP, g = pm.tilt(x, f["pmf"], T, prem)
        out[s] = {"x": x, "Q": f["pmf"], "P": pP, "gamma": g, "source": f"{s} options"}
    # SHNY: no listed options -> map from GLD (Q and P separately; gold premium is 0 so they coincide)
    u, L, spread, fee = pm.LEV["SHNY"]
    y, _ = pm.leveraged_from(out[u]["x"], out[u]["Q"], T, L, spread, fee)
    out["SHNY"] = {"x": y, "Q": out[u]["Q"], "P": out[u]["P"], "gamma": 0.0, "source": "mapped from GLD options (3x, T-bill + 7%)"}
    # direct vs mapped check for the funds that have both
    check = {}
    for s in ["TQQQ", "QLD", "UPRO", "SSO", "UGL"]:
        u, L, spread, fee = pm.LEV[s]
        y, p = pm.leveraged_from(out[u]["x"], out[u]["Q"], T, L, spread, fee)
        U = np.array([.05, .25, .5, .75, .95])
        check[s] = {"direct": pm.quantile_fn(out[s]["x"], out[s]["Q"])(U).tolist(), "mapped": pm.quantile_fn(y, p)(U).tolist()}
    return out, T, bench, check


def joint(marg, measure, n=20000, seed=0, months=15):
    """Historical-rank copula: block-bootstrap ranks of 15-month returns, pushed through each implied inverse CDF."""
    H = pd.DataFrame(bootstrap_annual(MON[UNIV].values, n=n, months=months, seed=seed), columns=UNIV)
    S = {}
    for a in UNIV:
        u = st.rankdata(H[a].values, method="ordinal") / (n + 1)
        S[a] = pm.quantile_fn(marg[a]["x"], marg[a][measure])(u) - 1
    return pd.DataFrame(S), H


def ann(m, T):
    return (1 + m) ** (1 / T) - 1


def stats(w, R, T):
    p = R @ w
    return {"mean_horizon": float(p.mean()), "mean_ann": float(ann(p.mean(), T)), "growth_ann": float(np.exp(np.log1p(p).mean() / T) - 1),
            "median_ann": float(ann(np.median(p), T)), "omega_risk": float(lpm1(w, R)), "omega": float(omega(w, R)),
            "p_loss": float((p < 0).mean()), "p5": float(np.percentile(p, 5))}


def max_growth(R, cap=None, w0=None):
    """max E[log(1 + R w)]  s.t.  1'w = 1, w >= 0  [and LPM1(w) <= cap].  10 weights, concave objective, convex
    constraint set, so SLSQP with exact gradients finds the global optimum (the LPM constraint is piecewise linear
    over 20,000 scenarios, smooth in practice)."""
    M, n = R.shape; G = 1 + R
    f = lambda w: -np.log(np.maximum(G @ w, 1e-12)).mean()
    df = lambda w: -(G / np.maximum(G @ w, 1e-12)[:, None]).mean(0)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(n)}]
    if cap is not None:
        cons.append({"type": "ineq", "fun": lambda w: cap - lpm1(w, R), "jac": lambda w: R[(R @ w) < 0].sum(0) / M})
    best = None
    for start in ([w0] if w0 is not None else []) + [np.full(n, 1 / n)]:
        r = minimize(f, start, jac=df, constraints=cons, bounds=[(0, 1)] * n, method="SLSQP", options={"maxiter": 500, "ftol": 1e-12})
        w = np.maximum(r.x, 0); w /= w.sum()
        if cap is None or lpm1(w, R) <= cap * 1.001 + 1e-6:
            if best is None or f(w) < f(best):
                best = w
    return best


def growth_frontier(R, T, points=25):
    """(omega risk, annualised expected log growth, w) along  max E log(1 + R w)  s.t.  LPM1(w) <= x."""
    n = R.shape[1]
    w_lo = min_lpm(R, float(R.mean(0).min()) - 1.0)                 # minimum omega-risk portfolio
    w_hi = max_growth(R)                                             # unconstrained growth-optimal portfolio
    out, w = [], w_lo
    for x in np.linspace(lpm1(w_lo, R), lpm1(w_hi, R), points):
        w = max_growth(R, x, w)
        if w is not None:
            out.append((float(lpm1(w, R)), float(np.exp(np.log1p(R @ w).mean() / T) - 1), w))
        else:
            w = w_lo
    return out


def block(S, T, label):
    """Max-Omega, reference mixes and both frontiers for one scenario set."""
    res = {"assets": {a: stats(np.array([1.0]), S[[a]].values, T) for a in UNIV}, "portfolios": {}, "frontier": {}, "growth_frontier": {}}
    w, _ = max_omega(S[UNIV].values)
    res["portfolios"]["max-Omega, all 10 funds"] = {"w": as_dict(w, UNIV), **stats(w, S[UNIV].values, T)}
    for name, cols in TRIOS.items():
        w, _ = max_omega(S[cols].values)
        res["portfolios"][f"max-Omega, {name}"] = {"w": as_dict(w, cols), **stats(w, S[cols].values, T)}
    for name, wd in REFS.items():
        cols = list(wd); res["portfolios"][name] = {"w": wd, **stats(np.array([wd[c] for c in cols]), S[cols].values, T)}
    for name, cols in [("all 10 funds", UNIV), ("TQQQ/UGL/HYG", TRIOS["TQQQ/UGL/HYG"])]:
        R = S[cols].values
        res["frontier"][name] = [{"omega_risk": float(a), "mean_ann": float(ann(b, T)), "w": as_dict(w, cols)} for a, b, w in frontier(R, 0.0, 30)]
        g = growth_frontier(R, T)
        res["growth_frontier"][name] = [{"omega_risk": a, "growth_ann": b, "w": as_dict(w, cols)} for a, b, w in g]
    best = max(res["growth_frontier"]["all 10 funds"], key=lambda r: r["growth_ann"])
    res["max_growth"] = best
    print(f"[{label}] max-Ω all: {res['portfolios']['max-Omega, all 10 funds']['w']} Ω {res['portfolios']['max-Omega, all 10 funds']['omega']:.2f}; "
          f"trio: {res['portfolios']['max-Omega, TQQQ/UGL/HYG']['w']} Ω {res['portfolios']['max-Omega, TQQQ/UGL/HYG']['omega']:.2f}; "
          f"max growth {best['growth_ann']:+.1%} {best['w']}", flush=True)
    return res


def walk_forward(train_years=10, start=1997, n=5000):
    """Each January: max-Omega on 1-year bootstrap scenarios from the trailing `train_years` of months; hold 12 months."""
    rows, paths = [], {k: [1.0] for k in ["max-Omega, all 10 funds", "max-Omega, TQQQ/UGL/HYG", "max-growth, all 10 funds",
                                          "30 TQQQ/30 UGL/40 HYG", "100 SPY", "100 QQQ", "100 TQQQ"]}
    years = sorted({int(m[:4]) for m in MON.index if int(m[:4]) >= start})
    for y in years:
        tr = MON[(MON.index >= f"{y - train_years}-01") & (MON.index <= f"{y - 1}-12")]
        te = MON[(MON.index >= f"{y}-01") & (MON.index <= f"{y}-12")]
        S = pd.DataFrame(bootstrap_annual(tr[UNIV].values, n=n, seed=y), columns=UNIV)
        wa, _ = max_omega(S[UNIV].values); wt, _ = max_omega(S[TRIOS["TQQQ/UGL/HYG"]].values)
        wg = max_growth(S[UNIV].values)
        W = {"max-Omega, all 10 funds": dict(zip(UNIV, wa)), "max-Omega, TQQQ/UGL/HYG": dict(zip(TRIOS["TQQQ/UGL/HYG"], wt)),
             "max-growth, all 10 funds": dict(zip(UNIV, wg)), **REFS}
        g = {}
        for k in paths:
            cols = [c for c in W[k] if W[k][c] > 1e-6]; w = np.array([W[k][c] for c in cols]); w = w / w.sum()
            g[k] = float(np.prod(1 + te[cols].values, axis=0) @ w)       # buy-and-hold for the year
            paths[k].append(paths[k][-1] * g[k])
        rows.append({"year": y, "months": len(te), "returns": {k: v - 1 for k, v in g.items()},
                     "w_all": as_dict(wa, UNIV), "w_trio": as_dict(wt, TRIOS["TQQQ/UGL/HYG"]), "w_growth": as_dict(W["max-growth, all 10 funds"].values(), UNIV)})
    yrs = (len(MON[MON.index >= f"{start}-01"])) / 12
    summ = {}
    for k, p in paths.items():
        p = np.array(p); summ[k] = {"cagr": float(p[-1] ** (1 / yrs) - 1), "final": float(p[-1]),
                                    "worst_year": float(min(r["returns"][k] for r in rows)), "maxdd_yearly": float((p / np.maximum.accumulate(p) - 1).min())}
    return {"start": start, "train_years": train_years, "years": rows, "summary": summ, "paths": {k: list(map(float, v)) for k, v in paths.items()}}


if __name__ == "__main__":
    marg, T, bench, check = marginals()
    for s, b in bench.items():
        print(s, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in b.items()}, flush=True)
    res = {"expiry": EXPIRY, "valuation": str(VALUATION), "T": T, "r": pm.R_TBILL, "premium": PREMIUM, "benchmark": bench, "direct_vs_mapped": check,
           "marginals": {}}
    U = np.array([.01, .05, .25, .5, .75, .95, .99])
    for a in UNIV:
        m = marg[a]
        res["marginals"][a] = {"source": m["source"], "gamma": m["gamma"],
                               **{f"{ms}_quantiles": (pm.quantile_fn(m["x"], m[ms])(U) - 1).tolist() for ms in ("Q", "P")},
                               **{f"{ms}_mean_ann": float(ann(m[ms] @ m["x"] - 1, T)) for ms in ("Q", "P")}}
    res["quantile_levels"] = U.tolist()
    sets = {}
    for ms in ("Q", "P"):
        S, H = joint(marg, ms); sets[ms] = S
        res[ms] = block(S, T, ms)
    res["hist15"] = block(H, T, "history, 15-month bootstrap")
    for a in UNIV:
        res["marginals"][a]["hist_quantiles"] = np.quantile(H[a], U).tolist()
    res["walk_forward"] = walk_forward()
    for k, v in res["walk_forward"]["summary"].items():
        print(f"WF {k:28s} CAGR {v['cagr']:+.1%}  worst year {v['worst_year']:+.0%}  DD {v['maxdd_yearly']:.0%}")
    (OUT / "implied_summary.json").write_text(json.dumps(res, indent=1, default=float))
    print("saved", OUT / "implied_summary.json")
