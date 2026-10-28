"""Omega study for leveraged-ETF lazy portfolios (tau = 0: a loss is any negative 1-year return).

Joint distribution: 20,000 one-year scenarios from a stationary block bootstrap (mean block 6 months) of the joint
monthly returns in data/monthly_returns.csv. Holding fixed weights for a year = rebalancing once a year, so w'R_j is
the yearly-rebalanced portfolio's return in scenario j.
Writes results/summary.json (used by plots.py).  Run: python omega/study.py
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from omega_lp import max_omega, frontier, lpm1, omega, bootstrap_annual, rolling_annual

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"; OUT.mkdir(exist_ok=True)
MON = pd.read_csv(ROOT / "data" / "monthly_returns.csv", index_col=0)
UNIV = ["TQQQ", "QLD", "UPRO", "SSO", "SHNY", "UGL", "HYG", "QQQ", "SPY", "GLD"]   # cash left out: it never loses, so Omega is unbounded
TRIOS = {"TQQQ/SHNY/HYG": ["TQQQ", "SHNY", "HYG"], "TQQQ/UGL/HYG": ["TQQQ", "UGL", "HYG"]}
REFS = {"30 TQQQ/30 UGL/40 HYG": {"TQQQ": .3, "UGL": .3, "HYG": .4}, "50 TQQQ/10 SHNY/40 HYG": {"TQQQ": .5, "SHNY": .1, "HYG": .4},
        "100 QQQ": {"QQQ": 1}, "100 SPY": {"SPY": 1}, "100 TQQQ": {"TQQQ": 1}}


def scenarios(months, seed):
    return pd.DataFrame(bootstrap_annual(months.values, n=20000, seed=seed), columns=months.columns)


def stats(w, R):
    p = R @ w
    return {"mean": float(p.mean()), "median": float(np.median(p)), "omega_risk": float(lpm1(w, R)), "omega": float(omega(w, R)),
            "p_loss": float((p < 0).mean()), "p5": float(np.percentile(p, 5))}


def history(wd, months):
    """Yearly-rebalanced lump sum on the actual monthly path: CAGR and worst drop."""
    cols = list(wd); w = np.array([wd[c] for c in cols]); X = months[cols].values; v = w.copy(); path = []
    for k, m in enumerate(months.index):
        if k and m.endswith("-01"):
            v = v.sum() * w
        v = v * (1 + X[k]); path.append(v.sum())
    path = np.array(path)
    return {"cagr": float(path[-1] ** (12 / len(path)) - 1), "maxdd": float((path / np.maximum.accumulate(path) - 1).min())}


def as_dict(w, cols):
    return {c: round(float(x), 4) for c, x in zip(cols, w) if x > 5e-4}


def solve(S, months):
    """Max-Omega portfolios for the full universe and each trio, plus the reference mixes."""
    out = {}
    w, _ = max_omega(S[UNIV].values)
    out["max-Omega, all 10 funds"] = {"w": as_dict(w, UNIV), **stats(w, S[UNIV].values)}
    for name, cols in TRIOS.items():
        w, _ = max_omega(S[cols].values)
        out[f"max-Omega, {name}"] = {"w": as_dict(w, cols), **stats(w, S[cols].values)}
    for name, wd in REFS.items():
        cols = list(wd)
        out[name] = {"w": wd, **stats(np.array([wd[c] for c in cols]), S[cols].values)}
    for v in out.values():
        v["history"] = history(v["w"], months)
    return out


if __name__ == "__main__":
    S = scenarios(MON, 0)
    res = {"months": [MON.index[0], MON.index[-1]], "n_scenarios": len(S), "assets": {}}
    for a in UNIV:
        res["assets"][a] = stats(np.array([1.0]), S[[a]].values)
    res["full"] = solve(S, MON)

    # frontiers (omega risk vs expected 1-year return)
    res["frontier"] = {}
    for name, cols in [("all 10 funds", UNIV)] + list(TRIOS.items()):
        res["frontier"][name] = [{"omega_risk": float(a), "mean": float(b), "w": as_dict(w, cols)} for a, b, w in frontier(S[cols].values, 0.0, 40)]

    # robustness 1: historical overlapping 12-month windows instead of the bootstrap
    H = rolling_annual(MON[UNIV].values); wh, _ = max_omega(H)
    res["historical_windows"] = {"w": as_dict(wh, UNIV), **stats(wh, H), "n_windows": int(len(H))}

    # robustness 2: fit on 1986-2005, evaluate on 2006-2026
    ins, oos = MON[MON.index <= "2005-12"], MON[MON.index >= "2006-01"]
    Si, So = scenarios(ins, 1), scenarios(oos, 2)
    fit = solve(Si, ins)
    res["out_of_sample"] = {}
    for name, v in fit.items():
        cols = list(v["w"]); w = np.array([v["w"][c] for c in cols])
        res["out_of_sample"][name] = {"w": v["w"], "fit_1986_2005": stats(w, Si[cols].values), "test_2006_2026": stats(w, So[cols].values),
                                      "history_2006_2026": history(v["w"], oos)}
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    for k, v in res["full"].items():
        print(f"{k:28s} Ω {v['omega']:.2f}  mean {v['mean']:+.1%}  ω-risk {v['omega_risk']:.1%}  P(loss) {v['p_loss']:.0%}  "
              f"hist CAGR {v['history']['cagr']:+.1%} DD {v['history']['maxdd']:.0%}  {v['w']}")
    print("historical windows:", res["historical_windows"]["w"], round(res["historical_windows"]["omega"], 2))
    for k, v in res["out_of_sample"].items():
        print(f"OOS {k:28s} fit Ω {v['fit_1986_2005']['omega']:.2f} -> test Ω {v['test_2006_2026']['omega']:.2f}, "
              f"2006-26 CAGR {v['history_2006_2026']['cagr']:+.1%} DD {v['history_2006_2026']['maxdd']:.0%}  {v['w']}")
