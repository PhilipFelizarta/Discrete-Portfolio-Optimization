"""Taxable-account study: key portfolios, rebalancing rules, and after-tax growth-maximising mixes.

Three views of the future, all with the same joint dependence (historical copula):
  hist_path  the actual monthly path, Jan 1987 - Oct 2026 (simulated funds before inception; see data/README.md)
  hist_mc    10,000 twenty-year paths: stationary block bootstrap (mean block 6 months) of joint monthly returns
  options_mc 10,000 twenty-year paths: 16 i.i.d. draws of the option-implied P joint scenarios (15 months each,
             from implied_study.py), so every rebalance is > 1 year apart

Tax model (US federal, no state). Average-cost basis per fund; tax for each tax year paid from the portfolio:
  distributions  taxed when paid, reinvested (add basis). Yields below; character: qualified (SPY, QQQ) or ordinary
                 (HYG interest; leveraged-ETF payouts come from T-bill interest)
  sales          long-term rate if the position was last rebalanced >= 12 months ago, else ordinary;
                 GLD (grantor trust) is a collectible: 28% cap on long-term gains
  UGL            ProShares Trust II commodity pool (K-1): Section 1256, marked to market every year at 60/40
  capital losses net against capital gains; net losses carry forward (the $3k/yr offset vs income is ignored)
  harvesting     sell a fund whose value is >= 10% below basis and hold a non-identical substitute for 31 days
                 (TQQQ <-> UPRO/QLD, SPY <-> VOO-like index fund, ...). Modeled as resetting basis with unchanged returns
  end            everything sold, gains taxed (no step-up)
Run: python omega/taxable.py   -> results/taxable_summary.json
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).resolve().parent))
from study import MON, UNIV

ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT / "results"; CACHE = ROOT.parent / "cache"; CACHE.mkdir(exist_ok=True)
BRACKETS = {"top": {"ord": .408, "lt": .238, "coll": .318}, "mid": {"ord": .24, "lt": .15, "coll": .24}}
YIELD = {"SPY": .013, "QQQ": .006, "TQQQ": .010, "UPRO": .006, "QLD": .005, "SSO": .006, "HYG": .065, "GLD": 0, "UGL": 0, "SHNY": 0}
DIST = {"SPY": "lt", "QQQ": "lt"}                      # qualified dividends; everything else that pays is ordinary
GAIN = {"GLD": "coll", "UGL": "s1256"}                 # default: "lt"
HORIZON = 20


def rates(bracket, assets):
    b = dict(BRACKETS[bracket]); b["s1256"] = .6 * b["lt"] + .4 * b["ord"]
    d = np.array([b[DIST.get(a, "ord")] for a in assets])
    lt = np.array([b[GAIN.get(a, "lt")] for a in assets])
    st = np.array([b["s1256"] if GAIN.get(a) == "s1256" else b["ord"] for a in assets])
    return d, lt, st, np.array([GAIN.get(a) == "s1256" for a in assets])


def simulate(G, dt, w, assets, rule="yearly", bracket="top", harvest=False, band=0.10, steps_per_year=12):
    """G: (paths, steps, n) gross total return per step. Returns after-tax final wealth (paths,), pre-liquidation
    wealth path (paths, steps), total taxes paid. rule: none | yearly | band.  bracket None = tax-free account."""
    P, S, n = G.shape; w = np.asarray(w, float)
    y = np.array([YIELD[a] for a in assets]) * dt
    if bracket is None:
        rd = rlt = rst = np.zeros(n); mtm = np.zeros(n, bool)
    else:
        rd, rlt, rst, mtm = rates(bracket, assets)
    V = np.tile(w, (P, 1)); B = V.copy(); carry = np.zeros(P); liab = np.zeros(P); dtax = np.zeros(P); paid = np.zeros(P)
    since = np.zeros(P); path = np.empty((P, S)); yr = max(int(round(steps_per_year)), 1)
    for s in range(S):
        Vp = V; V = V * G[:, s, :]; since += dt
        D = y * Vp; dtax += D @ rd; B = B + D
        year_end = (s + 1) % yr == 0 if yr > 1 else True
        if harvest and bracket is not None:
            loss = (V < 0.9 * B) & ~mtm[None, :]
            liab += np.where(loss, (V - B) * rlt, 0).sum(1); B = np.where(loss, V, B)
        tot = V.sum(1, keepdims=True); cur = V / np.maximum(tot, 1e-300)
        if rule == "yearly":
            do = np.full(P, year_end)
        elif rule.startswith("every"):                              # every k steps, e.g. every3 = quarterly on monthly data
            do = np.full(P, (s + 1) % int(rule[5:]) == 0)
        elif rule == "band":
            do = np.abs(cur - w).max(1) > band
        else:
            do = np.zeros(P, bool)
        if do.any():
            tgt = w * tot
            sell = np.where(do[:, None], np.maximum(V - tgt, 0), 0)
            f = np.divide(sell, V, out=np.zeros_like(V), where=V > 0)
            g = (V - B) * f
            r = np.where((since >= 1 - 1e-9)[:, None], rlt, rst)
            liab += (g * np.where(mtm, 0, r)).sum(1)                   # UGL gains are taxed by the mark below
            B = np.where(do[:, None], B * (1 - f) + np.maximum(tgt - V, 0), B)
            V = np.where(do[:, None], tgt, V); since = np.where(do, 0, since)
        if year_end and bracket is not None:
            m = (V - B) * mtm; liab += (m * rlt).sum(1); B = np.where(mtm, V, B)
            cap = liab + carry; pay = np.maximum(cap, 0) + dtax; carry = np.minimum(cap, 0)
            liab[:] = 0; dtax[:] = 0
            tot = V.sum(1); k = np.where(tot > 0, 1 - np.minimum(pay / np.maximum(tot, 1e-300), 1), 0)[:, None]
            V = V * k; B = B * k; paid += pay
        path[:, s] = V.sum(1)
    if bracket is not None:
        cap = liab + carry + ((V - B) * rlt).sum(1)
        final = V.sum(1) - np.maximum(cap, 0) - dtax; paid += np.maximum(cap, 0) + dtax
    else:
        final = V.sum(1)
    return np.maximum(final, 1e-12), path, paid


def summarize(final, path, years):
    g = np.log(final) / years
    peak = np.maximum.accumulate(np.concatenate([np.ones((len(path), 1)), path], 1), 1)[:, 1:]
    dd = (path / peak - 1).min(1)
    return {"growth": float(np.expm1(g.mean())), "median_cagr": float(np.expm1(np.median(g))), "p10_cagr": float(np.expm1(np.quantile(g, .1))),
            "p90_cagr": float(np.expm1(np.quantile(g, .9))), "p_loss": float((final < 1).mean()), "median_maxdd": float(np.median(dd)),
            "p_dd_over_75": float((dd < -0.75).mean())}


def hist_mc(n=10000, months=HORIZON * 12, block=6, seed=11):
    rng = np.random.default_rng(seed); X = MON[UNIV].values; T = len(X)
    idx = np.empty((n, months), int); idx[:, 0] = rng.integers(T, size=n)
    for k in range(1, months):
        jump = rng.random(n) < 1 / block
        idx[:, k] = np.where(jump, rng.integers(T, size=n), (idx[:, k - 1] + 1) % T)
    return 1 + X[idx]


def options_mc(n=10000, seed=12):
    f = CACHE / "options_P_scen.npz"
    if not f.exists():
        from implied_study import marginals, joint
        marg, T, _, _ = marginals(); S, _ = joint(marg, "P")
        np.savez(f, R=S[UNIV].values, T=T)
    z = np.load(f); R, T = z["R"], float(z["T"])
    steps = int(round(HORIZON / T))
    idx = np.random.default_rng(seed).integers(len(R), size=(n, steps))
    return 1 + R[idx], T, steps * T


def optimise(views, assets, rule="yearly", harvest=True, bracket="top", n_sub=1000, w0=None, robust=False, blend=False):
    """max E[log after-tax wealth]/yr over long-only weights: one view, the worse of several (robust), or their
    average (blend: equal belief in each view)."""
    subs = [(G[:n_sub], dt, spy, yrs) for G, dt, spy, yrs in views]
    def g_of(w, v):
        G, dt, spy, yrs = v
        f, _, _ = simulate(G, dt, np.maximum(w, 0) / max(np.maximum(w, 0).sum(), 1e-9), assets, rule, bracket, harvest, steps_per_year=spy)
        return np.log(f).mean() / yrs
    n = len(assets); best = None
    starts = [w0] if w0 is not None else []
    starts += [np.full(n, 1 / n), np.eye(n)[assets.index("SPY")] * .9 + .1 / n]
    for x0 in starts:
        if robust:
            z0 = np.r_[x0, min(g_of(x0, v) for v in subs)]
            cons = [{"type": "eq", "fun": lambda z: z[:-1].sum() - 1}] + \
                   [{"type": "ineq", "fun": (lambda z, v=v: g_of(z[:-1], v) - z[-1])} for v in subs]
            r = minimize(lambda z: -z[-1], z0, constraints=cons, bounds=[(0, 1)] * n + [(None, None)], method="SLSQP",
                         options={"maxiter": 200, "ftol": 1e-9, "eps": 1e-4})
            w = np.maximum(r.x[:-1], 0); w /= w.sum(); val = min(g_of(w, v) for v in subs)
        else:
            obj = (lambda x: -np.mean([g_of(x, v) for v in subs])) if blend else (lambda x, v=subs[0]: -g_of(x, v))
            r = minimize(obj, x0, constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1}], bounds=[(0, 1)] * n,
                         method="SLSQP", options={"maxiter": 200, "ftol": 1e-10, "eps": 1e-4})
            w = np.maximum(r.x, 0); w /= w.sum(); val = -obj(w)
        if best is None or val > best[1]:
            best = (w, val)
    return best[0]


def as_dict(w, assets):
    return {a: round(float(x), 3) for a, x in sorted(zip(assets, w), key=lambda t: -t[1]) if x > 0.005}


if __name__ == "__main__":
    t0 = time.time()
    A = UNIV
    GH = hist_mc(); GO, dtO, yrsO = options_mc()
    HP = MON[MON.index >= "1987-01"]                                  # calendar tax years: rebalance and pay each December
    Gp = (1 + HP[A].values)[None]; yrs_p = Gp.shape[1] / 12
    views = {"hist_path": (Gp, 1 / 12, 12, yrs_p), "hist_mc": (GH, 1 / 12, 12, HORIZON), "options_mc": (GO, dtO, 1, yrsO)}
    print(f"scenarios ready {time.time() - t0:.0f}s", flush=True)

    # growth-maximising mixes (taxable, top bracket, yearly rebalance + harvesting). The tax-free optima came out within
    # 3 points of these in a first run, so taxes barely move the mix; the views do.
    opt = {"growth-max, history": optimise([views["hist_mc"]], A), "growth-max, options": optimise([views["options_mc"]], A)}
    opt["growth-max, 50/50 history + options"] = optimise([views["hist_mc"], views["options_mc"]], A, blend=True, w0=opt["growth-max, options"])
    for k, w in opt.items():
        print(k, as_dict(w, A), f"{time.time() - t0:.0f}s", flush=True)

    KEY = {"100 SPY": {"SPY": 1}, "100 QQQ": {"QQQ": 1}, "30 TQQQ/30 UGL/40 HYG (green)": {"TQQQ": .3, "UGL": .3, "HYG": .4},
           }
    for k, w in opt.items():
        KEY[k] = dict(zip(A, w))
    RULES = [("IRA, yearly", "yearly", None, False), ("taxable, never rebalance", "none", "top", False),
             ("taxable, yearly", "yearly", "top", False), ("taxable, yearly + harvest", "yearly", "top", True),
             ("taxable, 10pp band + harvest", "band", "top", True), ("taxable mid bracket, yearly + harvest", "yearly", "mid", True)]
    res = {"assumptions": {"brackets": BRACKETS, "yield": YIELD, "horizon_years": HORIZON, "hist_path": ["1987-01", MON.index[-1]],
                           "options_step_years": dtO}, "portfolios": {}}
    for name, wd in KEY.items():
        w = np.array([wd.get(a, 0) for a in A]); res["portfolios"][name] = {"w": as_dict(w, A), "views": {}}
        for vk, (G, dt, spy, yrs) in views.items():
            res["portfolios"][name]["views"][vk] = {}
            for lab, rule, br, hv in RULES:
                f, path, paid = simulate(G, dt, w, A, rule, br, hv, steps_per_year=spy)
                s = summarize(f, path, yrs); s["tax_paid_share"] = float(np.median(paid / (paid + f)))
                res["portfolios"][name]["views"][vk][lab] = s
        r = res["portfolios"][name]["views"]
        print(f"{name:40s} " + "  ".join(f"{vk}: IRA {r[vk]['IRA, yearly']['growth']:+.1%} tax {r[vk]['taxable, yearly + harvest']['growth']:+.1%} "
                                          f"hold {r[vk]['taxable, never rebalance']['growth']:+.1%}" for vk in views), flush=True)

    # syrup's question: how often to rebalance, tax-free vs taxable (monthly data views only)
    res["rebalance_frequency"] = {}
    for name in ("30 TQQQ/30 UGL/40 HYG (green)", "growth-max, 50/50 history + options", "growth-max, options"):
        w = np.array([KEY[name].get(a, 0) for a in A]); res["rebalance_frequency"][name] = {}
        for lab, rule in (("monthly", "every1"), ("quarterly", "every3"), ("yearly", "yearly"), ("every 2 years", "every24"), ("never", "none")):
            row = {}
            for vk in ("hist_path", "hist_mc"):
                G, dt, spy, yrs = views[vk]
                row[vk] = {br or "IRA": float(np.expm1(np.log(simulate(G, dt, w, A, rule, br, br is not None, steps_per_year=spy)[0]).mean() / yrs))
                           for br in (None, "top")}
            res["rebalance_frequency"][name][lab] = row
            print("freq", name[:30], lab, {vk: {k: round(v, 3) for k, v in r.items()} for vk, r in row.items()}, flush=True)

    # where the green mix's tax drag comes from: switch off one tax feature at a time
    w = np.array([KEY["30 TQQQ/30 UGL/40 HYG (green)"].get(a, 0) for a in A]); res["green_tax_drag"] = {}
    for lab, patch in (("as modeled", {}), ("HYG income taxed like deferred gains", {"YIELD": ("HYG", 0)}),
                       ("UGL not marked to market", {"GAIN": ("UGL", "lt")})):
        old_y, old_g = dict(YIELD), dict(GAIN)
        if "YIELD" in patch: YIELD[patch["YIELD"][0]] = patch["YIELD"][1]
        if "GAIN" in patch: GAIN[patch["GAIN"][0]] = patch["GAIN"][1]
        res["green_tax_drag"][lab] = {vk: float(np.expm1(np.log(simulate(G, dt, w, A, "yearly", "top", True, steps_per_year=spy)[0]).mean() / yrs))
                                      for vk, (G, dt, spy, yrs) in views.items()}
        YIELD.clear(); YIELD.update(old_y); GAIN.clear(); GAIN.update(old_g)
        print("drag", lab, {k: round(v, 3) for k, v in res["green_tax_drag"][lab].items()}, flush=True)
    (OUT / "taxable_summary.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"saved ({time.time() - t0:.0f}s)")
