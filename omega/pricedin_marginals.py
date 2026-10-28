"""Option-implied marginals, computed the PricedIn way.

PricedIn's implied-distribution step (pricedin-algo-layer `get_distr`, a private repo) fits a discrete distribution
directly: the unknown is a PMF p on a strike grid (p >= 0, sum p = 1). Model call and put curves are double integrals
of p, fitted to the market curves by least squares. The production code solves this with SciPy SLSQP (maxiter = 100).

Both model curves are LINEAR in p, so the problem is a convex quadratic programme:
  - `fit_qp`: the same objective and constraints, solved exactly with Clarabel (via cvxpy) in about 0.02-0.7 s
    per chain, versus 1.5-23 s for SLSQP, with an equal or lower objective.
  - `fit_extended`: the version used in the study. Same unknown and solver, with four changes listed at the function.

Inputs are put on a forward basis so the formulation applies at long maturities:
  - option mids / DF, with DF = exp(-r T) and r = the 3-month T-bill rate;
  - the price anchor is the forward F from put-call parity on near-the-money strikes (median over strikes).

If PRICEDIN_ALGO_PATH points to a local checkout of pricedin-algo-layer, `compare_with_algo_layer` runs the original
SLSQP code on the same inputs. That code is imported at run time and is not copied into this repo.
"""
import json, os, re, sys, time
from datetime import date
from pathlib import Path
import numpy as np
import cvxpy as cp
from scipy.optimize import brentq

CHAINS = Path(__file__).resolve().parent.parent / "data" / "chains"
R_TBILL = 0.0408           # 3-month T-bill (FRED DTB3, latest print before the quotes)

# Leveraged funds: underlying, L, financing spread over T-bills, expense ratio (fitted to fund history; COPULA.md section 1)
LEV = {"TQQQ": ("QQQ", 3, .0055, .0086), "QLD": ("QQQ", 2, .0050, .0095), "UPRO": ("SPY", 3, .0065, .0091),
       "SSO": ("SPY", 2, .0065, .0089), "UGL": ("GLD", 2, .02, .0095), "SHNY": ("GLD", 3, .07, .0095)}


def load_chain(sym, expiry):
    """CBOE delayed-quote JSON -> spot, timestamp, {strike: {"C"/"P": (bid, ask)}} for one expiry (YYMMDD)."""
    j = json.loads((CHAINS / f"{sym}.json").read_text())
    d = j["data"]; rows = {}
    for o in d["options"]:
        m = re.match(r"[A-Z]+(\d{6})([CP])(\d{8})", o["option"])
        if m and m.group(1) == expiry:
            rows.setdefault(int(m.group(3)) / 1000, {})[m.group(2)] = (float(o["bid"] or 0), float(o["ask"] or 0))
    return float(d["current_price"]), j["timestamp"], rows


def chain_inputs(sym, expiry, valuation):
    spot, ts, rows = load_chain(sym, expiry)
    T = (date(2000 + int(expiry[:2]), int(expiry[2:4]), int(expiry[4:])) - valuation).days / 365.25
    DF = np.exp(-R_TBILL * T)
    mid = lambda q: (q[0] + q[1]) / 2
    near = [k + (mid(v["C"]) - mid(v["P"])) / DF for k, v in rows.items()
            if "C" in v and "P" in v and min(v["C"][0], v["P"][0]) > 0 and 0.9 * spot <= k <= 1.1 * spot]
    F = float(np.median(near))
    c = {"sym": sym, "expiry": expiry, "spot": spot, "timestamp": ts, "T": T, "DF": DF, "F": F,
         "q": R_TBILL - np.log(F / spot) / T, "k_call": [], "u_call": [], "h_call": [], "k_put": [], "u_put": [], "h_put": []}
    for k, v in sorted(rows.items()):
        for side, tag in (("C", "call"), ("P", "put")):
            if side in v and v[side][0] > 0 and v[side][1] > v[side][0]:
                c[f"k_{tag}"].append(k); c[f"u_{tag}"].append(mid(v[side]) / DF); c[f"h_{tag}"].append((v[side][1] - v[side][0]) / 2 / DF)
    # Total return to a holder = price ratio * e^{dT}. The parity carry q is dividends + stock-borrow cost. For leveraged
    # funds the borrow part prices the fund's own decay (financing + fee), which a holder never receives, so subtract it.
    drag = (LEV[sym][1] - 1) * LEV[sym][2] + LEV[sym][3] if sym in LEV else 0.0
    c["d"] = max(c["q"] - drag, 0.0)
    return c


# --------------------------------------------- as-is formulation ---------------------------------------------------
def _grid_and_truth(c):
    """The PricedIn grid (strikes plus 0 and last + 1) and interpolated market curves, anchored at the forward."""
    s0 = c["F"]; kc, uc, kp, up = c["k_call"], c["u_call"], c["k_put"], c["u_put"]
    last = max(max(kc), max(kp)) + 1
    k = np.unique(np.concatenate(([0], kc, kp, [last])))
    tc = np.interp(k, [0] + kc + [last], [s0] + uc + [0])
    tp = np.interp(k, [0.0] + kp + [last], [0.0] + up + [last - s0])
    return k, tc, tp, s0


def _curves(k, s0):
    """Linear maps p -> normalised call and put curves: CDF - 1 = cumsum(p) - 1, integrated once more along the grid."""
    n = len(k); dk = np.diff(k / s0)
    D = np.zeros((n, n - 1)); D[1:, :] = np.tril(np.ones((n - 1, n - 1))); D *= dk[None, :]
    A = D @ np.tril(np.ones((n, n)))[:-1, :]
    return A, 1 - D.sum(1)                                  # call = A p + b,  put = A p


def fit_qp(c):
    """Exact minimiser of the PricedIn objective: p >= 0, sum p = 1, least squares on both curves."""
    k, tc, tp, s0 = _grid_and_truth(c)
    A, b = _curves(k, s0)
    p = cp.Variable(len(k), nonneg=True)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ p + b - tc / s0) + cp.sum_squares(A @ p - tp / s0)), [cp.sum(p) == 1])
    t0 = time.perf_counter(); prob.solve(solver=cp.CLARABEL); dt = time.perf_counter() - t0
    pv = np.maximum(p.value, 0); pv /= pv.sum()
    return {"k": k, "pmf": pv, "objective": objective_value(c, pv), "seconds": dt}


def objective_value(c, pmf):
    k, tc, tp, s0 = _grid_and_truth(c)
    A, b = _curves(k, s0)
    return float(np.sum((A @ pmf + b - tc / s0) ** 2) + np.sum((A @ pmf - tp / s0) ** 2))


def compare_with_algo_layer(c):
    """Run PricedIn's own get_distr (SLSQP) on the same inputs, if a local checkout is configured."""
    path = os.environ.get("PRICEDIN_ALGO_PATH")
    if not path:
        return None
    sys.path.insert(0, path)
    import core.market_implications as mi
    t0 = time.perf_counter()
    p = mi.get_distr((c["u_call"], c["u_put"]), (c["k_call"], c["k_put"]), c["F"])
    dt = time.perf_counter() - t0
    if p is None:
        return {"seconds": dt, "converged": False}
    p = np.maximum(p, 0); p = p / p.sum()
    return {"seconds": dt, "converged": True, "pmf": p, "objective": objective_value(c, p)}


# --------------------------------------------- extended fit (used in the study) -------------------------------------
def fit_extended(c, lam=1.0, n=400):
    """Same unknown (PMF, prices linear in it), same solver, with four changes:
      1. A log-spaced grid that extends well past the listed strikes. On the as-is grid, all mass beyond the listed
         range piles onto 0 and last + 1 (25% of TQQQ's mass sits at 1.75x spot), which truncates the tails Omega
         measures.
      2. A martingale constraint, sum p x = F (risk-neutral mean = forward). It is linear, so this is still a QP.
      3. Out-of-the-money quotes only (puts below F, calls above), with errors in units of the half bid-ask spread.
         Listed ETF options are American, and ITM ones carry an early-exercise premium.
      4. A small curvature penalty on p, so that mass beyond the last strike spreads smoothly instead of spiking."""
    F = c["F"]
    q = [(k, u, h, True) for k, u, h in zip(c["k_call"], c["u_call"], c["h_call"]) if k >= F] + \
        [(k, u, h, False) for k, u, h in zip(c["k_put"], c["u_put"], c["h_put"]) if k < F]
    K, U, H, isc = (np.array(z) for z in zip(*q))
    W = 1 / np.maximum(H, 5e-4 * F)
    atm = np.argmin(np.abs(K - F)); s = max(2 * U[atm] / (0.8 * F), 0.05)          # straddle ~ 0.8 F sigma sqrt(T)
    lo = min(max(np.exp(-8 * s), 0.002) * F, 0.8 * K.min()); hi = max(np.exp(5 * s) * F, 1.25 * K.max())
    x = np.exp(np.linspace(np.log(lo), np.log(hi), n))
    A = np.where(isc[:, None], np.maximum(x[None, :] - K[:, None], 0), np.maximum(K[:, None] - x[None, :], 0))
    D2 = np.diff(np.eye(n), 2, axis=0) * n
    p = cp.Variable(n, nonneg=True)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(cp.multiply(W, A @ p - U)) / len(U) + lam * cp.sum_squares(D2 @ p)),
                      [cp.sum(p) == 1, x @ p == F])
    t0 = time.perf_counter(); prob.solve(solver=cp.CLARABEL); dt = time.perf_counter() - t0
    pv = np.maximum(p.value, 0); pv /= pv.sum()
    m = A @ pv
    return {"k": x, "pmf": pv, "seconds": dt, "n_quotes": int(len(U)), "strike_range": (K.min() / c["spot"], K.max() / c["spot"]),
            "rmse_half_spreads": float(np.sqrt(np.mean(((m - U) * W) ** 2))), "share_within_spread": float(np.mean(np.abs(m - U) * W <= 1))}


def gross_tr(c, fit):
    """Support points of the gross total return over the horizon, (S_T / S_0) e^{dT}, aligned with fit['pmf']."""
    return fit["k"] / c["spot"] * np.exp(c["d"] * c["T"])


def tilt(x, p, T, premium):
    """Risk-neutral -> real-world by an Esscher (power-utility pricing kernel) tilt p_P ~ p_Q x^g, with g chosen so that
    the expected return rises by `premium` per year: E_P[X] = E_Q[X] e^{premium T}. Returns (p_P, g)."""
    if premium == 0:
        return p, 0.0
    target = (p @ x) * np.exp(premium * T); lx = np.log(x)
    def mean(g):
        w = p * np.exp(g * (lx - lx.max())); return (w / w.sum()) @ x
    g = brentq(lambda g: mean(g) - target, -30, 30)
    w = p * np.exp(g * (lx - lx.max())); return w / w.sum(), float(g)


def quantile_fn(x, p):
    """Inverse CDF of a discrete distribution on a fine grid (midpoint rule, interpolated in log space)."""
    o = np.argsort(x); x, p = x[o], p[o]
    cdf = (np.cumsum(p) - p / 2) / p.sum()
    return lambda u: np.exp(np.interp(u, cdf, np.log(x)))


def leveraged_from(x, p, T, L, spread, fee, r=R_TBILL):
    """Map an underlying's gross-TR distribution to a daily-reset L-times fund (continuous-rebalancing identity,
    with the underlying's own variance rate):  ln Y = L ln X - (L-1)(r + spread) T - fee T - 1/2 (L^2 - L) var T."""
    lx = np.log(x); var = p @ (lx - p @ lx) ** 2 / T
    return np.exp(L * lx - (L - 1) * (r + spread) * T - fee * T - 0.5 * (L * L - L) * var * T), p
