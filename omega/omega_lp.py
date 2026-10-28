"""Omega-ratio portfolio optimisation on discrete (scenario) return distributions, solved exactly as linear programs.

For a portfolio with weights w and scenario returns R (M scenarios x n assets, equally likely), with loss threshold tau:

    omega risk     LPM1(w) = E[max(tau - R w, 0)]              (the denominator of Omega: expected shortfall below tau)
    Omega(w)       = E[max(R w - tau, 0)] / LPM1(w) = 1 + (E[R w] - tau) / LPM1(w)

Maximising Omega is a linear-fractional programme. The Charnes-Cooper substitution y = t w, with the scaling chosen so
that LPM1 = 1, turns it into an LP (Mausser, Saunders & Seco 2006; Kane, Bartholomew-Biggs, Cross & Dewar 2009):

    max   mu'y - tau t
    s.t.  (1/M) sum_j d_j = 1
          d_j >= tau t - R_j y,  d_j >= 0
          1'y = t,  y >= 0,  t >= 0
    ->    w = y / t,  Omega - 1 = optimal objective

The (omega risk, expected return) efficient frontier is the LP family

    min (1/M) sum_j d_j   s.t.  d_j >= tau - R_j w,  d >= 0,  mu'w >= m,  1'w = 1,  w >= 0.

`tau` may be a scalar or a per-scenario vector (e.g. the T-bill return in that scenario, giving Omega over cash).
Weights are long-only and fully invested (the repo's original constraints 1'w = 1 and 1'|w| = 1 imply exactly this).
"""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, vstack, identity


def _tau(tau, M):
    return np.full(M, float(tau)) if np.isscalar(tau) else np.asarray(tau, float)


def lpm1(w, R, tau=0.0):
    """Omega risk: expected shortfall of the portfolio return below tau."""
    return np.maximum(_tau(tau, len(R)) - R @ w, 0).mean()


def omega(w, R, tau=0.0):
    p = R @ w - _tau(tau, len(R))
    lo = np.maximum(-p, 0).mean()
    return np.inf if lo == 0 else np.maximum(p, 0).mean() / lo


def max_omega(R, tau=0.0):
    """Exact Omega-maximising long-only portfolio. Returns (w, Omega)."""
    M, n = R.shape; tv = _tau(tau, M); mu = R.mean(0)
    # variables: [y (n), d (M), t (1)]
    c = -np.r_[mu, np.zeros(M), -tv.mean()]                    # maximise mu'y - mean(tau) t
    # -R y - d + tau t <= 0
    A_ub = hstack([csr_matrix(-R), -identity(M, format="csr"), csr_matrix(tv[:, None])]).tocsr()
    b_ub = np.zeros(M)
    A_eq = vstack([csr_matrix(np.r_[np.zeros(n), np.full(M, 1 / M), 0.0][None, :]),
                   csr_matrix(np.r_[np.ones(n), np.zeros(M), -1.0][None, :])]).tocsr()
    b_eq = np.array([1.0, 0.0])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * (n + M + 1), method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    y, t = res.x[:n], res.x[-1]
    w = y / t
    return w, omega(w, R, tau)


def min_lpm(R, target, tau=0.0):
    """Minimum omega-risk long-only portfolio with expected return >= target. Returns w (or None if infeasible)."""
    M, n = R.shape; tv = _tau(tau, M); mu = R.mean(0)
    c = np.r_[np.zeros(n), np.full(M, 1 / M)]
    A_ub = vstack([hstack([csr_matrix(-R), -identity(M, format="csr")]),            # tau - R w - d <= 0
                   csr_matrix(np.r_[-mu, np.zeros(M)][None, :])]).tocsr()          # mu'w >= target
    b_ub = np.r_[-tv, -target]
    A_eq = csr_matrix(np.r_[np.ones(n), np.zeros(M)][None, :]); b_eq = np.array([1.0])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * (n + M), method="highs")
    return res.x[:n] if res.status == 0 else None


def frontier(R, tau=0.0, points=40):
    """Efficient (omega risk, expected return) frontier: list of (lpm1, mean, w) from the min-risk portfolio to the
    highest-mean asset."""
    mu = R.mean(0)
    w0 = min_lpm(R, float(mu.min()) - 1.0, tau)            # target below every asset = unconstrained min risk
    out = []
    for m in np.linspace(mu @ w0, mu.max(), points):
        w = min_lpm(R, m, tau)
        if w is not None:
            out.append((lpm1(w, R, tau), mu @ w, w))
    return out


def bootstrap_annual(monthly, n=20000, months=12, block=6, seed=0):
    """Stationary block bootstrap (Politis & Romano) of joint monthly return rows -> n scenarios of `months`-month
    compounded returns. Rows are resampled whole, so the cross-asset dependence of each month is kept; blocks keep
    some serial dependence (momentum, volatility clustering)."""
    rng = np.random.default_rng(seed)
    X = np.asarray(monthly); T = len(X)
    out = np.empty((n, X.shape[1]))
    p = 1 / block
    for s in range(n):
        i = rng.integers(T); g = np.ones(X.shape[1])
        for k in range(months):
            if k and rng.random() < p:
                i = rng.integers(T)
            g *= 1 + X[i]
            i = (i + 1) % T
        out[s] = g - 1
    return out


def rolling_annual(monthly, months=12):
    """Historical overlapping 12-month returns (one scenario per start month)."""
    G = np.log1p(np.asarray(monthly))
    c = np.vstack([np.zeros(G.shape[1]), np.cumsum(G, 0)])
    return np.expm1(c[months:] - c[:-months])
