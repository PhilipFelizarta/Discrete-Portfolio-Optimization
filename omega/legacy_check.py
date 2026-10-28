"""Audit of the original `Omega Portfolio.ipynb` algorithm (2022), run on the notebook's own toy assets.

1. Re-implements the notebook's h-gradient descent exactly (same formulas, same constants).
2. Shows that the weight update never sees the objective: two runs from the same start with *different* return
   samples follow the same weight path, so the result is decided by the starting point and the constraints.
3. Compares with the exact LP optimum (omega_lp.max_omega) and a 200k-portfolio brute-force search on the same samples.
Run: python omega/legacy_check.py
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from omega_lp import max_omega, omega as omega_exact

# ---- the notebook's toy assets (gross returns; tau = 1 means "losses are returns below 1") ----
x = [np.array([0.0, 0.5, 1.6, 2.0]), np.array([0.0, 0.3, 1.5, 2.5]), np.array([0.0, 0.5, 1.5, 2.1]),
     np.linspace(0, 1.3, num=50), np.linspace(0.8, 1.2, num=50), np.linspace(0.8, 1.2, num=50)]
px = [np.array([0.6, 0.3, 0.05, 0.05]), np.array([0.25] * 4), np.array([0.5, 0.0, 0.0, 0.5]),
      np.square(1.3 - x[3]), 1.0 - 25 * np.square(x[4] - 1.0), 1.0 - 25 * np.square(x[5] - 1.0)]
px = [p / p.sum() for p in px]
E = np.array([x[i] @ px[i] for i in range(6)])
M = 100


def sample(seed):
    rng = np.random.default_rng(seed)
    return np.array([rng.choice(x[i], M, p=px[i]) for i in range(6)])     # (assets, M) like R_sample


def h_gradient(w, r, constr, R_sample, tau=1.0, c=0.1):
    """Verbatim logic of the notebook's h_gradient (vectorised loop, same maths)."""
    slack = np.dot(w, r) - tau - c
    slack = np.sqrt(slack) if slack > 0 else 0.0
    dl_dlambda1 = np.sum(w) - 1
    dl_dlambda2 = np.sum(np.abs(w)) - 1
    dl_dtheta = np.dot(w, r) - tau - c - slack ** 2
    info = tau - w @ R_sample
    dl_dw = -R_sample[:, info > 0].sum(1) / M
    dl_dw = dl_dw + constr[0] + np.sign(w) * constr[1] + r * constr[2]
    grad_constr = [np.sum(dl_dw), np.dot(np.sign(w), dl_dw), np.dot(r, dl_dw)]
    grad_w = r * dl_dtheta + np.sign(w) * dl_dlambda2 + dl_dlambda1        # <- no term from the objective
    return grad_w, grad_constr


def notebook_descent(w0, R_sample, steps=10000, lr=1e-1, tau=1.0, c=0.05):
    w = np.array(w0, float); lam = np.array([0.0, 0.0, 1.0]); path = []
    for _ in range(steps):
        g, gc = h_gradient(w, E, lam, R_sample, tau, c)
        w -= lr * g; lam -= lr * np.array(gc); path.append(w.copy())
    return w, np.array(path)


def notebook_omega(w, R_sample, tau=1.0):
    """The notebook's omega(): exact expectation on top, sampled shortfall below."""
    return (w @ E - tau) / np.maximum(tau - w @ R_sample, 0).mean()


if __name__ == "__main__":
    RA, RB = sample(1), sample(2)
    rng = np.random.default_rng(7)
    w0 = np.maximum(rng.standard_normal(6), 0); w0 /= w0.sum()
    wa, pa = notebook_descent(w0, RA)
    wb, pb = notebook_descent(w0, RB)
    print("start weights           ", np.round(w0, 4))
    print("notebook result, sample A", np.round(wa, 4))
    print("notebook result, sample B", np.round(wb, 4))
    print("max |path A - path B| over 10k steps:", float(np.abs(pa - pb).max()), "(0 => the samples, i.e. the objective, never mattered)")
    # the notebook then clips negatives and renormalises
    wn = np.maximum(wa, 0); wn /= wn.sum()
    R = RA.T                                                                # scenarios x assets
    w_lp, om_lp = max_omega(R, tau=1.0)
    rng = np.random.default_rng(0)
    W = rng.dirichlet(np.full(6, 0.3), 200000)
    P = R @ W.T - 1.0
    om_bf = np.maximum(P, 0).mean(0) / np.maximum(np.maximum(-P, 0).mean(0), 1e-12)
    print("\nOmega(tau=1) on sample A, all measured the same way (sample mean on top and bottom):")
    print(f"  notebook gradient descent : {omega_exact(wn, R, 1.0):.3f}   w = {np.round(wn, 3)}")
    print(f"  exact LP (omega_lp)       : {om_lp:.3f}   w = {np.round(w_lp, 3)}")
    print(f"  brute force, 200k random  : {om_bf.max():.3f}   w = {np.round(W[om_bf.argmax()], 3)}")
    print(f"  notebook's own omega() of its result (exact mean / sampled shortfall, minus 1 form): {notebook_omega(wn, RA):.3f}")
