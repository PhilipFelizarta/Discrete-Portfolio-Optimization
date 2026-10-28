# How the joint distribution was built: marginals and copula

**Short version.** No parametric copula was fitted. The joint distribution of the 10 funds' 1-year returns is the
**empirical** one: whole historical months are resampled (all 10 funds' returns from the *same* month together), in
random blocks, and compounded to a year. By Sklar's theorem any joint distribution splits into marginals plus a
copula. Here both come out of the same resampled data, so the copula is the empirical copula implied by 40 years of
co-movement. Below is how each piece was made and what it looks like.

## 1. Marginals: one return series per fund

Each fund's **daily** return is reconstructed from its underlying, then compounded to months (`data/monthly_returns.csv`):

    leveraged fund:  r_t = L · u_t − (L − 1) · (rf_t + s) / 252 − fee / 252

where `u_t` is the underlying's daily total return, `rf_t` the 3-month T-bill rate, and `s` the financing spread over
T-bills. `s` was fitted so the simulated fund's compounded return matches the real fund over its whole life:

| Fund | Underlying | L | Spread s | Fee | Fitted on |
|---|---|---|---|---|---|
| TQQQ / QLD | Nasdaq-100 TR | 3 / 2 | 0.55% / 0.50% | 0.86% / 0.95% | 2010–26 / 2006–26: CAGR within 0.03%, daily corr ≥ 0.996 |
| UPRO / SSO | S&P 500 TR | 3 / 2 | 0.65% | 0.91% / 0.89% | 2009–26 / 2006–26 |
| SHNY | GLD (spot gold − 0.40%) | 3 | **7.0%** | 0.95% | 2023–26 (prime-rate-based ETN financing; same fit every year) |
| UGL | GLD | 2 | 2.0% | 0.95% | 2008–26 |
| HYG | ICE BofA US High Yield TR index − 0.49% fee − 0.66%/yr drag | 1 | – | – | drag fitted 2007–26 |
| QQQ, SPY, GLD | index TR / spot gold − 0.40% | 1 | – | – | – |

- **Daily reset is built in.** Leverage is applied every day before compounding, so each fund's volatility decay is
  already inside its monthly and yearly returns.
- **HYG's index history:** monthly in 1986–89 (spread evenly over the month's days), daily from 1990.

### What the marginals look like (`results/copula_marginals.png`)

Monthly returns, Oct 1986 – Oct 2026:

| | TQQQ | UGL | SHNY | HYG | SPY | GLD |
|---|---|---|---|---|---|---|
| Annualised vol | 71% | 31% | 47% | 8% | 15% | 16% |
| Skew | 0.04 | 0.47 | 0.60 | **−1.22** | −0.70 | 0.33 |
| Excess kurtosis | 1.3 | 1.5 | 1.6 | **10.0** | 2.1 | 1.4 |
| Worst month | −71% (Oct 1987) | −33% (Oct 2008) | −48% (Oct 2008) | −16% (Oct 2008) | −22% (Oct 1987) | −17% (Oct 2008) |

1-year returns, bootstrap vs the 480 overlapping historical 12-month windows (the bootstrap should reproduce history):

| | TQQQ | UGL | SHNY | HYG | SPY | GLD |
|---|---|---|---|---|---|---|
| Mean, boot / hist | 50% / 48% | 6% / 8% | −3% / −1% | 7% / 7% | 13% / 12% | 6% / 7% |
| Median, boot / hist | 34% / 38% | 0% / −1% | −14% / −16% | 7% / 6% | 14% / 14% | 5% / 4% |
| 5th percentile, boot / hist | −79% / −85% | −36% / −35% | −57% / −56% | −8% / −8% | −16% / −18% | −16% / −16% |
| P(loss), boot / hist | 33% / 28% | 50% / 51% | 63% / 62% | 20% / 21% | 20% / 18% | 38% / 40% |

**The leveraged marginals are strongly right-skewed at 1 year** (TQQQ skew 2.4, with a mean far above its median),
which is why a mean-based y-axis flatters them. Junk bonds are the opposite: small, steady gains with a fat left
tail (kurtosis 10, all of it in 2008).

## 2. Dependence: the copula

### Method: stationary block bootstrap of joint months (Politis & Romano, 1994)
1. **The joint sample:** the 480 monthly vectors r_t ∈ ℝ¹⁰. Each month is one draw from the joint distribution, so
   whatever happened together historically (Oct 2008: stocks −, junk −, gold briefly −) stays together. No
   correlation matrix, no copula family, no tail parameter is assumed.
2. **Building one 1-year scenario:** start at a random month. For each of the next 11 months, continue to the next
   calendar month with probability 5/6, or jump to a new random month with probability 1/6. Block lengths are
   geometric with mean 6 months; the sample wraps around at the end.
3. **Compound:** R_j = Π(1 + r) − 1 over the 12 drawn months, for every fund at once.
4. **Repeat** 20,000 times (seed 0).

Why blocks, not single months: momentum and volatility clustering (e.g. 2000–02 or 2008 unfolding over many months)
change how bad a year can get. Resampling single months independently would shrink the left tail of 1-year returns.

Why it's the right object for Omega: the optimiser only needs equally likely joint scenarios R_j. For fixed weights
held a year (= yearly rebalancing), the portfolio return is exactly wᵀR_j. The scenario cloud is the joint
distribution (marginals and copula at once), and the LP works on it directly, with no convolution and no density.

### What the copula looks like (`results/copula_pairs.png`)

Rank scatter of 1-year returns. Spearman's ρ, plus empirical lower-tail dependence
λ_L(10%) = P(both in their worst 10%) / 10%. Independence gives 0.10.

| Pair | Spearman ρ (1 yr) | λ_L(10%), empirical | λ_L(10%), Gaussian copula with same ρ |
|---|---|---|---|
| TQQQ – UGL | −0.05 | 0.09 | 0.08 |
| TQQQ – SHNY | −0.04 | 0.09 | 0.09 |
| TQQQ – HYG | **+0.50** | **0.46** | 0.33 |
| UGL – HYG | +0.08 | 0.11 | 0.13 |
| TQQQ – SPY | +0.86 | 0.70 | 0.61 |

- **Stocks and junk bonds crash together more than a Gaussian copula admits.** Their bad years coincide 46% of the
  time vs 33% under a Gaussian copula with the same rank correlation. This is the classic credit/equity
  lower-tail dependence (2000–02, 2008, 2020), and it's the main reason not to replace the empirical copula with a
  Gaussian one.
- **Gold is close to independent of stocks, including in the tails** (λ_L ≈ 0.09, about the 0.10 of independence).
  In TQQQ's worst 10% of months, UGL averaged +2.1% vs +0.4% in other months. The correlation does rise in those
  months (−0.04 overall → +0.19), driven by Oct 2008, when everything was sold for cash.
- **SHNY has the same dependence as UGL** (both are gold). Only its marginal differs, through its financing cost.

## 3. Limitations
- **Only 40 years:** about four big equity drawdowns and one credit crisis. The bootstrap can only recombine months
  that happened; it can't produce a joint extreme worse than history (e.g. a stocks-and-gold crash together, or a
  2008-style credit crash during a gold bear).
- **Most of the history is reconstructed:** simulated funds and index data before the real ETFs existed (TQQQ 2010,
  UGL 2008, SHNY 2023, HYG 2007). Calibration is in the table above.
- **Block length (6 months) is a judgement call.** Longer blocks fatten the 1-year tails of the leveraged funds.
- **Equal weighting of history:** the 1980s–90s gold bear and the 2010s–20s equity bull count as much as anything
  else.

**Parametric alternatives**, if smoother tails or stress scenarios are wanted:
- **t-copula:** captures symmetric tail dependence (a fitted ν would be low given the TQQQ–HYG result).
- **Clayton or vine copulas:** asymmetric lower-tail dependence, with each fund's marginal fitted separately.

Either can be dropped in: generate scenarios and pass them to `omega_lp.max_omega` / `frontier`.

Reproduce: `python omega/copula_report.py` → `results/copula_tables.json`, `copula_marginals.png`, `copula_pairs.png`.
