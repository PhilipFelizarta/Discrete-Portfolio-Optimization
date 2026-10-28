# data/monthly_returns.csv

Joint monthly total returns (decimal), Oct 1986 – Oct 2026, of **simulated** funds, built daily and compounded to months:

| column | construction | calibration vs the real fund |
|---|---|---|
| TQQQ / QLD | 3x / 2x daily Nasdaq-100 total return − (L−1)·(T-bill + 0.55% / 0.50%) − fee | CAGR within 0.03%/yr, daily corr 0.996–0.999 |
| UPRO / SSO | 3x / 2x daily S&P 500 total return − (L−1)·(T-bill + 0.65%) − fee | same |
| SHNY | 3x daily GLD − 2·(T-bill + 7.0%) − 0.95% | financing fitted to real SHNY 2023–26 (prime-rate-based ETN financing) |
| UGL | 2x daily GLD − (T-bill + 2.0%) − 0.95% | fitted to real UGL 2008–26 |
| QQQ / SPY / GLD | index total return; spot gold − 0.40% | GLD CAGR within 0.04%/yr |
| HYG | ICE BofA US High Yield TR index − 0.49% fee − 0.66%/yr drag | drag fitted to real HYG 2007–26 |
| CASH | 3-month T-bill | |

The gold and junk-bond legs before the real funds existed are reconstructions, not fund prices.
