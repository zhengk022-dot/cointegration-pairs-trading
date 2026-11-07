# Cointegration-Based Statistical Arbitrage: V/MA

A quantitative research project examining whether cointegrated equity pairs
exhibit statistically measurable mean-reversion, and whether that
mean-reversion is fast enough to be a profitable trading signal.

## Research Question

> Are Visa (V) and Mastercard (MA) cointegrated, and if so, is the speed of
> mean-reversion sufficient to generate edge after transaction costs?

The null hypothesis: *no cointegrating relationship exists between V and MA.*

## Methodology

### 1. Cointegration Testing
Engle-Granger cointegration test on 9 years of daily close prices (2015–2024).
Null rejected at p < 0.0001 (test statistic: −5.34, well past the 1% critical
value of −3.90). V and MA share a long-run equilibrium driven by their
identical business model: both earn interchange fees on the same consumer
card spending with no credit risk.

### 2. Ornstein-Uhlenbeck Process
The spread is modelled as an OU process:

```
dS = κ(μ − S)dt + σdW
```

Parameters estimated via AR(1) regression on the discretised process.
**Half-life: 23.3 days** (κ = 0.030) — fast enough to generate ~10 round
trips per year while remaining slow enough that 1bp/side transaction costs
do not erode the signal.

### 3. Kalman Filter — Dynamic Hedge Ratio
The hedge ratio β between V and MA is not constant. OLS assumes a fixed
relationship; in practice β drifted from ~0.80 (2015) to ~0.60 (2024) — a
25% shift. A static hedge ratio would systematically miscalculate the spread
during this transition, generating false signals.

A Kalman filter tracks β as a latent state variable that evolves as a random
walk, producing a hedge ratio that adapts continuously to structural drift.

For the trading signal, a 252-day rolling OLS window is used: responsive
enough to track drift, stable enough to avoid overfitting to daily noise.

### 4. Strategy
- Signal: z-score of rolling-OLS spread (60-day normalisation window)
- Entry: |z| > 2.0 — long spread when cheap, short when expensive
- Exit: z crosses 0 (mean reversion complete)
- Transaction costs: 1bp per side

### 5. Out-of-Sample Validation
Strict time-based split: trained on 2015–2020, tested on 2021–2024.
No parameter re-fitting on the test set.

## Results

| Metric | In-sample (2015–2020) | Out-of-sample (2021–2024) |
|---|---|---|
| Sharpe ratio | 0.80 | 0.59 |
| Total PnL | — | 82.34 |
| Max drawdown | — | −33.34 |
| Bootstrap 95% CI (Sharpe) | — | [−0.51, 1.70] |

The 26% Sharpe decay is modest. The wide bootstrap CI reflects the
statistical reality of estimating Sharpe from ~750 daily observations —
3 years of data is not enough for a tight estimate regardless of strategy
quality. A tighter CI requires 5+ years of out-of-sample data.

## Key Findings

1. V/MA passes cointegration testing at p < 0.0001; "textbook" pairs such
   as GLD/GDX and TLT/IEF do not cointegrate in the 2015–2024 regime.
2. The OU half-life of 23.3 days places V/MA in the tradeable window for
   daily strategies with realistic transaction costs.
3. The hedge ratio drifted 25% over the sample period. A Kalman filter
   makes this drift visible; a rolling OLS window corrects for it in
   the trading signal.
4. Out-of-sample Sharpe of 0.59 suggests the signal is real, but the
   bootstrap CI honestly reflects significant estimation uncertainty.

## Setup

```bash
pip install yfinance pandas numpy statsmodels pykalman matplotlib scipy
python "Cointegration research framework.py"
```

Output: `vma_beta.png`, `vma_zscore_comparison.png`, `vma_equity.png`
