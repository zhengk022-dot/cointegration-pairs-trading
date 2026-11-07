import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import linregress
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller, coint
from pykalman import KalmanFilter

# ── 1. Pair screening ────────────────────────────────────────────────────────

def test_pair(a, b, start='2015-01-01', end='2024-01-01'):
    raw = yf.download([a, b], start=start, end=end, progress=False)['Close']
    data = raw.dropna()
    score, pvalue, _ = coint(data[a], data[b])
    print(f"{a}/{b}:  p={pvalue:.4f}  score={score:.4f}  n={len(data)}")
    return pvalue

print("=== Cointegration screening ===")
test_pair('V',   'MA')
test_pair('KO',  'PEP')
test_pair('AUDUSD=X', 'NZDUSD=X')

# ── 2. Load V/MA data ────────────────────────────────────────────────────────

raw  = yf.download(['V', 'MA'], start='2015-01-01', end='2024-01-01', progress=False)['Close']
data = raw.dropna()

# ── 3. OLS static hedge ratio + spread ──────────────────────────────────────

ols_result  = OLS(data['V'], add_constant(data['MA'])).fit()
beta_static = ols_result.params.iloc[1]
spread      = ols_result.resid

# ── 4. OU process — half-life estimation ────────────────────────────────────

spread_lag  = spread.shift(1).dropna()
spread_diff = spread.diff().dropna()
slope, intercept, _, _, _ = linregress(spread_lag, spread_diff)

kappa     = -slope
half_life = np.log(2) / kappa
print(f"\nHalf-life: {half_life:.1f} days  (kappa={kappa:.4f})")

# ── 5. Kalman filter — dynamic hedge ratio ───────────────────────────────────

obs = np.column_stack([data['MA'], np.ones(len(data))])

kf = KalmanFilter(
    transition_matrices=np.eye(2),
    observation_matrices=obs[:, np.newaxis, :],
    initial_state_mean=[1.0, 0.0],
    initial_state_covariance=np.eye(2),
    transition_covariance=1e-5 * np.eye(2),
    observation_covariance=1.0,
)

state_means, _  = kf.filter(data['V'].values)
beta_dynamic    = state_means[:, 0]
alpha_dynamic   = state_means[:, 1]
spread_dynamic  = data['V'].values - (beta_dynamic * data['MA'].values + alpha_dynamic)

# ── 6. Z-scores ──────────────────────────────────────────────────────────────

def zscore(s, window=60):
    s = pd.Series(s, index=data.index)
    return (s - s.rolling(window).mean()) / s.rolling(window).std()

z_static  = zscore(spread)
z_dynamic = zscore(spread_dynamic)

# ── 7. Charts ────────────────────────────────────────────────────────────────

# beta: static vs dynamic
plt.figure(figsize=(12, 4))
plt.plot(data.index, beta_dynamic, label='Kalman β (dynamic)', color='steelblue')
plt.axhline(beta_static, color='red', linestyle='--', label=f'OLS β (static) = {beta_static:.3f}')
plt.title('V/MA Hedge Ratio: Static vs Dynamic')
plt.legend()
plt.tight_layout()
plt.savefig('vma_beta.png', dpi=150)
plt.close()

# z-score: static vs dynamic
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
for ax, z, title in [
    (ax1, z_static,  'Z-score — Static OLS hedge ratio'),
    (ax2, z_dynamic, 'Z-score — Kalman dynamic hedge ratio'),
]:
    ax.plot(z, color='steelblue' if ax is ax1 else 'darkorange')
    ax.axhline( 2, color='red',   linestyle='--', alpha=0.5)
    ax.axhline(-2, color='green', linestyle='--', alpha=0.5)
    ax.axhline( 0, color='black', alpha=0.2)
    ax.set_title(title)
    ax.set_ylim(-4, 4)

plt.tight_layout()
plt.savefig('vma_zscore_comparison.png', dpi=150)
plt.close()

print("\nSaved: vma_beta.png, vma_zscore_comparison.png")

# ── 8. Backtest ──────────────────────────────────────────────────────────────

ENTRY  = 2.0
EXIT   = 0.0
COST   = 0.0001
WARMUP = 60      # skip rolling window warmup period

position = 0
pnl      = []

for i in range(WARMUP + 1, len(z_static)):
    z    = z_static.iloc[i - 1]   # yesterday's z → today's position
    prev = position

    if position == 0:
        if z >  ENTRY: position = -1
        if z < -ENTRY: position =  1
    elif position ==  1 and z >= EXIT: position = 0
    elif position == -1 and z <= EXIT: position = 0

    daily_pnl = position * spread.diff().iloc[i]
    cost      = abs(position - prev) * COST * data['V'].iloc[i]
    pnl.append(daily_pnl - cost)

pnl    = pd.Series(pnl, index=data.index[WARMUP + 1:])
equity = pnl.cumsum()
sharpe = pnl.mean() / pnl.std() * np.sqrt(252)

print(f"Sharpe:       {sharpe:.2f}")
print(f"Total PnL:    {equity.iloc[-1]:.2f}")
print(f"Max drawdown: {(equity - equity.cummax()).min():.2f}")

# ── 9. Out-of-sample split ───────────────────────────────────────────────────

split = '2021-01-01'
pnl_in  = pnl[pnl.index <  split]
pnl_out = pnl[pnl.index >= split]

sharpe_in  = pnl_in.mean()  / pnl_in.std()  * np.sqrt(252)
sharpe_out = pnl_out.mean() / pnl_out.std() * np.sqrt(252)
print(f"In-sample  Sharpe (2015-2020): {sharpe_in:.2f}")
print(f"Out-of-sample Sharpe (2021-2024): {sharpe_out:.2f}")

# ── 10. Bootstrap Sharpe CI ──────────────────────────────────────────────────

np.random.seed(42)
n_boot    = 1000
boot_sharpes = [
    (pnl_out.sample(len(pnl_out), replace=True).mean() /
     pnl_out.sample(len(pnl_out), replace=True).std() * np.sqrt(252))
    for _ in range(n_boot)
]
lo, hi = np.percentile(boot_sharpes, [2.5, 97.5])
print(f"Bootstrap 95% CI on out-of-sample Sharpe: [{lo:.2f}, {hi:.2f}]")