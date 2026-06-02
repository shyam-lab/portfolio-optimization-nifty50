# Assignment 5: Classical, ML-Enhanced, and QUBO Portfolio Construction

## Objective

This assignment compares three portfolio-construction paradigms on NIFTY 50 processed data:
classical mean-variance optimization, ML-enhanced selection via gradient-boosted classification,
and discrete binary selection via QUBO solved with simulated annealing. The central question
is whether the discrete, cardinality-constrained QUBO formulation provides a measurable
advantage over continuous optimization, and under what conditions. Sharpe ratios in this
report assume a zero risk-free rate.

## 1. Introduction

Markowitz mean-variance optimization selects continuous weights to maximize risk-adjusted return.
It is the standard approach in quantitative portfolio construction but suffers from estimation
sensitivity: small changes in estimated returns produce large changes in optimal weights,
particularly when the covariance matrix is ill-conditioned or the number of assets is large
relative to the estimation window.

An alternative framing poses the problem in two stages: first select a subset of $K$ stocks,
then optimize weights within that subset. The selection step is combinatorial. QUBO (Quadratic
Unconstrained Binary Optimization) encodes this combinatorial selection as a quadratic objective
over binary variables, which can be solved by simulated annealing or, in principle, by quantum
hardware.

All three pipelines are built on the same data and evaluated over the same out-of-sample
window. Performance differences across methods reflect the combined effect of stock
selection rule (Sharpe screen vs ML classifier vs QUBO), weighting scheme (MVO vs
return-tilted inverse-volatility), cardinality (continuous vs fixed $K = 10$), and
solver behaviour. An equal-weight baseline provides a model-free sanity check.

## 2. Data and Universes

The raw data file is `data/raw/raw_data.csv`, containing NSE OHLC data. The data pipeline
(`data/Data_Pipeline.ipynb`) produces two wide price matrices:

- **Full Universe**: `data/processed/price_matrix_full.csv`. 49 stocks, 2598 trading days,
  2010-11-04 to 2021-04-30. All stocks have complete data over this window.
- **Extended Universe**: `data/processed/price_matrix_extended.csv`. 37 stocks, 4286 trading
  days, 2004-01-23 to 2021-04-30. Stocks with excessive missing data are dropped, giving a
  longer history that includes the 2008 financial crisis.

Daily simple returns are computed as $r_{i,t} = P_{i,t}/P_{i,t-1} - 1$. Returns exceeding
50% in absolute value are treated as data errors and set to NaN.

## 3. Setup and Parameters

All backtests share these parameters:

- **Test period**: January 2014 to April 2021 (7.3 years out-of-sample).
- **Training window**: Rolling 3-year trailing window at each rebalance date.
- **Minimum data requirement**: 504 live trading days per stock to enter the available set
  at any rebalance.
- **Expected return estimator**: EWMA with 252-day (1-year) halflife, annualized.
- **Covariance estimator**: Sample covariance over trailing 756 days (3 years), annualized.
- **Rebalance cadences**: Monthly (88 rebalance dates) and yearly (8 rebalance dates).
- **Weight cap**: 20% maximum per stock, long-only ($0 \le w_i \le 0.20$).
- **Cardinality**: $K = 10$ for Classical and QUBO stock selection.
- **Risk aversion**: $\gamma = 5.0$ in the main backtest MVO objective.
- **Transaction costs**: None charged. Turnover is reported as a diagnostic only.

## 4. Methods

### 4.1 Classical MVO

At each rebalance date, trailing Sharpe ratios $\mu_i / \sigma_i$ are computed for all
available stocks. The top $K = 10$ stocks by Sharpe ratio form the candidate set. Weights
within this set are chosen by solving

$$
\min_w \; \tfrac{1}{2}\, w^\top \Sigma w \;-\; \gamma\, \mu^\top w
\quad \text{s.t.} \quad \sum_i w_i = 1,\; 0 \le w_i \le 0.20
$$

using the SCS solver via CVXPY. The weight cap prevents extreme concentration.

### 4.2 ML-Enhanced MVO

The ML path replaces the trailing-Sharpe screen with an XGBClassifier that predicts which
stocks will outperform the cross-sectional median over the next horizon.

**Features.** Five trailing features per (stock, date) pair, computed from log returns:

$$
\text{ret}_{21},\; \text{ret}_{63},\; \text{ret}_{252},\; \text{vol}_{21},\; \text{vol}_{63}
$$

**Target.** The binary label is derived from the forward cumulative log return:

$$
y_{i,t} = \mathbf{1}\!\left[\sum_{s=t+1}^{t+h} \log(1+r_{i,s}) \;>\; \text{cross-sectional median at day } t\right]
$$

Cross-sectional demeaning removes the daily market factor so the classifier learns relative
outperformance rather than market direction.

**Training.** Each observation receives a half-life sample weight with a 1-year decay:

$$
w_i = 0.5^{\,(t_{\text{end}} - t_i)\,/\,365.25}
$$

The classifier is retrained quarterly. Between quarters, the existing booster is warm-started
with 100 additional trees on the updated training set. XGBClassifier hyperparameters:
300 estimators (cold start), max depth 4, learning rate 0.05, subsample 0.8.

**Selection.** At each rebalance, $P(\text{buy})$ is predicted for all available stocks.
Stocks with $P(\text{buy}) > 0.51$ are kept; if more than $K$ pass the threshold, the top
$K$ by probability are selected. If fewer than 3 pass, the method falls back to the Classical
Sharpe screen.

**Weighting.** The same EWMA $\mu$ and $\Sigma$ used by Classical are passed to the MVO solver.
The classifier determines which stocks enter the optimizer; the optimizer determines how much
capital each stock receives.

### 4.3 QUBO + Simulated Annealing

QUBO encodes the stock selection problem as a quadratic minimization over binary variables
$x_i \in \{0, 1\}$:

$$
C(x) = -\mu^\top x + \gamma\, x^\top \Sigma\, x + \lambda\!\left(\sum_i x_i - K\right)^2
$$

The first term rewards high expected return. The second term penalizes portfolio variance.
The third term enforces cardinality: $\lambda$ is set large enough that any solution with
$\sum x_i \ne K$ has higher cost than any feasible solution.

**Q-matrix construction.** Expanding the penalty using $x_i^2 = x_i$:

$$
Q_{ii} = -\mu_i + \gamma\,\Sigma_{ii} + \lambda(1 - 2K), \qquad
Q_{ij} = 2\gamma\,\Sigma_{ij} + 2\lambda \quad (i \ne j)
$$

with $\lambda = 10 \cdot \text{mean}(|\mu|)$.

**Simulated annealing.** The solver performs swap moves: at each iteration, one selected stock
is replaced by one unselected stock, preserving exactly $K$ active positions throughout the
search. The cardinality constraint is satisfied by construction, not by penalty alone. The
acceptance rule follows the Metropolis criterion with geometric cooling
($T \leftarrow 0.997 \cdot T$). Each run uses 1200 iterations and 4 independent restarts;
the first restart is seeded with the top $K$ Sharpe stocks, and the remaining three start
from random selections.

**Weighting.** After SA selects $K$ stocks, weights are assigned by return-tilted inverse
volatility:

$$
w_i \propto \frac{\max(\mu_i, 0) + \epsilon}{\sigma_i}
$$

followed by iterative cap-and-normalize to enforce the 20% constraint.

### 4.4 Equal Weight

$w_i = 1/N$ for all $N$ available stocks. No selection, no optimization. This is the
model-free baseline.

## 5. Walk-Forward Backtest

### 5.1 Rebalance Date Generation

Monthly rebalance dates are the last trading day of each calendar month within the test
window. The first eligible month-end falls in late December 2013 (just before the test
start), giving 88 monthly rebalance points through March 2021. Yearly rebalance dates are
the last trading day of each calendar year, starting December 2013, giving 8 yearly
rebalance points. April 2021 is the final holding period end date for both cadences.

### 5.2 Stock Availability Filter

At each rebalance date, the system looks back 3 years and counts non-NaN return observations
per stock within that window. A stock must have at least `TRAIN_DAYS = 504` live observations
(approximately 2 trading years) to be considered available. This threshold ensures that every
stock entering the optimizer has enough history for the EWMA estimator to produce stable
$\mu$ and $\Sigma$ estimates. Stocks that were listed recently or had extended trading halts
are excluded until they accumulate sufficient data. The available set can change from one
rebalance to the next as stocks cross the 504-day threshold.

For the Full universe (data from Nov 2010), the first monthly rebalance in late December 2013
has roughly 3 years of data, so most of the 49 stocks pass the filter. For the Extended
universe (data from Jan 2004), all 37 stocks have over 9 years of history by the first
rebalance, so the filter is never binding.

If fewer than $\max(K, 2) = 10$ stocks pass the filter at a given rebalance, that date is
skipped entirely. In practice this never occurs because both universes have well over 10
eligible stocks at every rebalance date.

### 5.3 Backtest Loop

At each rebalance date $t$:

1. Build the training sample from returns observed strictly before $t$ (the rebalance-day
   return itself is excluded to avoid look-ahead).
2. Apply the availability filter to determine which stocks can be traded.
3. Estimate $\mu$ and $\Sigma$ from the trailing training sample using EWMA and sample
   covariance respectively.
4. Compute portfolio weights using the selected method (Classical, ML, QUBO, or EqualWeight).
5. Hold those weights fixed until the next rebalance date $t_{\text{next}}$.
6. Record daily portfolio returns over the holding period $(t, t_{\text{next}}]$.

The portfolio daily return between rebalance dates is

$$
r^{\text{port}}_{\tau} = \sum_i w_{i,t}\, r_{i,\tau}, \qquad \tau \in (t, t_{\text{next}}]
$$

Weights are not drift-adjusted within a holding period. The portfolio is buy-and-hold between
rebalances, so intra-period price movements cause the effective weights to drift from their
target values.

### 5.4 Boundary Handling

The final holding period runs from the last rebalance date to `TEST_END = 2021-04-30`. For
monthly rebalancing this is roughly one month; for yearly it is approximately four months
(last yearly rebalance is late December 2020). This partial final period is included in all
performance calculations. Turnover at the first rebalance is measured as the L1 norm of the
initial weight vector (equivalent to a full portfolio build from cash).

### 5.5 Performance Metrics

Daily returns from all holding periods are concatenated into a single time series. From this
series the following metrics are computed:

- **Annualized return**: $\bar{r} \times 252$
- **Annualized volatility**: $\text{std}(r) \times \sqrt{252}$
- **Sharpe ratio**: annualized return / annualized volatility
- **CAGR**: $(V_T / V_0)^{1/T} - 1$ where $T$ is measured in years
- **Maximum drawdown**: largest peak-to-trough decline in the cumulative equity curve

## 6. Results

### 6.1 Summary Table

| Cadence | Universe | Method | AnnReturn | AnnVol | Sharpe | CAGR | MaxDD | AvgTurnover | MedCard |
|---------|----------|--------|-----------|--------|--------|------|-------|-------------|---------|
| monthly | Full | EqualWeight | 17.4% | 17.8% | 0.98 | 17.1% | −41.3% | 0.01 | 49 |
| monthly | Full | Classical | 27.6% | 22.3% | 1.24 | 28.5% | −41.1% | 0.43 | 5 |
| monthly | Full | ML | 25.9% | 21.5% | 1.20 | 26.5% | −38.9% | 1.41 | 5 |
| monthly | Full | QUBO | 21.2% | 17.3% | 1.22 | 21.7% | −26.8% | 0.33 | 10 |
| monthly | Extended | EqualWeight | 17.4% | 18.4% | 0.95 | 17.0% | −43.1% | 0.01 | 37 |
| monthly | Extended | Classical | 28.8% | 21.7% | 1.33 | 30.3% | −38.8% | 0.35 | 5 |
| monthly | Extended | ML | 23.0% | 22.0% | 1.05 | 22.9% | −48.7% | 1.24 | 5 |
| monthly | Extended | QUBO | 20.0% | 18.2% | 1.10 | 20.1% | −36.4% | 0.30 | 10 |
| yearly | Full | EqualWeight | 17.4% | 17.8% | 0.98 | 17.1% | −41.3% | 0.13 | 49 |
| yearly | Full | Classical | 30.1% | 23.5% | 1.28 | 31.4% | −44.2% | 0.95 | 5 |
| yearly | Full | ML | 26.3% | 22.3% | 1.18 | 26.8% | −38.9% | 1.58 | 6 |
| yearly | Full | **QUBO** | **25.6%** | **17.5%** | **1.47** | **27.2%** | **−28.3%** | 0.89 | 10 |
| yearly | Extended | EqualWeight | 17.4% | 18.4% | 0.95 | 17.0% | −43.1% | 0.13 | 37 |
| yearly | Extended | Classical | 29.5% | 23.5% | 1.25 | 30.6% | −43.6% | 0.98 | 5 |
| yearly | Extended | ML | 27.1% | 22.7% | 1.20 | 27.8% | −41.5% | 1.38 | 5 |
| yearly | Extended | **QUBO** | **24.8%** | **19.0%** | **1.30** | **25.8%** | **−33.1%** | 0.89 | 9.5 |

### 6.2 Equity Curves

The notebook plots growth-of-one curves for all four methods across both cadences and
universes. Classical and QUBO consistently outperform EqualWeight. Classical reaches the
highest terminal value in all panels. QUBO's growth path is smoother, with visibly shallower
pullbacks during the 2020 COVID drawdown.

### 6.3 Drawdown Analysis

The drawdown (underwater) plots show that QUBO's maximum drawdown is consistently the
shallowest of any active method: −27% to −36% versus −39% to −44% for Classical. During
the March 2020 crash, QUBO portfolios lost roughly 10-15 percentage points less from peak than
Classical portfolios. EqualWeight drawdowns track the broad market at −41% to −43%.

## 7. Interpretation

### 7.1 Where Classical Wins

Classical MVO delivers the highest absolute returns in every universe and cadence combination.
Within its top $K$ Sharpe-screened set it has continuous weights subject to the 20% per-asset
cap, so the optimizer can tilt heavily toward whichever stocks have the strongest trailing
signal up to that cap. CAGR ranges from 28.5% (monthly Full) to 31.4% (yearly Full).

### 7.2 Where QUBO Wins

QUBO's advantage is risk-adjusted performance, not raw return. With yearly rebalancing:

- **Full universe**: Sharpe 1.47 (best in table), vol 17.5%, MaxDD −28.3%
- **Extended universe**: Sharpe 1.30 (best in table), vol 19.0%, MaxDD −33.1%

QUBO has the highest Sharpe ratio of any method in both yearly universes. The SA solver's
hard cardinality constraint ($K = 10$) enforces diversification at the selection stage. The
inverse-volatility weighting reinforces this by tilting capital toward lower-risk names
within the selected set. QUBO also has the lowest turnover among the active methods at
either cadence (0.89 yearly, 0.33 monthly, versus Classical's 0.95 yearly / 0.43 monthly and
ML's 1.4–1.6 yearly / 1.2–1.4 monthly); lower turnover would make QUBO less exposed to
transaction costs, though no costs are applied in this backtest.

The structural reason QUBO wins on risk-adjusted metrics is that the hard cardinality
constraint acts as implicit regularization. Continuous MVO can concentrate into 5 effective
positions (median cardinality 5), amplifying estimation error. QUBO's $K = 10$ selection
(exact on three of four cells; yearly Extended drifts to 9.5 when one rebalance drops a
name) spreads risk more evenly, which reduces volatility and drawdowns even though it
sacrifices some return.

### 7.3 ML Positioning

The XGBClassifier-screened path produces higher Sharpe and CAGR than EqualWeight in every
universe and cadence combination in this backtest. Drawdowns are mixed: ML is shallower
than EqualWeight on monthly Full but deeper on monthly Extended. ML Sharpe sits below
both Classical and QUBO in most settings, and its average turnover (1.2-1.6x per rebalance)
is the highest of any method. Under transaction costs, ML's gross-of-cost edge would
likely erode the fastest.

### 7.4 Monthly vs Yearly

Monthly rebalancing provides more opportunities to update portfolio weights but introduces
more turnover. Yearly rebalancing reduces trading and produces QUBO's strongest result in
this backtest (Sharpe 1.47 on Full, 1.30 on Extended). EqualWeight's return, vol, Sharpe,
CAGR, and max drawdown are identical across monthly and yearly cadences (since the
holdings vector is the same: every available stock at $1/N$). Only EqualWeight's
turnover changes (0.01 monthly vs 0.13 yearly), reflecting the larger drift between
re-equalizations under the longer holding period.

## 8. Limitations

- **Frictionless backtests.** No transaction costs, market impact, or slippage are modeled.
  Reported turnover shows ML would be most affected by realistic friction.
- **Single test window.** All results rest on a 7.3-year test period (2014-2021) that
  covers a predominantly bullish Indian equity market. The 2020 COVID drawdown is the only
  significant stress event.
- **Long-only, fully invested.** No cash allocation or short positions are permitted.
- **Survivorship-adjacent data.** The processed price matrices use historical NIFTY 50
  constituents; stocks that were delisted or removed from the index during the sample
  period are not present.
- **Fixed hyperparameters.** $K = 10$, $\gamma = 5.0$, the EWMA halflife (252 days), and
  the SA schedule (1200 iterations, 4 restarts) were chosen for conceptual consistency
  across assignments, not cross-validated for this specific comparison.
- **No shrinkage on covariance.** The sample covariance matrix is used without Ledoit-Wolf
  or other shrinkage, which may disadvantage the MVO-based methods on the larger Full
  universe.

## 9. Conclusion

The comparison supports a nuanced answer to the question of when quantum-inspired optimization
provides an advantage. QUBO with simulated annealing does not maximize absolute returns;
Classical MVO consistently achieves higher CAGR. Where QUBO adds value is in risk-adjusted
performance:

1. QUBO has the highest Sharpe ratio in the table under yearly rebalancing:
   **1.47 (Full)** and **1.30 (Extended)**, with the shallowest drawdowns among the
   active methods.
2. The hard cardinality constraint acts as implicit regularization, producing portfolios
   with lower concentration and more stable risk profiles than the MVO-selected
   alternatives in this backtest.
3. QUBO's low turnover (0.89) would reduce its exposure to transaction costs if any were
   charged, though this is not modelled here.

The practical case for QUBO is strongest when the investor's priority is drawdown control and
Sharpe stability rather than return maximization, and when the stock universe is large enough
(49 stocks in Full) to give the combinatorial solver meaningful room to diversify.
