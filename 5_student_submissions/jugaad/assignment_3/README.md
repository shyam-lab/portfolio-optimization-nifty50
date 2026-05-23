# Assignment 3: Machine Learning Driven Portfolio Optimization

## Objective

This assignment examines whether a machine learning classifier can improve portfolio selection on
NIFTY 50 stock data. A gradient-boosted classifier is trained to identify which stocks are likely
to outperform the cross-sectional average over the next month. The selected stocks then enter a
Markowitz mean-variance optimizer. The key question is whether this ML-driven screen improves
risk-adjusted performance compared to running the optimizer on the full universe.

## 1. Introduction

Classical Markowitz optimization uses all available stocks and estimates expected returns and
covariances from historical data. The approach is sensitive to estimation error: stocks with
high historical returns dominate the optimizer even if those estimates are unreliable.

This assignment introduces a two-stage pipeline. First, an XGBClassifier predicts which stocks
are likely to outperform their peers over the next month. Second, the Markowitz optimizer runs
on only the classifier-selected subset, using the same EWMA-weighted return and covariance
estimates as the baseline. The same data, the same solver, and the same objective are used -
only the stock universe fed into the optimizer differs.

Three portfolio families are compared on both the Full and Extended universes:

1. **Equal Weight** - $w_i = 1/N$, model-free baseline.
2. **Markowitz (EWMA)** - mean-variance optimizer on the full universe.
3. **ML class + Markowitz** - same optimizer restricted to classifier-selected stocks.

## 2. Data and Two Universes

The raw data file is `data/raw/raw_data.csv`. Only rows with `Series == "EQ"` are retained;
pseudo-symbol rows are removed. Close prices are pivoted into a wide date x symbol matrix.

Two universes are constructed:

- **Full Universe**: all stocks, restricted to the common window where every stock has data.
  49 stocks, 2598 trading days, 2010-11-04 to 2021-04-30.
- **Extended Universe**: stocks with more than 1,100 missing days are dropped, giving a longer
  window. 37 stocks, 4286 trading days, 2004-01-23 to 2021-04-30.

The trade-off is between breadth (more stocks, shorter history) and depth (fewer stocks, longer
history that includes the 2008 financial crisis and other stress periods).

## 3. Train and Test Split

All model parameters are estimated on data up to `2018-12-31`. The test window is
`2019-01-01` to `2021-04-30` (~2.5 years), which includes the 2020 COVID drawdown and recovery.
No test-period information enters any estimation step.

## 4. Feature Construction

### 4.1 Forecast Target

For each (stock, date) pair, the target is whether the stock's 21-day forward return beats the
cross-sectional average:

$$
R_{i,t} = \frac{P_{i,t+21}}{P_{i,t}} - 1, \qquad
y_{i,t} = \mathbf{1}\left[ R_{i,t} - \bar{R}_t > 0 \right]
$$

Cross-sectional demeaning removes the common market factor so the classifier learns stock
selection, not market direction. Because consecutive dates share 20 of 21 days in their
forward windows, the labels are serially correlated. Classifier accuracy should be read as
an upper bound; the effective sample size is smaller than the panel row count.

### 4.2 Features

The initial feature pool (20 features per (stock, date), constructed from past data only):

- **Daily lags** $r_{t-k}$, $k = 1, 2, 3, 4, 5$ (short-term momentum/reversal).
- **Monthly lags** $r_{t-k}$, $k = 21, 42, \ldots, 252$ at monthly spacing (medium-term momentum).
- **Rolling means** over 63, 126, and 252-day windows (long-term trend).

Walk-forward CV (Section 6) selects the final subset per universe. The Extended universe retains
only 6 features (5 daily lags + 252-day rolling mean); the Full universe retains all 20.

## 5. ML Direction Classifier

### 5.1 XGBClassifier

An XGBoost gradient-boosted classifier is fit on the 20-feature pooled panel with the binary
cross-sectional target. Each observation is assigned a half-life sample weight (see §5.2).
The classifier's test-period output is a probability score per (stock, date) pair, which is
averaged per stock over all test dates to produce a single outperformance probability $\hat{p}_i$.
Stocks with $\hat{p}_i > 0.51$ form the filtered universe.

### 5.2 Half-Life Sample Weights

Each training observation is weighted by recency with a 2-year half-life:

$$
w_i = 0.5^{\,(T_{\text{end}} - t_i)\;/\;(365.25 \times 2)}
$$

This matches the EWMA decay used for $\mu$ and $\Sigma$, so all weighted estimators in the
notebook apply the same rate. An observation 2 years before the cutoff receives half the weight
of the most recent observation.

### 5.3 Classifier Metrics

| Universe | Buy Accuracy | Buy-Set Mean Return | Sell-Set Mean Return | Stocks Kept (P > 0.51) |
| --- | --- | --- | --- | --- |
| Full | 53.44% | - | - | 14 / 49 |
| Extended | 51.09% | - | - | 14 / 37 |

Full has a positive buy-sell spread (+1.84pp). Extended is near zero (-0.04pp). Both universes
keep only 14 stocks, cutting the optimizer's feasible set roughly in half.

## 6. Walk-Forward Feature Selection

The 20 features are selected using walk-forward cross-validation to avoid overfitting the
feature choice to the single 2018-12-31 cutoff. Six expanding (train, val) folds are constructed
with val years 2013-2018; the 2019-2021 holdout is never used in CV.

### 6.1 Why Walk-Forward

A single cutoff offers no held-out signal for model selection. Any implicit feature choice tuned
to the test window is a form of overfitting. Six independent folds within the training era
validate feature choices against years the classifier has not yet seen.

### 6.2 Why Warm-Start

Each fold adds one year of data. A cold refit discards signal already captured in the prior
booster. Warm-starting from fold $n-1$ and appending 100 new trees produces a chronologically-
stacked ensemble that is more data-efficient and faster than a full cold refit each fold.

### 6.3 Stage A: Group Ablation Results

Leave-one-out ablation over three feature groups. A group is dropped if removing it improves
mean val Sharpe by $\ge 0.05$.

Full universe:

| Config | Features | Mean Val Sharpe | Delta vs Baseline |
| --- | --- | --- | --- |
| Baseline (all groups) | 20 | 1.2400 | +0.0000 |
| Drop daily\_lags | 15 | 1.2411 | +0.0011 |
| Drop monthly\_lags | 8 | 0.9581 | -0.2819 |
| Drop rolling\_means | 17 | 0.4653 | -0.7747 |

All three groups kept for Full (no group improves val Sharpe by $\ge 0.05$ when dropped).

Extended universe:

| Config | Features | Mean Val Sharpe | Delta vs Baseline |
| --- | --- | --- | --- |
| Baseline (all groups) | 20 | 1.1901 | +0.0000 |
| Drop daily\_lags | 15 | 1.1802 | -0.0099 |
| Drop monthly\_lags | 8 | 1.6351 | +0.4450 |
| Drop rolling\_means | 17 | 0.6292 | -0.5609 |

Monthly lags dropped for Extended (removing them improves val Sharpe by +0.45, well above 0.05).

### 6.4 Stage B: Rolling-Mean Subset Search

All non-empty subsets of the surviving rolling-mean windows are scored. Best subset chosen by
max mean val Sharpe.

**Full universe (all RM windows survive Stage A):**

| RM Subset | Features | Mean Val Sharpe |
| --- | --- | --- |
| {63, 126, 252} | 20 | 1.2400 |
| {252} | 18 | 1.1180 |
| {126, 252} | 19 | 0.9797 |
| {63, 126} | 19 | 0.9591 |
| {63} | 18 | 0.9557 |
| {63, 252} | 19 | 0.9449 |
| {126} | 18 | 0.9005 |

Full: all three RM windows kept (val Sharpe 1.24 beats any subset).

**Extended universe (only rolling\_means survive Stage A; monthly lags already dropped):**

| RM Subset | Features | Mean Val Sharpe |
| --- | --- | --- |
| {252} | 6 | 1.8236 |
| {63, 126, 252} | 8 | 1.6351 |
| {63} | 6 | 1.6330 |
| {63, 126} | 7 | 1.6017 |
| {126} | 6 | 1.5491 |
| {63, 252} | 7 | 1.3845 |
| {126, 252} | 7 | 1.3457 |

Extended: only 252-day rolling mean kept (val Sharpe 1.82, clearly best).

### 6.5 Selected Features per Universe

- **Full**: all 20 features (5 daily lags + 12 monthly lags + $rm_{63}$ + $rm_{126}$ + $rm_{252}$).
- **Extended**: 6 features (5 daily lags + $rm_{252}$ only).

The Extended universe benefits from a simpler model. Its longer, more volatile history (2004-2021)
contains more regime changes; medium-term lags and shorter rolling means add noise rather than
signal over a 6-fold CV.

## 7. Portfolio Construction

### 7.1 EWMA $\mu$ and $\Sigma$

Expected returns and covariance are estimated with exponential weighting using a 504-day
(2 trading year) half-life, matching the classifier decay rate:

$$
\mu_i = 252 \cdot \sum_t w_t \, r_{i,t}, \qquad
\Sigma_{ij} = 252 \cdot \sum_t w_t \, (r_{i,t} - \mu_i/252)(r_{j,t} - \mu_j/252)
$$

### 7.2 Mean-Variance Solver

$$
\min_w \; \tfrac{1}{2} \, w^\top \Sigma w - \gamma \, \mu^\top w
\quad \text{s.t.} \quad \mathbf{1}^\top w = 1,\; w \ge 0
$$

Low $\gamma$ favors variance reduction; high $\gamma$ concentrates in high-return stocks.
Solver: SCS (same as Assignment 2).

### 7.3 Three Portfolio Families

Only the stock universe fed into the optimizer differs across the three families. The estimator
($\mu$, $\Sigma$), the solver, and the objective are identical.

## 8. Results

### 8.1 Full Universe Results

| Gamma | Portfolio | Ann. Return | Ann. Volatility | Sharpe | CAGR | N\_eff |
| --- | --- | --- | --- | --- | --- | --- |
| - | Equal Weight | 7.91% | 21.71% | 0.364 | 6.10% | 49.0 |
| 0.10 | Markowitz | 20.15% | 25.75% | 0.782 | 18.28% | 4.7 |
| 0.50 | Markowitz | 32.43% | 39.92% | 0.812 | 27.44% | 1.6 |
| 1.00 | Markowitz | 34.34% | 43.63% | 0.787 | 27.82% | 1.2 |
| 2.00 | Markowitz | 33.61% | 43.74% | 0.768 | 26.83% | 1.1 |
| 5.00 | Markowitz | 33.18% | 43.84% | 0.757 | 26.23% | 1.0 |
| 0.10 | ML class + Markowitz | 12.23% | 24.76% | 0.494 | 9.59% | 4.4 |
| 0.50 | ML class + Markowitz | 24.44% | 28.01% | 0.873 | 22.76% | 1.8 |
| 1.00 | ML class + Markowitz | 30.30% | 32.53% | 0.931 | 28.32% | 2.0 |
| 2.00 | ML class + Markowitz | 42.01% | 46.34% | 0.907 | 36.41% | 1.1 |
| 5.00 | ML class + Markowitz | 43.58% | 48.47% | 0.899 | 37.14% | 1.0 |

### 8.2 Extended Universe Results

| Gamma | Portfolio | Ann. Return | Ann. Volatility | Sharpe | CAGR | N\_eff |
| --- | --- | --- | --- | --- | --- | --- |
| - | Equal Weight | 8.42% | 22.42% | 0.376 | 6.59% | 37.0 |
| 0.10 | Markowitz | 12.39% | 24.87% | 0.498 | 9.72% | 5.1 |
| 0.50 | Markowitz | 27.16% | 27.92% | 0.973 | 26.14% | 2.8 |
| 1.00 | Markowitz | 34.23% | 34.65% | 0.988 | 32.48% | 2.4 |
| 2.00 | Markowitz | 41.91% | 45.34% | 0.924 | 36.91% | 1.2 |
| 5.00 | Markowitz | 43.58% | 48.47% | 0.899 | 37.14% | 1.0 |
| 0.10 | ML class + Markowitz | 7.89% | 25.68% | 0.307 | 4.66% | 4.0 |
| 0.50 | ML class + Markowitz | 22.75% | 28.54% | 0.797 | 20.51% | 2.1 |
| 1.00 | ML class + Markowitz | 32.32% | 34.56% | 0.935 | 30.03% | 1.9 |
| 2.00 | ML class + Markowitz | 43.58% | 48.47% | 0.899 | 37.14% | 1.0 |
| 5.00 | ML class + Markowitz | 43.58% | 48.47% | 0.899 | 37.14% | 1.0 |

### 8.3 Realized Risk-Return Trace

The notebook figure plots realized (vol, return) pairs for each $\gamma$ in a 1000-point grid
from 0.1 to 5.0, evaluated on the 2019-2021 test set. This is a realized (ex-post) trace, not
the classical ex-ante efficient frontier.

At low $\gamma$ both traces start close together, but as $\gamma$ increases the Full universe
traces diverge: ML achieves higher Sharpe at comparable or lower volatility. On the Extended
universe, Markowitz sits above ML throughout the trace. The overlap-band comparison at minimum
volatility shows a -0.41% gap on Full and -11.34% on Extended, but this understates the ML
advantage on Full because it measures only the low $\gamma$ region where Markowitz dominates.

## 9. Interpretation

### 9.1 Where ML Wins (Full Universe)

On the **Full universe**, the ML screen beats Markowitz on Sharpe at four of five $\gamma$
values ($\gamma \ge 0.5$). The sole exception is $\gamma = 0.1$, where Markowitz wins
(0.782 vs 0.494, gap -0.289).

The pattern follows from how $\gamma$ interacts with the reduced stock set. At $\gamma = 0.1$
the optimizer prioritizes variance reduction; ML's $N_{\text{eff}} = 4.4$ vs Markowitz's 4.7
means fewer effective positions, and the diversification loss from filtering 49 stocks down to
14 dominates. As $\gamma$ increases, the optimizer tilts toward expected return, and deliberate
concentration becomes the goal. The classifier's stock-picking signal (53% buy accuracy,
+1.84pp buy-sell spread) focuses capital on higher-conviction names. By $\gamma = 1.0$ ML
reaches Sharpe 0.931 vs 0.787 for Markowitz, a +0.144 advantage.

CV kept all 20 features for Full. Extended needed only 6 (5 daily lags + 252-day rolling mean);
monthly lags hurt (+0.45 val Sharpe when removed).

### 9.2 Where ML Does Not Help (Extended)

On the **Extended universe**, Markowitz wins at every $\gamma$. The Sharpe gap ranges from
-0.191 ($\gamma = 0.1$) to -0.025 ($\gamma = 2.0$), narrowing as both methods converge
toward the same dominant stock. At $\gamma \ge 2.0$ both portfolios collapse to
$N_{\text{eff}} \approx 1.0$ and identical Sharpe (0.899).

The Extended universe starts with only 37 stocks. Filtering to 14 removes 62% of the
investable set, compared to 71% on Full (14 of 49). In relative terms the Extended screen
is more aggressive. The classifier is also weaker on Extended (51.09% buy accuracy,
buy-sell spread near zero at -0.04pp), so the selection signal does not compensate for
the lost breadth.

### 9.3 Concentration Mechanics

The $N_{\text{eff}}$ column explains why the ML advantage is gamma-dependent. At $\gamma = 1.0$
on Full, ML achieves Sharpe 0.931 with $N_{\text{eff}} = 2.0$ while Markowitz gets 0.787 with
$N_{\text{eff}} = 1.2$. ML holds two effective positions versus one for Markowitz, yet
delivers better risk-adjusted return because those two stocks are better-selected by the
classifier. At $\gamma = 5.0$ on Extended, both methods converge to $N_{\text{eff}} = 1.0$
and identical performance (Sharpe 0.899), confirming that at extreme concentration the
screen and the unconstrained optimizer pick the same dominant name (BAJFINANCE).

## 10. Limitations

- **Single test window.** All results rest on a 2.5-year test period. The COVID drawdown (March
  2020) dominates test-period statistics and makes it difficult to separate model effects from
  the macroeconomic event.
- **No transaction costs.** Weights are set once at the start of the test period and held fixed.
  Rebalancing with costs is modeled in Assignment 5.
- **Long-only, fully invested.** No cash or short positions.
- **Free parameters.** The 504-day EWMA half-life and 0.51 classifier threshold were chosen for
  conceptual consistency, not cross-validated. Results are sensitive to both choices.
- **Overlapping labels.** The 21-day forward return target means consecutive training
  observations share most of the same realized return window, inflating apparent sample
  size and classifier confidence.
- **Two-stage feature selection is greedy.** Stage A ablates groups independently; Stage B
  searches within the surviving RM group only. The globally optimal feature set may differ.

## 11. Conclusion

Whether the ML screen adds value depends on the universe and on $\gamma$. On the Full universe
(49 stocks), ML class + Markowitz beats plain Markowitz on Sharpe at every $\gamma \ge 0.5$,
with the largest gain at $\gamma = 1.0$ (+0.144). At $\gamma = 0.1$ Markowitz wins because the
optimizer needs the diversification that 49 stocks provide and filtering to 14 removes too much.

On the Extended universe (37 stocks), Markowitz wins at every $\gamma$. The smaller starting
universe means the ML filter removes proportionally more breadth, and the classifier's weaker
accuracy on Extended (51% vs 53%) does not compensate.

CV selected 6 features for Extended (daily lags + 252-day rolling mean) and all 20 for Full.

The central finding is that the ML screen is not uniformly helpful or harmful. Its value is
conditional on the optimizer's risk-return preference: at high $\gamma$, where the optimizer
deliberately concentrates, the classifier's ability to identify the highest-conviction names
becomes an advantage. At low $\gamma$, where diversification matters most, shrinking the
stock set is a net cost.
