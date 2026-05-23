# Assignment 2: Portfolio Optimization using SCS

## Objective

This assignment examines portfolio optimization using the **SCS** solver on NIFTY 50 stock data. The analysis:

- the data is cleaned from `raw_data.csv`
- two stock universes are constructed from the cleaned data
- portfolio weights are estimated for different values of $\gamma$
- expected return, risk, and selected stocks are reported
- solver settings are changed to test stability

## 1. Introduction

Portfolio optimization is concerned with selecting weights that balance return and risk.

This assignment uses a classical mean-variance framework and solves it with **SCS**. The objective is to examine how the portfolio changes as the optimizer places more or less emphasis on return.

## 2. Data Processing and Two Universes

The raw file used in this assignment is `data/raw/raw_data.csv`. Before building the portfolio dataset, the file is cleaned in the following way:

- only rows with `Series == "EQ"` are kept
- the pseudo-symbol rows `NIFTY50_all` and `stock_metadata` are removed
- the working columns are `Date`, `Symbol`, `Close`, and `Prev Close`
- daily returns are computed gap-safe per row as $R_t = \text{Close}_t / \text{PrevClose}_t - 1$, using the exchange-reported previous close (split- and bonus-adjusted by NSE)
- a small number of single-day returns with $|R_t| \ge 0.5$ are dropped as unadjusted corporate-action artifacts (e.g. a +5438% UPL row on 2004-01-23 caused by a pre-split price mismatch)
- the cleaned prices are reshaped into a stock-by-date price matrix and the returns into a stock-by-date return matrix

Not all stocks have the same trading history. Accordingly, two universes are constructed:

- **Full Universe**: keep all cleaned stocks and use the common period where every stock has data
- **Extended Universe**: remove the highest-missingness stocks and keep a longer sample window

The final universes are:

- **Full Universe**: 49 stocks, 2598 trading days, from `2010-11-04` to `2021-04-30`
- **Extended Universe**: 37 stocks, 4286 trading days, from `2004-01-23` to `2021-04-30`

This comparison distinguishes between a shorter common history and a longer but smaller stock universe.

## 3. Mean-Variance Formulation

Daily returns are computed gap-safe per row using the exchange-reported previous close:

$$
R_t = \frac{\text{Close}_t}{\text{PrevClose}_t} - 1
$$

These returns are annualized using the 252-trading-day convention:

$$
\mu = 252 \cdot \mathbb{E}[R_t], \qquad \Sigma = 252 \cdot \mathrm{Cov}(R_t)
$$

The optimization problem solved in the notebook is:

$$
\min_w \; \frac{1}{2} w^\top \Sigma w - \gamma \mu^\top w
$$

subject to

$$
\mathbf{1}^\top w = 1, \qquad w \ge 0
$$

Here:

- $w$ is the portfolio weight vector
- $\mu$ is the annualized expected return vector
- $\Sigma$ is the annualized covariance matrix
- $\gamma$ controls the trade-off between return and risk

When $\gamma$ is small, the optimizer places greater weight on risk control. When $\gamma$ is large, the portfolio typically becomes more concentrated.

## 4. Why SCS Was Used

SCS is a first-order conic solver. It is appropriate here because the portfolio problem is convex, the constraints are linear, and the same model must be solved repeatedly for different values of $\gamma$.

### Cone Reformulation

Introduce auxiliary $t$:

$$
\frac{1}{2} w^\top \Sigma w = \frac{1}{2} \|L^\top w\|_2^2 \le t.
$$

The problem becomes

$$
\min_{w,\, t} \; \frac{1}{2} t - \gamma \mu^\top w
\quad \text{s.t.} \quad
\|L^\top w\|_2^2 \le t, \;\;
\mathbf{1}^\top w = 1, \;\; w \ge 0.
$$

Objective is linear in $(w, t)$; risk constraint is a rotated SOC. CVXPY passes this directly to SCS.

### ADMM Updates

Each SCS iteration:

1. **Primal**: solve for $(w, t)$ ignoring cone:
$$
(w^{k+1}, t^{k+1}) = \arg\min_{w,t} \; \left(\tfrac{1}{2}t - \gamma\mu^\top w\right) + \tfrac{\rho}{2}\left\|\binom{w}{t} - z^k + u^k\right\|_2^2
$$
Coefficient matrix involves $L^\top L = \Sigma$, fixed across all $\gamma$; same factorization reused every iteration.

2. **Project**: $z^{k+1} = \Pi\bigl(w^{k+1}+u^k,\; t^{k+1}+u^k_t\bigr)$ onto $\mathbf{1}^\top w=1$, $w\ge 0$, $(t, L^\top w)\in\mathcal{Q}_{\text{rot}}$.

3. **Dual**: $u^{k+1} = u^k + (w^{k+1}, t^{k+1}) - z^{k+1}$

Because $L$ is fixed, the factorization is paid once; SCS only iterates the cheap primal-project-dual cycle across the $\gamma$ sweep.

### Why Cholesky?

Cholesky-factor $\Sigma = L L^\top$:

- **Cone form**: converts $w^\top \Sigma w$ into $\|L^\top w\|_2^2$, a squared norm SCS can express as a rotated second-order cone.
- **Speed**: $L$ depends only on $\Sigma$, not on $\gamma$ or $\mu$. Computed **once**, reused across every $\gamma$ and every ADMM iteration.

## 5. Portfolio Results

Before the $\gamma$ sweep, the global minimum variance (GMV) portfolio is computed as the risk floor no $\gamma$-portfolio can achieve lower volatility than this. The GMV risk floors are **13.43%** for the Full universe and **17.00%** for the Extended universe. The higher Extended floor reflects the inclusion of the 2004-2010 period, which contains greater market stress and raises the estimated covariance.

In the tables below, **Return** means annualized expected return and **Risk** means annualized volatility.

### 5.1 Full Universe Results

| Gamma | Annualized Expected Return | Annualized Volatility | Sharpe | Stocks Selected | Top Holdings |
| --- | --- | --- | --- | --- | --- |
| 0.1 | 25.51% | 16.73% | 1.525 | 12 | HINDUNILVR / SHREECEM / BAJAJFINSV |
| 0.5 | 34.57% | 26.77% | 1.292 | 4 | BAJAJFINSV / BAJFINANCE / SHREECEM |
| 1.0 | 37.59% | 33.66% | 1.117 | 2 | BAJAJFINSV / BAJFINANCE |
| 2.0 | 37.86% | 34.83% | 1.087 | 2 | BAJAJFINSV / BAJFINANCE |
| 5.0 | 38.66% | 42.16% | 0.917 | 2 | BAJFINANCE / BAJAJFINSV |
| 10.0 | 39.03% | 46.85% | 0.833 | 1 | BAJFINANCE |

### 5.2 Extended Universe Results

| Gamma | Annualized Expected Return | Annualized Volatility | Sharpe | Stocks Selected | Top Holdings |
| --- | --- | --- | --- | --- | --- |
| 0.1 | 27.44% | 19.64% | 1.397 | 12 | SHREECEM / HINDUNILVR / ASIANPAINT |
| 0.5 | 39.17% | 29.70% | 1.319 | 3 | SHREECEM / BAJFINANCE / TITAN |
| 1.0 | 39.87% | 31.43% | 1.269 | 3 | SHREECEM / BAJFINANCE / TITAN |
| 2.0 | 40.73% | 35.04% | 1.162 | 2 | BAJFINANCE / SHREECEM |
| 5.0 | 42.45% | 49.09% | 0.865 | 1 | BAJFINANCE |
| 10.0 | 42.46% | 49.09% | 0.865 | 1 | BAJFINANCE |

## 6. Interpretation of Results

The results show a clear pattern in both universes.

At low $\gamma$, the optimizer is more careful about risk, so it spreads the portfolio across more stocks and more sectors. This is why the `0.1` portfolios in both universes hold 12 stocks and have the highest or near-highest Sharpe values.

At middle values of $\gamma$, the optimizer starts giving more importance to return. The portfolio becomes smaller and moves toward a few stronger names. In the Full universe, this concentration happens quickly and the solution moves toward BAJAJFINSV and BAJFINANCE. In the Extended universe, SHREECEM and TITAN remain important for longer.

At high $\gamma$, the solution becomes a corner portfolio or close to it. In both universes, BAJFINANCE dominates the final solution because it has the strongest expected return estimate in the sample.

Another important point is that the best risk-adjusted region is not the most concentrated one. The Sharpe values are strongest in the lower or middle gamma range. This indicates that pushing the optimizer too far toward return can reduce diversification faster than it improves overall risk-adjusted performance.

The difference between the Full and Extended universes comes from the data window. The Extended universe uses a longer sample, so the expected return and covariance estimates change. As a result, the selected holdings and the speed of concentration are not the same.

The efficient frontier comparison (notebook §6) shows a crossover near 23% risk. Below that level, the Full universe achieves higher return at the same risk because its broader 49-stock universe provides more diversification options within the shorter window. Above it, the Extended universe overtakes the longer sample captures stronger long-run return signals for names like BAJFINANCE and SHREECEM that dominate at higher risk budgets.

## 7. Solver Parameter Sensitivity

The assignment also asks for parameter changes and re-estimation. In this section, the optimization problem is held fixed and only the SCS settings are changed.

Using the Full universe at $\gamma = 1.0$, the results were:

| Configuration | Return | Risk | Main Observation |
| --- | --- | --- | --- |
| Standard | 37.59% | 33.66% | Baseline solution |
| Loose | 37.59% | 33.66% | Almost unchanged |
| High Precision | 37.59% | 33.66% | Almost unchanged |
| Aggressive Alpha | 37.59% | 33.66% | Almost unchanged |

The return drift, risk drift, and weight drift are negligible. Therefore, SCS is numerically stable for this portfolio problem.

## 8. Solver Time Comparison

The principal practical benefit of SCS in this setting is speed. On the Full universe at $\gamma = 1.0$, SCS was compared with a high-precision SCS run and with other convex solvers on the same problem.

| Solver | Solve Time (s) | Return | Risk |
| --- | ---: | ---: | ---: |
| SCS | 0.000097 | 37.59% | 33.66% |
| SCS (High Precision) | 0.000137 | 37.59% | 33.66% |
| ECOS | 0.000861 | 37.59% | 33.66% |
| OSQP | 0.000460 | 37.59% | 33.66% |

The portfolio is almost identical across all four runs, but SCS is faster. That is the main reason it is an appropriate solver for this assignment.

## 9. Limitations

- expected returns and covariance are estimated from historical data only
- transaction costs are not included
- the optimization is long-only and fully invested
- the final portfolio depends on the chosen universe and sample window

These results should be treated as a classroom-style portfolio-optimization study, not as a full real-world trading system.

## 10. Conclusion

SCS performs well for solving the classical mean-variance portfolio problem. It gives stable solutions and makes it easy to study how the portfolio changes as the trade-off between return and risk is adjusted.

The two-universe setup also matters. The data window affects the inputs, and once the inputs change, the optimized portfolio changes as well.

Overall, the main pattern is clear:

- low $\gamma$ gives broader diversification
- middle $\gamma$ concentrates the portfolio into a few stronger names
- high $\gamma$ gives corner solutions

The assignment shows both the usefulness of SCS as a solver and the importance of understanding how data choices affect portfolio optimization results.
