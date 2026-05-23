# Assignment 1: Understanding the Markowitz Portfolio Optimization Problem

## Objective

This assignment examines the Markowitz portfolio-optimization problem in a practical setting. The analysis:

- 10 stocks were selected from 3 different sectors
- optimized portfolios were constructed for 3 risk levels
- Monte Carlo search was compared with convex optimization

## 1. Introduction

Portfolio construction is concerned with balancing return and risk. Markowitz portfolio theory provides a formal framework for studying that trade-off.

The central idea is that a portfolio should not be assessed only by examining each asset in isolation. The co-movement of assets is equally important. This is the basis of diversification.

## 2. Before Markowitz

Before Markowitz, investors often focused on selecting individual stocks that appeared attractive on their own. That approach did not properly measure the risk of the overall portfolio.

Markowitz reframed the problem. Instead of asking which stock is best, the more useful question is which combination of assets provides the best balance of return and risk.

## 3. Markowitz and Diversification

In the Markowitz framework, a portfolio is represented by a weight vector $w$, where each element gives the share of capital invested in one asset.

The main quantities are:

- expected portfolio return:

  $$
  \mathbb{E}[R_p] = \mu^\top w
  $$

- portfolio variance:

  $$
  \mathrm{Var}(R_p) = w^\top \Sigma w
  $$

- portfolio volatility:

  $$
  \sigma_p = \sqrt{w^\top \Sigma w}
  $$

Here:

- $\mu$ is the vector of expected asset returns
- $\Sigma$ is the covariance matrix
- $w$ is the vector of portfolio weights

The covariance matrix is important because it captures how assets move relative to one another. This is what makes diversification possible. If assets do not move in exactly the same way, losses in one part of the portfolio can be offset by gains in another.

## 4. Mathematical Formulation

The target-risk formulation used in the notebook is:

$$
\begin{aligned}
\max_{w} \quad & \mu^\top w \\
\text{subject to} \quad & w^\top \Sigma w \le \sigma_{\text{target}}^2 \\
& \sum_{i=1}^{n} w_i = 1 \\
& w_i \ge 0 \quad \text{for all } i
\end{aligned}
$$

These constraints mean:

- the portfolio must remain within a chosen risk budget
- all capital must be invested
- short selling is not allowed

Another common version of the same idea is:

$$
\begin{aligned}
\max_{w} \quad & \mu^\top w - \gamma w^\top \Sigma w \\
\text{subject to} \quad & \sum_{i=1}^{n} w_i = 1 \\
& w_i \ge 0 \quad \text{for all } i
\end{aligned}
$$

In this version, $\gamma$ controls how strongly risk is penalized. A larger $\gamma$ implies a more conservative investor.

## 5. Why the Two Forms Are Equivalent

These two formulations appear different, but they represent the same underlying trade-off between return and risk. In one case, the investor chooses a target risk level directly. In the other case, the investor chooses a risk-aversion parameter.

If we vary $\sigma_{\text{target}}$ in the first model or vary $\gamma$ in the second model, we trace the same efficient frontier. Thus, the difference is mainly one of interpretation rather than basic theory.

The two formulations are also connected mathematically through the **Lagrangian**. For the target-risk problem, the Lagrangian is

$$
\mathcal{L}(w,\lambda,\eta)
=
\mu^\top w
-
\lambda \left(w^\top \Sigma w - \sigma_{\text{target}}^2\right)
-
\eta \left(\sum_{i=1}^{n} w_i - 1\right).
$$

Here, $\lambda$ is the multiplier on the risk constraint. This gives the same return-minus-risk structure as

$$
\mu^\top w - \gamma w^\top \Sigma w.
$$

So mathematically, the multiplier $\lambda$ plays a role similar to the risk-aversion parameter $\gamma$, which is why both formulations generate the same efficient frontier.

## 6. Data and Portfolio Setup

The portfolio universe consists of 10 large-cap U.S. stocks drawn from three sectors:

- Technology: `AAPL`, `MSFT`, `NVDA`, `ORCL`
- Finance: `JPM`, `V`, `BAC`
- Health Care: `JNJ`, `PFE`, `UNH`

Adjusted closing prices were obtained from Yahoo Finance through `yfinance`. The notebook used:

- requested download window: `2020-01-01` to `2025-12-03`
- saved output window: `2020-01-02` to `2025-12-02`
- simple daily returns computed as

  $$
  r_t = \frac{P_t}{P_{t-1}} - 1
  $$

- annualization based on `252` trading days

The annualized inputs were constructed as:

$$
\hat{\mu}_{\text{ann}} = 252 \, \bar{r}
$$

$$
\hat{\Sigma}_{\text{ann}} = 252 \, \hat{\Sigma}_{\text{daily}}
$$

where $\hat{\Sigma}_{\text{daily}}$ is the sample covariance matrix computed with an unbiased denominator (N−1).

The three target annual risk levels were:

- `20%` for low risk
- `28%` for medium risk
- `40%` for high risk

## 7. Methods Used

Two methods were used.

### Convex Optimization

#### What ECOS Solves

ECOS is a primal-dual interior-point solver for second-order cone programs (SOCPs). Its objective must be linear, so the quadratic risk constraint $w^\top \Sigma w \le \sigma_{\text{target}}^2$ has to be lifted into a cone before the problem can be passed to ECOS.

#### Casting the Markowitz Target-Risk Problem as an SOCP

The target-risk problem is

$$
\max_w \; \mu^\top w
\quad \text{s.t.} \quad
w^\top \Sigma w \le \sigma_{\text{target}}^2, \;
\mathbf{1}^\top w = 1, \;
w \ge 0.
$$

The quadratic risk constraint is a Euclidean ball. Cholesky-factor $\Sigma = L L^\top$, then

$$
w^\top \Sigma w \le \sigma_{\text{target}}^2
\;\Longleftrightarrow\;
\|L^\top w\|_2 \le \sigma_{\text{target}}
\;\Longleftrightarrow\;
(\sigma_{\text{target}},\, L^\top w) \in \mathcal{Q}^{n+1}.
$$

Negating the objective (ECOS minimizes), the SOCP ECOS actually solves is

$$
\min_w \; -\mu^\top w
\quad \text{s.t.} \quad
\mathbf{1}^\top w = 1, \;
w \ge 0, \;
(\sigma_{\text{target}},\, L^\top w) \in \mathcal{Q}^{n+1}.
$$

Cones: one zero cone (budget equality), $n$ nonneg rows (long-only), one second-order cone of dimension $n+1$ (risk cap).

#### Equivalence with the Lagrangian / Penalty Form

Section 5 shows that the target-risk form and the penalty form $\max_w \mu^\top w - \gamma w^\top \Sigma w$ trace the same efficient frontier, with the risk-constraint multiplier $\lambda$ playing the role of $\gamma$. ECOS solves the target-risk SOCP directly. At the optimum, the KKT stationarity condition gives

$$
\mu = 2\lambda \Sigma w + \eta \mathbf{1} - \nu, \qquad \nu \ge 0,
$$

which matches the gradient of $\mu^\top w - \gamma w^\top \Sigma w$ when $\gamma = \lambda$. The same optimal $w$ is recovered regardless of which form is solved only the parametrization differs.

### Monte Carlo Search

Random long-only portfolios were generated using a Dirichlet sampler (100 000 draws) and evaluated in return-risk space. For each risk budget, the highest-return sampled portfolio satisfying the risk limit was retained.

The principal comparison uses the standard Dirichlet sampler. Three alternative samplers (plain random, uniform simplex, and 3-asset concentrated Dirichlet) are retained in the notebook appendix.

Monte Carlo simulation is informative for visualizing the feasible set, but it does not guarantee the true optimum. Results are seed-dependent and may vary slightly across re-executions.

## 8. Results

### 8.1 Convex Optimization Results

The convex optimizer matched the target risk levels exactly and produced the following expected returns:

| Risk Level | Target Risk | Achieved Risk | Expected Return |
| :--- | :--- | :--- | :--- |
| Low Risk | 20.00% | 20.00% | 25.05% |
| Medium Risk | 28.00% | 28.00% | 39.74% |
| High Risk | 40.00% | 40.00% | 55.82% |

The corresponding portfolio weights were:

| Ticker | Low Risk | Medium Risk | High Risk |
| :--- | ---: | ---: | ---: |
| AAPL | 4.6% | 0.0% | 0.0% |
| MSFT | 0.0% | 0.0% | 0.0% |
| NVDA | 18.4% | 44.1% | 71.0% |
| ORCL | 8.1% | 8.0% | 7.7% |
| JPM | 6.9% | 3.9% | 0.0% |
| V | 0.0% | 0.0% | 0.0% |
| BAC | 0.0% | 0.0% | 0.0% |
| JNJ | 62.1% | 44.0% | 21.3% |
| PFE | 0.0% | 0.0% | 0.0% |
| UNH | 0.0% | 0.0% | 0.0% |

### 8.2 Monte Carlo Results

The best feasible Dirichlet portfolios were close to the efficient frontier, but more concentrated than the convex solution. The results below use the primary Dirichlet sampler (Appendix B contains the alternative samplers).

| Risk Level | Target Risk | Achieved Risk | Expected Return |
| :--- | :--- | :--- | :--- |
| Low Risk | 20.00% | 19.93% | 23.64% |
| Medium Risk | 28.00% | 27.95% | 39.40% |
| High Risk | 40.00% | 39.99% | 55.70% |

The corresponding Monte Carlo weights were:

| Ticker | Low Risk | Medium Risk | High Risk |
| :--- | ---: | ---: | ---: |
| AAPL | 5.4% | 0.0% | 0.0% |
| MSFT | 1.8% | 0.0% | 0.0% |
| NVDA | 15.3% | 46.8% | 73.3% |
| ORCL | 10.5% | 0.0% | 0.0% |
| JPM | 0.8% | 0.0% | 0.0% |
| V | 8.2% | 0.0% | 0.0% |
| BAC | 1.6% | 0.0% | 0.0% |
| JNJ | 52.0% | 53.2% | 26.7% |
| PFE | 0.2% | 0.0% | 0.0% |
| UNH | 4.3% | 0.0% | 0.0% |

### 8.3 Direct Comparison

The final comparison between the two methods is shown below:

| Risk Level | Convex Return | Monte Carlo Return | Return Gap | Convex Risk | Monte Carlo Risk |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Low Risk | 25.05% | 23.64% | 1.41 pp | 20.00% | 19.93% |
| Medium Risk | 39.74% | 39.40% | 0.34 pp | 28.00% | 27.95% |
| High Risk | 55.82% | 55.70% | 0.12 pp | 40.00% | 39.99% |

## 9. Interpretation

The results follow a clear pattern.

- At low risk, both methods concentrate heavily in `JNJ`, which offers meaningful expected return at much lower standalone volatility than `NVDA`. The convex solution retains small allocations to `ORCL`, `JPM`, and `AAPL` at the low-risk budget; the Monte Carlo solution spreads across more names at low risk but collapses to a two-name `JNJ`-`NVDA` boundary at the medium and high risk levels.
- As the risk budget rises, both methods shift weight toward `NVDA`, whose much higher expected return justifies greater exposure once the volatility cap is relaxed.
- Convex optimization outperforms Monte Carlo at all three risk levels because it solves the problem exactly rather than sampling.

The return gap is largest at the low-risk level. Under tighter constraints, random search has greater difficulty locating the optimal boundary of the feasible region.

**On concentration:** the optimizer collapses primarily to `JNJ` and `NVDA` even in a 10-stock, 3-sector universe. This is a direct consequence of the 2020-2025 sample window: the period captures both `JNJ`'s defensive stability through COVID and `NVDA`'s exceptional return driven by the AI hardware cycle. Raw historical means over this window make `NVDA`'s estimated $\mu$ very high; without shrinkage or a prior, the optimizer exploits this signal aggressively. A Black-Litterman prior or a regularized covariance estimator would distribute weights more broadly.

## 10. Limitations of the Markowitz Framework

Even though the Markowitz model is very useful, it has some limitations.

- It depends heavily on estimated returns and covariances. The concentrated weights in this notebook are a direct consequence of the 2020-2025 sample period and would likely change substantially over a different window.
- Small changes in the input data can lead to large changes in weights.
- Variance treats upside and downside volatility in the same way.
- The model does not include many real-world frictions such as transaction costs or dynamic rebalancing.

Thus, Markowitz should be treated as a strong starting point for portfolio construction, not as a complete description of real-world investing.

## 11. Conclusion

The Markowitz problem provides a clear framework for studying diversification, risk, and return jointly. It offers a mathematical basis for selecting portfolio weights instead of relying only on intuition.

Both methods produced sensible portfolios across the three risk levels, but convex optimization outperformed Monte Carlo at every target risk. Monte Carlo nevertheless remained informative because it illustrated the feasible set and showed how closely random search can approach the efficient frontier.

## Appendix Pointers

The notebook ([markowitz.ipynb](./markowitz.ipynb)) contains the full analysis. The appendix sections are:

- **Appendix A** (cells 38-46): gradient descent and OSQP solver experiments, including a solver comparison table and a note on the OSQP risk-budget overshoot at the 40% target.
- **Appendix B** (cells 48-59): three alternative Monte Carlo samplers plain random, uniform simplex, and 3-asset concentrated Dirichlet.
- **Appendix C** (cells 61-67): additional comparison between ECOS and the 3-asset Dirichlet sampler, plus an efficient frontier visualization with GMVP and individual asset markers.
