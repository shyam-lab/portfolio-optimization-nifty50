# Assignment 4: QUBO Portfolio Selection

## Objective

This assignment formulates the problem of selecting exactly $K = 10$ assets from the NIFTY 50
universe as a Quadratic Unconstrained Binary Optimization (QUBO) problem. The cardinality
constraint is encoded as a penalty term, converting the constrained selection problem into an
unconstrained quadratic minimization over binary variables. Two solvers are compared: brute
force on a reduced subset and simulated annealing on the full universe.

## 1. Introduction

Classical portfolio optimization selects weights from a continuous weight space subject to
linear constraints. A different framing asks a discrete question: given $N$ stocks, which
$K$ should be held at all? This is a combinatorial problem. QUBO is one way to pose it
in a form that quantum-inspired solvers can handle.

In the return-only case, the QUBO objective maximizes expected return subject to a cardinality
constraint, with the constraint encoded as a quadratic penalty. Adding a risk term introduces
pairwise covariance interactions that naturally push the solver toward diversified selections.

## 2. Problem Formulation

### 2.1 Decision Variables and Objective

For each stock $i$, define a binary variable $x_i \in \{0, 1\}$ (1 = selected). The goal
is to maximize total expected return:

$$\max \sum_{i=1}^{N} \mu_i x_i \quad \text{s.t.} \quad \sum_{i=1}^{N} x_i = K$$

where $\mu_i$ is the annualized expected return of stock $i$.

### 2.2 Cardinality Constraint as Penalty

QUBO requires no explicit constraints. The constraint $\sum x_i = K$ is absorbed into the
objective via a quadratic penalty:

$$C(x) = -\sum_i \mu_i x_i + \lambda \left(\sum_i x_i - K\right)^2$$

The penalty is zero when exactly $K$ stocks are selected and positive otherwise. Choosing
$\lambda$ large enough relative to $\mu_i$ ensures the penalty dominates whenever cardinality
is violated.

### 2.3 QUBO Matrix Entries

Expand the penalty using $x_i^2 = x_i$ (binary idempotence):

$$\left(\sum_i x_i - K\right)^2 = (1 - 2K)\sum_i x_i + 2\sum_{i < j} x_i x_j + K^2$$

The constant $K^2$ does not affect optimization. Collecting terms:

$$C(x) = \sum_i Q_{ii} x_i + \sum_{i < j} Q_{ij} x_i x_j + \text{const}$$

Diagonal terms:

$$Q_{ii} = -\mu_i + \lambda(1 - 2K)$$

With $K = 10$: $Q_{ii} = -\mu_i - 19\lambda$.

Off-diagonal terms:

$$Q_{ij} = 2\lambda \quad \text{for } i < j$$

### 2.4 Why Quadratic Interactions?

The cardinality constraint $\sum x_i = K$ is a linear constraint. Squaring it to form the
penalty introduces every pairwise product $x_i x_j$. This is not a modeling choice but a
mathematical consequence: any quadratic penalty on a linear constraint is quadratic in the
variables. Each co-selected pair $(i, j)$ adds $2\lambda$ to the cost, which the solver
minimizes by selecting exactly $K$ stocks.

## 3. Data

Source: `data/processed/price_matrix_full.csv` (the shared repository processed dataset).

- 49 stocks, 2598 trading days, 2010-11-04 to 2021-04-30.
- Daily simple returns: $r_{i,t} = P_{i,t} / P_{i,t-1} - 1$.
- Annualized expected return: $\mu_i = 252 \cdot \bar{r}_i$.

## 4. Return Signal

Historical annualized mean returns from the full 2010-2021 window. This matches the
assignment specification (historical mean returns option). The QUBO formulation is signal-
agnostic: any return vector can be substituted into the diagonal of $Q$.

Lambda is set as $\lambda = 20 \cdot \text{mean}(|\mu_i|)$, placing the penalty at
20 times the average signal magnitude. This ensures cardinality is enforced while the
return signal still differentiates between stocks.

## 5. Solvers

### 5.1 Brute Force (Subset)

Exact enumeration over all $\binom{N}{K}$ combinations is infeasible for $N = 49$, $K = 10$
($\binom{49}{10} \approx 8.2$ billion). The brute-force solver is restricted to the first
$M = 20$ stocks, where $\binom{20}{10} = 184{,}756$ combinations are tractable. For each
combination, the QUBO cost is computed directly and the minimum is recorded.

### 5.2 Simulated Annealing with Swap Moves

For the full 49-stock problem, simulated annealing with swap moves is used. Each step
exchanges one selected stock for one unselected stock, keeping cardinality fixed at $K = 10$
throughout. This means the search space is restricted to feasible solutions; the penalty
term is present in $Q$ but only affects the relative cost landscape.

Parameters: 200,000 iterations, 20 restarts, $T_0 = 10$, $\alpha = 0.99997$ (geometric
cooling).

### 5.3 Lambda Sensitivity (Bit-Flip SA)

A second SA variant uses single bit flips (cardinality not preserved). With this solver,
the penalty is the only mechanism enforcing $\sum x_i = K$. Sweeping $\lambda$ across
25 log-spaced values from $10^{-3}$ to $10^{3}$ shows three regimes:

- **Too small:** penalty weak, solver selects more or fewer than $K = 10$ stocks.
- **Right range:** exactly 10 stocks selected, total return near the top $K$ optimum.
- **Too large:** still 10 stocks but return signal $\mu_i$ is drowned out; selection
  becomes nearly arbitrary.

## 6. Return + Risk Extension

Adding an annualized sample covariance term with weight $\gamma$:

$$C(x) = -\sum_i \mu_i x_i + \gamma \sum_{i,j} \Sigma_{ij} x_i x_j
         + \lambda\left(\sum_i x_i - K\right)^2$$

Using $x_i^2 = x_i$:

$$Q_{ii} = -\mu_i + \gamma \sigma_i^2 + \lambda(1 - 2K)$$

$$Q_{ij} = 2\gamma \Sigma_{ij} + 2\lambda \quad \text{for } i < j$$

High positive covariance between stocks $i$ and $j$ increases $Q_{ij}$, making their
co-selection more costly. The solver is pushed toward stock pairs with low or negative
pairwise correlations, producing diversification pressure through the QUBO structure.

Lambda is rescaled with $\gamma$ as $\lambda_\gamma = \max(\lambda, 10\gamma \cdot
\max|\Sigma_{ij}|)$ to maintain penalty dominance as the covariance term grows.

## 7. Results

### 7.1 Brute Force Validation

On the 20-stock subset, brute force selects the same stocks as a naive top $K$ sort by
$\mu_i$:

- Brute force: ADANIPORTS, ASIANPAINT, BAJAJFINSV, BAJFINANCE, BRITANNIA, DRREDDY,
  EICHERMOT, HCLTECH, HDFC, HINDALCO
- Naive top $K$ (first 20): identical
- QUBO cost: -260.0706

This validates the formulation. With a sufficiently large $\lambda$, the QUBO correctly
encodes the constrained problem: the penalty kills all infeasible solutions, leaving the
return-maximizing feasible solution as the global minimum.

### 7.2 Simulated Annealing (Full Universe)

SA on all 49 stocks recovers a near-optimal selection:

- SA selected: ADANIPORTS, BAJAJFINSV, BAJFINANCE, BRITANNIA, EICHERMOT, HINDUNILVR,
  INDUSINDBK, KOTAKBANK, SHREECEM, ULTRACEMCO
- Total return: 2.5876 (annualized)
- QUBO cost gap vs naive top-10: 0.0009

The SA-selected set differs from the 20-stock brute-force set because the full 49-stock
universe contains stocks (HINDUNILVR, INDUSINDBK, KOTAKBANK, SHREECEM, ULTRACEMCO)
with higher $\mu_i$ than some stocks in the first 20.

### 7.3 Lambda Sensitivity

| Lambda | Stocks selected | Total Return | Regime |
| --- | --- | --- | --- |
| 0.001 | 35 | 5.37 | Penalty too weak, infeasible |
| 0.01 | 19 | 3.67 | Still over-selecting |
| 0.10 | 11 | 2.40 | Near-feasible |
| 0.32 | 10 | 2.40 | Feasible, near top-K optimum |
| 1.00 | 10 | 1.64 | Feasible, return starting to degrade |
| 10.0 | 10 | 1.57 | Signal partially drowned |
| 1000 | 10 | 1.76 | Signal fully drowned, arbitrary selection |

The working range for this problem is $\lambda \approx 0.3$ to $1$. The chosen value
$\lambda = 20 \cdot \text{mean}(|\mu_i|)$ falls in this range.

### 7.4 Risk-Aversion Sweep

| Gamma | Ann. Return | Ann. Variance | Ann. Vol | Sharpe | Num Sectors |
| --- | --- | --- | --- | --- | --- |
| 0 | 25.45% | 3.92% | 19.81% | 1.285 | 6 |
| 0.5 | 23.76% | 2.72% | 16.49% | 1.441 | 6 |
| 1 | 20.30% | 2.21% | 14.87% | 1.365 | 7 |
| 2 | 18.97% | 2.09% | 14.45% | 1.313 | 6 |
| 5 | 16.49% | 1.98% | 14.06% | 1.173 | 6 |
| 10 | 13.16% | 1.94% | 13.92% | 0.945 | 5 |

At $\gamma = 0$ (return only), the portfolio concentrates in 6 sectors. At $\gamma = 0.5$,
the Sharpe peaks at 1.441 as the risk penalty diversifies into lower-covariance stocks without
large return sacrifice. Beyond $\gamma = 5$ the selection stabilizes: further variance
reduction is marginal and the return penalty grows large.

## 8. Reflection

### 8.1 Lambda Too Small

The penalty $\lambda(\sum x_i - K)^2$ is too weak. The return benefit of adding an extra
stock outweighs the penalty cost of violating the constraint. The solver may select more
or fewer than $K = 10$ stocks. The QUBO solution is infeasible with respect to the original
constrained problem.

### 8.2 Lambda Too Large

The penalty terms $-19\lambda$ (diagonal) and $2\lambda$ (off-diagonal) dominate $Q$. The
return signal $\mu_i$ contributes negligibly: all diagonal entries look almost equal and all
off-diagonal entries look almost equal. The solver picks exactly $K$ stocks but is nearly
indifferent to which ones. Raising $\lambda$ beyond the working range does not improve
feasibility but degrades solution quality.

### 8.3 Why Quadratic Interactions?

The cardinality constraint $\sum x_i = K$ is linear. Encoding it as a penalty requires
squaring the residual $(\sum x_i - K)^2$, which produces every cross-term $x_i x_j$. This
is a mathematical consequence of using a squared penalty on a linear constraint. There is
no way to encode a cardinality constraint as a linear (QUBO-incompatible) function of
binary variables while keeping the problem unconstrained.

### 8.4 Risk Extension

The covariance matrix $\Sigma$ adds pairwise interactions with economic content. A high
positive covariance between stocks $i$ and $j$ increases $Q_{ij}$, penalizing their joint
selection. The solver seeks stock pairs with low or negative $\Sigma_{ij}$, which corresponds
to seeking diversified combinations. This is the QUBO equivalent of the mean-variance
risk term $w^\top \Sigma w$ in classical Markowitz.

## 9. Limitations

- Expected returns and covariance are estimated from the full historical window (2010-2021),
  with no train/test split. Results cannot be interpreted as out-of-sample.
- Equal weights within the selected portfolio are assumed. No within-portfolio weight
  optimization is performed.
- Simulated annealing is a heuristic; optimality is not guaranteed. Results may vary across
  runs (mitigated by 20 restarts).
- The sector map is manually defined and may not match current index classification.

## 10. Conclusion

QUBO provides a clean framework for discrete portfolio selection. The cardinality constraint
is absorbed into the objective via a squared penalty, converting a constrained combinatorial
problem into unconstrained quadratic minimization. Both brute force and simulated annealing
correctly recover the top $K$ selection under the return-only objective, validating the
formulation.

Adding a covariance penalty ($\gamma > 0$) introduces diversification pressure through the
off-diagonal structure of $Q$, pushing the solver toward lower-variance sector-diversified
portfolios at the cost of expected return. The best Sharpe in this study is achieved at
$\gamma = 0.5$ (Sharpe 1.441, 6 sectors), not at the pure return-maximizing $\gamma = 0$.

The central lesson is that the QUBO structure is modular: any return signal can be substituted
into the diagonal, and any pairwise cost (covariance, correlation penalty, sector
concentration) can be added to the off-diagonal. The cardinality enforcement mechanism is
unchanged across all variants.
