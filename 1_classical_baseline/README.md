# Classical Baseline: Markowitz Mean-Variance Portfolios

Long-only Markowitz mean-variance optimisation on Indian equities. Four
notebooks: closed-form theory and active-set enumeration on a 20-asset
selection, numerical solver comparison on the same selection, Monte
Carlo approximation of the long-only frontier, and a walk-forward
application on the 49-asset NIFTY universe with an at-most-K cardinality
study.

## Aim

Build a self-contained classical baseline against which any ML or
discrete-optimisation method can be compared. The mean-variance problem,
its solvers, and its known failure modes (estimation noise, cardinality,
turnover) are studied in isolation so that downstream comparisons rest
on transparent assumptions.

## Folder contents

```
1_classical_baseline/
  README.md
  notebooks/
    01_theory.ipynb               # closed-form + active-set enumeration
    02_solvers.ipynb              # OSQP / CLARABEL / SCS / ECOS comparison
    03_monte_carlo.ipynb          # simplex samplers vs the long-only frontier
    04_nifty_application.ipynb    # walk-forward + at-most-K MIQP on NIFTY
  results/                        # CSVs and PNGs produced by the notebooks
```

## Data sources

Two data paths, both Indian equities.

**20-asset selection (notebooks 1–3).** Downloaded live from Yahoo
Finance with `auto_adjust=True` over a fixed window (2019-01-01 to
2024-12-31). Tickers are NSE-listed (`.NS` suffix), grouped roughly by
sector:

- Information technology: TCS, INFY, WIPRO
- Banking and financials: HDFCBANK, ICICIBANK, KOTAKBANK
- Energy: RELIANCE, ONGC, BPCL
- Consumer staples: HINDUNILVR, ITC, NESTLEIND
- Industrials and autos: MARUTI, LT
- Pharma: SUNPHARMA, DRREDDY
- Metals: TATASTEEL, JSWSTEEL
- Telecom: BHARTIARTL, INDUSTOWER

The calendar window is fixed in code, but Yahoo Finance can revise
historical data, so numerical results may shift slightly across
download dates.

**49-asset NIFTY universe (notebook 4).** Read from
`data/processed/price_matrix_full.csv` (produced by
`data/Data_Pipeline.ipynb`). Daily closes, 2010-11-04 to 2021-04-30,
2,598 rows. The index column `NIFTY50_all` is dropped before modelling.
Twenty-six observations with `|r| > 0.5` are unadjusted corporate
actions (splits, bonus issues for TITAN, JSWSTEEL, ASIANPAINT, and
others) and are set to NaN before return estimation.

## Method summary

The same long-only mean-variance problem is studied in four forms across the four notebooks.

**Unconstrained mean-variance (01).** Closed form for `min ½ wᵀΣw` subject to `1ᵀw = 1` and `μᵀw = μ*`. The two-fund identity gives `w(μ*) = Σ⁻¹(α 1 + β μ)` for scalars built from `Σ⁻¹`. Reference frontier; can have short positions.

**Long-only mean-variance (01–02).** Add `w ≥ 0`. Traced over a γ grid three ways in 01 (exact active-set enumeration over `2ᴺ − 1` non-empty subsets, iterative active-set, CLARABEL via CVXPY) and across four CVXPY backends in 02 plus the iterative active-set method (OSQP, CLARABEL, SCS, ECOS, active-set). Enumeration is a correctness benchmark, not a scalable method.

**Walk-forward long-only with position cap (04).** Add `wᵢ ≤ 0.20`. Run on the 49-asset NIFTY universe across 88 monthly rebalance windows. Includes target-risk SOCP variant `max μᵀw s.t. ‖Lᵀw‖₂ ≤ σ*`, `Lᵀ = chol(Σ)`, with `σ*` per-window matched to the capped-MVO solution.

**At-most-K MIQP (04).** Add binaries `yᵢ ∈ {0,1}`, `wᵢ ≤ yᵢ · u`, `Σ yᵢ ≤ K`. Mixed-integer QP solved via CVXPY → SCIP. With cap `u = 0.20`, the budget forces `K · u ≥ 1`, so `K ≥ 5`. Compared against a greedy μ/σ screen heuristic.

## Notebook summaries

**`01_theory.ipynb`.** Markowitz problem statement on `N` assets and its closed-form analytical solution (two-fund identity, `A`/`B`/`C`/`D` scalars, parabolic frontier), then the two-asset specialisation with a controlled `ρ` sweep. Long-only frontier traced three ways on a 200-point γ grid: brute-force enumeration over the `2²⁰ − 1` non-empty subsets (~1M subsets in ~50 s), iterative active-set (~0.014 s), and CLARABEL via CVXPY (~0.3 s). The three agree to four decimals on `(μ_p, σ_p)` at every γ; active-set converges (KKT) at every grid point. Per-asset scatter coloured by sector.

**`02_solvers.ipynb`.** Long-only utility QP `min ½wᵀΣw − γμᵀw` run through five solvers: OSQP, CLARABEL, SCS, ECOS, and an iterative active-set method. Each backend's internal canonical form and per-iteration update equations are derived in Markowitz notation. Per-solver demo cells print primal-dual triples; comparison table reports return, volatility, top-5 holdings, solve time at γ=1. Efficient frontiers swept across the full γ grid plotted with per-asset sector scatter.

**`03_monte_carlo.ipynb`.** Monte Carlo is a sampler, not a solver. Five simplex samplers — Dirichlet(α=1) (uniform on `Δ_N`), Dirichlet(α=0.3) (corner-biased), sparse-face at k ∈ {2,3,5,8}, tilted Dirichlet (Sharpe-weighted α vector using `μ̂/σ̂`), and a mixed sampler (20% D₁ + 20% D₀.₃ + 30% sparse + 30% tilted) — compared against the iterative active-set long-only frontier swept inline across γ ∈ [10⁻³, 10²]. Comparison table includes draw wall time plus the active-set solver as the speed reference. Convergence study at M ∈ {1k, 5k, 10k, 50k, 100k} with nested prefixes for monotonicity. Seed `20260523`.

**`04_nifty_application.ipynb`.** Walk-forward application on the 49-asset NIFTY universe with strict no-lookahead protocol (EWMA `μ̂` halflife 252, sample `Σ̂`, 504-day live-observation filter). Four methods (equal-weight, min-variance, capped MVO, target-risk SOCP) evaluated over 88 monthly windows (2014–2021); turnover, max drawdown, mean per-window solve time reported. Cardinality study via at-most-K MIQP at K ∈ {5, 7, 10, 15, 20} compared against a greedy μ/σ screen; walk-forward MIQP vs greedy at K=10. Single combined plot at the end: $1 growth from test start + underwater drawdown for all six strategies in a light pastel palette. Final summary table aggregates all six walk-forward strategies (4 baseline + 2 cardinality) into one row set.

## Outputs

All artefacts go under `results/` as CSVs and PNGs.

| File                                        | Source notebook | Content                                                                                                          |
| ------------------------------------------- | --------------- | ---------------------------------------------------------------------------------------------------------------- |
| `theory_two_asset_frontier.png`             | 01              | Two-asset long-only frontier (RELIANCE × TCS)                                                                    |
| `theory_correlation_controlled.csv`         | 01              | Min-var vol per ρ in controlled experiment (σ₁=σ₂=0.25)                                                          |
| `theory_correlation_controlled.png`         | 01              | Frontier fan for ρ ∈ {−0.8, −0.4, 0, 0.4, 0.8}                                                                   |
| `theory_correlation_sweep.png`              | 01              | Frontier overlay across 4 real partners by ρ ranking                                                             |
| `theory_analytic_frontier.csv`              | 01              | Unconstrained analytic frontier (20-asset, target-return grid)                                                   |
| `theory_analytic_frontier.png`              | 01              | Unconstrained frontier plot with single-asset scatter                                                            |
| `theory_brute_force_frontier.csv`           | 01              | Long-only frontier over 200 γ — brute force, active-set, CLARABEL                                                |
| `theory_brute_force_timing.csv`             | 01              | Wall-clock totals per method (200-point γ sweep)                                                                 |
| `theory_brute_force_scaling.csv`            | 01              | Brute-force timing vs asset count `n`                                                                            |
| `theory_brute_force_scaling.png`            | 01              | Seconds vs N (log-y)                                                                                             |
| `theory_brute_force_scaling_by_subsets.png` | 01              | Seconds vs 2ⁿ−1 subset count (log-log)                                                                           |
| `theory_active_set_vs_cvxpy.csv`            | 01              | Active-set vs CLARABEL at seven reference γ                                                                      |
| `theory_three_frontier_overlay.png`         | 01              | Three-method long-only frontier overlay vs analytical                                                            |
| `theory_gamma_sigma_map.png`                | 01              | γ ↔ σ mapping for the three long-only methods                                                                    |
| `solver_comparison.csv`                     | 02              | Five-solver row table: return, vol, top-5 holdings, solve time at γ=1                                            |
| `solver_capability_table.csv`               | 02              | Solver × problem-class coverage                                                                                  |
| `frontier_overlay.csv`                      | 02              | Per-γ rows for each of the five solvers (method, gamma, return, volatility)                                      |
| `frontier_overlay.png`                      | 02              | Five efficient frontiers + sector-coloured per-asset scatter                                                     |
| `monte_carlo_cloud.csv`                     | 03              | Per-sampler downsampled cloud (5k rows × 5 samplers)                                                             |
| `monte_carlo_sampler_comparison.csv`        | 03              | Five samplers + active-set solver baseline: Sharpe, gap, draw time at M=100k                                     |
| `monte_carlo_target_risk_comparison.csv`    | 03              | Return gap vs frontier at 8 target volatilities                                                                  |
| `monte_carlo_convergence.csv`               | 03              | Best Sharpe vs M (nested prefix, monotone) for Dirichlet α=1, tilted, mixed                                      |
| `monte_carlo_dirichlet_alpha1_cloud.png`    | 03              | Dirichlet α=1 cloud (uniform on simplex)                                                                         |
| `monte_carlo_concentrated_cloud.png`        | 03              | Dirichlet α=0.3 cloud (corner-biased)                                                                            |
| `monte_carlo_sparse_face_clouds.png`        | 03              | Sparse face 2×2 panel, k ∈ {2, 3, 5, 8}                                                                          |
| `monte_carlo_tilted_cloud.png`              | 03              | Tilted Dirichlet (Sharpe-weighted α) cloud                                                                       |
| `monte_carlo_mixed_cloud.png`               | 03              | Mixed sampler cloud                                                                                              |
| `monte_carlo_sampler_comparison.png`        | 03              | Best Sharpe bar chart vs frontier reference                                                                      |
| `monte_carlo_convergence.png`               | 03              | Convergence curves: best Sharpe + gap to frontier vs M                                                           |
| `oos_walkforward.csv`                       | 04              | Four-method walk-forward OOS: return, vol, Sharpe, turnover, drawdown, mean solve time, cardinality (88 windows) |
| `miqp_comparison.csv`                       | 04              | In-sample MIQP vs continuous relaxation vs greedy (K=10)                                                         |
| `miqp_K_sweep.csv`                          | 04              | In-sample at-most-K sweep, K ∈ {5, 7, 10, 15, 20}                                                                |
| `miqp_walkforward.csv`                      | 04              | At-most-K MIQP vs greedy at K=10 (OOS walk-forward + cardinality + max weight)                                   |
| `miqp_walkforward_solvetime.png`            | 04              | Mean solve time per window: MIQP vs greedy                                                                       |
| `drawdown_paths.png`                        | 04              | Combined $1-growth + underwater drawdown for all six walk-forward strategies                                     |
| `summary_table.csv`                         | 04              | Consolidated 6-row table: 4 baseline + 2 cardinality, all OOS metrics                                            |

## Known limitations

- **Estimation noise.** Sample means and covariances are noisy
  estimators. The Monte Carlo study isolates random-search
  approximation quality for fixed inputs; estimation error in those
  inputs is a separate question and is not addressed here.
- **At-most-K, not exact-K.** The MIQP uses `Σ yᵢ ≤ K`. With the 20%
  per-asset cap, the unconstrained solution already concentrates on a
  small number of names (~6 at γ=5 on this universe), so the
  cardinality constraint rarely binds at K ≥ 7. Exact-K would only
  differ if a minimum-position-size constraint were also added.
- **MIQP overfits relative to greedy.** Notebook 4 shows MIQP K=10
  wins in-sample by ~10⁻⁴ Sharpe but loses out-of-sample to the
  greedy μ/σ screen (1.194 vs 1.236). The integer optimiser exploits
  noise in the EWMA `μ̂` estimate; the greedy screen is coarser but
  more robust. Shrinkage estimators may narrow the gap.
- **Corporate-action winsorisation.** `price_matrix_full.csv` is
  sourced from unadjusted NSE prices. Returns with `|r| > 0.5` are set
  to NaN at the affected observation. Surrounding returns are
  untouched. Adjusting at source in the data pipeline is the cleaner
  fix.
- **Frictionless walk-forward.** Notebook 4 reports gross-of-friction
  Sharpe and drawdown. Turnover is reported as a diagnostic, not
  penalised in the objective.
- **Survivorship-adjacent universe.** The 49-asset list is inherited
  from the processed pipeline. Stocks delisted between 2010 and 2021
  are absent, so OOS results carry mild survivorship bias.
- **SOCP redundant by construction in notebook 4.** The target-risk
  SOCP's `σ*` is set per window to the capped-MVO volatility, so by
  Lagrangian duality the two land on the same portfolio. SOCP's value
  would emerge with an independent risk target (e.g. fund mandate).
- **No shorting.** All numerical formulations enforce `w ≥ 0`,
  `1ᵀw = 1`. The unconstrained analytical frontier in notebook 1 may
  admit short positions; it is a theoretical reference.
