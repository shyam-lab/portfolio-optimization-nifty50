# Methodology Overview

This project follows a layered approach to portfolio optimization:

## 1. Classical Baseline

We begin with the Markowitz mean-variance framework:

- Expected return: μᵀw
- Risk: wᵀΣw

Optimization is performed under:
- full investment constraint
- long-only weights
- target risk levels

This establishes the efficient frontier and baseline performance.

---

## 2. Machine Learning Enhancements

Machine learning is used to improve return estimation.

Approach:
- Predict forward returns (15-day horizon)
- Use lagged returns, momentum, and volatility features
- Train models on historical data (time-split)

Key insight:
ML improves signal stability but does not eliminate noise or structural limitations.

---

## 3. QUBO Formulation

Portfolio selection is reformulated as a Quadratic Unconstrained Binary Optimization problem.

Key elements:
- Binary variables for asset selection
- Cardinality constraint enforced via penalty term
- Objective combines return maximization and constraint satisfaction

Solvers:
- Brute force (small scale)
- Simulated annealing (large scale)

---

## 4. Benchmarking

All approaches are evaluated using:
- Expected return
- Volatility
- Sharpe ratio
- Portfolio concentration

Comparisons are made across:
- Classical vs ML-enhanced
- Continuous vs discrete optimization
- Exact vs approximate solutions