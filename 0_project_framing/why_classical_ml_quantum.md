# Why Classical + ML + Quantum?

Portfolio optimization is not a single-method problem.

Each approach addresses a different limitation of the system.

---

## Classical Optimization (Markowitz)

Strengths:
- Convex and efficiently solvable
- Clear economic interpretation
- Strong theoretical foundation

Limitations:
- Highly sensitive to input estimates
- Struggles with discrete constraints
- Assumes continuous weights

---

## Machine Learning

Role:
- Improve estimation of expected returns

Strengths:
- Captures patterns in historical data
- Reduces noise at appropriate horizons

Limitations:
- Weak predictive power in financial markets
- Does not fundamentally change optimization structure

---

## Quantum / QUBO Approach

Role:
- Handle combinatorial optimization problems

Strengths:
- Naturally encodes discrete decisions
- Suitable for constraints like "select k assets"
- Scales via approximate solvers

Limitations:
- Approximation gap vs optimal
- Requires tuning (e.g., penalty parameters)
- Still emerging in practice

---

## Key Insight

These approaches are not competing—they are complementary.

- Classical methods define the structure
- ML improves inputs
- QUBO enables new constraint handling

---

## Project Thesis

The future of portfolio optimization is likely hybrid:

Classical + ML + Quantum-inspired methods

Each layer contributes to solving a different part of the problem.