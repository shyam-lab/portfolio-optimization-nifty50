# Problem Statement

Portfolio optimization lies at the intersection of finance, statistics, and optimization.

The classical Markowitz framework provides a mathematically elegant solution to balancing
risk and return. However, real-world portfolio construction introduces several challenges:

- Noisy and unstable return estimates
- Practical constraints (long-only, cardinality, weight caps)
- High sensitivity to small input changes
- Limited ability to handle discrete decisions

This project explores the following core question:

## Core Question

Can modern approaches (Machine Learning and Quantum-Inspired Optimization)
address the limitations of classical portfolio optimization?

## Objectives

1. Establish a classical baseline using mean-variance optimization
2. Improve return estimation using machine learning
3. Reformulate the problem into a QUBO framework for combinatorial optimization
4. Benchmark all approaches under consistent constraints

## Scope

- Dataset: NIFTY 50 equities
- Constraints: long-only, full investment, optional cardinality constraints
- Evaluation: return, volatility, Sharpe ratio, diversification

## Key Hypothesis

While classical optimization provides a strong foundation,
ML and QUBO methods may offer advantages under:
- noisy data conditions
- discrete selection constraints
- complex optimization landscapes