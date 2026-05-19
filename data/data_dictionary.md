# Data Dictionary

## Overview
This project processes historical NIFTY 50 stock data for portfolio optimization and risk analysis.

The pipeline includes:
- Data cleansing and consolidation
- Price matrix construction
- Missing value handling
- Creation of two analysis universes
- Return and covariance estimation

---

# 1. Data Ingestion and Consolidation

The dataset consists of individual CSV files, where each file represents one stock.

## Processing Steps
- Load all stock-wise CSV files
- Remove completely empty columns
- Add a `Symbol` column using the file name
- Concatenate all stock datasets into a single dataframe

## Output Dataset: `raw_data`

| Column | Type | Description |
|---|---|---|
| `Date` | object → datetime | Trading date |
| `Open` | float | Opening price |
| `High` | float | Highest price |
| `Low` | float | Lowest price |
| `Close` | float | Closing price |
| `Volume` | float/int | Trading volume |
| `Symbol` | string | Stock ticker |

---

# 2. Date Cleaning and Sorting

## Processing Steps
- Convert `Date` to datetime format
- Remove invalid or missing dates
- Sort data by `Symbol` and `Date`
- Retain only:
  - `Date`
  - `Symbol`
  - `Close`

---

# 3. Duplicate Handling

Some stocks contain multiple entries for the same trading date.

## Resolution
Data is grouped by:
- `Date`
- `Symbol`

The last available closing price is retained.

```python
price_data.groupby(["Date", "Symbol"]).last()
```

---

# 4. Price Matrix Construction

The cleaned dataset is reshaped into a pivoted price matrix.

```python
price_matrix = price_data.pivot(
    index="Date",
    columns="Symbol",
    values="Close"
)
```

## Matrix Structure

| Component | Description |
|---|---|
| Rows | Trading dates |
| Columns | Stock symbols |
| Values | Closing prices |

## Matrix Shape

```python
(number_of_days, number_of_stocks)
```

Example:

```python
(5306, 50)
```

Meaning:
- 5306 trading days
- 50 stocks

---

# 5. Missing Data Analysis

Different stock listing dates create missing values in the price matrix.

## Diagnostic Summary

| Metric | Value |
|---|---|
| Original trading days | 5306 |
| Days after dropping all NaNs | 2598 |
| Data loss | ~51% |

A small number of recently listed stocks caused most missing observations.

---

# 6. Analysis Universes

To balance stock coverage and historical depth, two universes were created.

---

## Approach 1: Full Universe (`price_matrix_full`)

### Description
- Retains all available stocks
- Drops rows containing any missing values

### Purpose
Maintains maximum stock coverage.

### Characteristics

| Property | Value |
|---|---|
| Stocks | 50 |
| Historical Window | Shorter |
| Missing Values | None |

### Matrix Shape

```python
(2598, 50)
```

### Timeline
Starts only when all stocks have valid data.

---

## Approach 2: Extended History (`price_matrix_extended`)

### Description
Stocks with excessive missing values are removed.

Criteria:
```python
missing_days > 1100
```

After removing these stocks:
- Remaining rows with NaNs are dropped

### Purpose
Extends the historical analysis period.

### Characteristics

| Property | Value |
|---|---|
| Stocks | 38 |
| Historical Window | Longer |
| Missing Values | None |

### Matrix Shape

```python
(longer_history, 38)
```

### Advantage
Provides more historical observations for return and covariance estimation.

---

# 7. Return and Risk Estimation

## Daily Returns

```python
daily_ret = matrix.pct_change().dropna()
```

| Variable | Description |
|---|---|
| `daily_ret` | Daily percentage returns |

---

## Expected Returns

Annualized mean returns:

```python
mu = daily_ret.mean() * 252
```

| Variable | Description |
|---|---|
| `mu` | Annualized expected returns |

### Shape

```python
(number_of_stocks,)
```

---

## Covariance Matrix

Annualized covariance matrix:

```python
Sigma = daily_ret.cov() * 252
```

| Variable | Description |
|---|---|
| `Sigma` | Annualized covariance matrix |

### Shape

```python
(number_of_stocks, number_of_stocks)
```

Example:

```python
(50, 50)
```

---

# 8. Optimization Variables

## Portfolio Weights (`w`)

Represents allocation weight for each stock.

### Constraints

```python
sum(w) = 1
w >= 0
```

| Constraint | Meaning |
|---|---|
| Fully Invested | All capital allocated |
| Long-only | No short selling |

---

# 9. Optimization Objective

Minimum variance portfolio optimization:

```python
Minimize:
w.T @ Sigma @ w
```

## Solver
- CVXPY
- Clarabel

---

# 10. Data Quality Rules

| Rule | Action |
|---|---|
| Empty columns | Removed |
| Invalid dates | Dropped |
| Duplicate records | Deduplicated |
| Missing values | Handled before analysis |
| Chronological order | Enforced |
