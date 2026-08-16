# QuantPortfolioMCP

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.11+-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)


A quantitative portfolio optimization MCP server that exposes portfolio construction, covariance estimation, risk attribution, and statistical analysis tools to AI assistants through the Model Context Protocol (MCP).

The core idea is to keep numerical computation deterministic and executable in Python while allowing an AI assistant to orchestrate the analytical workflow and interpret the results. MCP tools are designed to be callable by an LLM rather than requiring the model to perform the underlying financial mathematics itself.

---

## Current Architecture

```text
AI Assistant / MCP Client
          │
          ▼
   QuantPortfolioMCP
          │
          ├── Market Data
          │      └── yfinance
          │
          ├── Statistical Analysis
          │      └── scipy.stats / NumPy
          │
          ├── Covariance Engine
          │      ├── Sample
          │      ├── EWMA
          │      └── Ledoit-Wolf
          │
          ├── Portfolio Optimization
          │      ├── Mean-Variance
          │      ├── Black-Litterman
          │      ├── HRP
          │      ├── CVaR
          │      └── L1/L2 Regularization
          │
          └── Risk Attribution
                 ├── Portfolio Volatility
                 ├── Marginal Risk Contribution
                 ├── Component Risk Contribution
                 └── Diversification Ratio
```

---

# Current Features

## 1. Market Data Ingestion

`get_market_data()`

Fetches historical market data and converts it into quantitative inputs required by the portfolio engine.

**Current flow:**

```text
Ticker List
    ↓
Historical Prices
    ↓
Daily Returns
    ↓
Annualized Mean Returns
    ↓
Covariance Matrix
    ↓
Correlation Matrix
```

Supports ticker formats for:

* NSE — `.NS`
* BSE — `.BO`
* US equities — standard tickers

The tool returns the underlying daily returns together with the calculated statistics.

---

## 2. Covariance Estimation

`estimate_covariance_matrix()`

Three covariance estimators are currently available:

### Sample Covariance

Traditional historical covariance estimator.

### EWMA

Exponentially weights recent observations more heavily.

### Ledoit-Wolf Shrinkage

Shrinks the empirical covariance matrix toward a structured target to improve stability when the matrix is noisy or poorly conditioned.

```text
Historical Returns
       │
       ├── Sample
       ├── EWMA
       └── Ledoit-Wolf
              ↓
       Covariance Matrix
```

---

# 3. Portfolio Optimization

The MCP currently exposes multiple portfolio construction methodologies.

## Mean-Variance Optimization

`optimize_mean_variance()`

Supports:

* Minimum volatility
* Maximum Sharpe ratio
* Target-return efficient portfolio
* Long-only constraints
* Custom portfolio bounds

```text
Expected Returns + Covariance
              ↓
       Optimization
              ↓
       Portfolio Weights
```

---

## Black-Litterman

`optimize_black_litterman()`

Combines:

```text
Market Equilibrium
       +
Investor Views
       +
View Confidence
       ↓
Posterior Expected Returns
       ↓
Optimal Portfolio
```

The output also reports:

* Equilibrium returns
* Posterior returns
* Market weights
* Optimized weights
* Active bets

---

## Hierarchical Risk Parity

`optimize_hierarchical_risk_parity()`

Uses correlation-based hierarchical clustering to construct a portfolio without directly relying on covariance-matrix inversion.

Current implementation includes:

* Correlation-based distance
* Hierarchical clustering
* Quasi-diagonalization
* Cluster-based allocation
* Risk contribution analysis

---

## CVaR / Expected Shortfall Optimization

`optimize_cvar()`

Uses a linear-programming formulation of Conditional Value at Risk to optimize the portfolio against downside/tail losses.

Current functionality includes:

* Configurable confidence level
* Portfolio bounds
* Historical return scenarios
* Expected Shortfall minimization

---

## Regularized Optimization

`optimize_regularized()`

Adds L1 and L2 penalties to portfolio optimization.

This is intended to reduce unstable or extreme portfolio allocations.

```text
Expected Return
      +
Risk
      +
L1 Penalty
      +
L2 Penalty
      ↓
Regularized Portfolio
```

---

# 4. Portfolio Risk Attribution

`calculate_portfolio_attribution()`

Decomposes portfolio risk into:

* Total portfolio volatility
* Marginal Risk Contribution (MRC)
* Component Risk Contribution (CRC)
* Diversification Ratio

This allows the portfolio to be examined not only by **return**, but by **where the risk is actually coming from**.

```text
Portfolio
    │
    ├── Asset A → Risk Contribution
    ├── Asset B → Risk Contribution
    ├── Asset C → Risk Contribution
    └── Asset D → Risk Contribution
```

---

# 5. Statistical Analysis MCP

The project also contains a statistical-analysis MCP containing tools for quantitative diagnostics.

### Descriptive Statistics

* Mean
* Median
* Variance
* Standard deviation
* Range
* Quartiles
* IQR
* Skewness
* Kurtosis
* Coefficient of variation
* Percentiles
* Quantiles
* Frequency tables

### Statistical Tests

* One-sample t-test
* ANOVA
* Chi-square
* Mann-Whitney U
* Wilcoxon
* Binomial test
* Shapiro-Wilk normality test

### Dependence Analysis

* Pearson correlation
* Spearman correlation
* Kendall's Tau
* Covariance
* Linear regression

### Statistical Utilities

* Z-scores
* Moving averages
* Bootstrap confidence intervals
* Trimmed mean
* Outlier detection
* Geometric mean
* Harmonic mean

This provides a foundation for adding statistical validation to portfolio research rather than relying exclusively on optimization outputs.

---

# End-to-End Current Workflow

A typical portfolio analysis can currently follow this sequence:

```text
1. Select Assets
       ↓
2. Fetch Historical Market Data
       ↓
3. Calculate Returns
       ↓
4. Examine Statistical Properties
       ↓
5. Estimate Covariance
       ↓
6. Select Portfolio Construction Method
       │
       ├── Mean-Variance
       ├── Black-Litterman
       ├── HRP
       ├── CVaR
       └── Regularized
       ↓
7. Generate Portfolio Weights
       ↓
8. Calculate Risk Attribution
       ↓
9. Interpret Portfolio Characteristics
```

The LLM acts primarily as the **orchestration and interpretation layer**, while NumPy/SciPy/Pandas perform the underlying numerical calculations.

---

# Future Scope

The current project is intentionally structured so additional quantitative research layers can be added on top of the existing optimization engine.

## Phase 1 — Portfolio Diagnostics

Add a comprehensive portfolio analytics layer:

* Sharpe Ratio
* Sortino Ratio
* Calmar Ratio
* Maximum Drawdown
* VaR
* CVaR
* Beta
* Tracking Error
* Information Ratio
* Downside Deviation
* Turnover
* Gross Exposure
* Net Exposure

---

## Phase 2 — Robust Portfolio Construction

Expand beyond the current estimators:

* Robust covariance estimation
* Oracle Approximating Shrinkage
* Minimum Correlation portfolios
* Risk Parity
* Equal Risk Contribution
* Maximum Diversification
* Entropy-based portfolios
* Factor-constrained optimization

---

## Phase 3 — Factor Risk Engine

Introduce systematic factor analysis:

```text
Portfolio
    ↓
Factor Exposure
    ├── Market
    ├── Size
    ├── Value
    ├── Momentum
    ├── Quality
    └── Sector
```

Future versions can calculate factor betas, factor contribution to risk and factor-neutral portfolios.

---

## Phase 4 — Stress Testing & Scenario Analysis

Introduce historical and hypothetical stress scenarios.

Examples:

* Global Financial Crisis
* COVID crash
* Interest-rate shocks
* Inflation shocks
* Equity-market crashes
* Sector-specific shocks
* Currency shocks
* Custom user-defined scenarios

```text
Portfolio
    ↓
Scenario Engine
    ↓
Shocked Returns
    ↓
Portfolio P&L
    ↓
Risk Attribution
```

---

## Phase 5 — Backtesting Engine

Add a complete walk-forward portfolio backtesting framework.

```text
Historical Data
      ↓
Training Window
      ↓
Portfolio Optimization
      ↓
Out-of-Sample Period
      ↓
Rebalance
      ↓
Repeat
      ↓
Performance Analysis
```

Metrics would include:

* CAGR
* Sharpe
* Sortino
* Maximum Drawdown
* Calmar
* Win Rate
* Turnover
* Transaction Costs
* Tail Loss
* Risk-adjusted performance

This would allow optimization methods to be evaluated based on **out-of-sample performance rather than in-sample portfolio statistics alone**.

---

## Phase 6 — Robustness & Statistical Validation

Automatically test whether portfolio conclusions survive changes in assumptions.

Examples:

```text
Lookback Period
     ×
Covariance Model
     ×
Rebalance Frequency
     ×
Transaction Cost
     ×
Optimization Method
```

The system can then determine whether a portfolio allocation is:

* Stable
* Parameter-sensitive
* Regime-dependent
* Statistically significant
* Potentially overfit

---

## Phase 7 — Regime Detection

Add market-regime analysis using:

* Volatility regimes
* Correlation regimes
* Bull/bear regimes
* Hidden Markov Models
* Clustering
* Regime-dependent covariance matrices

This would allow portfolio construction to adapt to changing market conditions.

---

## Phase 8 — Advanced Data Sources

Expand market-data capabilities beyond basic historical equity prices.

Potential integrations:

* NSE/BSE data
* ETFs
* Bonds
* Government securities
* FX
* Commodities
* Crypto
* Central-bank data
* Macroeconomic indicators
* Interest-rate curves
* Volatility indices
* Options data

The goal is to evolve from an **equity portfolio optimizer** into a broader **multi-asset quantitative research engine**.

---

# Long-Term Architecture

The eventual system can evolve toward:

```text
                     AI / LLM
                         │
                         ▼
                 MCP Orchestration
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
      Market Data    Macro Data     Alternative Data
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                Research & Statistics
                         │
                         ▼
                 Risk / Factor Engine
                         │
                         ▼
                Portfolio Optimizers
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
       Backtest      Stress Test    Robustness
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                 Portfolio Decision
                         │
                         ▼
                Risk Attribution
                         │
                         ▼
                 AI Interpretation
```

The long-term objective is therefore not simply to provide an optimizer, but to build an **agent-accessible quantitative research and portfolio construction engine** where the AI orchestrates deterministic statistical, optimization, risk, and backtesting tools.

MCP is particularly suitable for this architecture because its tools are explicitly designed to expose callable functions that AI clients can invoke and compose into workflows.

---
