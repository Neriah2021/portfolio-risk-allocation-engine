# Institutional Portfolio Risk & Allocation Engine

A mean-variance portfolio optimization and risk analytics platform built on real multi-asset market data. Simulates the workflow of a wealth management or insurance investment team: constructing risk-mandated portfolios, quantifying tail risk, and stress-testing against historical crisis scenarios.

**[Live dashboard demo — insert screenshot or GIF here]**

## What it does

- **Data pipeline** — pulls historical daily prices for an 8-asset multi-class universe (US/International/EM equities, corporate/government/high-yield bonds, gold, REITs) via Yahoo Finance
- **Portfolio optimization** — builds three risk-mandate portfolios (Conservative, Moderate, Aggressive) using mean-variance optimization, and plots the full efficient frontier
- **Risk analytics** — computes historical VaR and CVaR (Expected Shortfall) at 95%/99% confidence for each mandate
- **Stress testing** — applies stylized shock scenarios (2008 Financial Crisis, 2020 COVID Crash, 2022 Rate Shock) to estimate portfolio-level losses under crisis conditions
- **Interactive dashboard** — a Streamlit app tying all of the above into one investment-committee-style view

## Why this project

Built to demonstrate applied CFA Level 1 portfolio management concepts (mean-variance optimization, efficient frontier, Sharpe ratio) alongside the risk management practices used by real investment/insurance teams (VaR/CVaR, scenario stress testing) — implemented as working software rather than a static analysis.

## Tech stack

- `yfinance` — market data
- `PyPortfolioOpt` — mean-variance optimization
- `pandas` / `numpy` — data processing
- `matplotlib` / `plotly` — visualization
- `streamlit` — interactive dashboard

## Project structure

```
├── data_layer.py           # Pulls and prepares market data
├── optimizer.py             # Builds risk-mandate portfolios + efficient frontier
├── risk_analytics.py        # VaR/CVaR and stress testing
├── dashboard.py              # Interactive Streamlit dashboard
├── requirements.txt
└── README.md
```

## How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the pipeline in order
python data_layer.py       # generates prices.csv, daily_returns.csv
python optimizer.py        # generates portfolio_mandates.csv, efficient_frontier.png
python risk_analytics.py   # generates var_cvar_results.csv, stress_test_results.csv

# 3. Launch the dashboard
streamlit run dashboard.py
```

## Sample results

| Mandate | Expected Return | Volatility | Sharpe Ratio | 2008-Style Shock |
|---|---|---|---|---|
| Conservative | 2.4% | 5.4% | -0.12 | -1.2% |
| Moderate | 12.6% | 12.5% | 0.77 | -18.6% |
| Aggressive | 8.9% | 8.9% | 0.66 | -8.5% |

*(Results are based on a real historical data pull; will vary slightly on re-run due to updated market data.)*

## Key insight

The stress testing reveals that "safe" portfolios aren't safe against every risk type — the Conservative mandate, while resilient to equity-driven crashes (2008, 2020), is more exposed to rate-shock scenarios (2022-style) due to its heavy government bond allocation. This highlights why single-metric risk assessment (like volatility alone) is insufficient for institutional risk management.

## Limitations & assumptions

- Stress scenarios use stylized historical shock magnitudes per asset class rather than live crisis-period data, since the ETF universe doesn't have continuous history back to 2008
- Historical VaR/CVaR reflects the specific data-pull period and will shift with new data
- This is a demonstration/portfolio project, not investment advice

## Author

[Your name] — MFin candidate, I.H. Asper School of Business, University of Manitoba | CFA Level 1
