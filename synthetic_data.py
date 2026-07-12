"""
Synthetic Data Generator
-------------------------
Used ONLY for local dev/testing in environments without market-data access.
Generates realistic, correlated daily returns for the same 8-asset universe
as data_layer.py, calibrated to plausible real-world return/vol/correlation
profiles (based on long-run historical behavior of these asset classes).

When you run this project on your own machine, use data_layer.py with
yfinance instead - this file is a stand-in so we can build/test the rest
of the pipeline (risk engine, optimizer, dashboard) without waiting on
live data access.
"""

import numpy as np
import pandas as pd

ASSETS = ["US_EQUITY", "INTL_EQUITY", "EM_EQUITY", "CORP_BOND",
          "GOV_BOND", "HY_BOND", "GOLD", "REIT"]

# Roughly realistic annualized return/vol assumptions (long-run historical ballpark)
ANN_RETURN = np.array([0.10, 0.07, 0.08, 0.05, 0.03, 0.06, 0.05, 0.08])
ANN_VOL    = np.array([0.16, 0.17, 0.22, 0.08, 0.07, 0.11, 0.15, 0.20])

# Plausible correlation structure: equities correlate with each other,
# bonds correlate with each other, gold is a diversifier (low/negative corr)
CORR = np.array([
    # US_EQ  INTL   EM    CORP  GOV   HY    GOLD  REIT
    [1.00,  0.85,  0.75, 0.20, -0.10, 0.55, 0.00, 0.65],  # US_EQUITY
    [0.85,  1.00,  0.80, 0.20, -0.10, 0.55, 0.05, 0.55],  # INTL_EQUITY
    [0.75,  0.80,  1.00, 0.15, -0.15, 0.55, 0.10, 0.45],  # EM_EQUITY
    [0.20,  0.20,  0.15, 1.00,  0.65, 0.60, 0.10, 0.30],  # CORP_BOND
    [-0.10, -0.10, -0.15, 0.65, 1.00, 0.20, 0.25, 0.00],  # GOV_BOND
    [0.55,  0.55,  0.55, 0.60,  0.20, 1.00, 0.05, 0.45],  # HY_BOND
    [0.00,  0.05,  0.10, 0.10,  0.25, 0.05, 1.00, 0.15],  # GOLD
    [0.65,  0.55,  0.45, 0.30,  0.00, 0.45, 0.15, 1.00],  # REIT
])


def generate_synthetic_returns(n_days: int = 2000, seed: int = 42,
                                start_date: str = "2017-01-01") -> pd.DataFrame:
    """
    Generate correlated daily returns via Cholesky decomposition of the
    target correlation matrix, scaled to target annualized return/vol.
    """
    rng = np.random.default_rng(seed)

    daily_mean = ANN_RETURN / 252
    daily_vol = ANN_VOL / np.sqrt(252)

    L = np.linalg.cholesky(CORR)
    z = rng.standard_normal((n_days, len(ASSETS)))
    correlated_z = z @ L.T

    daily_returns = daily_mean + correlated_z * daily_vol

    dates = pd.bdate_range(start=start_date, periods=n_days)
    df = pd.DataFrame(daily_returns, index=dates, columns=ASSETS)
    return df


def returns_to_prices(returns: pd.DataFrame, start_price: float = 100.0) -> pd.DataFrame:
    """Convert a returns series to a cumulative price index (base=100)."""
    return start_price * (1 + returns).cumprod()


if __name__ == "__main__":
    returns = generate_synthetic_returns()
    prices = returns_to_prices(returns)

    returns.to_csv("daily_returns.csv")
    prices.to_csv("prices.csv")

    print("Synthetic dataset generated.")
    print(f"Shape: {returns.shape}, Date range: {returns.index.min().date()} to {returns.index.max().date()}")
    print("\n--- Sanity check: realized annualized stats ---")
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    check = pd.DataFrame({"Ann. Return": ann_ret.round(3), "Ann. Vol": ann_vol.round(3)})
    print(check)
    print("\nSaved: prices.csv, daily_returns.csv")
