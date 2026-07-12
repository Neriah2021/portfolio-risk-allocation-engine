"""
Data Layer - Institutional Portfolio Risk & Allocation Engine
Pulls historical price data for a multi-asset universe and prepares
returns for downstream risk/optimization modules.
"""

import yfinance as yf
import pandas as pd
import numpy as np


# --- Universe definition -----------------------------------------------
# A representative multi-asset universe: equities, bonds, and commodities.
# You can swap tickers for anything relevant (e.g. Canadian banks: RY.TO, TD.TO)
DEFAULT_UNIVERSE = {
    "US_EQUITY": "SPY",       # S&P 500 ETF
    "INTL_EQUITY": "EFA",     # Developed international equities
    "EM_EQUITY": "EEM",       # Emerging markets equities
    "CORP_BOND": "LQD",       # Investment grade corporate bonds
    "GOV_BOND": "IEF",        # 7-10yr US Treasuries
    "HY_BOND": "HYG",         # High yield bonds
    "GOLD": "GLD",            # Gold
    "REIT": "VNQ",            # Real estate
}


def fetch_price_data(universe: dict = None, start: str = "2015-01-01", end: str = None) -> pd.DataFrame:
    """
    Fetch adjusted close prices for a dict of {label: ticker}.
    Returns a DataFrame with columns = asset labels, index = dates.
    """
    universe = universe or DEFAULT_UNIVERSE
    tickers = list(universe.values())

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]

    # Handle single-ticker edge case (yfinance returns Series if len==1)
    if isinstance(raw, pd.Series):
        raw = raw.to_frame()

    # Rename columns from tickers back to friendly labels
    ticker_to_label = {v: k for k, v in universe.items()}
    raw = raw.rename(columns=ticker_to_label)

    raw = raw.dropna(how="all")
    return raw


def compute_returns(price_df: pd.DataFrame, freq: str = "daily") -> pd.DataFrame:
    """
    Convert price series to returns.
    freq: 'daily' or 'monthly'
    """
    if freq == "monthly":
        price_df = price_df.resample("ME").last()

    returns = price_df.pct_change().dropna(how="all")
    return returns


def summary_stats(returns: pd.DataFrame, annualize_factor: int = 252) -> pd.DataFrame:
    """
    Quick annualized return/vol summary per asset - sanity check before
    feeding into the optimizer.
    """
    ann_return = returns.mean() * annualize_factor
    ann_vol = returns.std() * np.sqrt(annualize_factor)
    sharpe = ann_return / ann_vol

    stats = pd.DataFrame({
        "Annualized Return": ann_return,
        "Annualized Volatility": ann_vol,
        "Sharpe (rf=0)": sharpe,
    })
    return stats.round(4)


if __name__ == "__main__":
    print("Fetching multi-asset universe...")
    prices = fetch_price_data()
    print(f"\nData shape: {prices.shape}")
    print(prices.tail())

    daily_returns = compute_returns(prices, freq="daily")
    print("\n--- Asset Summary Stats (annualized) ---")
    print(summary_stats(daily_returns))

    # Save for use by other modules
    prices.to_csv("prices.csv")
    daily_returns.to_csv("daily_returns.csv")
    print("\nSaved prices.csv and daily_returns.csv")
