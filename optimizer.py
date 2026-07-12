"""
Optimization Engine - Institutional Portfolio Risk & Allocation Engine
Builds mean-variance optimized portfolios for different risk mandates
(conservative, moderate, aggressive) and computes the efficient frontier.
"""

import pandas as pd
import numpy as np
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import plotting


def load_returns(path: str = "daily_returns.csv") -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def get_expected_returns_and_cov(returns: pd.DataFrame):
    """
    Use PyPortfolioOpt's estimators on daily returns.
    mu: annualized expected returns (mean historical return)
    S: annualized covariance matrix (sample covariance)
    """
    prices_like = (1 + returns).cumprod()  # reconstruct a price-like series
    mu = expected_returns.mean_historical_return(prices_like, returns_data=False)
    S = risk_models.sample_cov(prices_like, returns_data=False)
    return mu, S


def build_portfolio(mu, S, mandate: str = "moderate", risk_free_rate: float = 0.03):
    """
    Build an optimized portfolio for a given risk mandate.

    mandate options:
      - "conservative": minimize volatility
      - "moderate": maximize Sharpe ratio
      - "aggressive": target a higher return, minimizing risk for that return
    """
    ef = EfficientFrontier(mu, S)

    if mandate == "conservative":
        weights = ef.min_volatility()
    elif mandate == "moderate":
        weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
    elif mandate == "aggressive":
        # Target return: 75th percentile of individual asset returns (ambitious but bounded)
        target = float(np.percentile(mu.values, 75))
        weights = ef.efficient_return(target_return=target)
    else:
        raise ValueError(f"Unknown mandate: {mandate}")

    cleaned_weights = ef.clean_weights()
    perf = ef.portfolio_performance(risk_free_rate=risk_free_rate, verbose=False)

    return {
        "mandate": mandate,
        "weights": cleaned_weights,
        "expected_annual_return": perf[0],
        "annual_volatility": perf[1],
        "sharpe_ratio": perf[2],
    }


def build_all_mandates(mu, S, risk_free_rate: float = 0.03):
    """Build all three risk-mandate portfolios and return as a comparison table."""
    results = {}
    for mandate in ["conservative", "moderate", "aggressive"]:
        # Fresh EfficientFrontier object needed for each (state gets consumed on solve)
        results[mandate] = build_portfolio(mu, S, mandate, risk_free_rate)
    return results


def results_to_dataframe(results: dict) -> pd.DataFrame:
    """Flatten the mandate results dict into a clean comparison table."""
    rows = []
    for mandate, res in results.items():
        row = {"Mandate": mandate.capitalize(),
               "Expected Return": res["expected_annual_return"],
               "Volatility": res["annual_volatility"],
               "Sharpe Ratio": res["sharpe_ratio"]}
        row.update(res["weights"])
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Mandate")
    return df.round(4)


def plot_efficient_frontier(mu, S, results: dict, save_path: str = "efficient_frontier.png"):
    """Plot the efficient frontier with the three mandate portfolios marked."""
    import matplotlib.pyplot as plt

    ef = EfficientFrontier(mu, S)
    fig, ax = plt.subplots(figsize=(9, 6))
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)

    # Overlay our three chosen portfolios
    colors = {"conservative": "green", "moderate": "blue", "aggressive": "red"}
    for mandate, res in results.items():
        ax.scatter(res["annual_volatility"], res["expected_annual_return"],
                    marker="*", s=300, c=colors[mandate], label=mandate.capitalize(),
                    edgecolors="black", zorder=5)

    ax.set_title("Efficient Frontier with Risk-Mandate Portfolios")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved chart to {save_path}")


if __name__ == "__main__":
    returns = load_returns()
    mu, S = get_expected_returns_and_cov(returns)

    print("--- Expected Annual Returns (per asset) ---")
    print(mu.round(4))
    print("\n--- Annualized Covariance Matrix ---")
    print(S.round(4))

    results = build_all_mandates(mu, S)

    print("\n--- Portfolio Comparison: Conservative vs Moderate vs Aggressive ---")
    comparison = results_to_dataframe(results)
    print(comparison)

    comparison.to_csv("portfolio_mandates.csv")
    print("\nSaved portfolio_mandates.csv")

    plot_efficient_frontier(mu, S, results)
