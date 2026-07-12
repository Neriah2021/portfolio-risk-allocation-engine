"""
Risk Analytics Module - Institutional Portfolio Risk & Allocation Engine
Computes VaR, CVaR (Expected Shortfall), and runs historical-style stress
tests on the three mandate portfolios built in optimizer.py.

This is the module that speaks directly to insurance risk teams and bank
Product Control / Risk teams - it answers "how bad could this get?"
"""

import pandas as pd
import numpy as np


def load_data():
    returns = pd.read_csv("daily_returns.csv", index_col=0, parse_dates=True)
    mandates = pd.read_csv("portfolio_mandates.csv", index_col=0)
    return returns, mandates


def get_portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """
    Given asset weights for one portfolio, compute the historical daily
    return series that portfolio WOULD have earned over the sample period.
    This is the basis for all risk metrics below.
    """
    # Align weights to the same asset columns as returns
    w = weights.reindex(returns.columns).fillna(0)
    portfolio_returns = returns.dot(w)
    return portfolio_returns


def compute_var_cvar(portfolio_returns: pd.Series, confidence: float = 0.95,
                      horizon_days: int = 1) -> dict:
    """
    Historical VaR and CVaR (Expected Shortfall).

    VaR: the loss that will NOT be exceeded with (confidence)% probability
         over the given horizon.
    CVaR: the AVERAGE loss in the worst (1-confidence)% of cases -
          the more informative "tail risk" number regulators/insurers care about.
    """
    scaled_returns = portfolio_returns * np.sqrt(horizon_days)  # simple sqrt-time scaling

    var_percentile = (1 - confidence) * 100
    var = -np.percentile(scaled_returns, var_percentile)

    tail_losses = scaled_returns[scaled_returns <= -var]
    cvar = -tail_losses.mean() if len(tail_losses) > 0 else var

    return {
        "confidence": confidence,
        "horizon_days": horizon_days,
        "VaR": var,
        "CVaR": cvar,
    }


def compute_risk_metrics_all_mandates(returns: pd.DataFrame, mandates: pd.DataFrame,
                                       weight_columns: list) -> pd.DataFrame:
    """Compute VaR/CVaR at 95% and 99% confidence for all three mandates."""
    results = []
    for mandate_name in mandates.index:
        weights = mandates.loc[mandate_name, weight_columns]
        port_returns = get_portfolio_returns(returns, weights)

        var95 = compute_var_cvar(port_returns, confidence=0.95)
        var99 = compute_var_cvar(port_returns, confidence=0.99)

        results.append({
            "Mandate": mandate_name,
            "Daily VaR (95%)": var95["VaR"],
            "Daily CVaR (95%)": var95["CVaR"],
            "Daily VaR (99%)": var99["VaR"],
            "Daily CVaR (99%)": var99["CVaR"],
            "Max Daily Loss (historical)": -port_returns.min(),
        })

    return pd.DataFrame(results).set_index("Mandate").round(4)


# --- Stress Testing ------------------------------------------------------
# Since our asset universe (ETFs) doesn't have data back to 2008, we apply
# STYLIZED shock scenarios calibrated to real historical asset-class moves
# during major crises. This is a standard technique when live data doesn't
# span the crisis period - clearly label assumptions in your writeup.

STRESS_SCENARIOS = {
    "2008_Financial_Crisis": {
        "US_EQUITY": -0.38, "INTL_EQUITY": -0.43, "EM_EQUITY": -0.53,
        "CORP_BOND": -0.05, "GOV_BOND": 0.14, "HY_BOND": -0.26,
        "GOLD": 0.05, "REIT": -0.38,
    },
    "2020_COVID_Crash": {
        "US_EQUITY": -0.34, "INTL_EQUITY": -0.33, "EM_EQUITY": -0.31,
        "CORP_BOND": -0.10, "GOV_BOND": 0.08, "HY_BOND": -0.13,
        "GOLD": 0.04, "REIT": -0.42,
    },
    "2022_Rate_Shock": {
        "US_EQUITY": -0.19, "INTL_EQUITY": -0.16, "EM_EQUITY": -0.20,
        "CORP_BOND": -0.15, "GOV_BOND": -0.12, "HY_BOND": -0.11,
        "GOLD": 0.00, "REIT": -0.26,
    },
}


def run_stress_test(mandates: pd.DataFrame, weight_columns: list) -> pd.DataFrame:
    """
    Apply each stress scenario's asset-level shocks to each portfolio's
    weights to get an estimated portfolio-level loss under that scenario.
    """
    results = []
    for mandate_name in mandates.index:
        weights = mandates.loc[mandate_name, weight_columns]
        row = {"Mandate": mandate_name}
        for scenario_name, shocks in STRESS_SCENARIOS.items():
            shock_series = pd.Series(shocks)
            aligned_weights = weights.reindex(shock_series.index).fillna(0)
            portfolio_shock = (aligned_weights * shock_series).sum()
            row[scenario_name] = portfolio_shock
        results.append(row)

    return pd.DataFrame(results).set_index("Mandate").round(4)


if __name__ == "__main__":
    returns, mandates = load_data()

    # The weight columns are all columns in mandates.csv except the metric columns
    metric_cols = ["Expected Return", "Volatility", "Sharpe Ratio"]
    weight_columns = [c for c in mandates.columns if c not in metric_cols]

    print("--- VaR / CVaR by Mandate (daily, historical method) ---")
    risk_table = compute_risk_metrics_all_mandates(returns, mandates, weight_columns)
    print(risk_table)
    risk_table.to_csv("var_cvar_results.csv")

    print("\n--- Stress Test Results: Estimated Portfolio Loss by Scenario ---")
    stress_table = run_stress_test(mandates, weight_columns)
    print(stress_table)
    stress_table.to_csv("stress_test_results.csv")

    print("\nSaved var_cvar_results.csv and stress_test_results.csv")
