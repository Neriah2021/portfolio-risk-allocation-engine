"""
Dashboard - Institutional Portfolio Risk & Allocation Engine
Interactive Streamlit app that ties together the data layer, optimizer,
and risk analytics modules into a single investment-committee-style view.

Run with:  streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Portfolio Risk & Allocation Engine", layout="wide")


# --- Data loading (cached so it doesn't reload on every interaction) ----
@st.cache_data
def load_all_data():
    prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)
    returns = pd.read_csv("daily_returns.csv", index_col=0, parse_dates=True)
    mandates = pd.read_csv("portfolio_mandates.csv", index_col=0)
    var_cvar = pd.read_csv("var_cvar_results.csv", index_col=0)
    stress = pd.read_csv("stress_test_results.csv", index_col=0)
    return prices, returns, mandates, var_cvar, stress


try:
    prices, returns, mandates, var_cvar, stress = load_all_data()
except FileNotFoundError as e:
    st.error(f"Missing data file: {e}. Run data_layer.py, optimizer.py, and "
             f"risk_analytics.py first to generate the required CSVs.")
    st.stop()

metric_cols = ["Expected Return", "Volatility", "Sharpe Ratio"]
weight_columns = [c for c in mandates.columns if c not in metric_cols]

# --- Header ---------------------------------------------------------------
st.title("📊 Institutional Portfolio Risk & Allocation Engine")
st.caption("A mean-variance optimization and risk analytics platform across "
           "conservative, moderate, and aggressive risk mandates.")

# --- Sidebar: mandate selector --------------------------------------------
st.sidebar.header("Controls")
selected_mandate = st.sidebar.selectbox("Select Risk Mandate", mandates.index.tolist())
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**About this tool**\n\n"
    "Built on real multi-asset market data (equities, bonds, gold, REITs). "
    "Portfolios are constructed via mean-variance optimization "
    "(PyPortfolioOpt) and evaluated using historical VaR/CVaR and "
    "scenario-based stress testing."
)

# --- Top row: KPI cards ----------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
row = mandates.loc[selected_mandate]
col1.metric("Expected Annual Return", f"{row['Expected Return']*100:.2f}%")
col2.metric("Annual Volatility", f"{row['Volatility']*100:.2f}%")
col3.metric("Sharpe Ratio", f"{row['Sharpe Ratio']:.2f}")
col4.metric("Daily VaR (95%)", f"{var_cvar.loc[selected_mandate, 'Daily VaR (95%)']*100:.2f}%")

st.markdown("---")

# --- Row: Allocation pie + Efficient frontier -------------------------------
left, right = st.columns(2)

with left:
    st.subheader(f"{selected_mandate} — Asset Allocation")
    weights = mandates.loc[selected_mandate, weight_columns]
    weights = weights[weights > 0.001]  # drop near-zero weights for clarity
    fig_pie = px.pie(values=weights.values, names=weights.index, hole=0.4)
    fig_pie.update_traces(textinfo="label+percent")
    st.plotly_chart(fig_pie, use_container_width=True)

with right:
    st.subheader("All Mandates — Risk/Return Comparison")
    fig_scatter = go.Figure()
    colors = {"Conservative": "green", "Moderate": "blue", "Aggressive": "red"}
    for m in mandates.index:
        fig_scatter.add_trace(go.Scatter(
            x=[mandates.loc[m, "Volatility"]],
            y=[mandates.loc[m, "Expected Return"]],
            mode="markers+text",
            marker=dict(size=22, color=colors.get(m, "gray"),
                        line=dict(width=2, color="black")),
            text=[m], textposition="top center",
            name=m,
        ))
    # Pad axis ranges so markers/labels never get clipped at the edges
    vol_vals = mandates["Volatility"]
    ret_vals = mandates["Expected Return"]
    vol_pad = (vol_vals.max() - vol_vals.min()) * 0.4 + 0.01
    ret_pad = (ret_vals.max() - ret_vals.min()) * 0.4 + 0.01
    fig_scatter.update_layout(
        xaxis_title="Volatility (annualized)",
        yaxis_title="Expected Return (annualized)",
        xaxis_range=[vol_vals.min() - vol_pad, vol_vals.max() + vol_pad],
        yaxis_range=[ret_vals.min() - ret_pad, ret_vals.max() + ret_pad],
        showlegend=False,
        margin=dict(l=40, r=40, t=30, b=40),
        autosize=True,
    )
    st.plotly_chart(fig_scatter, use_container_width=True, config={"responsive": True})

st.markdown("---")

# --- Row: VaR/CVaR comparison across mandates -------------------------------
st.subheader("Tail Risk: VaR & CVaR Across Mandates")
var_plot_df = var_cvar[["Daily VaR (95%)", "Daily CVaR (95%)", "Daily VaR (99%)", "Daily CVaR (99%)"]] * 100
fig_var = px.bar(var_plot_df, barmode="group",
                  labels={"value": "Loss (%)", "Mandate": "", "variable": "Metric"})
st.plotly_chart(fig_var, use_container_width=True)

st.markdown("---")

# --- Row: Stress test heatmap -----------------------------------------------
st.subheader("Stress Test: Estimated Portfolio Loss by Scenario")
stress_pct = stress * 100
fig_stress = px.imshow(
    stress_pct,
    text_auto=".1f",
    color_continuous_scale="RdYlGn",
    labels=dict(color="Loss (%)"),
    aspect="auto",
)
fig_stress.update_layout(coloraxis_colorbar_title="Loss (%)")
st.plotly_chart(fig_stress, use_container_width=True)

st.caption(
    "Stress scenarios apply stylized historical shock magnitudes per asset "
    "class (2008 Financial Crisis, 2020 COVID Crash, 2022 Rate Shock) to "
    "each portfolio's current weights."
)

st.markdown("---")

# --- Row: Historical performance backtest -----------------------------------
st.subheader("Historical Backtest — Growth of $100")
weights_full = mandates.loc[selected_mandate, weight_columns].reindex(returns.columns).fillna(0)
port_daily_returns = returns.dot(weights_full)
growth = 100 * (1 + port_daily_returns).cumprod()

fig_growth = go.Figure()
fig_growth.add_trace(go.Scatter(x=growth.index, y=growth.values, mode="lines",
                                 line=dict(color="steelblue", width=2)))
fig_growth.update_layout(xaxis_title="Date", yaxis_title="Portfolio Value ($)")
st.plotly_chart(fig_growth, use_container_width=True)

# --- Footer: raw data tables (expandable) -----------------------------------
with st.expander("View raw portfolio weights"):
    st.dataframe(mandates)

with st.expander("View raw VaR/CVaR table"):
    st.dataframe(var_cvar)

with st.expander("View raw stress test table"):
    st.dataframe(stress)
