import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from plotly.subplots import make_subplots
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Regime-Shift Engine", page_icon="📈", layout="wide")

# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Running pipeline — this takes roughly 2-3 minutes on first load.")
def load():
    from pipeline import pipeline_exists, run_pipeline
    if not pipeline_exists():
        run_pipeline()

    features = pd.read_csv("data/raw/features.csv", index_col="Date", parse_dates=True)
    regimes = pd.read_csv("data/raw/wf_regimes.csv", index_col="Date", parse_dates=True)["regime"]
    portfolio = pd.read_csv("data/raw/portfolio.csv", index_col=0, parse_dates=True)
    raw = pd.read_csv("data/raw/market_data.csv", index_col="Date", parse_dates=True)
    tear_sheet = pd.read_csv("data/raw/tear_sheet.csv", index_col=0)
    weights = pd.read_csv("data/raw/weights.csv", index_col="date", parse_dates=True)

    # Align portfolio/regimes/raw on common dates
    common_idx = portfolio.index.intersection(regimes.index).intersection(raw.index)
    portfolio_aligned = portfolio.loc[common_idx]
    regimes_aligned = regimes.loc[common_idx]
    raw_aligned = raw.loc[common_idx]

    tear_sheet = tear_sheet.apply(pd.to_numeric, errors="coerce")

    return raw_aligned, features, regimes_aligned, portfolio_aligned, tear_sheet, weights

raw, features_df, regimes, portfolio, tear_sheet, weights = load()

# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════
COLOURS = {"BULL": "#2ecc71", "BEAR": "#e67e22", "CRISIS": "#e74c3c"}

ASSET_COLOURS = {
    "spy": "#3498db",   # blue = equities
    "tlt": "#9b59b6",   # purple = bonds
    "gld": "#f1c40f"    # yellow = gold
}

STRATEGY_COLOURS = {
    "Strategy": "#2ecc71",
    "60/40 Benchmark": "#3498db",
    "Equal Weight": "#f39c12",
    "Buy & Hold SPY": "#e74c3c"
}

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
st.sidebar.title("Regime-Shift Engine")
st.sidebar.markdown("Multi-asset HMM-based tactical allocation")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["Regime Chart", "Tear Sheet", "Weights History", "Regime Statistics"]
)

# ══════════════════════════════════════════════════════════════
# PAGE 1 — REGIME CHART
# ══════════════════════════════════════════════════════════════
if page == "Regime Chart":
    st.title("📈 Market Regime Detection")
    st.markdown("HMM-detected Bull / Bear / Crisis regimes overlaid on S&P 500 price history.")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", value=regimes.index.min())
    with col2:
        end_date = st.date_input("End date", value=raw.index.max())

    date_mask = (raw.index >= pd.Timestamp(start_date)) & (raw.index <= pd.Timestamp(end_date))
    raw_f = raw.loc[date_mask]
    regimes_f = regimes.reindex(raw_f.index).dropna()

    raw_weekly = raw_f.resample("W").last().dropna()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        subplot_titles=("S&P 500 Price", "VIX (Fear Index)"),
        vertical_spacing=0.08
    )

    fig.add_trace(
        go.Scatter(
            x=raw_weekly.index, y=raw_weekly["spy"],
            line=dict(color="#2c3e50", width=1.2),
            name="S&P 500"
        ),
        row=1, col=1
    )

    # Group consecutive same-regime days
    blocks = []
    regimes_list = list(regimes_f.items())
    if len(regimes_list) > 0:
        block_start, current_regime = regimes_list[0]
        for date, regime in regimes_list[1:]:
            if regime != current_regime:
                blocks.append((block_start, date, current_regime))
                block_start = date
                current_regime = regime
        blocks.append((block_start, regimes_list[-1][0], current_regime))

    # Batch shading
    shapes = []
    for block_start, block_end, regime_name in blocks:
        shapes.append(dict(
            type="rect",
            xref="x", yref="paper",
            x0=block_start, x1=block_end,
            y0=0, y1=1,
            fillcolor=COLOURS.get(regime_name, "grey"),
            opacity=0.15,
            line=dict(width=0),
            layer="below"
        ))
    fig.update_layout(shapes=shapes)

    fig.add_trace(
        go.Scatter(
            x=raw_weekly.index, y=raw_weekly["vix"],
            line=dict(color="#8e44ad", width=1),
            name="VIX"
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=600, showlegend=True,
        hovermode="x unified",
        plot_bgcolor="white", paper_bgcolor="white"
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    st.plotly_chart(fig, width="stretch")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("🟢 BULL — Positive returns, low fear")
    with col2:
        st.warning("🟠 BEAR — Uncertain, moderate fear")
    with col3:
        st.error("🔴 CRISIS — Negative returns, high fear")

# ══════════════════════════════════════════════════════════════
# PAGE 2 — TEAR SHEET
# ══════════════════════════════════════════════════════════════
elif page == "Tear Sheet":
    st.title("📊 Performance Tear Sheet")
    st.markdown("Strategy vs three benchmarks across all risk-adjusted metrics.")

    st.subheader("Key metrics summary")

    col1, col2, col3, col4 = st.columns(4)

    strategy_sharpe = float(tear_sheet.loc["Sharpe Ratio", "Strategy"])
    strategy_dd = float(tear_sheet.loc["Max Drawdown (%)", "Strategy"])
    strategy_return = float(tear_sheet.loc["Annual Return (%)", "Strategy"])
    strategy_vol = float(tear_sheet.loc["Annual Volatility (%)", "Strategy"])

    spy_sharpe = float(tear_sheet.loc["Sharpe Ratio", "Buy & Hold SPY"])
    spy_dd = float(tear_sheet.loc["Max Drawdown (%)", "Buy & Hold SPY"])
    spy_return = float(tear_sheet.loc["Annual Return (%)", "Buy & Hold SPY"])
    spy_vol = float(tear_sheet.loc["Annual Volatility (%)", "Buy & Hold SPY"])

    col1.metric(
        "Sharpe Ratio",
        f"{strategy_sharpe:.3f}",
        delta=f"{strategy_sharpe - spy_sharpe:+.3f} vs SPY"
    )
    col2.metric(
        "Max Drawdown",
        f"{strategy_dd:.2f}%",
        delta=f"{strategy_dd - spy_dd:+.2f}% vs SPY"
    )
    col3.metric(
        "Annual Return",
        f"{strategy_return:.2f}%",
        delta=f"{strategy_return - spy_return:+.2f}% vs SPY"
    )
    col4.metric(
        "Annual Volatility",
        f"{strategy_vol:.2f}%",
        delta=f"{strategy_vol - spy_vol:+.2f}% vs SPY",
        delta_color="inverse"
    )

    st.markdown("---")

    st.subheader("Full comparison table")
    st.dataframe(
        tear_sheet.style.format("{:.4f}").background_gradient(cmap="RdYlGn", axis=1),
        width="stretch"
    )

    st.markdown("---")

    st.subheader("Equity curves — $1 invested")

    portfolio_weekly = portfolio[[
        "strategy_equity", "benchmark_6040_eq",
        "benchmark_equal_eq", "benchmark_spy_eq"
    ]].resample("W").last().dropna()

    fig2 = go.Figure()

    curve_map = {
        "Strategy": "strategy_equity",
        "60/40 Benchmark": "benchmark_6040_eq",
        "Equal Weight": "benchmark_equal_eq",
        "Buy & Hold SPY": "benchmark_spy_eq"
    }

    for name, col in curve_map.items():
        fig2.add_trace(go.Scatter(
            x=portfolio_weekly.index,
            y=portfolio_weekly[col],
            name=name,
            line=dict(color=STRATEGY_COLOURS[name], width=2)
        ))

    fig2.update_layout(
        height=450, hovermode="x unified",
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis_title="Portfolio Value ($)",
        xaxis_title="Date",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig2.update_xaxes(showgrid=False)
    fig2.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    st.plotly_chart(fig2, width="stretch")

    st.markdown("---")

    st.subheader("Transaction cost analysis")
    col1, col2 = st.columns(2)

    with col1:
        if "Total Transaction Cost (%)" in tear_sheet.index:
            total_cost = float(tear_sheet.loc["Total Transaction Cost (%)", "Strategy"])
            annual_drag = float(tear_sheet.loc["Avg Annual Drag (%)", "Strategy"])

            st.info(f"""
            **Total transaction costs paid**: {total_cost:.2f}%  
            **Average annual drag**: {annual_drag:.2f}%
            
            The strategy has been penalized 10 basis points per unit of turnover
            on each monthly rebalance. Despite this friction, risk-adjusted
            performance exceeds all benchmarks.
            """)

    with col2:
        if "transaction_cost" in portfolio.columns:
            cost_series = portfolio["transaction_cost"]
            cost_series_quaterly = cost_series[cost_series > 0].resample("QE").sum() * 100

            fig_cost = go.Figure()
            fig_cost.add_trace(go.Bar(
                x=cost_series_quaterly.index,
                y=cost_series_quaterly.values,
                marker_color="#e74c3c",
                name="Quarterly cost"
            ))
            fig_cost.update_layout(
                height=250,
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis_title="Cost (%)",
                showlegend=False,
                margin=dict(t=20, b=20)
            )
            st.plotly_chart(fig_cost, width="stretch")

# ══════════════════════════════════════════════════════════════
# PAGE 3 — WEIGHTS HISTORY
# ══════════════════════════════════════════════════════════════
elif page == "Weights History":
    st.title("⚖️ Portfolio Weights Over Time")
    st.markdown("How the CVXPY optimizer allocated between SPY, TLT, and GLD as regimes shifted.")

    # Stacked area chart of weights
    fig_weights = go.Figure()

    for asset in ["spy", "tlt", "gld"]:
        fig_weights.add_trace(go.Scatter(
            x=weights.index,
            y=weights[asset] * 100,
            name=asset.upper(),
            stackgroup="one",
            fillcolor=ASSET_COLOURS[asset],
            line=dict(width=0.5, color=ASSET_COLOURS[asset]),
            hovertemplate="%{y:.1f}%<extra></extra>"
        ))

    fig_weights.update_layout(
        height=500,
        hovermode="x unified",
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis_title="Portfolio Weight (%)",
        xaxis_title="Rebalance Date",
        yaxis=dict(range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_weights.update_xaxes(showgrid=False)
    fig_weights.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    st.plotly_chart(fig_weights, width="stretch")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Average allocation by regime")
        avg_by_regime = weights.groupby("regime")[["spy", "tlt", "gld"]].mean() * 100
        avg_by_regime = avg_by_regime.round(1)

        fig_bar = go.Figure()
        for asset in ["spy", "tlt", "gld"]:
            fig_bar.add_trace(go.Bar(
                x=avg_by_regime.index,
                y=avg_by_regime[asset],
                name=asset.upper(),
                marker_color=ASSET_COLOURS[asset]
            ))

        fig_bar.update_layout(
            height=350,
            barmode="stack",
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis_title="Average Weight (%)",
            xaxis_title="Regime"
        )
        st.plotly_chart(fig_bar, width="stretch")

    with col2:
        st.subheader("Rebalance turnover distribution")

        fig_turnover = go.Figure()
        fig_turnover.add_trace(go.Histogram(
            x=weights["turnover"] * 100,
            nbinsx=30,
            marker_color="#3498db"
        ))
        fig_turnover.update_layout(
            height=350,
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis_title="Turnover per rebalance (%)",
            yaxis_title="Number of rebalances"
        )
        st.plotly_chart(fig_turnover, width="stretch")

    st.markdown("---")

    st.subheader("Recent rebalances")
    recent = weights.tail(10).copy()
    recent["spy"] = (recent["spy"] * 100).round(1).astype(str) + "%"
    recent["tlt"] = (recent["tlt"] * 100).round(1).astype(str) + "%"
    recent["gld"] = (recent["gld"] * 100).round(1).astype(str) + "%"
    recent["turnover"] = (recent["turnover"] * 100).round(2).astype(str) + "%"
    recent["cost"] = (recent["cost"] * 10000).round(1).astype(str) + " bps"
    st.dataframe(recent, width="stretch")

# ══════════════════════════════════════════════════════════════
# PAGE 4 — REGIME STATISTICS
# ══════════════════════════════════════════════════════════════
elif page == "Regime Statistics":
    st.title("🔍 Regime Statistics")
    st.markdown("Distribution and characteristics of each detected regime.")

    col1, col2 = st.columns(2)

    counts = regimes.value_counts()

    with col1:
        st.subheader("Regime distribution")
        fig3 = go.Figure(go.Pie(
            labels=counts.index,
            values=counts.values,
            marker_colors=[COLOURS[r] for r in counts.index],
            hole=0.4
        ))
        fig3.update_layout(height=350, showlegend=True)
        st.plotly_chart(fig3, width="stretch")

    with col2:
        st.subheader("Days per regime")
        counts_df = counts.reset_index()
        counts_df.columns = ["Regime", "Days"]
        counts_df["% of time"] = (counts_df["Days"] / counts_df["Days"].sum() * 100).round(1)
        st.dataframe(counts_df, width="stretch", hide_index=True)

        st.markdown("---")
        st.subheader("Average daily SPY return per regime")
        df_combined = pd.DataFrame({
            "simple_returns": np.exp(features_df["returns"]) - 1,
            "regime": regimes.reindex(features_df.index)
        }).dropna()
        avg_returns = (df_combined.groupby("regime")["simple_returns"].mean() * 100).round(4)
        st.dataframe(
            avg_returns.reset_index().rename(
                columns={"simple_returns": "Avg Daily Return (%)"}
            ),
            width="stretch", hide_index=True
        )

    st.markdown("---")
    st.subheader("VIX distribution by regime")
    fig4 = go.Figure()
    for regime in ["BULL", "BEAR", "CRISIS"]:
        mask = regimes == regime
        vix_vals = raw.loc[raw.index[raw.index.isin(regimes[mask].index)], "vix"].dropna()
        if not len(vix_vals):
            continue
        fig4.add_trace(go.Box(
            y=vix_vals.values,
            name=regime,
            marker_color=COLOURS[regime],
            boxpoints=False
        ))
    fig4.update_layout(
        height=400,
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis_title="VIX",
        boxmode="group"
    )
    st.plotly_chart(fig4, width="stretch")
