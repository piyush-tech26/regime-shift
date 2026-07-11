import numpy as np
import pandas as pd
from model.optimizer import RegimeOptimizer
def run_backtest(
    asset_returns: pd.DataFrame,
    regimes: pd.Series,
    lookback_days: int = 252,
    rebalance_freq: int = 21,
    transaction_bps: float = 10.0,
    max_weight: float = 0.80,
    risk_free: float = 0.04)->tuple:
    # Aligning the asset returns with regime labeled data
    common_idx = asset_returns.index.intersection(regimes.index)
    asset_returns = asset_returns.loc[common_idx]
    regimes = regimes.loc[common_idx]
    optimizer = RegimeOptimizer(max_weight=max_weight, risk_free=risk_free)
    assets = [c.replace("_return", "") for c in asset_returns.columns]
    n_assets = len(assets)
    # Simple returns for portfolio math(log returns dont sum correctly)
    simple_returns = np.exp(asset_returns) - 1
    # Storage
    current_weights = np.array([1/n_assets] * n_assets)  # starting with equal weight
    weights_history = []
    strategy_returns = []
    transaction_costs = []
    dates = asset_returns.index
    days_since_rebalance = 0

    for i, date in enumerate(dates):
        # Rebalance if enough data AND rebalance day reached
        can_rebalance = i >= lookback_days
        is_rebalance_day = days_since_rebalance >= rebalance_freq
        if can_rebalance and is_rebalance_day:
            lookback = asset_returns.iloc[i - lookback_days:i]
            current_regime = regimes.iloc[i]
            try:
                result = optimizer.optimize(current_regime, lookback)
                new_weights = result["raw_weights"]
                # Compute transaction cost from turnover
                turnover = np.sum(np.abs(new_weights - current_weights))
                cost = turnover * (transaction_bps / 10000)
                current_weights = new_weights
                transaction_costs.append(cost)
                days_since_rebalance = 0
                weights_history.append({
                    "date": date,
                    "regime": current_regime,
                    **{asset: w for asset, w in zip(assets, new_weights)},
                    "turnover": turnover,
                    "cost": cost
                })
            except Exception as e:
                # Optimizer failed — keep old weights
                transaction_costs.append(0.0)
                print(f"Warning: optimizer failed on {date}: {e}")
        else:
            transaction_costs.append(0.0)
        # Daily portfolio return using current weights
        daily_asset_returns = simple_returns.iloc[i].values
        portfolio_return = np.dot(current_weights, daily_asset_returns)-transaction_costs[-1]
        strategy_returns.append(portfolio_return)
        days_since_rebalance += 1

    # Assemble output
    portfolio = pd.DataFrame({
        "strategy_returns": strategy_returns,
        "transaction_cost": transaction_costs
    }, index=dates)

    # Benchmarks
    # 60/40: 60% SPY, 40% TLT
    portfolio["benchmark_6040"] = 0.6 * simple_returns["spy_return"] + 0.4 * simple_returns["tlt_return"]
    # Equal weight: 1/3 each
    portfolio["benchmark_equal"] = simple_returns.mean(axis=1)
    # Buy and hold SPY
    portfolio["benchmark_spy"] = simple_returns["spy_return"]

    # Equity curves
    portfolio["strategy_equity"]    = (1 + portfolio["strategy_returns"]).cumprod()
    portfolio["benchmark_6040_eq"]  = (1 + portfolio["benchmark_6040"]).cumprod()
    portfolio["benchmark_equal_eq"] = (1 + portfolio["benchmark_equal"]).cumprod()
    portfolio["benchmark_spy_eq"]   = (1 + portfolio["benchmark_spy"]).cumprod()
    weights_df = pd.DataFrame(weights_history).set_index("date") if weights_history else pd.DataFrame()
    print(f"Backtest complete:")
    print(f"  Total days: {len(portfolio)}")
    print(f"  Rebalances: {len(weights_history)}")
    print(f"  Avg turnover: {weights_df['turnover'].mean():.2%}")
    print(f"  Total transaction costs: {sum(transaction_costs):.4%}")
    return portfolio, weights_df

if __name__ == "__main__":
    from data.preprocessor import get_asset_returns
    features = pd.read_csv("data/raw/features.csv", index_col="Date", parse_dates=True)
    regimes = pd.read_csv("data/raw/wf_regimes.csv", index_col="Date", parse_dates=True)["regime"]
    asset_returns = get_asset_returns(features)
    portfolio, weights_history = run_backtest(
        asset_returns=asset_returns,
        regimes=regimes,
        lookback_days=252,
        rebalance_freq=21,
        transaction_bps=10.0
    )
    print("\nStrategy vs Benchmarks (final $1 invested):")
    print(f"  Strategy:       ${portfolio['strategy_equity'].iloc[-1]:.2f}")
    print(f"  60/40:          ${portfolio['benchmark_6040_eq'].iloc[-1]:.2f}")
    print(f"  Equal weight:   ${portfolio['benchmark_equal_eq'].iloc[-1]:.2f}")
    print(f"  Buy&hold SPY:   ${portfolio['benchmark_spy_eq'].iloc[-1]:.2f}")
    print("\nWeights history (first 5):")
    print(weights_history.head())
    portfolio.to_csv("data/raw/portfolio.csv")
    weights_history.to_csv("data/raw/weights.csv")
