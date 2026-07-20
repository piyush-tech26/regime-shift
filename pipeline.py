import os
import numpy as np
import pandas as pd
from scipy import stats
from data.fetcher import fetch_all
from data.preprocessor import preprocess, get_asset_returns
from model.hmm_engine import RegimeHMM
from model.regime_labeler import label_regimes, apply_labels
from validation.walk_forward import walk_forward_validation, combine_results
from backtest.engine import run_backtest
from metrics.tear_sheet import compute_tear_sheet
DIR="data/raw"
FROZEN_DIR="data/frozen"
SNAPSHOT_FILE=f"{FROZEN_DIR}/market_data_sanpshot.parquet"

def smooth_regimes(series, window=10):
    regime_map = {"BULL": 0, "BEAR": 1, "CRISIS": 2}
    reverse_map = {0: "BULL", 1: "BEAR", 2: "CRISIS"}
    numeric = series.map(regime_map)
    smoothed = numeric.rolling(window=window, center=False, min_periods=1).apply(lambda x: stats.mode(x, keepdims=True)[0][0], raw=True)
    return smoothed.map(reverse_map)
def get_market_data(force_refresh: bool = False) -> pd.DataFrame:
    """
    Load market data from frozen snapshot if available, else fetch and save.
    Ensures reproducibility — same data regardless of when pipeline is run.
    """
    os.makedirs(FROZEN_DIR, exist_ok=True)

    if os.path.exists(SNAPSHOT_FILE) and not force_refresh:
        print(f"Loading frozen market data from {SNAPSHOT_FILE}")
        df = pd.read_parquet(SNAPSHOT_FILE)
        print(f"Data range: {df.index[0].date()} to {df.index[-1].date()} ({len(df)} rows)")
        return df

    print("Fetching fresh market data...")
    df = fetch_all()
    df.to_parquet(SNAPSHOT_FILE)
    print(f"Snapshot saved to {SNAPSHOT_FILE}")
    print("Commit this file to git to freeze data for reproducibility.")
    return df
def ensemble_regimes(regime_series_list: list) -> pd.Series:
    combined = pd.concat(regime_series_list, axis=1)
    combined.columns = [f"seed_{i}" for i in range(len(regime_series_list))]
    def majority_vote(row):
        votes = row.value_counts()
        max_count = votes.max()
        winners = votes[votes == max_count].index.tolist()
        if len(winners) > 1:
            if "CRISIS" in winners:
                return "CRISIS"
            elif "BEAR" in winners:
                return "BEAR"
            else:
                return "BULL"
        return winners[0]
    ensemble = combined.apply(majority_vote, axis=1)
    ensemble.name = "regime"
    return ensemble
def compute_seed_agreement(regime_series_list: list) -> pd.Series:
    """
    Compute agreement percentage across seeds for each day.
    Returns fraction of seeds that agreed with the majority vote.
    """
    combined = pd.concat(regime_series_list, axis=1)
    combined.columns = [f"seed_{i}" for i in range(len(regime_series_list))]
    def agreement(row):
        votes = row.value_counts()
        return votes.max() / len(row)
    return combined.apply(agreement, axis=1)
def pipeline_exists() -> bool:
    required=[f"{DIR}/market_data.csv",f"{DIR}/features.csv",f"{DIR}/wf_regimes.csv",f"{DIR}/portfolio.csv",f"{DIR}/weights.csv",f"{DIR}/tear_sheet.csv"]
    return all(os.path.exists(f) for f in required)

def run_pipeline(train_years=7,test_years=1,n_seeds=10, force_refresh_data=False):
    os.makedirs(DIR,exist_ok=True)
    print("Step 1:  Fetching market data...")
    raw=get_market_data(force_refresh=force_refresh_data)
    raw.to_csv(f"{DIR}/market_data.csv")
    print("Step 2: Preprocessing features...")
    features=preprocess(raw)
    features.to_csv(f"{DIR}/features.csv")
    print(f"Step 3: Running walk_forward validation across {n_seeds} seeds...")
    all_weights_by_seed={}
    per_seed_metrics=[]
    all_regimes_by_seed={}
    all_portfolios_by_seed={}
    for seed in range(n_seeds):
        print(f"---- Seed {seed+1}/{n_seeds} ----")
        results=walk_forward_validation(features, train_years=train_years, test_years=test_years,random_state=seed)
        regimes=combine_results(results)
        regimes=smooth_regimes(regimes)
        all_regimes_by_seed[seed]=regimes
        asset_returns=get_asset_returns(features)
        portfolio, weight_history=run_backtest(asset_returns=asset_returns, regimes=regimes,lookback_days=252,rebalance_freq=21,transaction_bps=10.0)
        all_portfolios_by_seed[seed] = portfolio
        all_weights_by_seed[seed] = weight_history
        tear=compute_tear_sheet(portfolio,weight_history)
        per_seed_metrics.append({"seed":seed, "sharpe": float(tear.loc["Sharpe Ratio", "Strategy"]),"sortino": float(tear.loc["Sortino Ratio", "Strategy"]),"max_dd": float(tear.loc["Max Drawdown (%)","Strategy"]),"annual_return": float(tear.loc["Annual Return (%)", "Strategy"]),"final_value": float(portfolio["strategy_equity"].iloc[-1]), "turnover": float(tear.loc["Avg Turnover per Rebalance (%)", "Strategy"])})
    print(f"\n\nStep 4: Aggreagating results across seeds...")
    seed_df = pd.DataFrame(per_seed_metrics)
    seed_df.to_csv(f"{DIR}/seed_analysis.csv", index=False)
    median_seed_idx = (seed_df["sharpe"] - seed_df["sharpe"].median()).abs().idxmin()
    representative_seed = int(seed_df.iloc[median_seed_idx]["seed"])
    # Save representative seed's outputs for dashboard
    regimes = all_regimes_by_seed[representative_seed]
    portfolio = all_portfolios_by_seed[representative_seed]
    weight_history = all_weights_by_seed[representative_seed]
    regimes.to_csv(f"{DIR}/wf_regimes.csv", header=["regime"])
    portfolio.to_csv(f"{DIR}/portfolio.csv")
    weight_history.to_csv(f"{DIR}/weights.csv")
    # Compute tear sheet with representative seed's data
    tear_sheet = compute_tear_sheet(portfolio, weight_history)
    tear_sheet.to_csv(f"{DIR}/tear_sheet.csv")
    # Step 5: Report distributional results
    print("PIPELINE COMPLETE")
    print(f"\nMulti-Seed Analysis ({n_seeds} seeds):")
    print(f"  Sharpe Ratio:")
    print(f"    Mean:    {seed_df['sharpe'].mean():.3f}")
    print(f"    Std:     {seed_df['sharpe'].std():.3f}")
    print(f"    Range:   [{seed_df['sharpe'].min():.3f}, {seed_df['sharpe'].max():.3f}]")
    
    print(f"\n  Sortino Ratio:")
    print(f"    Mean:    {seed_df['sortino'].mean():.3f}")
    print(f"    Std:     {seed_df['sortino'].std():.3f}")
    
    print(f"\n  Max Drawdown:")
    print(f"    Mean:    {seed_df['max_dd'].mean():.2f}%")
    print(f"    Std:     {seed_df['max_dd'].std():.2f}%")
    
    print(f"\n  Annual Return:")
    print(f"    Mean:    {seed_df['annual_return'].mean():.2f}%")
    print(f"    Std:     {seed_df['annual_return'].std():.2f}%")

    print(f"\nRepresentative Run (Seed {representative_seed}, median-Sharpe):")
    print(f"  Strategy Value:  ${portfolio['strategy_equity'].iloc[-1]:.2f}")
    print(f"  60/40:           ${portfolio['benchmark_6040_eq'].iloc[-1]:.2f}")
    print(f"  Equal Weight:    ${portfolio['benchmark_equal_eq'].iloc[-1]:.2f}")
    print(f"  Buy & Hold:      ${portfolio['benchmark_spy_eq'].iloc[-1]:.2f}")
    print(f"\nBenchmark Comparison (Sharpe):")
    print(f"  Strategy (mean): {seed_df['sharpe'].mean():.3f}")
    print(f"  60/40:           {tear_sheet.loc['Sharpe Ratio', '60/40 Benchmark']:.3f}")
    print(f"  Buy & Hold:      {tear_sheet.loc['Sharpe Ratio', 'Buy & Hold SPY']:.3f}")
    return raw, features, regimes, portfolio, weight_history, tear_sheet, seed_df
if __name__=="__main__":
    raw, features, regimes, portfolio, weights, tear_sheet, seed_df=run_pipeline(n_seeds=10)
    print(f"\nRegime distribution:\n{regimes.value_counts()}")
