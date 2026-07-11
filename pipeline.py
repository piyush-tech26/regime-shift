import os
import numpy as np
from scipy import stats
from data.fetcher import fetch_all
from data.preprocessor import preprocess, get_asset_returns
from model.hmm_engine import RegimeHMM
from model.regime_labeler import label_regimes, apply_labels
from validation.walk_forward import walk_forward_validation, combine_results
from backtest.engine import run_backtest
from metrics.tear_sheet import compute_tear_sheet
DIR="data/raw"
def smooth_regimes(series, window=10):
    regime_map = {"BULL": 0, "BEAR": 1, "CRISIS": 2}
    reverse_map = {0: "BULL", 1: "BEAR", 2: "CRISIS"}
    numeric = series.map(regime_map)
    smoothed = numeric.rolling(window=window, center=True, min_periods=1).apply(lambda x: stats.mode(x, keepdims=True)[0][0], raw=True)
    return smoothed.map(reverse_map)
def pipeline_exists() -> bool:
    required=[f"{DIR}/market_data.csv",f"{DIR}/features.csv",f"{DIR}/wf_regimes.csv",f"{DIR}/portfolio.csv",f"{DIR}/weights.csv",f"{DIR}/tear_sheet.csv"]
    return all(os.path.exists(f) for f in required)

def run_pipeline(train_years=7,test_years=1):
    os.makedirs(DIR,exist_ok=True)
    print("Step 1:  Fetching market data...")
    raw=fetch_all()
    raw.to_csv(f"{DIR}/market_data.csv")
    print("Step 2: Preprocessing features...")
    features=preprocess(raw)
    features.to_csv(f"{DIR}/features.csv")
    print("Step 3: Running walk_forward validation...")
    results=walk_forward_validation(features, train_years=train_years, test_years=test_years)
    regimes=combine_results(results)
    regimes = smooth_regimes(regimes)
    regimes.to_csv(f"{DIR}/wf_regimes.csv", header=["regime"])
    print("Step 4: Running optimizer backtest...")
    asset_returns=get_asset_returns(features)
    portfolio,weight_history=run_backtest(asset_returns=asset_returns, regimes=regimes,lookback_days=252,rebalance_freq=21,transaction_bps=10.0)
    portfolio.to_csv(f"{DIR}/portfolio.csv")
    weight_history.to_csv(f"{DIR}/weights.csv")
    print("Step 5: Computing performance metrics...")
    tear_sheet=compute_tear_sheet(portfolio,weight_history)
    tear_sheet.to_csv(f"{DIR}/tear_sheet.csv")
    print("\nPipeline Complete")
    print(f"\nFinal Portfolio Values ($1 invested):")
    print(f"  Strategy:     ${portfolio['strategy_equity'].iloc[-1]:.2f}")
    print(f"  60/40:        ${portfolio['benchmark_6040_eq'].iloc[-1]:.2f}")
    print(f"  Equal Weight: ${portfolio['benchmark_equal_eq'].iloc[-1]:.2f}")
    print(f"  Buy & Hold:   ${portfolio['benchmark_spy_eq'].iloc[-1]:.2f}")
    print(f"\nRisk-Adjusted Performance (Sharpe):")
    print(f"  Strategy:     {tear_sheet.loc['Sharpe Ratio', 'Strategy']:.3f}")
    print(f"  60/40:        {tear_sheet.loc['Sharpe Ratio', '60/40 Benchmark']:.3f}")
    print(f"  Buy & Hold:   {tear_sheet.loc['Sharpe Ratio', 'Buy & Hold SPY']:.3f}")
    return raw,features,regimes,portfolio,weight_history,tear_sheet
if __name__=="__main__":
    raw, features, regimes, portfolio, weights, tear_sheet=run_pipeline()
    print(f"\nRegime distribution:\n{regimes.value_counts()}")
