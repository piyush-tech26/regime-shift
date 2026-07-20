import numpy as np
import pandas as pd
def sharpe_ratio(returns: pd.Series,risk_free=0.02)-> float:
    excess = returns-risk_free/252
    if excess.std()==0:
        return np.nan
    return float(np.sqrt(252)*excess.mean()/excess.std())
def sortino_ratio(returns: pd.Series, risk_free=0.02)-> float:
    excess=returns-risk_free/252
    downside=excess[excess < 0].std()
    if downside == 0 or np.isnan(downside):
        return np.nan
    return float(np.sqrt(252)*excess.mean()/downside)
def max_drawdown(returns: pd.Series)->float:
    cumulative =(1+returns).cumprod()
    rolling_max= cumulative.cummax()
    drawdown =(cumulative-rolling_max)/rolling_max
    return float(drawdown.min())
def calmar_ratio(returns: pd.Series)->float:
    annual_return=(1+returns.mean())**252-1
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return np.nan
    return float(annual_return/mdd)
def annualised_return(returns: pd.Series) -> float:
    return float((1+returns.mean())**252-1)
def annualised_vol(returns: pd.Series) -> float:
    return float(returns.std()*np.sqrt(252))

def compute_tear_sheet(portfolio: pd.DataFrame, weights_history: pd.DataFrame=None)->pd.DataFrame:
    columns_to_analyze = {"Strategy": portfolio["strategy_returns"],
                          "60/40 Benchmark":portfolio["benchmark_6040"],
                          "Equal Weight": portfolio["benchmark_equal"],
                          "Buy & Hold SPY": portfolio["benchmark_spy"]}
    metrics = {}
    for name, returns in columns_to_analyze.items():
        metrics[name]={"Annual Return (%)":round(annualised_return(returns)*100,2),"Annual Volatility (%)": round(annualised_vol(returns)*100,2),"Sharpe Ratio": round(sharpe_ratio(returns),4), "Sortino Ratio":round(sortino_ratio(returns), 4), "Max Drawdown (%)":round(max_drawdown(returns)*100,2),"Calmar Ratio": round(calmar_ratio(returns),4)}
    tear=pd.DataFrame(metrics)
    if "transaction_cost" in portfolio.columns:
        total_cost=portfolio["transaction_cost"].sum()
        avg_annual_cost=total_cost/(len(portfolio)/252)
        tear.loc["Total Transaction Cost (%)"]=[round(total_cost*100,2),0.0,0.0,0.0]
        tear.loc["Avg Annual Drag (%)"]=[round(avg_annual_cost*100,2),0.0,0.0,0.0]
        # Turnover analysis
        if weights_history is not None and len(weights_history) > 0:
            tear.loc["Avg Turnover per Rebalance (%)"] = [round(weights_history["turnover"].mean() * 100, 2), 0.0, 0.0, 0.0]
            tear.loc["Rebalances per Year"] = [round(len(weights_history) / (len(portfolio) / 252), 1), 0.0, 0.0, 0.0]
    return tear
if __name__ == "__main__":
    portfolio=pd.read_csv("data/raw/portfolio.csv", index_col=0,parse_dates=True)
    weights_history=pd.read_csv("data/raw/weights.csv",index_col=0, parse_dates=True)
    print("Computing tear sheet...\n")
    tear= compute_tear_sheet(portfolio,weights_history)
    print(tear.to_string())
    tear.to_csv("data/raw/tear_sheet.csv")
    print("\nSaved to data/raw/tear_sheet.csv")
