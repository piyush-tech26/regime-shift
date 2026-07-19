import pandas as pd
import numpy as np

ASSETS = ["spy", "tlt", "gld"]
def expanding_vix_norm(vix_series: pd.Series, warmup_days: int = 252) -> pd.Series:
    """
    Normalize VIX using expanding window min/max.
    First `warmup_days` use raw VIX scaled by initial window stats.
    """
    vix_norm = pd.Series(index=vix_series.index, dtype=float)
    running_min = vix_series.iloc[:warmup_days].min()
    running_max = vix_series.iloc[:warmup_days].max()
    for i, (date, vix) in enumerate(vix_series.items()):
        if i < warmup_days:
            # Use first year's stats for warmup period
            vix_norm[date] = np.nan
        else:
            # Update running min/max with today's value
            running_min = min(running_min, vix)
            running_max = max(running_max, vix)
            vix_norm[date] = (vix - running_min) / (running_max - running_min + 1e-8)
    return vix_norm
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares data for the HMM and optimizer.
    - HMM features: SPY returns, VIX, yield_spread, CPI change
    - Asset returns: SPY, TLT, GLD returns for the optimizer
    """
    features = pd.DataFrame(index=df.index)
    features["returns"] = np.log(df["spy"] / df["spy"].shift(1))
    features["vix_norm"] =expanding_vix_norm( df["vix"])
    features["yield_spread"] = df["yield_spread"]
    features["cpi_change"] = df["cpi"].pct_change()
    for asset in ASSETS:
        features[f"{asset}_return"] = np.log(df[asset] / df[asset].shift(1))
    features = features.dropna()
    print(f"Preprocessed: {len(features)} rows, {features.shape[1]} features")
    print(f"HMM features: returns, vix_norm, yield_spread, cpi_change")
    print(f"Asset returns: {[f'{a}_return' for a in ASSETS]}")
    print(features.describe().round(4))
    return features
def get_hmm_features(features: pd.DataFrame) -> np.ndarray:
    """Extracting only the columns the HMM should see."""
    hmm_cols = ["returns", "vix_norm", "yield_spread", "cpi_change"]
    return features[hmm_cols].values

def get_asset_returns(features: pd.DataFrame) -> pd.DataFrame:
    """Extracting only the asset return columns for the optimizer."""
    return_cols = [f"{a}_return" for a in ASSETS]
    return features[return_cols]

if __name__ == "__main__":
    raw = pd.read_csv("data/raw/market_data.csv", index_col="Date", parse_dates=True)
    features = preprocess(raw)
    features.to_csv("data/raw/features.csv")
    print("\nFirst 5 rows:")
    print(features.head())
