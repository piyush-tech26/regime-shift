import numpy as np
import pandas as pd
from model.hmm_engine import RegimeHMM
def label_regimes(engine: RegimeHMM, features: np.ndarray) -> dict:
    """
    Labeling states using the rules below:
    - Bull - highest returns
    - Crisis - highest VIX
    - Bear - everything else
    """
    state_means=engine.get_state_means()
    bull_state=state_means["returns"].idxmax()
    remaining=state_means.drop(bull_state)
    crisis_state=remaining["vix_norm"].idxmax()
    bear_state=remaining.drop(crisis_state).index[0]
    sorted_states=state_means["returns"].sort_values(ascending=False)
    labels={int(bull_state.split("_")[1]):"BULL",int(bear_state.split("_")[1]):"BEAR",int(crisis_state.split("_")[1]):"CRISIS"}
    print("\nRegime Labels Assigned:")
    for state_num, regime in sorted(labels.items()):
        means=state_means.loc[f"state_{state_num}"]
        print(f" state_{state_num} -> {regime} | returns {means['returns']:.6f} |vix: {means['vix_norm']:.4f}")
    return labels
def apply_labels(state: np.ndarray, labels: dict) -> pd.Series:
    return pd.Series(state).map(labels)
if __name__ == "__main__":
    from data.preprocessor import get_hmm_features
    features_df = pd.read_csv("data/raw/features.csv",index_col="Date",parse_dates=True)
    features=get_hmm_features(features_df)
    engine=RegimeHMM(n_states=3)
    engine.fit(features)
    states=engine.predict(features)
    labels=label_regimes(engine, features)
    regime_series=apply_labels(states, labels)
    regime_series.index=features_df.index
    print(f"\nRegime distribution:")
    print(regime_series.value_counts())
    print(f"\nFirst 10 days:")
    print(pd.DataFrame({"sp500_returns": features_df["returns"],"vix_norm":features_df["vix_norm"],"regime":regime_series}).head(10))
    regime_series.to_csv("data/raw/regimes.csv",header=["regime"])
    print("\nRegimes saved to data/raw/regimes.csv")
