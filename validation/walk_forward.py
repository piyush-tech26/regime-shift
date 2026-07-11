import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.hmm_engine import RegimeHMM
from model.regime_labeler import label_regimes, apply_labels
from data.preprocessor import get_hmm_features
def walk_forward_validation(features_df: pd.DataFrame, train_years: int=7, test_years: int=1):
    results=[]
    dates=features_df.index
    total_days=len(features_df)
    train_days=train_years*252
    test_days=test_years*252
    print(f"Starting walk-forward validation..")
    print(f"Train window: {train_years} years ({train_days} days)")
    print(f"Test window: {test_years} year ({test_days} days)")
    print(f"Total data: {total_days} days\n")
    window_num=1
    start=0
    while start+train_days+test_days<=total_days:
        train_end=start+train_days
        test_end=train_end+test_days
        train_data=features_df.iloc[start:train_end]
        test_data=features_df.iloc[train_end:test_end]
        train_start_date=dates[start].date()
        train_end_date=dates[train_end-1].date()
        test_start_date=dates[train_end].date()
        test_end_date=dates[test_end-1].date()
        print(f"Window {window_num}:")
        print(f"    Train: {train_start_date} -> {train_end_date}")
        print(f"    Train: {test_start_date} -> {test_end_date}")
        engine=RegimeHMM(n_states=3)
        engine.fit(get_hmm_features(train_data))
        labels=label_regimes(engine, get_hmm_features(train_data))
        test_states=engine.predict(get_hmm_features(test_data))
        test_regimes=apply_labels(test_states, labels)
        test_regimes.index=test_data.index
        test_score=engine.score(get_hmm_features(test_data))
        print(f"    Test log-likelihood: {test_score:.2f}")
        print(f"    Regime distribution: {test_regimes.value_counts().to_dict()}\n")
        results.append({"window": window_num, "train_start": train_start_date, "train_end": train_end_date, "test_start": test_start_date, "test_end": test_end_date, "test_score": test_score, "regimes":test_regimes})
        start+=test_days
        window_num+=1
    print(f"Walk-forward complete - {len(results)} windows")
    return results
def combine_results(results: list) -> pd.Series:
    all_regimes=pd.concat([r["regimes"] for r in results])
    all_regimes=all_regimes[~all_regimes.index.duplicated(keep="first")]
    alll_regimes=all_regimes.sort_index()
    return all_regimes
if __name__=="__main__":
    features_df= pd.read_csv("data/raw/features.csv",index_col="Date",parse_dates=True)
    results=walk_forward_validation(features_df,train_years=7, test_years=1)
    combined=combine_results(results)
    print(f"Combined regime series: {len(combined)} days")
    print(f"\nOverall regime distribution:")
    print(combined.value_counts())
    combined.to_csv("data/raw/wf_regimes.csv",header=["regime"])
    print("\nSaved to data/raw/wf_regimes.csv")
