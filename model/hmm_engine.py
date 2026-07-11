import numpy as np
import pandas as pd
from hmmlearn import hmm
import warnings
warnings.filterwarnings("ignore")
class RegimeHMM:
    def __init__(self, n_states=3, n_iter=100, random_state=42):
        self.n_states=n_states
        self.n_iter=n_iter
        self.random_state=random_state
        self.model=None
        self.is_fitted = False
    def fit(self, features:np.ndarray):
        self.model=hmm.GaussianHMM(n_components=self.n_states, covariance_type="full",n_iter=self.n_iter,random_state=self.random_state)
        self.model.fit(features)
        self.is_fitted=True
        print(f"HMM fitted-log likelihood: {self.model.score(features):.2f}")
        return self
    def predict(self, features: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.model.predict(features)
    def score(self, features:np.ndarray) -> float:
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.model.score(features)
    def get_transition_matrix(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return pd.DataFrame(self.model.transmat_, columns=[f"state_{i}" for i in range(self.n_states)], index=[f"state_{i}" for i in range(self.n_states)])
    def get_state_means(self)->pd.DataFrame:
        if not self.is_fitted:
            raise  ValueError("Model not fitted yet. Call fit() first.")
        return pd.DataFrame(self.model.means_,columns=["returns","vix_norm","yield_spread","cpi_change"],index=[f"state_{i}" for i in range(self.n_states)])
if __name__=="__main__":
    from data.preprocessor import get_hmm_features

    features_df=pd.read_csv("data/raw/features.csv",index_col="Date",parse_dates=True)
    features=get_hmm_features(features_df)
    engine=RegimeHMM(n_states=3)
    engine.fit(features)
    states=engine.predict(features)
    print(f"\nUnique states found: {np.unique(states)}")
    print(f"\nState counts:\n{pd.Series(states).value_counts().sort_index()}")
    print(f"\nTransition Matrix:")
    print(engine.get_transition_matrix().round(4))
    print(f"\nState Means:")
    print(engine.get_state_means().round(6))
