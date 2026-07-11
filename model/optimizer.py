import numpy as np
import pandas as pd
import cvxpy as cp
import warnings
warnings.filterwarnings("ignore")
class RegimeOptimizer:
    """
    Regime-aware portfolio optimizer using CVXPY.
    Different objective functions per regime:
    - BULL:   Maximum Sharpe ratio (aggressive)
    - BEAR:   Minimum Variance (defensive)
    - CRISIS: Minimum CVaR (capital preservation)
    """
    def __init__(self, max_weight: float = 0.80, risk_free: float = 0.04):
        self.max_weight = max_weight
        self.risk_free_daily = risk_free / 252
    def _base_constraints(self, w, n_assets):
        """Constraints common to all regimes."""
        return [
            cp.sum(w) == 1,           # fully invested
            w >= 0,                   # no shorts
            w <= self.max_weight      # concentration limit
        ]
    def optimize_bull(self, mean_returns: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        """
        Maximum Sharpe ratio (approximated as maximum return / risk).
        We maximize the tangency portfolio using a linear objective transform.
        """
        n = len(mean_returns)
        w = cp.Variable(n)
        # Maximize excess return per unit of risk (approximate Sharpe)
        # Use quadratic utility: maximize μ'w - λ w'Σw
        # λ is a risk aversion parameter (small λ = aggressive)
        risk_aversion = 1.0
        excess_returns = mean_returns - self.risk_free_daily
        objective = cp.Maximize(
            excess_returns @ w - risk_aversion * cp.quad_form(w, cp.psd_wrap(cov_matrix))
        )
        constraints = self._base_constraints(w, n)
        problem = cp.Problem(objective, constraints)
        problem.solve()
        return np.array(w.value).flatten()
    def optimize_bear(self, mean_returns: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        """Minimum variance portfolio."""
        n = len(mean_returns)
        w = cp.Variable(n)
        objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov_matrix)))
        constraints = self._base_constraints(w, n)
        problem = cp.Problem(objective, constraints)
        problem.solve()
        return np.array(w.value).flatten()
    def optimize_crisis(self, returns_history: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        """
        Minimum CVaR — minimize expected loss in the worst alpha% of scenarios.
        Uses historical returns directly (not mean/cov) for tail-risk protection.
        """
        n_days, n_assets = returns_history.shape
        w = cp.Variable(n_assets)
        var = cp.Variable()                    # Value at Risk (auxiliary variable)
        aux = cp.Variable(n_days, nonneg=True) # tail losses
        portfolio_returns = returns_history @ w
        constraints = self._base_constraints(w, n_assets)
        constraints += [
            aux >= -portfolio_returns - var
        ]
        cvar = var + (1 / (alpha * n_days)) * cp.sum(aux)
        objective = cp.Minimize(cvar)
        problem = cp.Problem(objective, constraints)
        problem.solve()
        return np.array(w.value).flatten()
    def optimize(self, regime: str, returns_history: pd.DataFrame) -> dict:
        """
        Main entry point — routes to the right optimizer based on regime.
        returns_history: DataFrame with columns [spy_return, tlt_return, gld_return]
        """
        returns_arr = returns_history.values
        mean_returns = returns_arr.mean(axis=0)
        cov_matrix = np.cov(returns_arr, rowvar=False)
        if regime == "BULL":
            weights = self.optimize_bull(mean_returns, cov_matrix)
        elif regime == "BEAR":
            weights = self.optimize_bear(mean_returns, cov_matrix)
        elif regime == "CRISIS":
            weights = self.optimize_crisis(returns_arr)
        else:
            # Equal weight fallback
            weights = np.ones(len(mean_returns)) / len(mean_returns)
        # Clean up numerical noise
        weights = np.clip(weights, 0, self.max_weight)
        weights = weights / weights.sum()
        return {
            "regime": regime,
            "weights": {
                col.replace("_return", ""): float(w)
                for col, w in zip(returns_history.columns, weights)
            },
            "raw_weights": weights
        }

if __name__ == "__main__":
    from data.preprocessor import get_asset_returns
    features_df = pd.read_csv(
        "data/raw/features.csv",
        index_col="Date",
        parse_dates=True
    )
    asset_returns = get_asset_returns(features_df)
    optimizer = RegimeOptimizer()
    print("Testing optimizer on last 252 days (1 year lookback):")
    lookback = asset_returns.tail(252)
    for regime in ["BULL", "BEAR", "CRISIS"]:
        result = optimizer.optimize(regime, lookback)
        print(f"\n{regime}:")
        for asset, weight in result["weights"].items():
            print(f"  {asset.upper()}: {weight:.2%}")
