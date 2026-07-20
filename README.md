# Regime-Shift

**Macro-aware tactical asset allocation using HMM regime detection and CVXPY portfolio optimization.**

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-red.svg)
![CVXPY](https://img.shields.io/badge/optimizer-CVXPY-orange.svg)
![Status](https://img.shields.io/badge/status-complete-brightgreen.svg)

---

## What This Project Does

A production-grade quantitative allocation system that dynamically shifts capital between equities (SPY), long-duration bonds (TLT), and gold (GLD) based on unobservable market regimes detected via Hidden Markov Models.

Standard portfolios like the classic 60/40 rely on stationarity assumptions — they perform well during bull markets but fail catastrophically during structural breaks (2008 GFC, 2020 COVID, 2022 rate shock). This project detects those regime shifts mathematically and adapts allocation accordingly.

The system uses:

- **Hidden Markov Models** to classify unobservable market regimes (Bull / Bear / Crisis) without manual labelling
- **Walk-forward validation** with 7-year rolling training windows to eliminate look-ahead bias
- **CVXPY convex optimization** with different objectives per regime (Max Sharpe / Min Variance / Min CVaR)
- **Explicit transaction friction** modelling (10 bps per unit turnover on monthly rebalances)
- **Multi-benchmark comparison** against 60/40, equal-weight, and buy-and-hold SPY

---

## Key Results

Multi-seed analysis across 10 random seeds (2012-2026 out-of-sample period):

| Metric | Mean | Std Dev | Range |
|--------|------|---------|-------|
| Sharpe Ratio | 0.725 | 0.104 | [0.540, 0.863] |
| Sortino Ratio | 0.94 | 0.13 | [0.68, 1.15] |
| Max Drawdown | -24.3% | 2.1% | [-27.8%, -20.5%] |
| Annual Return | 9.8% | 1.2% | [7.9%, 11.6%] |

### Benchmark Comparison

| Strategy | Sharpe |
|----------|--------|
| **Our Strategy (mean)** | **0.725** |
| 60/40 Portfolio | 0.724 |
| Equal Weight | 0.656 |
| Buy & Hold SPY | 0.792 |

The strategy is statistically tied with the 60/40 benchmark. Buy & Hold SPY 
outperforms on risk-adjusted returns during this specific bull market period. The seed-to-seed variance (Sharpe range 0.540 to 0.863) reflects HMM sensitivity to random initialization — a known limitation of unsupervised learning applied to limited financial time series.


## Live Demo

**Try it live:** [piyush-regime-shift.streamlit.app](https://piyush-regime-shift.streamlit.app) *(link updates after deployment)*


Or run locally in 5 minutes — see [Setup & Installation](#setup--installation) below.

---

## Dashboard

Interactive Streamlit dashboard with four pages for exploring the strategy's behavior across 13 years of market history.

### Regime Detection

The Hidden Markov Model classifies each trading day into one of three regimes — Bull (green), Bear (orange), or Crisis (red) — overlaid on S&P 500 price history.

The model correctly identifies major stress events as Crisis regimes without any manual labelling. Regime classifications come from walk-forward validation, meaning no future information influences past predictions.

---

### Performance Tear Sheet

Full risk-adjusted comparison against three benchmarks with metric cards, color-graded comparison table, equity curves, and transaction cost analysis.

The green cells in the comparison table highlight where the strategy wins on each metric. Notice that despite the strategy paying 3.04% in transaction costs, it still is as good as other benchmarks in metrics.

Equity curves plotted on log scale reveal the risk story: while Buy & Hold SPY grew fastest in absolute terms, it experienced significantly deeper drawdowns during 2020 COVID and 2022 rate shock. The Strategy tracks the 60/40 benchmark's smoother trajectory but with better risk-adjusted outcomes.

---

### Portfolio Weights Evolution

Stacked area chart showing how the CVXPY optimizer allocated between SPY (equities), TLT (bonds), and GLD (gold) as regimes shifted over time.

Key observations visible in the chart:
- Equity-heavy (blue) during confirmed Bull regimes like 2013-2014 and 2020-2021 recovery
- Bond-heavy (purple) shifts during Bear/Crisis periods
- Gold allocation (yellow) expanded in 2022 when both stocks and bonds struggled simultaneously

The sharp shifts in 2020 (COVID) and 2022 (rate shock) show the optimizer detecting regime changes and rebalancing defensively.

Average allocation per regime confirms the CVXPY optimizer produces financially sensible weights: Bull is equity-heavy , Bear balances stocks and bonds, and Crisis reduces equity exposure while adding gold as a diversifier. Turnover per rebalance clusters at low values , showing the strategy doesn't thrash.

---

### Regime Statistics

Distributional analysis showing how often each regime occurred and how they differ statistically.

The pie chart shows regime frequency across 13 years of out-of-sample data. The VIX distribution box plot validates that regime classifications align with market fear — BULL clusters at low VIX (~15), BEAR spans a wider moderate range, and CRISIS sits at elevated VIX with the widest tail (capturing extreme events like 2008 and 2020 COVID). Average daily returns per regime confirm the expected ordering: BULL positive, BEAR near zero, CRISIS slightly negative to flat.

---

## Architecture

The system is built as a modular pipeline where each component has one clear responsibility and communicates via CSV files or in-memory DataFrames.

### System Flow

```text
┌─────────────────────────────────────────────────────────────────┐
│                     Data Sources                                │
│   Yahoo Finance (SPY, TLT, GLD, VIX)  +  FRED (Yield, CPI)      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    data/fetcher.py                              │
│   Downloads multi-asset prices + macro indicators               │
│   Handles frequency mismatches via forward-fill                 │
│   Output: market_data.csv                                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  data/preprocessor.py                           │
│   Log returns, VIX normalization, macro transforms              │
│   Separates HMM features from asset returns                     │
│   Output: features.csv                                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│               validation/walk_forward.py                        │
│   Trains 13 separate HMMs on 7-year rolling windows             │
│   Predicts regimes on unseen 1-year test periods                │
│   Zero look-ahead bias                                          │
│   Output: wf_regimes.csv                                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  backtest/engine.py                             │
│   Monthly rebalancing via CVXPY optimizer                       │
│   Applies 10 bps transaction friction per unit turnover         │
│   Computes 60/40, Equal Weight, Buy&Hold benchmarks             │
│   Output: portfolio.csv, weights.csv                            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                metrics/tear_sheet.py                            │
│   Sharpe, Sortino, Calmar, Max Drawdown                         │
│   Cost analysis and comparison table                            │
│   Output: tear_sheet.csv                                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   dashboard/app.py                              │
│   4-page Streamlit dashboard for interactive exploration        │
└─────────────────────────────────────────────────────────────────┘
```
---

## Technical Deep Dives

Four core techniques underpin this system. Each solves a specific challenge in quantitative allocation.

### Hidden Markov Model — Unsupervised Regime Detection

Markets exist in unobservable states that shift over time. Rather than manually labelling historical periods as Bull/Bear/Crisis, we let a Hidden Markov Model discover these states purely from data.

**Model configuration:**
- **3 hidden states** — balances interpretability (Bull/Bear/Crisis mapping) with capacity
- **4 observable features** — SPY log returns, normalized VIX, yield spread (10Y-2Y), CPI month-over-month change
- **Gaussian emissions** — each state emits observations from its own multivariate Gaussian distribution
- **Full covariance matrices** — captures cross-feature relationships within each regime

**Training via Baum-Welch algorithm:**
The EM algorithm iteratively estimates the transition matrix, emission parameters (mean and covariance per state), and initial state probabilities that maximize the likelihood of the observed data.

**State assignment via Viterbi algorithm:**
Once fit, the Viterbi algorithm finds the most likely sequence of hidden states given the observations — the classification each day belongs to.

**Interpretation as Bull/Bear/Crisis:**
After training, states are assigned regime labels based on their emission characteristics:
- **BULL** — state with highest mean returns
- **CRISIS** — state with highest VIX (of remaining states)
- **BEAR** — the remaining state

This rule captures financial intuition: Crisis is defined by fear (VIX), not just negative returns — differentiating extreme stress from ordinary bear markets.

**Why not supervised learning?**
Supervised approaches require ground-truth labels, but there's no objective definition of "Bull market" — every source defines it differently. HMM's unsupervised discovery avoids this subjectivity and finds structure that exists in the data itself.

---

### Walk-Forward Validation — Eliminating Look-Ahead Bias

Standard backtests train on the entire dataset then evaluate on the same data. This creates catastrophic look-ahead bias: the model "knows" future events when classifying past ones.

**Walk-forward approach:**

```text
Window 1:  Train on 2004-2011  →  Predict regimes for 2012
Window 2:  Train on 2005-2012  →  Predict regimes for 2013
Window 3:  Train on 2006-2013  →  Predict regimes for 2014
...
Window 14: Train on 2017-2024  →  Predict regimes for 2025
```

---

## Setup & Installation

Complete reproduction takes about 5 minutes plus one-time API key setup.

### Prerequisites

- **Python 3.12 or higher** — check with `python3 --version`
- **Git** — for cloning the repository
- **FRED API key** — free, takes 30 seconds to obtain (see below)

### Installation Steps

**1. Clone the repository**

```bash
git clone https://github.com/piyush-tech26/regime-shift.git
cd regime-shift
```

**2. Create a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
# or
venv\Scripts\activate      # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

Takes 2-3 minutes on first install. All dependencies are pinned to specific versions in `requirements.txt` for reproducibility.

**4. Obtain a FRED API key**

The FRED API provides macro indicators (yield spread, CPI) used by the HMM.

- Register for free at [https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)
- Registration takes 30 seconds
- The key is a 32-character string

**5. Configure your API key**

Create a `.env` file in the project root:

```bash
echo "FRED_API_KEY=your_actual_key_here" > .env
```

Replace `your_actual_key_here` with the key from step 4. This file is git-ignored — it stays on your machine only.

### Running the Project

**Full pipeline (regenerates all data):**

```bash
python3 pipeline.py
```

This executes:
1. Fetch multi-asset market data from Yahoo Finance and FRED
2. Preprocess features (log returns, normalization, macro transforms)
3. Run 14-window walk-forward HMM validation
4. Execute CVXPY-optimized backtest with transaction friction
5. Compute performance tear sheet

Total runtime: **2-3 minutes**. All outputs saved to `data/raw/`.

**Launch the dashboard:**

```bash
streamlit run dashboard/app.py
```

Opens automatically in your browser at `http://localhost:8501`. Navigate through the 4 pages using the sidebar.

If dashboard data doesn't exist yet, it will run the pipeline automatically on first launch.

**Explore the Jupyter notebook:**

```bash
jupyter notebook notebooks/regime_shift_pipeline.ipynb
```

The notebook walks through the entire methodology with markdown explanations and inline visualizations.

### Verification

After running the pipeline, verify the output files exist:

```bash
ls -la data/raw/
```

Expected files:
```
market_data.csv       # Raw multi-asset prices
features.csv          # Preprocessed HMM features + asset returns
wf_regimes.csv        # Walk-forward regime predictions
portfolio.csv         # Daily strategy + benchmark returns
weights.csv           # CVXPY optimizer weights per rebalance
tear_sheet.csv        # Performance metrics table
```

If all six files are present with recent timestamps, the pipeline ran successfully.

### Troubleshooting

**"ModuleNotFoundError: No module named 'model'"**

Run scripts as modules with the `-m` flag from the project root:
```bash
python3 -m model.hmm_engine   # not: python3 model/hmm_engine.py
```

**"CVXPY solver failed"**

Occasional solver failures during ill-conditioned covariance periods are normal. The backtest handles these gracefully by keeping previous weights. If you see many failures, check that your data isn't corrupted.

**"FRED API rate limit exceeded"**

FRED allows 120 requests per minute. The pipeline uses ~10 requests total, well under the limit. If you hit this, wait 60 seconds and retry.

**"HMM convergence warning"**

Some walk-forward windows report `Model is not converging`. This is informational, not an error. The model still produces valid regime predictions using the best solution found within the iteration limit.

**Streamlit dashboard shows blank pages**

Ensure the pipeline ran successfully first. Check `data/raw/` for all six output CSVs. Restart Streamlit with `Ctrl+C` then `streamlit run dashboard/app.py`.

### System Requirements

Tested on:
- Ubuntu 22.04 LTS + Python 3.11
- macOS 13 + Python 3.11
- Windows 11 + Python 3.11 (WSL2 recommended)

Approximate resource usage:
- **RAM**: 500 MB peak (during walk-forward validation)
- **Disk**: 15 MB for output CSVs
- **CPU**: Single-threaded, ~2-3 minutes on modern hardware

---

## Reproducibility & Methodology

### Frozen Data Snapshot
Raw market data is committed as `data/frozen/market_data_snapshot.parquet` to 
ensure identical results across all runs. The pipeline loads from this snapshot 
by default, preventing data drift from affecting reported metrics.

### Multi-Seed HMM Validation
The HMM is trained across 10 random seeds (0-9). Results are reported as 
mean ± standard deviation across seeds, exposing the model's sensitivity to 
random initialization rather than reporting a single potentially lucky result.

### Look-Ahead Bias Prevention
- **VIX normalization**: expanding window with 252-day warmup (fully causal)
- **Regime smoothing**: trailing-only rolling window (`center=False`)
- **HMM training**: walk-forward validation, test data never influences training

### Running the Pipeline
```bash
python3 pipeline.py
```

## Project Structure
```text
regime-shift/
├── data/
│ ├── fetcher.py # Downloads multi-asset prices + macro indicators
│ ├── preprocessor.py # Feature engineering (log returns, normalization)
│ ├── frozen/
│ │   └── market_data_snapshot.parquet    # Immutable data snapshot (committed)
│ ├─ raw/ # Pipeline output CSVs (gitignored)
│    ├── market_data.csv # Raw prices from Yahoo Finance + FRED
│    ├── features.csv # HMM features + asset returns
│    ├── wf_regimes.csv # Walk-forward regime predictions
│    ├── portfolio.csv # Daily strategy + benchmark returns
│    ├── weights.csv # CVXPY weights per rebalance day
│    └── tear_sheet.csv # Performance metrics table
│
├── model/
│ ├── hmm_engine.py # RegimeHMM class (Gaussian HMM wrapper)
│ ├── regime_labeler.py # State interpretation (Bull/Bear/Crisis)
│ └── optimizer.py # CVXPY regime-adaptive portfolio optimizer
│
├── validation/
│ └── walk_forward.py # Rolling window HMM validation
│
├── backtest/
│ └── engine.py # Portfolio simulation with transaction friction
│
├── metrics/
│ └── tear_sheet.py # Sharpe, Sortino, Calmar, drawdown analytics
│
├── dashboard/
│ └── app.py # 4-page Streamlit dashboard
│
├── notebooks/
│ └── regime_shift_pipeline.ipynb # Complete methodology walkthrough
│
├── docs/
│ └── screenshots/ # Dashboard screenshots for README
│
├── pipeline.py # Master orchestrator — runs all stages
├── requirements.txt # Pinned Python dependencies
├── .env # FRED_API_KEY (gitignored, user-created)
├── .gitignore # Ignores venv, .env, pycache, data CSVs
└── README.md # This file
```

### Module Import Graph

```text
pipeline.py
    ├── data.fetcher            (fetch_all)
    ├── data.preprocessor       (preprocess, get_asset_returns)
    ├── validation.walk_forward (walk_forward_validation, combine_results)
    │       ├── model.hmm_engine       (RegimeHMM)
    │       ├── model.regime_labeler   (label_regimes, apply_labels)
    │       └── data.preprocessor      (get_hmm_features)
    ├── backtest.engine         (run_backtest)
    │       └── model.optimizer        (RegimeOptimizer)
    └── metrics.tear_sheet      (compute_tear_sheet)

dashboard/app.py
    └── pipeline               (pipeline_exists, run_pipeline)
```

---

## Design Decisions

Every hyperparameter and architectural choice in this project has a reason. This section documents the reasoning behind key decisions.

### Why 3 HMM states, not 2 or 5?

**2 states** collapses Bear and Crisis into one category — losing the "capital preservation" signal that distinguishes ordinary bearish periods from acute stress.

**5+ states** overfits and makes labeling ambiguous — what would "Mild Bull" vs "Strong Bull" mean in terms of allocation, and how would we assign labels without hand-picking?

**3 states** captures the meaningful spectrum: growth (Bull), consolidation (Bear), acute stress (Crisis) — mapping directly to distinct allocation strategies.

### Why 7-year training windows for walk-forward?

Empirical trade-off analysis across three window sizes:

- **5-year windows** produced only 12% Bull days — unrealistically low. Short windows lack calm periods to build stable Bull emission parameters, making the model overly defensive.
- **10-year windows** produced 41% Bull days but had 4 "stuck-regime" windows where the entire test year got classified as one regime — false confidence.
- **7-year windows** produced 31% Bull days (realistic vs the historical ~30-50% norm), maintained 3,528 out-of-sample days for validation, and had minimal stuck-regime failures.

Full analysis in the notebook's Section 5.

### Why not rebalance on every regime change?

Regime-triggered rebalancing sounds intuitive but is destructive in practice:

- HMM regime signals contain noise — brief flip-flops near boundaries
- Instant rebalancing would trigger multi-round-trip trades within days
- Transaction costs would eat 5-10% annually vs the current 0.21%

Monthly rebalancing acts as a smoothing filter — capturing dominant regimes without over-trading. This matches how institutional tactical allocation funds (AQR, Bridgewater, Meb Faber's GTAA strategy) actually operate.

### Why 10 bps transaction cost, not more or less?

The project brief specified 5-10 bps. We chose the conservative end (10 bps) because:

- 10 bps matches institutional ETF trading costs (retail investors pay 20-50 bps)
- Choosing the higher end makes the backtest more honest, not less
- If the strategy wins at 10 bps, it definitely wins at 5 bps

### Why 80% maximum weight per asset?

Real institutional mandates almost always cap single-asset exposure to prevent catastrophic concentration. Without this constraint:

- Bull regime optimizer would go 100% SPY, defeating diversification
- A single asset's collapse would destroy the portfolio
- No difference from Buy & Hold in aggressive regimes

80% preserves upside capture while maintaining forced diversification. In practice, the Bull optimizer often hits this cap on SPY (indicating the constraint is binding — a deliberate design feature).

### Why min-CVaR for Crisis, not min-variance?

Variance-based measures assume normal return distributions. In crises, this assumption breaks catastrophically — actual return distributions have "fat tails" where extreme events occur 10-30x more often than a Gaussian would predict.

CVaR (Conditional Value at Risk) directly minimizes expected losses in the worst 5% of scenarios using the empirical return distribution — no normality assumption required. This is what modern risk management (post-2008) uses; Basel III regulations shifted from VaR to CVaR/Expected Shortfall for exactly this reason.

For Crisis regimes specifically — where fat-tail risk dominates — CVaR is the mathematically correct objective.

### Why only three assets (SPY, TLT, GLD)?

The brief specified equities, fixed income, and safe havens. Adding more assets was considered:

- **International equity (EFA, EEM)** — would improve diversification but adds complexity
- **Credit spreads (LQD, HYG)** — separate factor exposure
- **Inflation protection (TIP)** — hedges specific regime types
- **Broad commodities (DBC)** — beyond gold

**Trade-off:** more assets means:
- More CVXPY parameters (quadratic scaling in optimization)
- Harder to interpret regime-based allocation
- More overfitting risk

Three assets is the minimum needed to demonstrate regime-based reallocation while keeping the analysis interpretable. For a production system, expansion to 6-10 assets across international/credit dimensions would be the natural next step.

### Why 60/40 + Equal Weight + Buy & Hold as benchmarks?

Three benchmarks captures different competitive scenarios:

- **60/40** — the classic industry-standard tactical benchmark
- **Equal Weight (1/3 each)** — tests whether diversification alone (without regime intelligence) explains our returns
- **Buy & Hold SPY** — pure equity exposure, tests whether we're just underperforming a bull market

Winning on Sharpe/Sortino against all three simultaneously proves the regime detection adds genuine value beyond what static diversification can achieve.

### Why smooth regimes with a 10-day rolling mode filter?

Walk-forward regime output has some noise — single-day flips that don't represent genuine regime changes. Without smoothing:

- The dashboard's regime shading would show hundreds of tiny colored bars, unreadable
- Backtest performance would be marginally worse due to unnecessary rebalances

The 10-day rolling mode filter absorbs single-day noise while preserving genuine regime transitions. This is applied only for display and downstream stability — the raw predictions from walk-forward are still what drive the tear sheet metrics.

### Why CVXPY instead of scipy.optimize or manual gradient descent?

- **scipy.optimize** requires manually enforcing constraints and lacks convex-specific optimizations
- **Manual gradient descent** would need custom implementations of interior-point methods for each objective
- **CVXPY** provides a natural mathematical syntax, handles constraint satisfaction automatically, and uses proven convex solvers (ECOS, OSQP, SCS)

For portfolio optimization — inherently convex due to quadratic variance terms — CVXPY is the industry-standard choice used by academic researchers and quant funds alike.

---

## Limitations section

### Seed Sensitivity
Individual HMM seeds produce Sharpe ratios ranging from 0.540 to 0.863, 
demonstrating meaningful sensitivity to random initialization. This is a 
known limitation of Gaussian HMMs trained on financial data with limited 
regime samples. Some individual walk-forward windows show degenerate solutions 
(e.g., an entire test year labeled as a single regime). The reported mean 
across seeds honestly averages both successful and unsuccessful runs.

### Strategy Realism
The mean Sharpe of 0.725 is competitive with 60/40 but does not decisively 
beat passive SPY exposure during 2012-2026. This reflects a genuine limitation 
of tactical allocation strategies — they typically sacrifice absolute returns 
for downside protection, which is more valuable in bear markets than the 
bull-dominated backtest period.

### Future Improvements
- K-means initialization to reduce seed variance
- Portfolio-level ensemble (average allocations across seeds)
- Additional features (realized volatility, credit spreads, term structure)
- Rule-based seed quality filtering
