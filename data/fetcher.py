import yfinance as yf
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv
import os
load_dotenv()
def fetch_sp500(start="2000-01-01", end=None):
    """Fetching S&P 500 ETF (SPY) — equities."""
    spy = yf.download("SPY", start=start, end=end, auto_adjust=True)
    spy.columns = spy.columns.get_level_values(0)
    spy = spy[["Close"]].rename(columns={"Close": "spy"})
    return spy
def fetch_bonds(start="2000-01-01", end=None):
    """Fetching 20+ Year Treasury ETF (TLT) — fixed income."""
    tlt = yf.download("TLT", start=start, end=end, auto_adjust=True)
    tlt.columns = tlt.columns.get_level_values(0)
    tlt = tlt[["Close"]].rename(columns={"Close": "tlt"})
    return tlt
def fetch_gold(start="2000-01-01", end=None):
    """Fetching Gold ETF (GLD) — safe haven."""
    gld = yf.download("GLD", start=start, end=end, auto_adjust=True)
    gld.columns = gld.columns.get_level_values(0)
    gld = gld[["Close"]].rename(columns={"Close": "gld"})
    return gld
def fetch_vix(start="2000-01-01", end=None):
    """Fetching VIX fear index."""
    vix = yf.download("^VIX", start=start, end=end, auto_adjust=True)
    vix.columns = vix.columns.get_level_values(0)
    vix = vix[["Close"]].rename(columns={"Close": "vix"})
    return vix
def fetch_macro(start="2000-01-01", end=None):
    """Fetching macro indicators from FRED."""
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))
    t10y   = fred.get_series("GS10", start, end)
    t2y    = fred.get_series("GS2", start, end)
    spread = (t10y - t2y).rename("yield_spread")
    cpi = fred.get_series("CPIAUCSL", start, end).rename("cpi")
    macro = pd.concat([spread, cpi], axis=1)
    return macro
def fetch_all(start="2000-01-01", end=None):
    """Fetching and combining all data sources."""
    print("Fetching SPY (equities)...")
    spy = fetch_sp500(start, end)
    print("Fetching TLT (bonds)...")
    tlt = fetch_bonds(start, end)
    print("Fetching GLD (gold/safe haven)...")
    gld = fetch_gold(start, end)
    print("Fetching VIX...")
    vix = fetch_vix(start, end)
    print("Fetching macro indicators...")
    macro = fetch_macro(start, end)
    # Combining all on common dates
    df = spy.join(tlt, how="left")
    df = df.join(gld, how="left")
    df = df.join(vix, how="left")
    df = df.join(macro, how="left")
    # Forward fill macro(monthly to daily)
    df = df.ffill()
    # Droping any remaining NaN
    df = df.dropna()
    print(f"Data fetched: {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}")
    return df

if __name__ == "__main__":
    df = fetch_all()
    df.to_csv("data/raw/market_data.csv")
    print(df.head())
    print(df.tail())
