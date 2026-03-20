"""
data_loader.py
--------------
Downloads and prepares daily return data for the regime-conditional tail risk study.

Assets:
  - SPY  : S&P 500 benchmark
  - 9 SPDR sector ETFs (XLF, XLE, XLV, XLK, XLI, XLP, XLU, XLY, XLB)
  - VIX  : CBOE Volatility Index (market stress proxy)

Sample window spans multiple stress regimes: GFC 2008, COVID 2020, rate-hike 2022.
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Asset universe
# ---------------------------------------------------------------------------

EQUITY_TICKERS = {
    "SPY": "S&P 500 ETF",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Health Care",
    "XLK": "Technology",
    "XLI": "Industrials",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLY": "Consumer Discretionary",
    "XLB": "Materials",
}

VIX_TICKER = "^VIX"
START_DATE = "2005-01-01"
END_DATE   = "2024-12-31"

# Known stress periods for labelling plots / sub-sample analysis
STRESS_PERIODS = {
    "GFC 2008-09":    ("2008-09-01", "2009-03-31"),
    "COVID 2020":     ("2020-02-15", "2020-04-30"),
    "Rate Hike 2022": ("2022-01-01", "2022-12-31"),
}

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Return adjusted close prices, forward-fill missing days."""
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)
    # yfinance >=0.2 returns MultiIndex; single ticker returns flat
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"][tickers]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
    prices = prices.ffill().dropna(how="all")
    return prices


def download_all(start: str = START_DATE, end: str = END_DATE):
    """
    Download equity prices and VIX.

    Returns
    -------
    prices : pd.DataFrame  (T x n_assets)  – adjusted close
    vix    : pd.Series     (T,)             – VIX level
    """
    equity_list = list(EQUITY_TICKERS.keys())
    prices = _download_prices(equity_list, start, end)

    vix_raw = yf.download(VIX_TICKER, start=start, end=end,
                          auto_adjust=True, progress=False)
    if isinstance(vix_raw.columns, pd.MultiIndex):
        vix = vix_raw["Close"][VIX_TICKER]
    else:
        vix = vix_raw["Close"]
    vix.name = "VIX"
    vix = vix.ffill()

    return prices, vix


# ---------------------------------------------------------------------------
# Return computation
# ---------------------------------------------------------------------------

def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns (drops first NaN row)."""
    return np.log(prices / prices.shift(1)).dropna()


def equal_weight_returns(returns: pd.DataFrame) -> pd.Series:
    """Simple equal-weight portfolio daily log return."""
    port = returns.mean(axis=1)
    port.name = "portfolio"
    return port


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def load_data(start: str = START_DATE, end: str = END_DATE):
    """
    Full data pipeline.

    Returns
    -------
    returns      : pd.DataFrame  – daily log returns (T x n_assets)
    port_returns : pd.Series     – equal-weight portfolio returns
    vix          : pd.Series     – VIX levels aligned to returns index
    prices       : pd.DataFrame  – raw adjusted close prices
    """
    prices, vix = download_all(start, end)
    returns = log_returns(prices)
    port_returns = equal_weight_returns(returns)

    # Align VIX to return dates
    vix = vix.reindex(returns.index).ffill()

    return returns, port_returns, vix, prices


# ---------------------------------------------------------------------------
# Utility: stress-period boolean mask
# ---------------------------------------------------------------------------

def stress_mask(index: pd.DatetimeIndex, period_key: str) -> pd.Series:
    """Return boolean Series True during the named stress period."""
    start, end = STRESS_PERIODS[period_key]
    mask = (index >= start) & (index <= end)
    return pd.Series(mask, index=index, name=period_key)


def all_stress_masks(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Return DataFrame of boolean columns, one per stress period."""
    return pd.DataFrame(
        {k: stress_mask(index, k) for k in STRESS_PERIODS},
        index=index,
    )
