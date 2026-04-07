"""Data fetching module — downloads historical stock data via yfinance."""

import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch historical stock data for the given ticker and date range.

    Args:
        ticker: Stock symbol (e.g. AAPL, TSLA).
        start: Start date string YYYY-MM-DD.
        end: End date string YYYY-MM-DD.

    Returns:
        DataFrame with OHLCV columns, or empty DataFrame on error.
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start, end=end)
        if df.empty:
            return pd.DataFrame()
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return pd.DataFrame()


def validate_ticker(ticker: str) -> bool:
    """Quick validation — tries to fetch 5 days of data."""
    try:
        end = datetime.today()
        start = end - timedelta(days=10)
        df = yf.Ticker(ticker).history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
        return not df.empty
    except Exception:
        return False
