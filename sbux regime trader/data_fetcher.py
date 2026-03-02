import yfinance as yf
import pandas as pd
from pathlib import Path
from typing import Union, Optional


def clean_yfinance_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Handle multi-ticker case (even if we usually download one)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)  # drop ticker level

    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[[c for c in cols if c in df.columns]]

    # Clean index
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]
    df = df.dropna()

    # Set proper timezone (America/New_York for US stocks in my case)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("America/New_York")

    print(f"Cleaned shape: {df.shape}")
    print(f"Date range: {df.index[0]} → {df.index[-1]}")
    print(f"Timezone: {df.index.tz}")

    return df


def download_and_save_data(
    ticker: str,
    interval: str,
    period_or_days: Union[str,int],
    data_path: Union[str,Path],
    force_download: bool = False
) -> pd.DataFrame:

    data_path = Path(data_path)
    data_path.parent.mkdir(exist_ok=True, parents=True)

    if data_path.exists() and not force_download:
        print(f"Loading existing clean data from {data_path}")
        return pd.read_parquet(data_path)

    print(f"Downloading {ticker} {interval} data...")

    # Normalize period
    if isinstance(period_or_days, int):
        period = f"{period_or_days}d"
    else:
        period = period_or_days

    # Download
    raw = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        auto_adjust=True,   # handles splits & dividends
        prepost=False,
        progress=False
    )

    if raw.empty:
        raise ValueError(f"No data returned for {ticker} {interval} {period}")

    clean_df = clean_yfinance_data(raw)
    clean_df.to_parquet(data_path, compression="gzip")
    print(f"Saved clean data → {data_path}")

    return clean_df