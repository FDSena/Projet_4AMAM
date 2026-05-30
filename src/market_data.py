import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime


def download_data(ticker, start_date, end_date, interval="1d"):
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
    if data.empty:
        print(f"Data not found for ticker: {ticker}")
        return None
    data.index = pd.to_datetime(data.index)
    return data.sort_index(ascending=True)


def clean_data(data):
    if data is None or data.empty:
        return None
    clean_df = data.copy()
    if isinstance(clean_df.columns, pd.MultiIndex):
        clean_df.columns = clean_df.columns.get_level_values(0)
    clean_df.index = pd.to_datetime(clean_df.index, errors="coerce")
    clean_df = clean_df[~clean_df.index.duplicated(keep="first")]
    clean_df = clean_df.sort_index(ascending=True)
    clean_df = clean_df.dropna(how="any")
    return clean_df


def get_close_prices(data):
    if data is None or data.empty:
        return None
    if "Close" in data.columns:
        return data["Close"].copy().dropna()
    elif "Adj Close" in data.columns:
        return data["Adj Close"].copy().dropna()
    else:
        raise ValueError("Colonne de prix de clôture introuvable.")


def compute_log_returns(prices):
    if prices is None:
        return None
    return np.log(prices / prices.shift(1)).dropna()


def build_dataset(ticker, start_date, end_date, interval="1d"):
    """
    Construit un DataFrame unique avec toutes les variables pour un actif.

    Colonnes produites
    ------------------
    Open, High, Low, Close, Volume : données OHLCV nettoyées
    log_return                     : rendement logarithmique journalier
    cum_return                     : rendement cumulé (base 100)
    rolling_vol_21                 : volatilité glissante 21 jours (annualisée)
    """
    raw = download_data(ticker, start_date, end_date, interval)
    df = clean_data(raw)
    close = get_close_prices(df)
    log_ret = compute_log_returns(close)

    # Aligner tout sur l'index des rendements (on perd la 1ère ligne, normal)
    df = df.loc[log_ret.index].copy()
    df["log_return"]     = log_ret
    df["cum_return"]     = (df["log_return"].cumsum().apply(np.exp)) * 100
    df["rolling_vol_21"] = df["log_return"].rolling(21).std() * np.sqrt(252)

    df.index.name = "Date"
    return df

def get_S0_and_log_returns(df):
    """
    Extrait S0 (dernier prix de clôture) et les log-returns depuis un dataset.
    C'est la fonction que crr.py et calibration.py vont utiliser.
    """
    S0 = df["Close"].iloc[-1]
    log_returns = df["log_return"].dropna()
    return S0, log_returns

