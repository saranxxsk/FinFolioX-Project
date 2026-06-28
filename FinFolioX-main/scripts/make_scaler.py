import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'V', 'JNJ']
FEATURE_COLS = ['log_return', 'vol_change', 'sma10_dist', 'sma20_dist', 'sma50_dist', 'RSI', 'macd_norm']

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss = -delta.clip(upper=0).ewm(com=period - 1, min_periods=period).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-9)))

def compute_macd(series, fast=12, slow=26, signal=9):
    macd_line = series.ewm(span=fast, adjust=False).mean() - series.ewm(span=slow, adjust=False).mean()
    return macd_line - macd_line.ewm(span=signal, adjust=False).mean()

print("Downloading data to build global scaler...")
all_features = []
for tk in TICKERS:
    df = yf.download(tk, period="7y", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    out = pd.DataFrame(index=df.index)
    out["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    out["vol_change"] = df["Volume"].pct_change().clip(-5.0, 5.0)
    out["sma10_dist"] = (df["Close"] - df["Close"].rolling(10).mean()) / df["Close"].rolling(10).mean()
    out["sma20_dist"] = (df["Close"] - df["Close"].rolling(20).mean()) / df["Close"].rolling(20).mean()
    out["sma50_dist"] = (df["Close"] - df["Close"].rolling(50).mean()) / df["Close"].rolling(50).mean()
    out["RSI"] = compute_rsi(df["Close"])
    out["macd_norm"] = compute_macd(df["Close"]) / df["Close"]
    
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    all_features.append(out[FEATURE_COLS])

# Combine all tickers and fit the scaler
combined = pd.concat(all_features)
scaler = StandardScaler()
scaler.fit(combined.values)

os.makedirs("saved_models", exist_ok=True)
joblib.dump(scaler, "saved_models/scaler.pkl")
print("[OK] scaler.pkl successfully saved to saved_models/!")