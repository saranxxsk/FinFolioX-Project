import pandas as pd
import numpy as np
import os

# -----------------------------
# PATHS
# -----------------------------
INPUT_PATH = os.path.join("data", "raw", "AAPL_historical.csv")
OUTPUT_PATH = os.path.join("data", "processed", "AAPL_features.csv")

# -----------------------------
# RSI CALCULATION
# -----------------------------
def compute_rsi(series, window=14):
    delta = series.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

# -----------------------------
# ADD TECHNICAL INDICATORS
# -----------------------------
def add_technical_indicators():
    if not os.path.exists(INPUT_PATH):
        print("[BAD] Raw data not found. Run fetch_stock_data.py first.")
        return

    # -----------------------------
    # LOAD CSV SAFELY
    # -----------------------------
    df = pd.read_csv(
        INPUT_PATH,
        index_col=0,
        parse_dates=True,
        infer_datetime_format=True
    )

    # Fix MultiIndex columns (yfinance issue)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # -----------------------------
    # CLEAN & VALIDATE DATA
    # -----------------------------
    # Convert price columns to numeric
    for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop invalid rows
    df.dropna(subset=['Close'], inplace=True)

    print("🛠 Adding Technical Indicators...")

    # -----------------------------
    # TECHNICAL INDICATORS
    # -----------------------------
    # Simple Moving Averages
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()

    # RSI
    df['RSI'] = compute_rsi(df['Close'])

    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # -----------------------------
    # CLEAN NA VALUES FROM ROLLING
    # -----------------------------
    df.dropna(inplace=True)

    # -----------------------------
    # SAVE OUTPUT
    # -----------------------------
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH)

    print(f"[OK] Features added successfully!")
    print(f"📁 Saved to: {OUTPUT_PATH}")
    print("\n📊 Sample Output:")
    print(df[['Close', 'SMA_50', 'RSI', 'MACD']].head())

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    add_technical_indicators()
