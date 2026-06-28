import yfinance as yf
import pandas as pd
import os

# 1. Define settings
TICKER = "AAPL"
START_DATE = "2015-01-01"
END_DATE = "2024-01-01"
SAVE_PATH = os.path.join("data", "raw", f"{TICKER}_historical.csv")

def fetch_data():
    print(f"📥 Fetching data for {TICKER}...")
    
    # 2. Download from Yahoo Finance
    df = yf.download(TICKER, start=START_DATE, end=END_DATE)
    
    # 3. Check if data is empty
    if df.empty:
        print("[BAD] No data found. Check your internet or ticker symbol.")
        return

    # 4. Save to CSV
    # Ensure the directory exists
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    df.to_csv(SAVE_PATH)
    print(f"[OK] Data saved successfully to: {SAVE_PATH}")
    print(f"📊 Total Rows: {len(df)}")
    print(df.head())

if __name__ == "__main__":
    fetch_data()