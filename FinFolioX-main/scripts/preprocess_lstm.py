import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

INPUT_PATH = os.path.join("data", "processed", "AAPL_features.csv")
SCALER_PATH = os.path.join("saved_models", "scaler.pkl")
FINAL_DATA_PATH = os.path.join("data", "processed", "LSTM_training_data.csv")

def preprocess_data():
    if not os.path.exists(INPUT_PATH):
        print("[BAD] Feature data not found. Run add_indicators.py first.")
        return

    df = pd.read_csv(INPUT_PATH, index_col=0)
    
    # Select columns we want to feed into the LSTM
    # We include 'Close' because that's what we want to predict
    features = ['Close', 'Volume', 'SMA_50', 'SMA_200', 'RSI', 'MACD']
    
    # Filter only these columns
    data = df[features].values

    # 1. Initialize Scaler (0 to 1)
    scaler = MinMaxScaler(feature_range=(0, 1))

    # 2. Fit and Transform
    scaled_data = scaler.fit_transform(data)

    # 3. Save the Scaler (Important for later!)
    joblib.dump(scaler, SCALER_PATH)
    print(f"💾 Scaler saved to {SCALER_PATH}")

    # 4. Save scaled data for Kaggle
    # Converting back to DataFrame just for saving
    scaled_df = pd.DataFrame(scaled_data, columns=features)
    scaled_df.to_csv(FINAL_DATA_PATH, index=False)
    print(f"[OK] Scaled data ready for Kaggle: {FINAL_DATA_PATH}")

if __name__ == "__main__":
    preprocess_data()