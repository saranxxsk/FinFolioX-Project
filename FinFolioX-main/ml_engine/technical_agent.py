"""
ml_engine/technical_agent.py  HOLD  TechnicalAgent (LSTM-only, single brain)
==========================================================================
CHANGES vs previous version:

  FIX HOLD probability stretching consistency with test script
    Problem: test_lstm.py compares raw LSTM output against BUY/SELL thresholds.
    Production predict() applies logit stretching (factor=3.5) before returning.
    A raw prob of 0.72 becomes ~0.91 after stretching HOLD thresholds mean
    different things in test vs production.
    Fix: added predict_raw() which returns the unscaled probability for use in
    the test harness and explainability pipeline.
    Production predict() is unchanged (still stretches for trading decisions).
    Also added stretch_enabled flag so callers can disable stretching cleanly.
"""

import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model


# ==============================================================================
# FEATURE ENGINEERING  (must match training exactly)
# ==============================================================================
LSTM_COLS = [
    "log_return", "vol_change", "sma10_dist",
    "sma20_dist", "sma50_dist", "RSI", "macd_norm",
]

SEQ_LEN = 100


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss  = -delta.clip(upper=0).ewm(com=period - 1, min_periods=period).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-9)))


def compute_macd(series: pd.Series, fast: int = 12,
                 slow: int = 26, signal: int = 9) -> pd.Series:
    macd_line = (
        series.ewm(span=fast, adjust=False).mean()
        - series.ewm(span=slow, adjust=False).mean()
    )
    return macd_line - macd_line.ewm(span=signal, adjust=False).mean()


def build_lstm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the 7-feature DataFrame from raw OHLCV.
    Output column order matches LSTM_COLS exactly.
    """
    out = pd.DataFrame(index=df.index)
    out["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    out["vol_change"] = df["Volume"].pct_change().clip(-5.0, 5.0)
    out["sma10_dist"] = (
        (df["Close"] - df["Close"].rolling(10).mean())
        / df["Close"].rolling(10).mean()
    )
    out["sma20_dist"] = (
        (df["Close"] - df["Close"].rolling(20).mean())
        / df["Close"].rolling(20).mean()
    )
    out["sma50_dist"] = (
        (df["Close"] - df["Close"].rolling(50).mean())
        / df["Close"].rolling(50).mean()
    )
    out["RSI"]       = compute_rsi(df["Close"])
    out["macd_norm"] = compute_macd(df["Close"]) / df["Close"]
    return out.replace([np.inf, -np.inf], np.nan).dropna()


# ==============================================================================
# TECHNICAL AGENT
# ==============================================================================
class TechnicalAgent:
    """
    Single-brain LSTM technical agent.

    Two prediction entry-points:
      predict(df)      HOLD returns STRETCHED probability (for trading decisions).
      predict_raw(df)  HOLD returns RAW probability (for test harness / expl agent).

    Use predict_raw() whenever you need the probability BEFORE logit stretching,
    e.g. in the explainability pipeline or the test back-tester, so that
    BUY/SELL thresholds have the same meaning as in test_lstm.py.
    """

    def __init__(
        self,
        lstm_model_path: str,
        lstm_scaler_path: str,
        # Optional kwargs kept for backward compatibility HOLD silently ignored
        trans_model_path: str = None,
        trans_scaler_path: str = None,
        stretch_factor: float = 3.5,
        stretch_enabled: bool = True,
    ):
        self.stretch_factor  = stretch_factor
        self.stretch_enabled = stretch_enabled

        self.lstm_scaler = joblib.load(lstm_scaler_path)
        self.lstm_model  = load_model(lstm_model_path)
        print("      [OK] Brain 1: Keras LSTM Loaded")

        if trans_model_path is not None:
            print("      ℹ️  Transformer (Brain 2) disabled HOLD LSTM-only mode active.")

    # ------------------------------------------------------------------
    # Logit stretching
    # ------------------------------------------------------------------
    def _stretch_probability(self, p: float, factor: float = None) -> float:
        """
        Logit stretching HOLD pushes mean-hugging probabilities away from 0.5.
        p_stretched = sigmoid(logit(p) * factor)
        """
        if factor is None:
            factor = self.stretch_factor
        p     = np.clip(p, 1e-5, 1.0 - 1e-5)
        logit = np.log(p / (1.0 - p))
        return float(1.0 / (1.0 + np.exp(-logit * factor)))

    # ------------------------------------------------------------------
    # Internal: build scaled sequence from DataFrame
    # ------------------------------------------------------------------
    def _prepare_sequence(self, recent_data_df: pd.DataFrame):
        """
        Builds feature DataFrame, takes last SEQ_LEN rows, applies scaler.
        Returns (scaled_seq_np, feature_df) or (None, None) if not enough data.
        """
        feature_df = build_lstm_features(recent_data_df)
        if len(feature_df) < SEQ_LEN:
            return None, None

        last_100  = feature_df[LSTM_COLS].tail(SEQ_LEN).values    # (100, 7)
        scaled    = self.lstm_scaler.transform(last_100)           # (100, 7)
        return scaled, feature_df

    # ------------------------------------------------------------------
    # predict_raw HOLD unscaled output (for test harness & explainability)
    # ------------------------------------------------------------------
    def predict_raw(self, recent_data_df: pd.DataFrame) -> float:
        """
        Returns the raw sigmoid output of the LSTM without any logit stretching.
        Use this in:
          - test_lstm.py  (thresholds were set against raw probs)
          - ExplainabilityAgent (IG baseline and base_prob comparisons)
        """
        scaled, _ = self._prepare_sequence(recent_data_df)
        if scaled is None:
            return 0.5

        lstm_seq = scaled.reshape(1, SEQ_LEN, len(LSTM_COLS))
        raw_prob = float(self.lstm_model.predict(lstm_seq, verbose=0)[0][0])
        return float(np.clip(raw_prob, 0.0, 1.0))

    # ------------------------------------------------------------------
    # predict HOLD stretched output (for production trading decisions)
    # ------------------------------------------------------------------
    def predict(self, recent_data_df: pd.DataFrame) -> float:
        """
        Returns the STRETCHED probability for use in production trading logic.
        The fusion engine, conflict resolver, and ASC module all consume this.
        Do NOT use this in test_lstm.py or explainability pipelines.
        """
        scaled, _ = self._prepare_sequence(recent_data_df)
        if scaled is None:
            return 0.5

        lstm_seq = scaled.reshape(1, SEQ_LEN, len(LSTM_COLS))
        raw_prob = float(self.lstm_model.predict(lstm_seq, verbose=0)[0][0])

        if self.stretch_enabled:
            stretched = self._stretch_probability(raw_prob)
            print(f"      - LSTM Brain : {raw_prob:.4f} -> Stretched: {stretched:.4f}")
            return float(np.clip(stretched, 0.0, 1.0))
        else:
            print(f"      - LSTM Brain : {raw_prob:.4f} (no stretch)")
            return float(np.clip(raw_prob, 0.0, 1.0))

    # ------------------------------------------------------------------
    # predict_signal HOLD alias for predict() (used by red_team / adversarial)
    # ------------------------------------------------------------------
    def predict_signal(self, recent_data_df: pd.DataFrame) -> float:
        """
        Alias for predict(). Red Team and Adversarial Tester call this.
        Returns stretched probability for consistency with live trading path.
        """
        return self.predict(recent_data_df)