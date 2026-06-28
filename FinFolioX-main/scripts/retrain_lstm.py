# ╔======================================================================╗
# ║  LSTM BRAIN 1 HOLD RETRAIN v2  (Fixed: temporal split, no leakage)   ║
# ║  Kaggle P100 / Local GPU / CPU compatible                         ║
# ╚======================================================================╝
#
# CRITICAL FIXES vs previous training code:
#   1. TEMPORAL train/val split HOLD NO shuffle. Train on older data, val on recent.
#      The old code used train_test_split(shuffle=True) which leaks future data
#      into training, inflating val_auc to 0.90 while real accuracy was 58%.
#   2. 10+ years of data HOLD covers 2015-2026 including 2018 correction,
#      2020 COVID crash, 2022 bear market, and 2025+ volatility.
#   3. Class weighting instead of downsampling HOLD keeps all training data.
#   4. Stronger regularization HOLD higher dropout, spatial dropout, gradient
#      clipping, and label smoothing to prevent overfitting.
#   5. Simpler architecture option HOLD less capacity = better generalization
#      on genuinely out-of-sample data.
#   6. Walk-forward validation awareness HOLD val period is the most recent
#      ~20% of data chronologically per ticker.
#
# Run on Kaggle:  Accelerator -> GPU P100, paste this entire file.
# Run locally:    python scripts/retrain_lstm.py  (GPU optional)

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
from sklearn.preprocessing import StandardScaler

# -- GPU setup ----------------------------------------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"🚀  GPU detected: {gpus}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("ℹ️  No GPU HOLD training will be slower but still works.")

# ┌---------------------------------------------------------------------┐
# │  CONFIG                                                             │
# └---------------------------------------------------------------------┘

# Detect if running on Kaggle or locally
IS_KAGGLE = os.path.exists("/kaggle/working")
if IS_KAGGLE:
    OUT_DIR = "/kaggle/working/saved_models"
else:
    OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "saved_models")

MODEL_PATH  = os.path.join(OUT_DIR, "lstm_model.keras")
SCALER_PATH = os.path.join(OUT_DIR, "lstm_scaler.pkl")
BACKUP_MODEL = MODEL_PATH + ".bak"
BACKUP_SCALER = SCALER_PATH + ".bak"
os.makedirs(OUT_DIR, exist_ok=True)

SEQ_LEN       = 100     # timesteps fed to LSTM (must match production)
HORIZON       = 5       # 5-day forward return target
VAL_FRACTION  = 0.20    # last 20% of each ticker's timeline for validation
DATA_PERIOD   = "10y"   # 10 years of history HOLD covers multiple regimes

# Training hyperparameters (tuned for generalization, not train accuracy)
EPOCHS        = 150
BATCH_SIZE    = 256
LEARNING_RATE = 5e-4
DROPOUT       = 0.35
L2_REG        = 3e-4
LABEL_SMOOTH  = 0.05    # prevents overconfident predictions
GRAD_CLIP     = 1.0     # gradient clipping norm
PATIENCE_ES   = 25      # early stopping patience
PATIENCE_LR   = 10      # LR reduction patience

# 7 features HOLD MUST match TechnicalAgent exactly
LSTM_COLS = [
    "log_return", "vol_change", "sma10_dist",
    "sma20_dist", "sma50_dist", "RSI", "macd_norm",
]
N_FEATURES = len(LSTM_COLS)

# Diverse ticker universe HOLD includes all test tickers + extras for regime coverage
TRAIN_TICKERS = [
    # Tech (growth, high beta)
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "NFLX",
    # Financials
    "JPM", "BAC", "GS", "V", "MA",
    # Consumer / defensive
    "WMT", "JNJ", "PFE", "UNH", "MCD", "KO", "PEP", "PG", "HD", "COST",
    # Industrials / energy
    "XOM", "CVX", "CAT", "BA", "DIS",
    # ETFs (market / sector / commodities / bonds)
    "SPY", "QQQ", "DIA", "IWM", "EEM", "TLT", "GLD", "SLV", "USO", "UNG",
    # Additional for diversity
    "CRM", "ADBE", "CSCO", "IBM", "NKE", "TGT", "VZ", "ORCL",
]

# ┌---------------------------------------------------------------------┐
# │  FEATURE ENGINEERING HOLD identical to production technical_agent.py   │
# └---------------------------------------------------------------------┘

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss  = -delta.clip(upper=0).ewm(com=period - 1, min_periods=period).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-9)))

def compute_macd(series: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
    macd_line = (series.ewm(span=fast, adjust=False).mean()
                 - series.ewm(span=slow, adjust=False).mean())
    return macd_line - macd_line.ewm(span=signal, adjust=False).mean()

def build_lstm_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    out["vol_change"] = df["Volume"].pct_change().clip(-5.0, 5.0)
    out["sma10_dist"] = (df["Close"] - df["Close"].rolling(10).mean()) / df["Close"].rolling(10).mean()
    out["sma20_dist"] = (df["Close"] - df["Close"].rolling(20).mean()) / df["Close"].rolling(20).mean()
    out["sma50_dist"] = (df["Close"] - df["Close"].rolling(50).mean()) / df["Close"].rolling(50).mean()
    out["RSI"]        = compute_rsi(df["Close"])
    out["macd_norm"]  = compute_macd(df["Close"]) / df["Close"]
    return out.replace([np.inf, -np.inf], np.nan).dropna()

# ┌---------------------------------------------------------------------┐
# │  DATA PIPELINE HOLD TEMPORAL SPLIT (the critical fix)                  │
# └---------------------------------------------------------------------┘

def generate_sequences_temporal(tickers: list):
    """
    Build train/val sequences with TEMPORAL SPLIT per ticker.
    For each ticker, the first 80% of its timeline goes to train,
    the last 20% goes to validation. NO shuffling across time.

    This prevents the data leakage that inflated the old model's
    validation AUC to 0.90 while real accuracy was only 58%.
    """
    import yfinance as yf

    print(f"📊  Downloading {DATA_PERIOD} data for {len(tickers)} tickers...")
    print(f"    Temporal split: first {100-int(VAL_FRACTION*100)}% train, last {int(VAL_FRACTION*100)}% val")

    train_X, train_y = [], []
    val_X,   val_y   = [], []
    stats = {"tickers_ok": 0, "tickers_skip": 0, "total_seqs": 0}

    for i, ticker in enumerate(tickers):
        try:
            hist = yf.download(ticker, period=DATA_PERIOD, progress=False)
            if hist.empty or len(hist) < SEQ_LEN + HORIZON + 100:
                stats["tickers_skip"] += 1
                continue

            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)

            feat_df    = build_lstm_features(hist)
            if len(feat_df) < SEQ_LEN + HORIZON:
                stats["tickers_skip"] += 1
                continue

            close_aligned = hist["Close"].reindex(feat_df.index)
            feat_vals     = feat_df[LSTM_COLS].values
            close_vals    = close_aligned.values

            n_samples  = len(feat_vals) - SEQ_LEN - HORIZON + 1
            split_idx  = int(n_samples * (1.0 - VAL_FRACTION))

            for j in range(n_samples):
                seq     = feat_vals[j : j + SEQ_LEN]
                curr_px = close_vals[j + SEQ_LEN - 1]
                fut_px  = close_vals[j + SEQ_LEN - 1 + HORIZON]

                if np.isnan(curr_px) or np.isnan(fut_px) or curr_px <= 0:
                    continue

                fwd_ret = (fut_px - curr_px) / curr_px
                label   = 1.0 if fwd_ret > 0.0 else 0.0

                if j < split_idx:
                    train_X.append(seq)
                    train_y.append(label)
                else:
                    val_X.append(seq)
                    val_y.append(label)

            stats["tickers_ok"] += 1
            stats["total_seqs"] += n_samples

            if (i + 1) % 10 == 0:
                print(f"    [{i+1}/{len(tickers)}] {stats['tickers_ok']} ok, "
                      f"{len(train_X)} train + {len(val_X)} val seqs")

        except Exception as e:
            stats["tickers_skip"] += 1

    X_train = np.array(train_X, dtype=np.float32)
    y_train = np.array(train_y, dtype=np.float32)
    X_val   = np.array(val_X,   dtype=np.float32)
    y_val   = np.array(val_y,   dtype=np.float32)

    # Compute class balance stats
    tr_up = y_train.sum(); tr_dn = len(y_train) - tr_up
    vl_up = y_val.sum();   vl_dn = len(y_val)   - vl_up

    print(f"\n[OK]  Data ready:")
    print(f"    Tickers: {stats['tickers_ok']} used, {stats['tickers_skip']} skipped")
    print(f"    Train: {len(X_train)} sequences  (up={int(tr_up)}, dn={int(tr_dn)}, "
          f"ratio={tr_up/len(y_train):.1%})")
    print(f"    Val:   {len(X_val)} sequences  (up={int(vl_up)}, dn={int(vl_dn)}, "
          f"ratio={vl_up/len(y_val):.1%})")

    return X_train, y_train, X_val, y_val

# ┌---------------------------------------------------------------------┐
# │  SCALER HOLD fit on training data only                                 │
# └---------------------------------------------------------------------┘

def fit_and_apply_scaler(X_train, X_val):
    N_tr, T, F = X_train.shape
    N_vl       = X_val.shape[0]

    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, F))  # fit on TRAIN only

    X_train_sc = scaler.transform(X_train.reshape(-1, F)).reshape(N_tr, T, F).astype(np.float32)
    X_val_sc   = scaler.transform(X_val.reshape(-1, F)).reshape(N_vl, T, F).astype(np.float32)

    return X_train_sc, X_val_sc, scaler

# ┌---------------------------------------------------------------------┐
# │  MODEL HOLD BiLSTM with stronger regularization                       │
# └---------------------------------------------------------------------┘

def build_model() -> keras.Model:
    """
    Bidirectional LSTM with focus on GENERALIZATION not train accuracy.

    Key differences from old model:
      - Spatial dropout after each LSTM (drops entire feature channels)
      - Gradient clipping in optimizer
      - Label smoothing in loss
      - Moderate capacity (128->64) to prevent memorization
    """
    reg = regularizers.l2(L2_REG)

    inputs = layers.Input(shape=(SEQ_LEN, N_FEATURES))

    # LSTM Block 1
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, kernel_regularizer=reg)
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout1D(DROPOUT)(x)  # drops entire feature maps

    # LSTM Block 2
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=False, kernel_regularizer=reg)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT)(x)

    # Dense head
    x = layers.Dense(32, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(DROPOUT * 0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
            clipnorm=GRAD_CLIP,           # gradient clipping
        ),
        loss=keras.losses.BinaryCrossentropy(
            label_smoothing=LABEL_SMOOTH,  # prevents overconfident outputs
        ),
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )

    model.summary()
    return model

# ┌---------------------------------------------------------------------┐
# │  TRAINING HOLD with proper callbacks and class weighting               │
# └---------------------------------------------------------------------┘

def train_model(model, X_train, y_train, X_val, y_val):
    """
    Train with class weighting (NOT downsampling) and temporal validation.
    """
    # Compute class weights to handle any imbalance
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    # Weight the minority class higher
    w_pos = len(y_train) / (2.0 * n_pos + 1e-9)
    w_neg = len(y_train) / (2.0 * n_neg + 1e-9)
    class_weight = {0: w_neg, 1: w_pos}
    print(f"\n    Class weights: 0(dn)={w_neg:.3f}, 1(up)={w_pos:.3f}")

    cb_list = [
        # Monitor val_loss (not val_auc) HOLD more stable for generalization
        callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=PATIENCE_ES,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=PATIENCE_LR,
            min_lr=1e-6,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print(f"\n🏋️  Training: {EPOCHS} epochs max, batch={BATCH_SIZE}, "
          f"lr={LEARNING_RATE}, dropout={DROPOUT}")
    print(f"    Label smoothing={LABEL_SMOOTH}, L2={L2_REG}, "
          f"grad_clip={GRAD_CLIP}")
    print(f"    Early stop patience={PATIENCE_ES}, LR patience={PATIENCE_LR}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=cb_list,
        verbose=1,
    )

    # Load best checkpoint
    best_model = keras.models.load_model(MODEL_PATH)
    return best_model, history

# ┌---------------------------------------------------------------------┐
# │  EVALUATION HOLD comprehensive checks                                  │
# └---------------------------------------------------------------------┘

def evaluate_model(model, X_val, y_val, scaler):
    """Full evaluation suite."""

    print("\n" + "=" * 65)
    print("  EVALUATION")
    print("=" * 65)

    # 1. Validation metrics
    val_loss, val_acc, val_auc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n  Val Loss: {val_loss:.4f}")
    print(f"  Val Acc:  {val_acc:.4f}")
    print(f"  Val AUC:  {val_auc:.4f}")

    # 2. Probability distribution
    probs = model.predict(X_val, verbose=0).flatten()
    print(f"\n  Probability distribution on validation set:")
    print(f"    Mean:   {np.mean(probs):.4f}")
    print(f"    Median: {np.median(probs):.4f}")
    print(f"    Std:    {np.std(probs):.4f}")
    print(f"    %> 0.5: {np.mean(probs > 0.5)*100:.1f}%")
    print(f"    %> 0.7: {np.mean(probs > 0.7)*100:.1f}%")
    print(f"    %< 0.3: {np.mean(probs < 0.3)*100:.1f}%")

    # 3. Baseline (zeros) check
    zero_input = np.zeros((1, SEQ_LEN, N_FEATURES), dtype=np.float32)
    zero_prob  = float(model.predict(zero_input, verbose=0)[0][0])
    print(f"\n  F(zeros) = {zero_prob:.4f}  (ideal: ~0.50)")
    if 0.35 < zero_prob < 0.65:
        print(f"    [OK] Baseline is near-neutral")
    else:
        print(f"    [WARN]  Baseline is biased")

    # 4. Directional sanity check with realistic synthetic data
    print(f"\n  Directional sanity checks:")
    T, F = SEQ_LEN, N_FEATURES

    def make_realistic_seq(trend: str) -> np.ndarray:
        """Generate synthetic sequence that mimics real market conditions."""
        rng = np.random.default_rng(42)
        seq = np.zeros((T, F), dtype=np.float32)

        if trend == "bull":
            # Positive returns, price above SMAs, high RSI, positive MACD
            seq[:, 0] = rng.normal(+0.005, 0.010, T)  # log_return
            seq[:, 1] = rng.normal(+0.05,  0.30,  T)  # vol_change
            seq[:, 2] = rng.normal(+0.015, 0.008, T)  # sma10_dist
            seq[:, 3] = rng.normal(+0.025, 0.010, T)  # sma20_dist
            seq[:, 4] = rng.normal(+0.040, 0.015, T)  # sma50_dist
            seq[:, 5] = rng.normal(62.0,   8.0,   T)  # RSI
            seq[:, 6] = rng.normal(+0.0003, 0.0002, T) # macd_norm
        elif trend == "bear":
            # Negative returns, price below SMAs, low RSI, negative MACD
            seq[:, 0] = rng.normal(-0.005, 0.012, T)
            seq[:, 1] = rng.normal(+0.10,  0.40,  T)  # higher vol in bear
            seq[:, 2] = rng.normal(-0.015, 0.008, T)
            seq[:, 3] = rng.normal(-0.025, 0.010, T)
            seq[:, 4] = rng.normal(-0.040, 0.015, T)
            seq[:, 5] = rng.normal(38.0,   8.0,   T)
            seq[:, 6] = rng.normal(-0.0003, 0.0002, T)
        else:  # flat
            seq[:, 0] = rng.normal(0.000, 0.008, T)
            seq[:, 1] = rng.normal(0.02,  0.25,  T)
            seq[:, 2] = rng.normal(0.000, 0.005, T)
            seq[:, 3] = rng.normal(0.000, 0.006, T)
            seq[:, 4] = rng.normal(0.000, 0.010, T)
            seq[:, 5] = rng.normal(50.0,  6.0,   T)
            seq[:, 6] = rng.normal(0.000, 0.0001, T)
        return seq

    checks_passed = 0
    for name, trend, want_above in [
        ("Strong Bull", "bull",  True),
        ("Strong Bear", "bear",  False),
        ("Flat Market", "flat",  None),
    ]:
        raw_seq = make_realistic_seq(trend)
        scaled  = scaler.transform(raw_seq)
        inp     = scaled.reshape(1, T, F).astype(np.float32)
        prob    = float(model.predict(inp, verbose=0)[0][0])

        if want_above is True:
            signal = "BUY" if prob > 0.52 else "SELL"
            ok = prob > 0.52
        elif want_above is False:
            signal = "BUY" if prob > 0.48 else "SELL"
            ok = prob < 0.48
        else:
            signal = "HOLD" if 0.42 < prob < 0.58 else ("BUY" if prob >= 0.58 else "SELL")
            ok = 0.35 < prob < 0.65  # flat should be near 0.5

        status = "[OK]" if ok else "[BAD]"
        if ok: checks_passed += 1
        print(f"    {status}  {name:12s}  -> prob: {prob:.4f}  [{signal}]")

    print(f"\n    Directional checks: {checks_passed}/3 passed")

    # 5. Per-threshold accuracy on validation
    print(f"\n  Threshold analysis on validation set:")
    for buy_t, sell_t in [(0.52, 0.48), (0.55, 0.45), (0.60, 0.40)]:
        preds = np.where(probs > buy_t, 1, np.where(probs < sell_t, 0, -1))
        mask  = preds >= 0  # exclude HOLD
        if mask.sum() > 0:
            active_acc = np.mean(preds[mask] == y_val[mask]) * 100
            coverage   = mask.mean() * 100
            print(f"    BUY>{buy_t:.2f} SELL<{sell_t:.2f}: "
                  f"acc={active_acc:.1f}%  coverage={coverage:.1f}%")

    return val_acc, val_auc

# ┌---------------------------------------------------------------------┐
# │  MAIN                                                               │
# └---------------------------------------------------------------------┘

def main():
    start_time = time.time()

    print("=" * 65)
    print("  LSTM BRAIN 1 HOLD RETRAIN v2 (Temporal Split, No Leakage)")
    print("=" * 65)
    print(f"\n  Config:")
    print(f"    SEQ_LEN={SEQ_LEN}, HORIZON={HORIZON}, DATA={DATA_PERIOD}")
    print(f"    EPOCHS={EPOCHS}, BATCH={BATCH_SIZE}, LR={LEARNING_RATE}")
    print(f"    DROPOUT={DROPOUT}, L2={L2_REG}, LABEL_SMOOTH={LABEL_SMOOTH}")
    print(f"    Output: {MODEL_PATH}")

    # -- Step 1: Generate data with temporal split ----------------------
    X_train, y_train, X_val, y_val = generate_sequences_temporal(TRAIN_TICKERS)

    if len(X_train) < 1000 or len(X_val) < 200:
        print("[BAD]  Not enough data. Check internet connection and tickers.")
        return

    # -- Step 2: Fit scaler on train, apply to both ---------------------
    print("\n📐  Fitting scaler on training data...")
    X_train_sc, X_val_sc, scaler = fit_and_apply_scaler(X_train, X_val)

    # Backup old files
    if os.path.exists(MODEL_PATH):
        import shutil
        shutil.copy2(MODEL_PATH, BACKUP_MODEL)
        print(f"    Backed up old model -> {BACKUP_MODEL}")
    if os.path.exists(SCALER_PATH):
        import shutil
        shutil.copy2(SCALER_PATH, BACKUP_SCALER)
        print(f"    Backed up old scaler -> {BACKUP_SCALER}")

    # Save scaler
    joblib.dump(scaler, SCALER_PATH)
    print(f"    [OK] Scaler saved -> {SCALER_PATH}")
    print(f"       Mean: {scaler.mean_[:3]}...")
    print(f"       Scale: {scaler.scale_[:3]}...")

    # -- Step 3: Build and train model ----------------------------------
    print("\n🔨  Building model...")
    model = build_model()

    best_model, history = train_model(
        model, X_train_sc, y_train, X_val_sc, y_val
    )

    # -- Step 4: Evaluate -----------------------------------------------
    val_acc, val_auc = evaluate_model(best_model, X_val_sc, y_val, scaler)

    # -- Step 5: Summary ------------------------------------------------
    elapsed = (time.time() - start_time) / 60

    print("\n" + "=" * 65)
    print(f"  [OK] TRAINING COMPLETE")
    print(f"  Time:   {elapsed:.1f} minutes")
    print(f"  Model:  {MODEL_PATH}")
    print(f"  Scaler: {SCALER_PATH}")
    print(f"  Val Acc: {val_acc:.4f}  |  Val AUC: {val_auc:.4f}")
    print(f"\n  NEXT STEP: Copy lstm_model.keras and lstm_scaler.pkl")
    print(f"  to D:\\FinFolioX\\saved_models\\ and run test_lstm.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
