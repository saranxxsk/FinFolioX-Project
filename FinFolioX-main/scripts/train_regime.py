"""
ml_engine/train_hybrid_regime.py
==================================
FinFolioX HOLD One-Time Training Script for Hybrid Regime Agent v2.3.1

Run ONCE from project root:
    python ml_engine/train_hybrid_regime.py

Creates: saved_models/hmm_regime_hybrid.pkl

What it does:
  1. Downloads ^GSPC (2003-2024) via yfinance HOLD falls back to synthetic if offline
  2. Runs BIC validation to confirm 3 states is optimal
  3. Trains the pure-NumPy GaussianHMM
  4. Saves model + scaler + regime_map to saved_models/
  5. Runs a quick sanity check on the saved model

After this, FinFolioSystem loads it automatically via:
    HybridRegimeAgent(hmm_model_path="saved_models/hmm_regime_hybrid.pkl")

You do NOT need to retrain unless you want to update the training window.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

# -- path setup -----------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from ml_engine.hybrid_regime_agent import HybridRegimeAgent, _synthetic_ohlcv

MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "saved_models", "hmm_regime_hybrid.pkl")
TRAIN_START     = "2003-01-01"
TRAIN_END       = "2024-12-31"
TICKER          = "^GSPC"


def main():
    print("=" * 60)
    print("  FinFolioX HOLD Hybrid Regime Agent v2.3.1 Training")
    print("=" * 60)

    # -- 1. Get training data ----------------------------------
    train_df = None
    print(f"\n📥 Downloading {TICKER} ({TRAIN_START} -> {TRAIN_END})...")
    try:
        import yfinance as yf
        df = yf.download(TICKER, start=TRAIN_START, end=TRAIN_END,
                         auto_adjust=True, progress=False)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.dropna(inplace=True)
        if len(df) > 200:
            train_df = df
            print(f"   [OK] Downloaded  |  {len(train_df)} trading days")
        else:
            print("   [WARN]  Too few rows HOLD using synthetic data")
    except Exception as e:
        print(f"   [WARN]  Download failed ({type(e).__name__}) HOLD using synthetic data")

    if train_df is None:
        print("   🔄 Generating synthetic OHLCV (2003–2024, seed=42)...")
        train_df = _synthetic_ohlcv(TRAIN_START, TRAIN_END, seed=42)
        print(f"   [OK] Synthetic data ready  |  {len(train_df)} trading days")

    # -- 2. Train ----------------------------------------------
    print("\n⚙️  Training...")
    agent = HybridRegimeAgent(hmm_model_path=None, verbose=True)
    agent.train_on_df(train_df, run_bic=True)

    # -- 3. Save -----------------------------------------------
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    agent.save(MODEL_SAVE_PATH)
    print(f"\n💾 Saved -> {MODEL_SAVE_PATH}")

    # -- 4. Sanity check ---------------------------------------
    print("\n[CHECK] Loading back and running sanity check...")
    loaded = HybridRegimeAgent(hmm_model_path=MODEL_SAVE_PATH, verbose=False)

    # Use last 500 days of training data as test slice
    test_df = train_df.iloc[-500:] if len(train_df) >= 500 else train_df
    label, vol, conf = loaded.detect(test_df)
    print(f"   detect() -> regime='{label}'  vol={vol:.5f}  conf={conf:.2f}")

    assert label in ("Bull", "Bear", "Sideways"), f"Bad label: {label}"
    assert 0.001 < vol < 0.20,  f"Vol out of range: {vol}"
    assert 0.0 < conf <= 1.0,   f"Conf out of range: {conf}"

    # Also test analyze_regime() (T1-T10 interface)
    l2, v2 = loaded.analyze_regime(test_df)
    assert l2 in ("Bull", "Bear", "Sideways"), f"Bad label from analyze_regime: {l2}"
    assert 0.001 < v2 < 0.20, f"Vol out of range from analyze_regime: {v2}"
    print(f"   analyze_regime() -> ('{l2}', {v2:.5f})")

    print("\n   [OK] All assertions passed.")
    print("\n" + "=" * 60)
    print("  Training complete.")
    print(f"  Model: {MODEL_SAVE_PATH}")
    print("  Next:  python validate_hybrid_regime.py")
    print("=" * 60)


if __name__ == "__main__":
    main()