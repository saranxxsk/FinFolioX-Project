"""
PHASE 11: ADVERSARIAL ROBUSTNESS (v7 HOLD Red Team)
-------------------------------------------------
Replaces the old AdversarialTester with v7 logic:

  - Correct per-window bias detection (confidently wrong, not crash-delta)
  - Score volatility tracking + weight penalty
  - Minimum scored-window floor before trusting accuracy
  - LSTM agent weight recommendations for the aggregator
  - Saturation check in diagnostic (full-history)
  - All feature engineering matches training pipeline exactly
"""

import logging
import numpy as np
import pandas as pd
import joblib
import os

import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

# ==============================================================================
# CONFIG
# ==============================================================================
SEQ_LEN   = 100
LSTM_COLS = [
    "log_return", "vol_change", "sma10_dist",
    "sma20_dist", "sma50_dist", "RSI", "macd_norm",
]

BUY_THRESHOLD   = 0.52
SELL_THRESHOLD  = 0.48
CRASH_MAGNITUDE = 0.40

# Crash reaction threshold
CRASH_DELTA_THRESHOLD = 0.01

# Diagnostic saturation (full-history run only)
DIAG_SATURATION_SCORE = 0.70
DIAG_SATURATION_DELTA = 0.015

# Per-window bias: "confidently wrong"
# score > this AND correct_loose == False
BIAS_SCORE_THRESHOLD = 0.70

# Agent weight config
MIN_SCOREABLE_FOR_WEIGHT = 2
WEIGHT_HIGH      = 1.0
WEIGHT_MEDIUM    = 0.6
WEIGHT_UNCERTAIN = 0.4
WEIGHT_LOW       = 0.2
WEIGHT_NONE      = 0.0
SCORE_STD_HIGH_THRESHOLD = 0.20

# Default FLAT threshold (±0.75% for 5-day windows)
DEFAULT_FLAT_THRESHOLD = 0.0075

# Minimum raw rows before a window is usable
MIN_RAW_ROWS = 250


# ==============================================================================
# FEATURE ENGINEERING  (identical to training pipeline)
# ==============================================================================
def _rsi(series: pd.Series, p: int = 14) -> pd.Series:
    d    = series.diff()
    gain = d.clip(lower=0).ewm(com=p - 1, min_periods=p).mean()
    loss = -d.clip(upper=0).ewm(com=p - 1, min_periods=p).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-9)))


def _macd(series: pd.Series, fast=12, slow=26, sig=9) -> pd.Series:
    line = (series.ewm(span=fast, adjust=False).mean()
            - series.ewm(span=slow, adjust=False).mean())
    return line - line.ewm(span=sig, adjust=False).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    out["vol_change"] = df["Volume"].pct_change().clip(-5.0, 5.0)
    out["sma10_dist"] = ((df["Close"] - df["Close"].rolling(10).mean())
                         / df["Close"].rolling(10).mean())
    out["sma20_dist"] = ((df["Close"] - df["Close"].rolling(20).mean())
                         / df["Close"].rolling(20).mean())
    out["sma50_dist"] = ((df["Close"] - df["Close"].rolling(50).mean())
                         / df["Close"].rolling(50).mean())
    out["RSI"]       = _rsi(df["Close"])
    out["macd_norm"] = _macd(df["Close"]) / df["Close"]
    return out.replace([np.inf, -np.inf], np.nan).dropna()


# ==============================================================================
# ADVERSARIAL TESTER CLASS
# ==============================================================================
class AdversarialTester:
    """
    Drop-in replacement for the old AdversarialTester.

    Usage (same as before):
        tester = AdversarialTester(master_system)
        tester.run_robustness_test("AAPL")

    New:
        tester.run_full_backtest(ticker, windows, raw_hist)
        -> returns dict with accuracy, bias rate, and lstm_weight
    """

    def __init__(self, master_system,
                 flat_threshold: float = DEFAULT_FLAT_THRESHOLD):
        self.system         = master_system
        self.flat_threshold = flat_threshold
        self.logger         = logging.getLogger("RedTeam.v7")

        # Load LSTM model + scaler from master_system paths
        self._model  = None
        self._scaler = None
        self._load_lstm()

    # -- Model loading ---------------------------------------------------------
    def _load_lstm(self):
        try:
            base       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path  = os.path.join(base, "saved_models", "lstm_model.keras")
            scaler_path = os.path.join(base, "saved_models", "lstm_scaler.pkl")
            self._model  = tf.keras.models.load_model(model_path)
            self._scaler = joblib.load(scaler_path)
            self.logger.info("LSTM model + scaler loaded.")
        except Exception as e:
            self.logger.warning(f"Could not load LSTM: {e}. "
                                "predict_direct() will fall back to master_system.")

    # -- Feature + prediction --------------------------------------------------
    def _predict_direct(self, raw_df: pd.DataFrame) -> float:
        """
        Runs raw OHLCV data through feature engineering -> LSTM -> float score.
        Falls back to master_system.tech_agent if LSTM not loaded.
        """
        if self._model is not None and self._scaler is not None:
            feat = build_features(raw_df)
            if len(feat) < SEQ_LEN:
                raise ValueError(
                    f"Only {len(feat)} feature rows after engineering "
                    f"(need {SEQ_LEN})")
            seq    = feat[LSTM_COLS].tail(SEQ_LEN).values
            scaled = (self._scaler.transform(seq)
                      .reshape(1, SEQ_LEN, len(LSTM_COLS))
                      .astype(np.float32))
            return float(
                self._model(tf.constant(scaled), training=False).numpy()[0][0])

        # Fallback: use master_system's tech agent
        agent = self.system.tech_agent
        if hasattr(agent, "predict_signal"):
            return agent.predict_signal(raw_df)
        return agent.predict(raw_df)

    # -- Crash injection -------------------------------------------------------
    def generate_flash_crash(self, hist_df: pd.DataFrame,
                             drop_pct: float = CRASH_MAGNITUDE) -> pd.DataFrame:
        """
        Pure price-based crash injection HOLD no synthetic indicator overrides.
        The model must react to the price movement itself, not hand-crafted RSI=5.
        (Removes the old RSI=5 / MACD=-10 nuclear override.)
        """
        c   = hist_df.copy()
        idx = c.index[-1]
        nc  = c.loc[idx, "Close"] * (1.0 - drop_pct)
        c.loc[idx, "Close"]  = nc
        c.loc[idx, "Low"]    = nc
        c.loc[idx, "Open"]   = nc * 1.01
        c.loc[idx, "High"]   = nc * 1.02
        c.loc[idx, "Volume"] = c.loc[idx, "Volume"] * 5.0
        return c

    # -- Signal helpers --------------------------------------------------------
    @staticmethod
    def _signal(s: float) -> str:
        if s >= BUY_THRESHOLD:  return "BUY"
        if s <= SELL_THRESHOLD: return "SELL"
        return "HOLD"

    @staticmethod
    def _direction(p0: float, p1: float, threshold: float) -> str:
        r = (p1 - p0) / p0
        if r >  threshold: return "UP"
        if r < -threshold: return "DOWN"
        return "FLAT"

    @staticmethod
    def _correct_strict(signal: str, direction: str) -> bool:
        return ((signal == "BUY"  and direction == "UP") or
                (signal == "SELL" and direction == "DOWN"))

    @staticmethod
    def _correct_loose(signal: str, direction: str):
        if direction == "FLAT":
            return None
        return AdversarialTester._correct_strict(signal, direction)

    @staticmethod
    def _crash_detected(normal: float, crashed: float) -> bool:
        return abs(normal - crashed) > CRASH_DELTA_THRESHOLD

    @staticmethod
    def _diag_saturated(score: float, delta: float) -> bool:
        return ((score > DIAG_SATURATION_SCORE
                 or score < (1 - DIAG_SATURATION_SCORE))
                and abs(delta) < DIAG_SATURATION_DELTA)

    @staticmethod
    def _window_bias(score: float, correct_loose_val) -> bool:
        """
        v7 definition: CONFIDENTLY WRONG.
        score far from neutral (>0.70 or <0.30) AND correct_loose is False.
        FLAT windows (None) are skipped HOLD can't be wrong on a FLAT move.
        """
        if correct_loose_val is None:
            return False
        if correct_loose_val:
            return False
        return (score > BIAS_SCORE_THRESHOLD
                or score < (1.0 - BIAS_SCORE_THRESHOLD))

    @staticmethod
    def _recommend_weight(loose_acc: float, bias_rate: float,
                          score_std: float, n_scoreable: int) -> float:
        if n_scoreable < MIN_SCOREABLE_FOR_WEIGHT:
            return WEIGHT_UNCERTAIN
        if bias_rate >= 0.50:
            return WEIGHT_NONE
        if   loose_acc >= 80: base = WEIGHT_HIGH
        elif loose_acc >= 55: base = WEIGHT_MEDIUM
        elif loose_acc >= 30: base = WEIGHT_LOW
        else:                 base = WEIGHT_NONE
        if score_std > SCORE_STD_HIGH_THRESHOLD:
            ladder = [WEIGHT_NONE, WEIGHT_LOW, WEIGHT_UNCERTAIN,
                      WEIGHT_MEDIUM, WEIGHT_HIGH]
            idx  = ladder.index(base) if base in ladder else 0
            base = ladder[max(0, idx - 1)]
        return base

    @staticmethod
    def _localize(ts: pd.Timestamp, ref: pd.DatetimeIndex) -> pd.Timestamp:
        if ref.tz is not None and ts.tz is None:
            return ts.tz_localize(ref.tz)
        if ref.tz is None and ts.tz is not None:
            return ts.tz_localize(None)
        return ts

    # =========================================================================
    # PUBLIC API HOLD replaces run_robustness_test()
    # =========================================================================
    def run_robustness_test(self, ticker: str):
        """
        Single-ticker robustness test HOLD same entry point as before.
        Now uses v7 feature engineering, clean crash injection,
        and v7 saturation detection. Prints a clear report card.
        """
        print("\n" + "!" * 60)
        print(f"🧪 PHASE 11 v7: STRESS TEST HOLD {ticker}")
        print("!" * 60)

        # Fetch data via master_system
        raw_df = self._fetch_raw(ticker)
        if raw_df is None:
            return

        # -- Diagnostic (full history) -----------------------------------------
        print("\n[CHECK] DIAGNOSTIC (most recent 100-bar window)")
        print("-" * 50)
        try:
            normal_score  = self._predict_direct(raw_df)
            crashed_score = self._predict_direct(
                self.generate_flash_crash(raw_df, CRASH_MAGNITUDE))
            delta         = normal_score - crashed_score
            sig           = self._signal(normal_score)
            sat           = self._diag_saturated(normal_score, delta)

            react = ("[OK] REACTED"
                     if abs(delta) > CRASH_DELTA_THRESHOLD
                     else "[WARN]  NO REACTION")
            dirn  = ("↑ MORE BULLISH (oversold-bounce)"
                     if delta < 0 else "↓ more bearish")
            b_flag = "[WARN]  SATURATION BIAS" if sat else "[OK] not saturated"

            print(f"   Normal    : {normal_score:.6f}  -> {sig}")
            print(f"   Crashed   : {crashed_score:.6f}  -> "
                  f"{self._signal(crashed_score)}")
            print(f"   Delta     : {delta:+.6f}  {react}  {dirn}")
            print(f"   Saturation: {b_flag}")
            if sat:
                print(f"   [WARN]  Score {normal_score:.3f} is far from 0.5 AND "
                      f"delta {abs(delta):.4f} < {DIAG_SATURATION_DELTA}.")
                print(f"      Reduce LSTM weight for {ticker} in aggregator.")
        except Exception as e:
            print(f"   [BAD] Diagnostic failed: {e}")
            return

        # -- Report card -------------------------------------------------------
        print("\n" + "=" * 50)
        print("🛡️  ROBUSTNESS REPORT CARD")
        print("=" * 50)

        score_drop = normal_score - crashed_score
        if self._crash_detected(normal_score, crashed_score):
            print(f"[OK] PASS: Model reacted to the crash.")
            dirn_str = ("↑ bullish (mean-reversion)"
                        if score_drop < 0 else "↓ bearish (danger detection)")
            print(f"   Score change: {score_drop:+.4f}  {dirn_str}")
        else:
            print(f"[BAD] FAIL: Model ignored the crash.")
            print(f"   Score change: {score_drop:+.4f} (below threshold "
                  f"{CRASH_DELTA_THRESHOLD})")

        if sat:
            print(f"\n[WARN]  SATURATION WARNING: Model is pinned at {normal_score:.3f}.")
            print(f"   Even a -{CRASH_MAGNITUDE*100:.0f}% crash moved it by "
                  f"only {abs(delta):.4f}. LSTM weight should be 0.0 for {ticker}.")

        print("\nTest complete.\n")

    def run_full_backtest(self, ticker: str,
                          windows: list,
                          raw_hist: pd.DataFrame) -> dict:
        """
        Multi-window adversarial backtest with v7 accuracy + bias scoring.

        Args:
            ticker    : e.g. "AAPL"
            windows   : list of (start_date, end_date, label) tuples
            raw_hist  : full OHLCV DataFrame (from yfinance or master_system)

        Returns:
            dict with keys:
              strict_acc, loose_acc, robustness_rate, bias_rate,
              score_std, lstm_weight, results (list of per-window dicts)
        """
        print(f"\n{'-'*60}")
        print(f"📈 FULL BACKTEST: {ticker}  ({len(windows)} windows)")
        print(f"{'-'*60}")

        results = []

        for (start, end, label) in windows:
            r = self._run_window(ticker, start, end, label,
                                 raw_hist, self.flat_threshold)
            results.append(r)

            if r["error"]:
                print(f"   [WARN]  {label}: {str(r['error']).split(chr(10))[0]}")
                continue

            dir_icon = {"UP": "📈", "DOWN": "📉", "FLAT": "➡️"}.get(
                r["actual_dir"], "?")
            s_ok = ("[OK]" if r["correct_strict"]
                    else ("➡️ FLAT" if r["actual_dir"] == "FLAT" else "[BAD]"))
            l_ok = ("[OK]" if r["correct_loose"]
                    else ("⬜ SKIP" if r["correct_loose"] is None else "[BAD]"))
            bias = "[WARN]  BIAS" if r["bias_detected"] else "HOLD"

            print(f"\n   🔬 {label}  [{start} -> {end}]")
            print(f"      Score  : {r['normal_score']:.4f}  -> "
                  f"{r['normal_signal']}")
            print(f"      Actual : {r['actual_return']:+.2%} "
                  f"{dir_icon} {r['actual_dir']}")
            print(f"      Strict:{s_ok}  Loose:{l_ok}  Bias:{bias}")

        # -- Aggregate stats ---------------------------------------------------
        valid   = [r for r in results
                   if not r.get("error") and r.get("actual_dir")]
        dir_r   = [r for r in valid if r["normal_signal"] != "HOLD"]
        l_act   = [r for r in dir_r if r["correct_loose"] is not None]
        l_cor   = [r for r in l_act if r["correct_loose"]]
        s_cor   = [r for r in dir_r if r["correct_strict"]]
        biased  = [r for r in valid if r["bias_detected"]]
        robust  = [r for r in valid if r["crash_detected"]]
        scores  = [r["normal_score"] for r in valid]

        s_acc   = len(s_cor) / len(dir_r)  * 100 if dir_r else 0.0
        l_acc   = len(l_cor) / len(l_act) * 100  if l_act else 0.0
        rob     = len(robust) / len(valid) * 100  if valid else 0.0
        b_rate  = len(biased) / len(valid)        if valid else 0.0
        std     = float(np.std(scores))            if scores else 0.0

        weight  = self._recommend_weight(l_acc, b_rate, std, len(l_act))

        print(f"\n{'-'*60}")
        print(f"   Strict Acc   : {s_acc:.1f}%  ({len(s_cor)}/{len(dir_r)})")
        print(f"   Loose Acc    : {l_acc:.1f}%  ({len(l_cor)}/{len(l_act)})")
        print(f"   Robustness   : {rob:.1f}%  ({len(robust)}/{len(valid)})")
        print(f"   Bias Rate    : {b_rate*100:.1f}%  ({len(biased)}/{len(valid)})")
        print(f"   Score Std    : {std:.3f}"
              f"{'  [WARN]  HIGH' if std > SCORE_STD_HIGH_THRESHOLD else ''}")
        bar = "█" * int(weight * 10) + "░" * (10 - int(weight * 10))
        print(f"   LSTM Weight  : {bar}  {weight:.1f}")
        print(f"{'-'*60}\n")

        return dict(
            ticker=ticker,
            strict_acc=s_acc,
            loose_acc=l_acc,
            robustness_rate=rob,
            bias_rate=b_rate,
            score_std=std,
            lstm_weight=weight,
            results=results,
        )

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================
    def _fetch_raw(self, ticker: str):
        """Fetch raw OHLCV from master_system using whatever method exists."""
        try:
            if hasattr(self.system, "_fetch_stock_data"):
                _, df = self.system._fetch_stock_data(ticker)
            elif hasattr(self.system, "fetch_market_data"):
                _, df = self.system.fetch_market_data(ticker)
            else:
                print("[BAD] master_system has no known fetch method.")
                return None
            if df is None or df.empty:
                print(f"[BAD] No data returned for {ticker}.")
                return None
            return df
        except Exception as e:
            print(f"[BAD] Data fetch failed: {e}")
            return None

    def _run_window(self, ticker, start, end, label,
                    raw_hist, flat_threshold) -> dict:
        """Run one adversarial window. Returns a result dict."""
        r = dict(
            ticker=ticker, window=label, start=start, end=end,
            price_start=None, price_end=None,
            actual_return=None, actual_dir=None,
            normal_score=None, normal_signal=None,
            crashed_score=None,
            score_delta=None, crash_detected=None,
            bias_detected=None,
            correct_strict=None, correct_loose=None, error=None,
        )
        try:
            s_dt = self._localize(pd.Timestamp(start), raw_hist.index)
            e_dt = self._localize(pd.Timestamp(end),   raw_hist.index)

            raw_as_of = raw_hist[raw_hist.index < s_dt].copy()
            if len(raw_as_of) < MIN_RAW_ROWS:
                r["error"] = (f"Only {len(raw_as_of)} rows before window "
                              f"(need {MIN_RAW_ROWS})")
                return r

            ns    = self._predict_direct(raw_as_of)
            cs    = self._predict_direct(
                self.generate_flash_crash(raw_as_of, CRASH_MAGNITUDE))
            sig   = self._signal(ns)
            delta = ns - cs

            wd = raw_hist[
                (raw_hist.index >= s_dt) & (raw_hist.index <= e_dt)]
            if wd.empty:
                r.update(normal_score=ns, normal_signal=sig,
                         crashed_score=cs, score_delta=delta,
                         crash_detected=self._crash_detected(ns, cs),
                         bias_detected=False,
                         error="No price data in window")
                return r

            p0        = float(wd["Close"].iloc[0])
            p1        = float(wd["Close"].iloc[-1])
            ret       = (p1 - p0) / p0
            direction = self._direction(p0, p1, flat_threshold)
            cl        = self._correct_loose(sig, direction)

            r.update(
                price_start=p0, price_end=p1,
                actual_return=ret, actual_dir=direction,
                normal_score=ns, normal_signal=sig,
                crashed_score=cs, score_delta=delta,
                crash_detected=self._crash_detected(ns, cs),
                bias_detected=self._window_bias(ns, cl),
                correct_strict=self._correct_strict(sig, direction),
                correct_loose=cl,
            )
        except Exception as exc:
            import traceback
            r["error"] = f"{exc}\n{traceback.format_exc()}"
        return r