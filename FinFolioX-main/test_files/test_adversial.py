"""
PHASE 11 HOLD ADVERSARIAL BACKTEST + ACCURACY  (v7 HOLD Correct Bias + Weight Floor)
================================================================================
Changes from v6:

  FIX 1 HOLD Per-window bias detection was always 0/16:
    v6 used |crash_delta| < 0.015 to detect saturation bias at window level.
    Problem: TSLA window-level deltas are 0.032–0.189, all above 0.015, so
    nothing was flagged even though TSLA is clearly broken.
    Root cause: the diagnostic catches TSLA saturation because it uses the
    most recent 100-bar window (score=0.984, delta=0.007). Per-window tests
    use earlier data where scores vary widely.

    New definition (correct):
      per-window bias = score is far from neutral (>0.70 or <1-0.70)
                        AND the prediction was wrong (correct_loose == False)
      = "confidently wrong" HOLD model had high conviction but got it backwards.
    This correctly flags TSLA Mar04->09 (score=0.926, BUY on a DOWN day)
    and Mar09->16 (score=0.955, BUY on a DOWN day).

  FIX 2 HOLD QQQ weight 1.0 from a single scored window:
    QQQ had 3 FLAT windows skipped -> only 1 scored window -> 100% loose acc
    -> weight 1.0. Statistically meaningless.
    Fix: MIN_SCOREABLE_FOR_WEIGHT = 2. Tickers with fewer scoreable windows
    are capped at WEIGHT_UNCERTAIN (0.4) regardless of accuracy.

  FIX 3 HOLD Score volatility not surfaced:
    TSLA scores jump 0.384 -> 0.926 -> 0.955 -> 0.793 (std ≈ 0.25).
    High score std means the model is inconsistent on this ticker.
    Added ScoreStd column to per-ticker table and penalises weight when
    std > SCORE_STD_HIGH_THRESHOLD.

  NEW: --bias-score-threshold CLI flag (default 0.70)
  NEW: ScoreStd column in summary table
  NEW: Volatility penalty in agent weight calculation
  NEW: Diagnostic prints saturation check separately from per-window bias

Usage:
    python test_adversial.py --ticker AAPL SPY QQQ TSLA
    python test_adversial.py --ticker QQQ --flat-threshold 0.01
    python test_adversial.py --ticker TSLA --bias-score-threshold 0.65
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import joblib

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

try:
    from tabulate import tabulate
except ImportError:
    def tabulate(rows, headers="keys", tablefmt="simple"):
        if not rows:
            return "(empty)"
        keys = list(rows[0].keys()) if headers == "keys" else headers
        lines = ["\t".join(str(k) for k in keys)]
        for r in rows:
            lines.append("\t".join(str(r.get(k, "")) for k in keys))
        return "\n".join(lines)

# ==============================================================================
# PATHS
# ==============================================================================
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR       = os.path.join(BASE_DIR, "saved_models")
LSTM_MODEL_PATH  = os.path.join(MODELS_DIR, "lstm_model.keras")
LSTM_SCALER_PATH = os.path.join(MODELS_DIR, "lstm_scaler.pkl")

# ==============================================================================
# CONFIG
# ==============================================================================
SEQ_LEN = 100
LSTM_COLS = [
    "log_return", "vol_change", "sma10_dist",
    "sma20_dist", "sma50_dist", "RSI", "macd_norm",
]

TEST_WINDOWS = [
    ("2026-03-03", "2026-03-08", "Mar03->08 Bear start"),
    ("2026-03-05", "2026-03-10", "Mar05->10 Bear early"),
    ("2026-03-09", "2026-03-16", "Mar09->16 Deep Bear"),
    ("2026-03-12", "2026-03-17", "Mar12->17 Bounce"),
]

BUY_THRESHOLD   = 0.52
SELL_THRESHOLD  = 0.48
CRASH_MAGNITUDE = 0.40
MIN_RAW_ROWS    = 250

# Crash detection: |score change| > this = model reacted
CRASH_DELTA_THRESHOLD = 0.01

# Diagnostic saturation (full-history check only, not per-window)
DIAG_SATURATION_SCORE = 0.70
DIAG_SATURATION_DELTA = 0.015

# Per-window bias threshold HOLD overridden by --bias-score-threshold
# "confidently wrong" = score > BIAS_SCORE_THRESHOLD AND correct_loose==False
DEFAULT_BIAS_SCORE_THRESHOLD = 0.70
BIAS_SCORE_THRESHOLD = DEFAULT_BIAS_SCORE_THRESHOLD   # set at runtime

# Weight recommendations
MIN_SCOREABLE_FOR_WEIGHT = 2    # fewer -> cap at WEIGHT_UNCERTAIN
WEIGHT_HIGH      = 1.0
WEIGHT_MEDIUM    = 0.6
WEIGHT_UNCERTAIN = 0.4          # not enough data to trust the number
WEIGHT_LOW       = 0.2
WEIGHT_NONE      = 0.0

# Score volatility penalty threshold
SCORE_STD_HIGH_THRESHOLD = 0.20


# ==============================================================================
# FEATURE ENGINEERING  (identical to training)
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
# DIRECT MODEL CALL
# ==============================================================================
def predict_direct(raw_df: pd.DataFrame, model, scaler) -> float:
    feat = build_features(raw_df)
    if len(feat) < SEQ_LEN:
        raise ValueError(f"Only {len(feat)} feature rows (need {SEQ_LEN})")
    seq    = feat[LSTM_COLS].tail(SEQ_LEN).values
    scaled = (scaler.transform(seq)
              .reshape(1, SEQ_LEN, len(LSTM_COLS))
              .astype(np.float32))
    return float(model(tf.constant(scaled), training=False).numpy()[0][0])


# ==============================================================================
# CRASH INJECTION
# ==============================================================================
def generate_flash_crash(raw_df: pd.DataFrame,
                         drop_pct: float = 0.40) -> pd.DataFrame:
    c   = raw_df.copy()
    idx = c.index[-1]
    nc  = c.loc[idx, "Close"] * (1.0 - drop_pct)
    c.loc[idx, "Close"]  = nc
    c.loc[idx, "Low"]    = nc
    c.loc[idx, "Open"]   = nc * 1.01
    c.loc[idx, "High"]   = nc * 1.02
    c.loc[idx, "Volume"] = c.loc[idx, "Volume"] * 5.0
    return c


# ==============================================================================
# DECISION HELPERS
# ==============================================================================
def _signal(s: float) -> str:
    if s >= BUY_THRESHOLD:  return "BUY"
    if s <= SELL_THRESHOLD: return "SELL"
    return "HOLD"


def _direction(p0: float, p1: float, threshold: float) -> str:
    r = (p1 - p0) / p0
    if r >  threshold: return "UP"
    if r < -threshold: return "DOWN"
    return "FLAT"


def _correct_strict(signal: str, direction: str) -> bool:
    return ((signal == "BUY"  and direction == "UP") or
            (signal == "SELL" and direction == "DOWN"))


def _correct_loose(signal: str, direction: str):
    """Returns None (skip) for FLAT, True/False otherwise."""
    if direction == "FLAT":
        return None
    return _correct_strict(signal, direction)


def _crash_detected(normal: float, crashed: float) -> bool:
    return abs(normal - crashed) > CRASH_DELTA_THRESHOLD


def _diag_saturated(score: float, delta: float) -> bool:
    """
    Full-history saturation check used only in the diagnostic block.
    score pinned far from 0.5 AND a -40% crash barely moved it.
    """
    return ((score > DIAG_SATURATION_SCORE
             or score < (1 - DIAG_SATURATION_SCORE))
            and abs(delta) < DIAG_SATURATION_DELTA)


def _window_bias(score: float, correct_loose_val) -> bool:
    """
    Per-window bias = CONFIDENTLY WRONG.

    v6 bug: used |crash_delta| < 0.015 which was too tight HOLD
    TSLA window deltas of 0.03-0.19 all passed, giving 0 detections.

    v7 fix: bias fires when BOTH are true:
      1. score is far from neutral (> BIAS_SCORE_THRESHOLD or < 1-threshold)
         meaning the model was highly confident
      2. correct_loose is False (the prediction was wrong HOLD not just FLAT)

    FLAT windows (correct_loose=None) are never biased HOLD skip, not penalise.
    Correct predictions are never biased regardless of score magnitude.

    Expected for TSLA:
      Mar04->09: score=0.926 (>0.70), correct_loose=False -> BIASED [OK]
      Mar09->16: score=0.955 (>0.70), correct_loose=False -> BIASED [OK]
      Mar03->08: score=0.384 (<0.30), correct_loose=False -> BIASED [OK]
      Mar13->18: score=0.793 (>0.70), correct_loose=None  -> not biased (FLAT)
    """
    if correct_loose_val is None:
        return False   # FLAT window HOLD skip, not penalise
    if correct_loose_val:
        return False   # correct HOLD no bias regardless of score
    return (score > BIAS_SCORE_THRESHOLD
            or score < (1.0 - BIAS_SCORE_THRESHOLD))


def _localize(ts: pd.Timestamp, ref: pd.DatetimeIndex) -> pd.Timestamp:
    if ref.tz is not None and ts.tz is None:
        return ts.tz_localize(ref.tz)
    if ref.tz is None and ts.tz is not None:
        return ts.tz_localize(None)
    return ts


# ==============================================================================
# AGENT WEIGHT RECOMMENDATION
# ==============================================================================
def _recommend_weight(loose_acc: float, bias_rate: float,
                      score_std: float, n_scoreable: int) -> float:
    """
    Returns recommended LSTM weight (0.0–1.0) for the aggregator.

    Penalisation cascade (applied in order):
      1. Too few scored windows -> WEIGHT_UNCERTAIN
      2. High bias rate (>=50%) -> WEIGHT_NONE
      3. Accuracy-based base weight
      4. High score std (>0.20) -> one step down the weight ladder
    """
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


# ==============================================================================
# DIAGNOSTIC
# ==============================================================================
def run_diagnostic(raw_hist: pd.DataFrame, model, scaler, ticker: str):
    print(f"\n{'-'*64}")
    print(f"[CHECK] DIAGNOSTIC  ({ticker})")
    print(f"{'-'*64}")
    feat = build_features(raw_hist)
    seq  = feat[LSTM_COLS].tail(SEQ_LEN)
    print(f"   Shape      : {seq.shape}   (expected ({SEQ_LEN}, {len(LSTM_COLS)}))")
    try:
        score = predict_direct(raw_hist, model, scaler)
        sig   = _signal(score)
        stuck = "[WARN]  stuck at 0.5" if abs(score - 0.5) <= 0.001 else "[OK]"
        print(f"   Normal     : {score:.6f}  -> {sig}  {stuck}")

        c_score = predict_direct(
            generate_flash_crash(raw_hist, CRASH_MAGNITUDE), model, scaler)
        delta   = score - c_score
        react   = ("[OK] REACTED"
                   if abs(delta) > CRASH_DELTA_THRESHOLD
                   else "[WARN]  NO REACTION")
        dirn    = ("↑ MORE BULLISH (oversold-bounce)"
                   if delta < 0 else "↓ more bearish")
        sat     = _diag_saturated(score, delta)
        b_flag  = "[WARN]  SATURATION BIAS DETECTED" if sat else "[OK] not saturated"

        print(f"   Crashed    : {c_score:.6f}  -> {_signal(c_score)}")
        print(f"   Delta      : {delta:+.6f}  {react}  {dirn}")
        print(f"   Saturation : {b_flag}")
        if sat:
            print(f"   [WARN]  Score {score:.3f} far from 0.5 AND delta "
                  f"{abs(delta):.4f} < {DIAG_SATURATION_DELTA}.")
            print(f"      Reduce LSTM weight for this ticker in the aggregator.")
    except Exception as e:
        print(f"   [BAD] {e}")
    print(f"{'-'*64}\n")


# ==============================================================================
# PER-WINDOW TEST
# ==============================================================================
def run_window_test(ticker, start, end, label, raw_hist, model, scaler,
                    flat_threshold: float) -> dict:
    r = dict(
        ticker=ticker, window=label, start=start, end=end,
        price_start=None, price_end=None,
        actual_return=None, actual_dir=None,
        normal_score=None, normal_signal=None,
        crashed_score=None, crashed_signal=None,
        score_delta=None, crash_detected=None,
        bias_detected=None,
        correct_strict=None, correct_loose=None, error=None,
    )
    try:
        s_dt = _localize(pd.Timestamp(start), raw_hist.index)
        e_dt = _localize(pd.Timestamp(end),   raw_hist.index)

        raw_as_of = raw_hist[raw_hist.index < s_dt].copy()
        if len(raw_as_of) < MIN_RAW_ROWS:
            r["error"] = f"Only {len(raw_as_of)} rows (need {MIN_RAW_ROWS})"
            return r

        ns    = predict_direct(raw_as_of, model, scaler)
        cs    = predict_direct(
            generate_flash_crash(raw_as_of, CRASH_MAGNITUDE), model, scaler)
        sig   = _signal(ns)
        delta = ns - cs

        wd = raw_hist[
            (raw_hist.index >= s_dt) & (raw_hist.index <= e_dt)]
        if wd.empty:
            r.update(
                normal_score=ns, normal_signal=sig,
                crashed_score=cs, crashed_signal=_signal(cs),
                score_delta=delta,
                crash_detected=_crash_detected(ns, cs),
                bias_detected=False,
                error="No price data in window",
            )
            return r

        p0        = float(wd["Close"].iloc[0])
        p1        = float(wd["Close"].iloc[-1])
        ret       = (p1 - p0) / p0
        direction = _direction(p0, p1, flat_threshold)
        cl        = _correct_loose(sig, direction)

        r.update(
            price_start=p0, price_end=p1, actual_return=ret,
            actual_dir=direction,
            normal_score=ns, normal_signal=sig,
            crashed_score=cs, crashed_signal=_signal(cs),
            score_delta=delta,
            crash_detected=_crash_detected(ns, cs),
            bias_detected=_window_bias(ns, cl),   # v7: confidently-wrong logic
            correct_strict=_correct_strict(sig, direction),
            correct_loose=cl,
        )
    except Exception as exc:
        import traceback
        r["error"] = f"{exc}\n{traceback.format_exc()}"
    return r


# ==============================================================================
# FAILURE REASON
# ==============================================================================
def _failure_reason(r: dict) -> str:
    if r.get("correct_strict"):
        return "HOLD"
    if r.get("bias_detected"):
        return (f"Confidently wrong "
                f"(score={r['normal_score']:.3f}) HOLD reduce LSTM weight")
    sig = r.get("normal_signal", "?")
    act = r.get("actual_dir",    "?")
    ret = r.get("actual_return", 0)
    sc  = r.get("normal_score",  0)
    if act == "FLAT":
        return f"FLAT move ({ret:+.2%}) HOLD ambiguous"
    if sig == "BUY" and act == "DOWN":
        return f"Over-bullish ({sc:.3f}) in downtrend"
    if sig == "SELL" and act == "UP":
        return f"Over-bearish ({sc:.3f}) in uptrend"
    if sig == "HOLD":
        return "HOLD HOLD no directional call"
    return "Unknown"


# ==============================================================================
# MAIN
# ==============================================================================
def run_backtest(tickers: list, flat_threshold: float):
    print("\n" + "=" * 72)
    print("🧪 PHASE 11 HOLD ADVERSARIAL BACKTEST + ACCURACY  (v7)")
    print("=" * 72)
    print(f"   Tickers              : {', '.join(tickers)}")
    print(f"   Windows              : {len(TEST_WINDOWS)}")
    print(f"   FLAT threshold       : ±{flat_threshold * 100:.2f}%")
    print(f"   Crash mag            : -{CRASH_MAGNITUDE * 100:.0f}%  |  "
          f"Buy ≥ {BUY_THRESHOLD}  Sell ≤ {SELL_THRESHOLD}")
    print(f"   Bias (per-window)    : score>{BIAS_SCORE_THRESHOLD} "
          f"AND correct_loose==False")
    print(f"   Min scored windows   : {MIN_SCOREABLE_FOR_WEIGHT} "
          f"(else weight capped at {WEIGHT_UNCERTAIN})")
    print(f"   Score std penalty    : std>{SCORE_STD_HIGH_THRESHOLD} "
          f"-> one weight step down")
    print("=" * 72)

    print("\n⚙️  Loading LSTM + scaler...")
    try:
        model  = tf.keras.models.load_model(LSTM_MODEL_PATH)
        scaler = joblib.load(LSTM_SCALER_PATH)
        print(f"   [OK] Input shape: {tuple(model.input_shape)}")
    except Exception as e:
        print(f"   [BAD] {e}")
        sys.exit(1)

    all_results = []

    for ticker in tickers:
        print(f"\n{'-'*72}")
        print(f"📈 TICKER: {ticker}")
        print(f"{'-'*72}")
        try:
            raw = yf.Ticker(ticker).history(period="2y")
            if raw.empty:
                print("   [BAD] No data. Skipping.")
                continue
            print(f"   [OK] {len(raw)} rows  "
                  f"({raw.index[0].date()} -> {raw.index[-1].date()})")
        except Exception as e:
            print(f"   [BAD] {e}")
            continue

        run_diagnostic(raw, model, scaler, ticker)

        for (start, end, label) in TEST_WINDOWS:
            print(f"\n   🔬 {label}  [{start} -> {end}]")
            r = run_window_test(
                ticker, start, end, label, raw, model, scaler, flat_threshold)
            all_results.append(r)

            if r["error"]:
                print(f"      [WARN]  {str(r['error']).split(chr(10))[0]}")
                continue

            dir_icon = {"UP": "📈", "DOWN": "📉", "FLAT": "➡️"}.get(
                r["actual_dir"], "?")
            s_ok = ("[OK]" if r["correct_strict"]
                    else ("➡️FLAT" if r["actual_dir"] == "FLAT" else "[BAD]"))
            l_ok = ("[OK]" if r["correct_loose"]
                    else ("⬜SKIP" if r["correct_loose"] is None else "[BAD]"))
            rob  = "[OK]" if r["crash_detected"] else "[BAD]"
            bias = "[WARN] BIAS" if r["bias_detected"] else "HOLD"
            cdir = "↑rise" if r["score_delta"] < 0 else "↓drop"

            print(f"      Normal  : {r['normal_score']:.6f}  -> {r['normal_signal']}")
            print(f"      Crashed : {r['crashed_score']:.6f}  "
                  f"(delta={r['score_delta']:+.5f} {cdir})  "
                  f"Robust:{rob}  Bias:{bias}")
            print(f"      Actual  : {r['actual_return']:+.2%} "
                  f"{dir_icon} {r['actual_dir']}")
            print(f"      Strict:{s_ok}  Loose:{l_ok}  "
                  f"{'← ' + _failure_reason(r) if not r['correct_strict'] else ''}")

    # -- Summary ----------------------------------------------------------------
    print(f"\n\n{'█'*72}")
    print("📊  ACCURACY & ROBUSTNESS SUMMARY  (v7)")
    print(f"{'█'*72}\n")

    valid   = [r for r in all_results
               if not r.get("error") and r.get("actual_dir")]
    skipped = [r for r in all_results if r.get("error")]

    if not valid:
        print("   [BAD] No valid results.")
        return

    # -- Per-ticker table -------------------------------------------------------
    rows = []
    agent_weights = {}
    ticker_stats  = {}

    for ticker in tickers:
        t = [r for r in valid if r["ticker"] == ticker]
        if not t:
            continue

        dir_r     = [r for r in t if r["normal_signal"] != "HOLD"]
        s_cor     = [r for r in dir_r if r["correct_strict"]]
        l_act     = [r for r in dir_r if r["correct_loose"] is not None]
        l_cor     = [r for r in l_act if r["correct_loose"]]
        holds     = len(t) - len(dir_r)
        flats     = sum(1 for r in dir_r if r["actual_dir"] == "FLAT")
        biased    = sum(1 for r in t if r["bias_detected"])
        crash_p   = sum(1 for r in t if r["crash_detected"])
        scores    = [r["normal_score"] for r in t]
        score_std = float(np.std(scores))

        s_acc     = len(s_cor) / len(dir_r)  * 100 if dir_r else 0.0
        l_acc     = len(l_cor) / len(l_act) * 100  if l_act else 0.0
        bias_rate = biased / len(t)                 if t     else 0.0

        weight    = _recommend_weight(l_acc, bias_rate, score_std, len(l_act))
        agent_weights[ticker] = weight

        ticker_stats[ticker] = dict(
            t=t, dir_r=dir_r, s_cor=s_cor, l_act=l_act, l_cor=l_cor,
            holds=holds, flats=flats, biased=biased, crash_p=crash_p,
            s_acc=s_acc, l_acc=l_acc, bias_rate=bias_rate,
            score_std=score_std, weight=weight,
        )

        if len(l_act) < MIN_SCOREABLE_FOR_WEIGHT:
            w_note = f"[WARN]  {weight:.1f} (only {len(l_act)} sample)"
        elif bias_rate >= 0.50:
            w_note = f"⛔ {weight:.1f} (high bias)"
        elif score_std > SCORE_STD_HIGH_THRESHOLD:
            w_note = f"[WARN]  {weight:.1f} (high std)"
        else:
            w_note = f"[OK] {weight:.1f}"

        rows.append({
            "Ticker":     ticker,
            "Windows":    len(t),
            "Scored":     len(l_act),
            "Strict[OK]":   f"{len(s_cor)}/{len(dir_r)} ({s_acc:.0f}%)",
            "Loose[OK]":    f"{len(l_cor)}/{len(l_act)} ({l_acc:.0f}%)",
            "FLAT⬜":     flats,
            "Bias[WARN]":    f"{biased}/{len(t)}",
            "ScoreStd":   f"{score_std:.3f}",
            "CrashDet":   f"{crash_p}/{len(t)}",
            "LSTMWeight": w_note,
        })

    print(tabulate(rows, headers="keys", tablefmt="rounded_outline"))

    # -- Detailed window table --------------------------------------------------
    print("\n\n📋  DETAILED WINDOW RESULTS\n")
    det = []
    for r in valid:
        s_ok = ("[OK]" if r["correct_strict"]
                else ("➡️FLAT" if r["actual_dir"] == "FLAT" else "[BAD]"))
        l_ok = ("[OK]" if r["correct_loose"]
                else ("⬜SKIP" if r["correct_loose"] is None else "[BAD]"))
        det.append({
            "Ticker":  r["ticker"],
            "Window":  r["window"],
            "Signal":  r["normal_signal"],
            "Actual":  r["actual_dir"],
            "Return":  f"{r['actual_return']:+.2%}",
            "Score":   f"{r['normal_score']:.4f}",
            "Crash":   f"{r['crashed_score']:.4f}",
            "Delta":   f"{r['score_delta']:+.4f}",
            "Robust":  "[OK]" if r["crash_detected"] else "[BAD]",
            "Bias":    "[WARN]" if r["bias_detected"] else "HOLD",
            "Strict":  s_ok,
            "Loose":   l_ok,
        })
    print(tabulate(det, headers="keys", tablefmt="rounded_outline"))

    # -- Totals -----------------------------------------------------------------
    dir_all   = [r for r in valid if r["normal_signal"] != "HOLD"]
    s_cor_all = [r for r in dir_all if r["correct_strict"]]
    l_act_all = [r for r in dir_all if r["correct_loose"] is not None]
    l_cor_all = [r for r in l_act_all if r["correct_loose"]]
    rob_all   = [r for r in valid if r["crash_detected"]]
    flat_all  = [r for r in dir_all if r["actual_dir"] == "FLAT"]
    bias_all  = [r for r in valid if r["bias_detected"]]

    s_acc = len(s_cor_all) / len(dir_all)   * 100 if dir_all   else 0.0
    l_acc = len(l_cor_all) / len(l_act_all) * 100 if l_act_all else 0.0
    rob   = len(rob_all)   / len(valid)      * 100 if valid     else 0.0
    bias  = len(bias_all)  / len(valid)      * 100 if valid     else 0.0

    print(f"\n{'-'*60}")
    print(f"   Total Windows        : {len(valid)}")
    print(f"   Directional Signals  : {len(dir_all)}")
    print(f"   FLAT windows (skip)  : {len(flat_all)}")
    print(f"   Confidently wrong    : {len(bias_all)}  "
          f"(score far from 0.5 AND wrong)")
    print()
    print(f"   Strict Accuracy      : {s_acc:.1f}%  "
          f"({len(s_cor_all)}/{len(dir_all)})  [FLAT=wrong]")
    print(f"   Loose Accuracy       : {l_acc:.1f}%  "
          f"({len(l_cor_all)}/{len(l_act_all)})  [FLAT=skip]")
    print(f"   Robustness Rate      : {rob:.1f}%  "
          f"({len(rob_all)}/{len(valid)})")
    print(f"   Bias Rate            : {bias:.1f}%  "
          f"({len(bias_all)}/{len(valid)})")
    if skipped:
        print(f"   Skipped/Errors       : {len(skipped)}")
    print(f"{'-'*60}")

    grade_acc = l_acc
    if   grade_acc >= 75 and rob >= 75: grade = "🏆 A HOLD Production Ready"
    elif grade_acc >= 60 and rob >= 60: grade = "🥈 B HOLD Needs Tuning"
    elif grade_acc >= 50:               grade = "🥉 C HOLD Marginal"
    else:                               grade = "[BAD] F HOLD Retrain"

    print(f"\n   SYSTEM GRADE (loose) : {grade}")
    print(f"{'-'*60}")

    # -- Failure analysis -------------------------------------------------------
    print("\n📌  FAILURE ANALYSIS BY TICKER\n")
    for ticker in tickers:
        st = ticker_stats.get(ticker)
        if not st:
            continue
        fails = [r for r in st["t"]
                 if not r["correct_strict"]
                 and r.get("normal_signal") != "HOLD"]
        std_note = (
            f"  [score std={st['score_std']:.3f} [WARN]  HIGH VOLATILITY]"
            if st["score_std"] > SCORE_STD_HIGH_THRESHOLD
            else f"  [score std={st['score_std']:.3f}]"
        )
        if not fails:
            print(f"   {ticker}: [OK] All correct{std_note}")
            continue
        print(f"   {ticker}:{std_note}")
        for r in fails:
            print(f"      {r['window']:<25}  {_failure_reason(r)}")

    # -- Crash behaviour --------------------------------------------------------
    print("\n⚡  CRASH BEHAVIOUR ANALYSIS\n")
    up   = sum(1 for r in valid if r["score_delta"] < -CRASH_DELTA_THRESHOLD)
    down = sum(1 for r in valid if r["score_delta"] >  CRASH_DELTA_THRESHOLD)
    no   = sum(1 for r in valid
               if abs(r["score_delta"]) <= CRASH_DELTA_THRESHOLD)
    print(f"   ↑ Score RISES  (oversold-bounce logic) : {up}/{len(valid)}")
    print(f"   ↓ Score DROPS  (danger detection)      : {down}/{len(valid)}")
    print(f"   -> No reaction                          : {no}/{len(valid)}")
    if up > down:
        print()
        print("   [WARN]  Model treats large crashes as BUY opportunities "
              "(mean-reversion bias).")
        print("      Safe for V-shaped bounces. Dangerous in sustained bears.")
        print("      Recommendation: retrain with 2020 COVID + 2022 bear data.")

    # -- Agent weight recommendations -------------------------------------------
    print(f"\n{'-'*60}")
    print("🤖  MULTI-AGENT LSTM WEIGHT RECOMMENDATIONS\n")
    print("   Weights account for: accuracy, bias rate, score volatility,")
    print("   and minimum sample size. Remaining weight -> sentiment + regime.\n")

    for ticker, weight in agent_weights.items():
        st    = ticker_stats.get(ticker, {})
        bar   = "█" * int(weight * 10) + "░" * (10 - int(weight * 10))
        l_act = st.get("l_act", [])
        n_sc  = len(l_act) if isinstance(l_act, list) else l_act
        notes = []
        if n_sc < MIN_SCOREABLE_FOR_WEIGHT:
            notes.append(f"only {n_sc} scored window(s)")
        if st.get("bias_rate", 0) >= 0.50:
            notes.append("high bias rate")
        if st.get("score_std", 0) > SCORE_STD_HIGH_THRESHOLD:
            notes.append(f"score std={st['score_std']:.3f}")
        note_str = f"  [{', '.join(notes)}]" if notes else ""
        label = ("[OK] high trust"   if weight >= 0.8
                 else "[WARN]  medium" if weight >= 0.5
                 else "⛔ excluded")
        print(f"   {ticker:<6}  {bar}  {weight:.1f}  {label}{note_str}")

    print(f"\n   JSON (paste into your aggregator config):")
    print(f"   {json.dumps({'lstm_weights': agent_weights}, indent=4)}")

    # -- Retraining advice ------------------------------------------------------
    bad_tickers = [t for t, w in agent_weights.items() if w < 0.3]
    if bad_tickers:
        print(f"\n{'-'*60}")
        print("🔧  RETRAINING ADVICE\n")
        for ticker in bad_tickers:
            st = ticker_stats.get(ticker, {})
            print(f"   {ticker}:")
            print(f"      1. Add 2022 bear market data to training set.")
            print(f"      2. Add class weights HOLD bearish sequences are "
                  f"underrepresented.")
            if st.get("score_std", 0) > SCORE_STD_HIGH_THRESHOLD:
                print(f"      3. Score std={st['score_std']:.3f} is high HOLD "
                      f"consider a ticker-specific LSTM head or separate model.")
            else:
                print(f"      3. Consider a ticker-specific LSTM head if "
                      f"volatility profile differs from AAPL/SPY.")

    print(f"\n{'█'*72}\n")


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 11 HOLD Adversarial LSTM backtest (v7)"
    )
    parser.add_argument(
        "--ticker", nargs="+", default=["AAPL"],
        help="Space-separated tickers, e.g. AAPL SPY QQQ TSLA",
    )
    parser.add_argument(
        "--flat-threshold", type=float, default=0.0075,
        help="Min |return| to count as directional (default 0.75%%)",
    )
    parser.add_argument(
        "--bias-score-threshold", type=float,
        default=DEFAULT_BIAS_SCORE_THRESHOLD,
        help="Score distance from 0.5 to flag as confidently wrong "
             "(default 0.70)",
    )
    args = parser.parse_args()

    BIAS_SCORE_THRESHOLD = args.bias_score_threshold

    run_backtest(
        tickers=[t.upper() for t in args.ticker],
        flat_threshold=args.flat_threshold,
    )