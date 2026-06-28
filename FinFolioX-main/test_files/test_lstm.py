"""
test_lstm.py  HOLD  LSTM + Explainability Backtest v5  |  5-Day Horizon
======================================================================
Uses production TechnicalAgent and ExplainabilityAgent directly.
Zero logic duplication HOLD all IG, reliability, and feature engineering
come from ml_engine/technical_agent.py and ml_engine/explainability_agent.py.

HOW THE TWO-PHASE RELIABILITY WORKS (matches original test_lstm logic):
  Phase 1 : explain_prediction() for every ticker in the window.
            Raw attributions + prob auto-stored in expl_agent._session_data.
  Phase 2 : set_batch_reliability(_session_data) HOLD computes reliability
            from all 30 tickers simultaneously (cross-sectional, not incremental).
            Overwrites the default 0.5 values with real window reliability.
  Phase 3 : Re-select top driver per ticker using _select_top_driver(raw_attrs)
            which now reads the correct batch-computed reliability.

This matches exactly what the original standalone test did with:
  reliability = compute_reliability(window_data)
  top_driver, eff_ig, _ = select_top_driver(attrs, reliability)

FIXES STILL ACTIVE (now enforced by production classes):
  FIX 1 HOLD Weekend snap via snap_to_trading_day()
  FIX 2 HOLD Real F(baseline) via ExplainabilityAgent._last_baseline_prob
  FIX 3 HOLD macd_norm gate via ExplainabilityAgent._select_top_driver()
  FIX 4 HOLD Raw (unstretched) prob via TechnicalAgent / IG forward pass
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# -- Import production modules -------------------------------------------------
# Assumes test_lstm.py lives at the project root (same level as ml_engine/)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_engine.technical_agent import (
    TechnicalAgent,
    build_lstm_features,
    LSTM_COLS,
    SEQ_LEN,
)
from ml_engine.explainability_agent import ExplainabilityAgent, MACD_REL_GATE

# -- Paths ---------------------------------------------------------------------
MODEL_PATH  = r"D:\FinFolioX\saved_models\lstm_model.keras"
SCALER_PATH = r"D:\FinFolioX\saved_models\lstm_scaler.pkl"

# -- Test windows --------------------------------------------------------------
TEST_WINDOWS = [
    
    ("2026-03-23", "2026-03-28", "Mar23->28  Bear start"),
    ("2026-03-04", "2026-03-09", "Mar04->09  Bear early"),
    ("2026-03-15", "2026-03-20", "Mar15->20  Deep Bear"),
    ("2026-03-05", "2026-03-10", "Mar05->10  Bounce"),
    ("2026-04-02", "2026-04-07", "Apr02->07  Iran Bear Lull"),  # ← ADD THIS
]

TICKERS = [
    "AAPL", "MSFT", "NVDA", "TSLA", "META", "AMZN", "GOOGL", "AMD", "INTC", "NFLX",
    "JPM", "V", "WMT", "JNJ", "XOM", "CAT", "DIS", "BA", "MCD", "KO",
    "SPY", "QQQ", "TLT", "GLD", "SLV", "USO", "UNG", "DIA", "IWM", "EEM",
]

# These thresholds apply to RAW (unstretched) probabilities.
# Production trading uses stretched probs + fused confidence -- different scale.
BUY_THRESHOLD  = 0.52
SELL_THRESHOLD = 0.48

# Historical baselines for comparison columns
V1_EXPL = [37.9, 43.3, 46.7, 70.0, 0.0]
V3_EXPL = [55.2, 66.7, 66.7, 70.0, 0.0]


# ==============================================================================
# FIX 1 -- Trading-day snap  (keeps test immune to weekend date strings)
# ==============================================================================
def snap_to_trading_day(date_str: str) -> str:
    """
    Advances any weekend / non-business date to the next business day.
    pd.bdate_range(start=dt, periods=1)[0] gives the first bday ON OR AFTER dt.
    """
    dt      = pd.to_datetime(date_str)
    snapped = pd.bdate_range(start=dt, periods=1)[0]
    if snapped != dt:
        print(f"   WARNING: {date_str} is not a trading day -> snapped to {snapped.date()}")
    return snapped.strftime("%Y-%m-%d")


# ==============================================================================
# DATA HELPERS  (test-specific -- not in production path)
# ==============================================================================
def fetch_history_up_to(ticker: str, test_date: str) -> pd.DataFrame:
    """Fetches OHLCV history up to and including test_date (already snapped)."""
    test_dt  = pd.to_datetime(test_date)
    yf_end   = (test_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    yf_start = (test_dt - pd.Timedelta(days=300)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=yf_start, end=yf_end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_actual_return(ticker: str, test_date: str, outcome_date: str) -> float:
    """
    Returns % price change between test_date and outcome_date.
    Returns float('nan') if prices unavailable -- surfaced as unknown, never silently wrong.
    """
    yf_end   = (pd.to_datetime(outcome_date) + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    yf_start = (pd.to_datetime(test_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=yf_start, end=yf_end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty or len(df) < 2:
        return float("nan")
    try:
        p_entry = float(df["Close"].asof(pd.to_datetime(test_date)))
        p_exit  = float(df["Close"].asof(pd.to_datetime(outcome_date)))
    except Exception:
        p_entry = float(df["Close"].iloc[0])
        p_exit  = float(df["Close"].iloc[-1])
    if np.isnan(p_entry) or np.isnan(p_exit) or p_entry == 0:
        return float("nan")
    return ((p_exit - p_entry) / p_entry) * 100.0


# ==============================================================================
# SINGLE WINDOW RUNNER
# ==============================================================================
def run_window(test_date, outcome_date, label, tech_agent, expl_agent,
               v1_acc=None, v3_acc=None):

    # FIX 1: snap both dates to valid trading days
    test_date    = snap_to_trading_day(test_date)
    outcome_date = snap_to_trading_day(outcome_date)

    print(f"\n{'*'*112}")
    print(f"  {label}  |  {test_date} -> {outcome_date}")
    print(f"{'*'*112}")

    # Reset agent session so this window starts fresh
    expl_agent.reset_session()

    # =========================================================================
    # PHASE 1: run explain_prediction() for every ticker
    # =========================================================================
    # explain_prediction() auto-fills expl_agent._session_data with
    # {feat: raw_ig, ..., 'prob': base_prob} for every ticker processed.
    # We also capture per-ticker diagnostics for Phase 3 reporting.

    ticker_data = {}  # ticker -> (last_100, prob, raw_attrs, actual_return, baseline_prob)

    for ticker in TICKERS:
        try:
            hist = fetch_history_up_to(ticker, test_date)
            if hist.empty or len(hist) < 200:
                continue

            # build_lstm_features is imported from ml_engine/technical_agent.py
            # -- identical to the function used during model training
            feat_df = build_lstm_features(hist)
            if len(feat_df) < SEQ_LEN:
                continue

            last_100 = feat_df[LSTM_COLS].tail(SEQ_LEN)

            # Production ExplainabilityAgent -- IG via GradientTape
            # explain_prediction() returns (importance_dict, preliminary_top_driver)
            # The preliminary top driver uses default rel=0.5 until Phase 2 sets
            # the real batch reliability. We discard it here and re-select in Phase 3.
            expl_agent.explain_prediction(last_100)

            # Read diagnostics stored by the last explain_prediction() call
            # _last_base_prob     = raw (unstretched) F(x) -- matches FIX 4
            # _last_baseline_prob = F(0) for completeness check -- FIX 2
            prob          = expl_agent._last_base_prob
            baseline_prob = expl_agent._last_baseline_prob

            # Raw (pre-reliability, pre-flip) attributions are in _session_data
            session_entry = expl_agent._session_data[-1]
            raw_attrs     = {f: session_entry[f] for f in LSTM_COLS}

            actual_return = fetch_actual_return(ticker, test_date, outcome_date)

            ticker_data[ticker] = (last_100, prob, raw_attrs, actual_return, baseline_prob)

        except Exception:
            pass

    # =========================================================================
    # PHASE 2: batch reliability -- all tickers contribute simultaneously
    # =========================================================================
    # set_batch_reliability() reads expl_agent._session_data (auto-filled in Phase 1)
    # and computes per-feature reliability cross-sectionally across all tickers.
    # This OVERWRITES the incremental 0.5 defaults with real window-level values.
    # After this call, expl_agent._reliability has the same values that the
    # original standalone test_lstm computed via compute_reliability(window_data).
    expl_agent.set_batch_reliability(expl_agent._session_data)
    reliability = dict(expl_agent._reliability)  # snapshot for this window's reporting

    print(f"\n  Feature reliability (IG, vs LSTM signal):")
    for feat in LSTM_COLS:
        rel  = reliability[feat]
        bar  = "=" * int(rel * 20)
        flip = "(flip)" if rel < 0.5 else ""
        gate = " <- gate active" if (feat == "macd_norm" and rel < MACD_REL_GATE) else ""
        print(f"    {feat:<14}: {rel:.2f}  {bar}  {flip}{gate}")

    # =========================================================================
    # PHASE 3: score and report using batch reliability
    # =========================================================================
    print(f"\n  {'Ticker':<7} {'Prob':>7} {'Sig':<5} {'Act%':>9} "
          f"{'LSTM':>5} {'TopDriver':<14} {'IG_attr':>10} {'Expl':>6} {'Both':>6}")
    print(f"  {'-'*88}")

    results        = []
    lstm_c = lstm_w = lstm_n = 0
    expl_c = expl_w = expl_n = 0
    both_c = 0
    driver_counts  = {f: 0 for f in LSTM_COLS}
    driver_correct = {f: 0 for f in LSTM_COLS}

    for ticker in TICKERS:
        if ticker not in ticker_data:
            continue
        try:
            last_100, prob, raw_attrs, actual_return, baseline_prob = ticker_data[ticker]

            # FIX 4: prob is raw (unstretched) -- BUY/SELL thresholds apply directly
            signal = ("BUY"  if prob > BUY_THRESHOLD  else
                      "SELL" if prob < SELL_THRESHOLD  else "HOLD")

            # LSTM accuracy vs actual market outcome
            if np.isnan(actual_return):
                lstm_res = "?"; lstm_n += 1; lstm_ok = False
            elif signal == "HOLD":
                lstm_res = "-"; lstm_n += 1; lstm_ok = False
            elif signal == "BUY"  and actual_return > 0:
                lstm_res = "OK"; lstm_c += 1; lstm_ok = True
            elif signal == "SELL" and actual_return < 0:
                lstm_res = "OK"; lstm_c += 1; lstm_ok = True
            else:
                lstm_res = "XX"; lstm_w += 1; lstm_ok = False

            # Re-select top driver using the now-correct batch reliability.
            # _select_top_driver() reads expl_agent._reliability (set in Phase 2).
            # FIX 3 macd_norm gate is enforced inside _select_top_driver().
            top_driver, eff_ig, _ = expl_agent._select_top_driver(raw_attrs)
            driver_counts[top_driver] += 1

            lstm_direction = prob - 0.5

            if signal == "HOLD" or lstm_direction == 0.0:
                expl_res = "-"; expl_n += 1; expl_ok = False
            elif ((eff_ig > 0 and lstm_direction > 0) or
                  (eff_ig < 0 and lstm_direction < 0)):
                expl_res = "OK"; expl_c += 1; driver_correct[top_driver] += 1
                expl_ok = True
            else:
                expl_res = "XX"; expl_w += 1; expl_ok = False

            if expl_ok and lstm_ok:
                both_c += 1; both_s = "BOTH"
            elif expl_ok or lstm_ok:
                both_s = "PART"
            else:
                both_s = "NONE"

            act_str = f"{actual_return:>+8.2f}%" if not np.isnan(actual_return) else "     nan%"
            print(f"  {ticker:<7} {prob:>7.4f} {signal:<5} {act_str}  "
                  f"{lstm_res:>4}  {top_driver:<14} {eff_ig:>+10.6f}  "
                  f"{expl_res:>4}  {both_s:>4}")

            results.append({
                "ticker":         ticker,
                "prob":           round(prob, 4),
                "signal":         signal,
                "actual_%":       round(actual_return, 2) if not np.isnan(actual_return) else None,
                "lstm_result":    lstm_res,
                "lstm_ok":        lstm_ok,
                "top_driver":     top_driver,
                "ig_attribution": round(eff_ig, 6),
                "expl_result":    expl_res,
                "expl_ok":        expl_ok,
            })

        except Exception as e:
            print(f"  {ticker:<7} ERROR: {e}")

    lstm_active = lstm_c + lstm_w
    expl_active = expl_c + expl_w
    lstm_acc    = (lstm_c / lstm_active * 100) if lstm_active > 0 else 0.0
    expl_acc    = (expl_c / expl_active * 100) if expl_active > 0 else 0.0
    both_acc    = (both_c / lstm_active * 100) if lstm_active > 0 else 0.0

    print(f"\n  -- LSTM     : {lstm_c}C/{lstm_w}W/{lstm_n}N  ->  {lstm_acc:.1f}%  (vs market)")
    print(f"  -- Expl v5  : {expl_c}C/{expl_w}W/{expl_n}N  ->  {expl_acc:.1f}%  (IG, vs LSTM signal)")
    print(f"  -- Both     : {both_c}  ->  {both_acc:.1f}%")
    if v3_acc is not None:
        print(f"  -- Expl v3  : {v3_acc:.1f}%  |  Expl v1: {v1_acc:.1f}%")

    # FIX 2: IG completeness using real F(0) stored per ticker in Phase 1
    sample_ticker = next(iter(ticker_data))
    _, sample_prob, sample_attrs, _, sample_baseline_prob = ticker_data[sample_ticker]
    ig_sum        = sum(sample_attrs.values())
    expected_diff = sample_prob - sample_baseline_prob
    close         = abs(ig_sum - expected_diff) < 0.15
    print(f"\n  IG completeness check ({sample_ticker}): "
          f"sum(IG)={ig_sum:+.4f}  "
          f"F(x)-F(0)={expected_diff:+.4f}  "
          f"F(0)={sample_baseline_prob:.4f}  "
          f"{'CLOSE' if close else 'GAP'}")

    print(f"\n  Top driver frequency (v5, IG):")
    for feat, cnt in sorted(driver_counts.items(), key=lambda x: x[1], reverse=True):
        if cnt == 0: continue
        corr = driver_correct[feat]
        acc  = (corr / cnt * 100) if cnt > 0 else 0
        rel  = reliability[feat]
        bar  = "#" * cnt
        ok   = "GOOD" if acc >= 70 else ("WARN" if acc >= 55 else "BAD ")
        gate = "  [gate]" if (feat == "macd_norm" and rel < MACD_REL_GATE) else ""
        print(f"    {feat:<14}: {cnt:2d}x  expl_acc={acc:.0f}% {ok}  rel={rel:.2f}  {bar}{gate}")

    return {
        "label":          label,
        "test_date":      test_date,
        "outcome_date":   outcome_date,
        "lstm_acc":       lstm_acc,
        "expl_acc":       expl_acc,
        "both_acc":       both_acc,
        "lstm_c":         lstm_c,
        "lstm_w":         lstm_w,
        "lstm_n":         lstm_n,
        "expl_c":         expl_c,
        "expl_w":         expl_w,
        "both_c":         both_c,
        "lstm_active":    lstm_active,
        "reliability":    reliability,
        "driver_counts":  driver_counts,
        "driver_correct": driver_correct,
        "results":        results,
    }


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("=" * 112)
    print("  LSTM + EXPLAINABILITY AGENT v5  |  5-Day Horizon")
    print(f"  Method: Integrated Gradients (steps=50, baseline=zeros)")
    print(f"  Reliability: batch/cross-sectional (all tickers per window simultaneously)")
    print(f"  Metric: sign(IG_attribution) vs sign(prob-0.5)")
    print(f"  Fixes: weekend_snap | real_F(0) | macd_gate={MACD_REL_GATE} | raw_prob_thresholds")
    print(f"  Source: ml_engine/technical_agent.py + ml_engine/explainability_agent.py")
    print("=" * 112)

    # -------------------------------------------------------------------------
    # Load production TechnicalAgent
    # Exposes: lstm_model, lstm_scaler, predict(), predict_raw(), predict_signal()
    # -------------------------------------------------------------------------
    print("\nLoading TechnicalAgent (LSTM + scaler)...")
    try:
        tech_agent = TechnicalAgent(
            lstm_model_path=MODEL_PATH,
            lstm_scaler_path=SCALER_PATH,
        )
        print(f"   OK Model  : {MODEL_PATH}")
        print(f"   OK Scaler : {SCALER_PATH}")
        print(f"   OK Input shape: {tuple(tech_agent.lstm_model.input_shape)}")
    except Exception as e:
        print(f"   FAILED: {e}"); return

    # -------------------------------------------------------------------------
    # Load production ExplainabilityAgent
    # background_data_df=None is fine -- constructor ignores it (API compat only)
    # _verify_gradienttape() runs in __init__ and prints result
    # -------------------------------------------------------------------------
    print("\nLoading ExplainabilityAgent (Integrated Gradients v5)...")
    expl_agent = ExplainabilityAgent(tech_agent, background_data_df=None)

    # -------------------------------------------------------------------------
    # Run all windows
    # -------------------------------------------------------------------------
    all_stats = []
    for i, (test_date, outcome_date, label) in enumerate(TEST_WINDOWS):
        s = run_window(
            test_date, outcome_date, label,
            tech_agent, expl_agent,
            V1_EXPL[i], V3_EXPL[i],
        )
        all_stats.append(s)

    # -------------------------------------------------------------------------
    # Consolidated summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 112)
    print("  CONSOLIDATED SUMMARY -- v5  (Integrated Gradients, production classes)")
    print("=" * 112)
    print(f"\n  {'Window':<30} {'LSTM acc':>9} {'Expl v5':>9} "
          f"{'Expl v3':>9} {'Expl v1':>9} {'Both':>7}")
    print(f"  {'-'*78}")

    for i, s in enumerate(all_stats):
        lstm_ok  = "OK" if s["lstm_acc"] >= 65 else "!!"
        expl_ok  = "OK" if s["expl_acc"] >= 70 else ("~" if s["expl_acc"] >= 60 else "XX")
        delta    = s["expl_acc"] - V3_EXPL[i]
        dsign    = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
        nan_note = f"  [nan_returns={s['lstm_n']}]" if s["lstm_n"] > 5 else ""
        print(f"  {s['label']:<30} "
              f"{s['lstm_acc']:>7.1f}%{lstm_ok} "
              f"{s['expl_acc']:>7.1f}%{expl_ok} "
              f"{V3_EXPL[i]:>7.1f}%   "
              f"{V1_EXPL[i]:>7.1f}%   "
              f"{s['both_acc']:>5.1f}%  ({dsign}pp vs v3){nan_note}")

    avg_lstm = sum(s["lstm_acc"] for s in all_stats) / len(all_stats)
    avg_expl = sum(s["expl_acc"] for s in all_stats) / len(all_stats)
    avg_both = sum(s["both_acc"] for s in all_stats) / len(all_stats)
    avg_v3   = sum(V3_EXPL) / len(V3_EXPL)

    print(f"\n  {'-'*78}")
    print(f"  {'AVERAGE':<30} {avg_lstm:>7.1f}%   {avg_expl:>7.1f}%   "
          f"{avg_v3:>6.1f}%   {sum(V1_EXPL)/len(V1_EXPL):>6.1f}%   {avg_both:>5.1f}%")

    delta_v3  = avg_expl - avg_v3
    target_ok = "TARGET MET" if avg_expl >= 70 else \
                f"{'UP' if delta_v3 > 0 else 'DOWN'} vs v3: {delta_v3:+.1f}pp"
    print(f"\n  Expl v5 (IG) avg : {avg_expl:.1f}%  {target_ok}")
    print(f"  LSTM avg         : {avg_lstm:.1f}%  (directional accuracy vs market, raw prob)")
    print(f"  Both avg         : {avg_both:.1f}%  (expl+LSTM both correct)")

    # Aggregate top driver and reliability
    agg_counts  = {f: 0 for f in LSTM_COLS}
    agg_correct = {f: 0 for f in LSTM_COLS}
    agg_rel     = {f: [] for f in LSTM_COLS}
    for s in all_stats:
        for f in LSTM_COLS:
            agg_counts[f]  += s["driver_counts"][f]
            agg_correct[f] += s["driver_correct"][f]
            agg_rel[f].append(s["reliability"][f])

    print(f"\n  Aggregate top driver frequency (v5, all windows):")
    print(f"  {'Feature':<16} {'Times':>7} {'Correct':>9} {'Acc':>8}  {'AvgRel':>8}")
    print(f"  {'-'*55}")
    for feat, cnt in sorted(agg_counts.items(), key=lambda x: x[1], reverse=True):
        if cnt == 0: continue
        corr  = agg_correct[feat]
        acc   = (corr / cnt * 100) if cnt > 0 else 0
        avg_r = np.mean(agg_rel[feat])
        ok    = "GOOD" if acc >= 70 else ("WARN" if acc >= 55 else "BAD ")
        bar   = "#" * min(cnt, 20)
        gate  = "  <- gate" if (feat == "macd_norm" and avg_r < MACD_REL_GATE) else ""
        print(f"  {feat:<16} {cnt:>7}  {corr:>8}  {acc:>6.0f}% {ok}  {avg_r:>7.2f}  {bar}{gate}")

    print(f"\n  Aggregate feature reliability (IG, vs LSTM signal, all windows):")
    for feat in LSTM_COLS:
        avg_r = np.mean(agg_rel[feat])
        bar   = "=" * int(avg_r * 20)
        note  = "  <- flipped in selection" if avg_r < 0.5 else ""
        gate  = "  <- gate active" if (feat == "macd_norm" and avg_r < MACD_REL_GATE) else ""
        print(f"    {feat:<14}: {avg_r:.2f}  {bar}{note}{gate}")

    # Save results
    all_rows = []
    for s in all_stats:
        for r in s["results"]:
            r["window"] = s["label"]
            all_rows.append(r)
    pd.DataFrame(all_rows).to_csv("lstm_expl_v5_backtest.csv", index=False)
    print(f"\n  Saved -> lstm_expl_v5_backtest.csv")
    print(f"\n  IG steps       : {expl_agent.ig_steps}")
    print(f"  Baseline       : zeros")
    print(f"  MACD gate      : {MACD_REL_GATE}")
    print(f"  Reliability    : batch (cross-sectional per window)")
    print(f"  Agent source   : ml_engine/technical_agent.py + explainability_agent.py")
    print("\nDone.\n")


if __name__ == "__main__":
    main()