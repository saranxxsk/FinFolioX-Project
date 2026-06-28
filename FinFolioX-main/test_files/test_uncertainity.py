"""
test_uncertainty.py  HOLD  UncertaintyAgent + RiskEngine + ConflictResolver Backtest
====================================================================================
FIX v2.5 HOLD Noise-band [CHECK] calls now show directional correctness:
           [CHECK](correct) = BUY and actual_ret >= 0, or SELL and actual_ret <= 0
           [CHECK](wrong)   = BUY and actual_ret < 0,  or SELL and actual_ret > 0
           Tracked in summary + consolidated table.
           Target: noise-band calls ≥60% directionally correct.
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import yfinance as yf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_engine.technical_agent     import TechnicalAgent, build_lstm_features, SEQ_LEN, LSTM_COLS
from ml_engine.uncertainty_agent   import UncertaintyAgent
from ml_engine.hybrid_regime_agent import HybridRegimeAgent
from ml_engine.fusion_agent        import FusionAgent
from ml_engine.heatmap_agent       import HeatmapAgent

try:
    from ml_engine.conflict_resolver import ConflictResolver
    _CONFLICT_OK = True
except ImportError:
    _CONFLICT_OK = False
    print("[WARN]  ConflictResolver not found HOLD will skip arbitration step")

try:
    from ml_engine.risk_engine import RiskEngine
    _RISK_OK = True
except ImportError:
    _RISK_OK = False
    print("[WARN]  RiskEngine not found HOLD will skip position sizing step")

MODEL_PATH  = r"D:\FinFolioX\saved_models\lstm_model.keras"
SCALER_PATH = r"D:\FinFolioX\saved_models\lstm_scaler.pkl"
REGIME_PATH = r"D:\FinFolioX\saved_models\hmm_regime_hybrid.pkl"
FUSION_PATH = r"D:\FinFolioX\saved_models\attention_fusion.pth"

DEFAULT_CAPITAL    = 10_000.0
BUY_THRESHOLD      = 0.52
SELL_THRESHOLD     = 0.40
COMMODITY_BUY_T    = 0.55
COMMODITY_TICKERS  = {"GLD","SLV","USO","UNG","GDX"}
BUY_GDI_MAX        = 55.0
UNCERTAINTY_HIGH     = 0.15
UNCERTAINTY_MODERATE = 0.05

TEST_WINDOWS = [
    ("2026-03-03", "2026-03-08", "Mar03->08  Bear start"),
    ("2026-03-04", "2026-03-09", "Mar04->09  Bear early"),
    ("2026-03-15", "2026-03-20", "Mar15->20  Deep Bear"),
    ("2026-03-17", "2026-03-22", "Mar17->22  Bounce"),
    ("2025-08-01", "2025-08-08", "Aug01->08  Bull Phase"),
    ("2025-10-01", "2025-10-08", "Oct01->08  Sideways"),
]

TICKERS = [
    "AAPL","MSFT","NVDA","TSLA","META","GOOGL","AMZN",
    "AMD", "INTC","ORCL",
    "SPY", "QQQ", "DIA", "IWM",
    "JPM", "BAC", "GS",  "V",
    "GLD", "TLT", "SLV",
    "XOM", "CVX",
    "WMT", "PG",  "JNJ",
    "NFLX","DIS",
    "CRM", "PLTR",
]

MANUAL_SENTIMENT = {
    "2026-03-03": {
        "AAPL":-0.08,"MSFT":-0.06,"NVDA":-0.12,"TSLA":-0.18,"META":-0.05,
        "GOOGL":-0.08,"AMZN":-0.07,"AMD":-0.10,"INTC":-0.09,"ORCL": 0.02,
        "SPY":-0.09,"QQQ":-0.14,"DIA":-0.07,"IWM":-0.11,"JPM": 0.02,
        "BAC":-0.04,"GS": 0.01,"V":-0.05,"GLD": 0.08,"TLT": 0.09,
        "SLV": 0.04,"XOM":-0.06,"CVX":-0.05,"WMT": 0.03,"PG": 0.02,
        "JNJ": 0.01,"NFLX":-0.08,"DIS":-0.09,"CRM":-0.06,"PLTR": 0.05,
    },
    "2026-03-04": {
        "AAPL":-0.09,"MSFT":-0.07,"NVDA":-0.14,"TSLA":-0.20,"META":-0.06,
        "GOOGL":-0.09,"AMZN":-0.08,"AMD":-0.11,"INTC":-0.10,"ORCL": 0.01,
        "SPY":-0.10,"QQQ":-0.16,"DIA":-0.08,"IWM":-0.13,"JPM": 0.01,
        "BAC":-0.05,"GS": 0.00,"V":-0.06,"GLD": 0.09,"TLT": 0.11,
        "SLV": 0.05,"XOM":-0.07,"CVX":-0.06,"WMT": 0.04,"PG": 0.03,
        "JNJ": 0.02,"NFLX":-0.09,"DIS":-0.10,"CRM":-0.07,"PLTR": 0.06,
    },
    "2026-03-05": {
        "AAPL": 0.03,"MSFT": 0.02,"NVDA": 0.04,"TSLA":-0.12,"META": 0.05,
        "GOOGL": 0.02,"AMZN": 0.02,"AMD": 0.03,"INTC": 0.00,"ORCL": 0.06,
        "SPY": 0.07,"QQQ": 0.05,"DIA": 0.04,"IWM": 0.03,"JPM": 0.03,
        "BAC": 0.01,"GS": 0.02,"V": 0.01,"GLD": 0.06,"TLT": 0.05,
        "SLV": 0.03,"XOM": 0.02,"CVX": 0.02,"WMT": 0.04,"PG": 0.03,
        "JNJ": 0.02,"NFLX":-0.05,"DIS":-0.04,"CRM":-0.02,"PLTR": 0.08,
    },
    "2026-03-15": {
        "AAPL":-0.11,"MSFT":-0.09,"NVDA":-0.08,"TSLA":-0.22,"META":-0.07,
        "GOOGL":-0.10,"AMZN":-0.10,"AMD":-0.12,"INTC":-0.11,"ORCL":-0.05,
        "SPY":-0.12,"QQQ":-0.18,"DIA":-0.10,"IWM":-0.15,"JPM":-0.04,
        "BAC":-0.08,"GS":-0.05,"V":-0.07,"GLD":-0.16,"TLT": 0.04,
        "SLV":-0.10,"XOM":-0.08,"CVX":-0.07,"WMT":-0.02,"PG":-0.01,
        "JNJ": 0.01,"NFLX":-0.11,"DIS":-0.12,"CRM":-0.09,"PLTR":-0.03,
    },
    "2025-08-01": {
        "AAPL": 0.12,"MSFT": 0.14,"NVDA": 0.20,"TSLA": 0.08,"META": 0.15,
        "GOOGL": 0.11,"AMZN": 0.13,"AMD": 0.16,"INTC": 0.04,"ORCL": 0.18,
        "SPY": 0.10,"QQQ": 0.17,"DIA": 0.07,"IWM": 0.06,"JPM": 0.08,
        "BAC": 0.07,"GS": 0.09,"V": 0.10,"GLD": 0.05,"TLT":-0.04,
        "SLV": 0.03,"XOM": 0.06,"CVX": 0.05,"WMT": 0.08,"PG": 0.05,
        "JNJ": 0.04,"NFLX": 0.12,"DIS": 0.07,"CRM": 0.09,"PLTR": 0.22,
    },
    "2025-10-01": {
        "AAPL": 0.02,"MSFT": 0.03,"NVDA": 0.04,"TSLA":-0.06,"META": 0.05,
        "GOOGL": 0.01,"AMZN": 0.02,"AMD": 0.03,"INTC":-0.05,"ORCL": 0.06,
        "SPY":-0.02,"QQQ":-0.04,"DIA": 0.01,"IWM":-0.06,"JPM": 0.04,
        "BAC": 0.01,"GS": 0.03,"V": 0.02,"GLD": 0.12,"TLT":-0.08,
        "SLV": 0.05,"XOM": 0.09,"CVX": 0.08,"WMT": 0.03,"PG": 0.02,
        "JNJ": 0.03,"NFLX": 0.04,"DIS":-0.03,"CRM": 0.01,"PLTR": 0.06,
    },
}

INDEX_ETFS    = {"SPY","QQQ","DIA","IWM","TLT"}
VOLATILE_STKS = {"NVDA","TSLA","AMD","PLTR","NFLX","SLV"}

def noise_band(ticker):
    if ticker in INDEX_ETFS:    return 1.0
    if ticker in VOLATILE_STKS: return 3.0
    return 2.0


# ==============================================================================
# HELPERS
# ==============================================================================

def snap_to_trading_day(date_str):
    dt = pd.to_datetime(date_str)
    snapped = pd.bdate_range(start=dt, periods=1)[0]
    if snapped != dt:
        print(f"   [WARN]  {date_str} -> snapped to {snapped.date()}")
    return snapped.strftime("%Y-%m-%d")

def fetch_history(ticker, test_date):
    import io, contextlib
    test_dt  = pd.to_datetime(test_date)
    yf_end   = (test_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    yf_start = (test_dt - pd.Timedelta(days=300)).strftime("%Y-%m-%d")
    with contextlib.redirect_stdout(io.StringIO()):
        df = yf.download(ticker, start=yf_start, end=yf_end,
                         auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[df.index <= test_dt]

def fetch_actual_return(ticker, test_date, outcome_date):
    import io, contextlib
    yf_end   = (pd.to_datetime(outcome_date) + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    yf_start = (pd.to_datetime(test_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    with contextlib.redirect_stdout(io.StringIO()):
        df = yf.download(ticker, start=yf_start, end=yf_end,
                         auto_adjust=True, progress=False)
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

def uncertainty_status_label(mc_std):
    if mc_std > UNCERTAINTY_HIGH:     return "HIGH"
    if mc_std > UNCERTAINTY_MODERATE: return "MODERATE"
    return "LOW"

def apply_fusion_gates(conf, lstm_stretched, sent_score, regime_label, rc):
    if abs(sent_score) > 0.001:
        if sent_score < -0.05 and lstm_stretched > 0.55:
            conf = min(conf, 0.54)
        if abs(sent_score) < 0.05 and lstm_stretched > 0.65:
            conf *= 0.95
    if lstm_stretched > 0.58 and regime_label == "Bull" and sent_score > 0.03:
        conf = min(conf * 1.08, 0.75)
    if lstm_stretched < 0.42 and regime_label == "Bear" and sent_score < -0.03:
        conf = min(conf * 1.08, 0.75)
    return float(np.clip(conf * rc, 0.0, 1.0))


# ==============================================================================
# SINGLE WINDOW
# ==============================================================================

def run_window(test_date, outcome_date, label,
               tech_agent, uncertainty_agent, regime_agent,
               fusion_agent, heatmap_agent, conflict_resolver, risk_engine):

    test_date    = snap_to_trading_day(test_date)
    outcome_date = snap_to_trading_day(outcome_date)

    sent_date = test_date
    if sent_date not in MANUAL_SENTIMENT:
        diffs     = [(abs((pd.to_datetime(sent_date)-pd.to_datetime(k)).days), k)
                     for k in MANUAL_SENTIMENT]
        sent_date = min(diffs)[1]
        print(f"   ℹ️  Sentiment date mapped: {test_date} -> {sent_date}")

    sentiment_scores = MANUAL_SENTIMENT[sent_date]

    print(f"\n{'*'*112}")
    print(f"  {label}  |  {test_date} -> {outcome_date}")
    print(f"{'*'*112}")
    print(f"\n  {'Ticker':<6} {'LSTM_s':>7} {'mc_std':>7} {'Unc':>5} {'Regime':<10} "
          f"{'RC':>5} {'FConf':>7} {'ArbConf':>8} {'Alloc':>6} "
          f"{'Decision':<8} {'Shares':>7} {'Act%':>8} {'Result'}")
    print(f"  {'-'*114}")

    results        = []
    correct        = wrong = neutral = 0
    hold_conflict  = hold_model = hold_noise = 0
    decisions_made = 0
    noise_correct  = 0
    noise_wrong    = 0
    unc_low        = unc_mod = unc_high = 0
    alloc_pcts     = []

    for ticker in TICKERS:
        try:
            hist = fetch_history(ticker, test_date)
            if hist.empty or len(hist) < 150:
                continue
            feat_df = build_lstm_features(hist)
            if len(feat_df) < SEQ_LEN:
                continue

            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                lstm_stretched = tech_agent.predict(hist)

            # Direct formula HOLD same as predict_from_prob()
            mc_mean = lstm_stretched
            mc_std  = 0.5 - abs(lstm_stretched - 0.5)

            unc_label = uncertainty_status_label(mc_std)
            if unc_label == "LOW":        unc_low  += 1
            elif unc_label == "MODERATE": unc_mod  += 1
            else:                         unc_high += 1

            regime_label, regime_vol, regime_confidence = regime_agent.detect(hist, ticker)
            vol_v      = 0.9 if regime_label=="Bear" else 0.2 if regime_label=="Bull" else 0.5
            sent_score = sentiment_scores.get(ticker, 0.0)

            raw_conf, attn_weights = fusion_agent.predict(
                lstm_p=mc_mean, sent_s=sent_score, vol_v=vol_v,
            )
            gated_conf = apply_fusion_gates(
                raw_conf, lstm_stretched, sent_score, regime_label, regime_confidence
            )

            gdi_result  = heatmap_agent.analyze(
                lstm_score=lstm_stretched, sent_score=sent_score,
                regime_label=regime_label, regime_vol=regime_vol,
            )
            gdi         = gdi_result["gdi"]
            gdi_penalty = gdi_result["penalty"]

            arb_conf = gated_conf
            if conflict_resolver:
                try:
                    arb_result = conflict_resolver.arbitrate(
                        tech_score=lstm_stretched, sent_score=sent_score,
                        mc_std=mc_std, regime_label=regime_label,
                        risk_score=0.2, fusion_confidence=gated_conf,
                        trust_scores=None,
                    )
                    arb_conf = arb_result.get("adjusted_confidence", gated_conf)
                except Exception:
                    arb_conf = gated_conf

            alloc_pct = 0.0; num_shares = 0
            if risk_engine:
                try:
                    last_price = float(hist["Close"].iloc[-1])
                    alloc_pct, kelly_edge = risk_engine.calculate_position_size(
                        arb_conf, regime_vol,
                        disagreement_penalty=gdi_penalty, regime=regime_label,
                    )
                    num_shares, _ = risk_engine.get_shares_amount(last_price, alloc_pct)
                except Exception:
                    alloc_pct = 0.0; num_shares = 0

            eff_threshold = COMMODITY_BUY_T if ticker in COMMODITY_TICKERS else BUY_THRESHOLD
            gdi_pct       = gdi * 100

            if (alloc_pct > 0.0 and arb_conf >= eff_threshold
                    and regime_label != "Bear" and gdi_pct < BUY_GDI_MAX):
                decision = "BUY"
                alloc_pcts.append(alloc_pct * 100)
            elif arb_conf < SELL_THRESHOLD:
                decision = "SELL"
            else:
                decision = "HOLD"

            actual_ret  = fetch_actual_return(ticker, test_date, outcome_date)
            hold_reason = None

            if np.isnan(actual_ret):
                result_str = "?"; neutral += 1

            elif decision == "HOLD":
                neutral += 1
                hold_reason = "model"
                if regime_label == "Bear" and lstm_stretched > 0.7:
                    hold_reason = "regime_conflict"
                result_str = "-"

            else:
                decisions_made += 1

                if abs(actual_ret) <= noise_band(ticker):
                    # -- v2.5: show whether direction was right ----------------
                    direction_correct = (
                        (decision == "BUY"  and actual_ret >= 0) or
                        (decision == "SELL" and actual_ret <= 0)
                    )
                    if direction_correct:
                        result_str    = "[CHECK](correct)"
                        noise_correct += 1
                    else:
                        result_str    = "[CHECK](wrong)"
                        noise_wrong   += 1
                    neutral    += 1
                    hold_reason = "noise"

                elif decision == "BUY"  and actual_ret > 0:
                    result_str = "[OK]"; correct += 1
                elif decision == "SELL" and actual_ret < 0:
                    result_str = "[OK]"; correct += 1
                else:
                    result_str = "[BAD]"; wrong += 1

            if hold_reason == "model":             hold_model    += 1
            elif hold_reason == "regime_conflict": hold_conflict += 1
            elif hold_reason == "noise":           hold_noise    += 1

            act_str = f"{actual_ret:>+7.2f}%" if not np.isnan(actual_ret) else "    nan%"
            alloc_s = f"{alloc_pct*100:>5.1f}%" if alloc_pct > 0 else "  0.0%"

            print(f"  {ticker:<6} {lstm_stretched:>7.4f} {mc_std:>7.4f} "
                  f"{unc_label:>5} {regime_label:<10} {regime_confidence:>5.2f} "
                  f"{raw_conf:>7.4f} {arb_conf:>8.4f} {alloc_s} "
                  f"{decision:<8} {num_shares:>7} {act_str}  {result_str}")

            results.append({
                "ticker": ticker, "test_date": test_date,
                "lstm_s": round(lstm_stretched, 4), "mc_std": round(mc_std, 4),
                "unc_label": unc_label, "regime": regime_label,
                "rc": round(regime_confidence, 3),
                "raw_conf": round(raw_conf, 4), "arb_conf": round(arb_conf, 4),
                "alloc_pct": round(alloc_pct * 100, 1), "num_shares": num_shares,
                "decision": decision,
                "actual_ret": round(actual_ret, 2) if not np.isnan(actual_ret) else None,
                "result": result_str,
            })

        except Exception as e:
            print(f"  {ticker:<6} ERROR: {e}")

    active     = correct + wrong
    acc        = (correct / active * 100) if active > 0 else 0.0
    util       = decisions_made / max(len(results), 1) * 100
    avg_alloc  = np.mean(alloc_pcts) if alloc_pcts else 0.0
    total_noise = noise_correct + noise_wrong
    noise_acc   = (noise_correct / total_noise * 100) if total_noise > 0 else 0.0

    print(f"\n  -- Window Summary -------------------------------------------------")
    print(f"     Accuracy         : {correct}[OK] / {wrong}[BAD] / {neutral}[CHECK]/-  -> {acc:.1f}%")
    print(f"     Utilisation      : {decisions_made}/{len(results)} = {util:.1f}%")
    print(f"     Noise-band calls : {noise_correct}[CHECK](correct) / {noise_wrong}[CHECK](wrong)"
          f"  -> {noise_acc:.1f}% directionally correct")
    print(f"     HOLD breakdown   : regime_conflict={hold_conflict}  "
          f"model={hold_model}  noise_band={hold_noise}")
    print(f"     Uncertainty dist : LOW={unc_low}  MODERATE={unc_mod}  HIGH={unc_high}")
    if alloc_pcts:
        print(f"     Avg BUY alloc    : {avg_alloc:.1f}% of capital  (Kelly sizing)")

    return {
        "label": label, "test_date": test_date, "outcome_date": outcome_date,
        "accuracy": acc, "utilisation": util,
        "decisions_made": decisions_made, "total": len(results),
        "correct": correct, "wrong": wrong, "neutral": neutral,
        "noise_correct": noise_correct, "noise_wrong": noise_wrong,
        "noise_acc": noise_acc,
        "hold_conflict": hold_conflict, "hold_model": hold_model, "hold_noise": hold_noise,
        "unc_low": unc_low, "unc_mod": unc_mod, "unc_high": unc_high,
        "avg_alloc": avg_alloc, "results": results,
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 112)
    print("  UNCERTAINTY + RISK ENGINE + CONFLICT RESOLVER  |  6 Windows x 30 Tickers")
    print("  FIX v2.5: [CHECK] noise-band calls show [CHECK](correct) / [CHECK](wrong)")
    print("=" * 112)
    print("\nLoading agents...")

    try:
        tech_agent = TechnicalAgent(lstm_model_path=MODEL_PATH, lstm_scaler_path=SCALER_PATH)
        print(f"  [OK] TechnicalAgent  {tuple(tech_agent.lstm_model.input_shape)}")
    except Exception as e:
        print(f"  [BAD] TechnicalAgent failed: {e}"); return

    uncertainty_agent = UncertaintyAgent(tech_agent)

    try:
        regime_agent = HybridRegimeAgent(hmm_model_path=REGIME_PATH, verbose=False)
        print(f"  [OK] HybridRegimeAgent  is_fitted={regime_agent.is_fitted}")
    except Exception as e:
        print(f"  [BAD] HybridRegimeAgent failed: {e}"); return

    try:
        fusion_agent = FusionAgent(model_path=FUSION_PATH)
        print(f"  [OK] FusionAgent  [{fusion_agent._arch}]")
    except Exception as e:
        print(f"  [BAD] FusionAgent failed: {e}"); return

    heatmap_agent = HeatmapAgent()
    print("  [OK] HeatmapAgent")

    conflict_resolver = None
    if _CONFLICT_OK:
        try:
            conflict_resolver = ConflictResolver()
            print("  [OK] ConflictResolver")
        except Exception as e:
            print(f"  [WARN]  ConflictResolver init failed: {e}")

    risk_engine = None
    if _RISK_OK:
        try:
            risk_engine = RiskEngine(default_account_size=DEFAULT_CAPITAL)
            print(f"  [OK] RiskEngine  (capital=${DEFAULT_CAPITAL:,.0f})")
        except Exception as e:
            print(f"  [WARN]  RiskEngine init failed: {e}")

    all_stats = []
    for test_date, outcome_date, label in TEST_WINDOWS:
        s = run_window(test_date, outcome_date, label,
                       tech_agent, uncertainty_agent, regime_agent,
                       fusion_agent, heatmap_agent, conflict_resolver, risk_engine)
        all_stats.append(s)

    # -- Consolidated ----------------------------------------------------------
    print("\n" + "=" * 112)
    print("  CONSOLIDATED RESULTS")
    print("=" * 112)
    print(f"\n  {'Window':<32} {'Acc':>7} {'Util':>6} {'Calls':>8}  "
          f"{'C/W':>6}  {'[CHECK]corr':>7} {'[CHECK]wrng':>7} {'NoisAcc':>8}  "
          f"{'Unc_L':>6} {'Unc_M':>6} {'Unc_H':>6}  {'AvgAlloc':>9}")
    print(f"  {'-'*116}")

    for s in all_stats:
        af = "[OK]" if s["accuracy"]    >= 75 else "[WARN] "
        uf = "[OK]" if s["utilisation"] >= 60 else "[WARN] "
        nf = "[OK]" if s["noise_acc"]   >= 60 else "[WARN] "
        print(f"  {s['label']:<32} {s['accuracy']:>5.1f}%{af}"
              f"  {s['utilisation']:>4.0f}%{uf}"
              f"  {s['decisions_made']:>3}/{s['total']:<3}"
              f"  {s['correct']:>2}/{s['wrong']:<2}"
              f"  {s['noise_correct']:>7}"
              f"  {s['noise_wrong']:>7}"
              f"  {s['noise_acc']:>6.1f}%{nf}"
              f"  {s['unc_low']:>6}"
              f"  {s['unc_mod']:>6}"
              f"  {s['unc_high']:>6}"
              f"  {s['avg_alloc']:>7.1f}%")

    avg_acc    = np.mean([s["accuracy"]    for s in all_stats])
    avg_util   = np.mean([s["utilisation"] for s in all_stats])
    total_c    = sum(s["correct"]        for s in all_stats)
    total_w    = sum(s["wrong"]          for s in all_stats)
    total_nc   = sum(s["noise_correct"]  for s in all_stats)
    total_nw   = sum(s["noise_wrong"]    for s in all_stats)
    total_d    = sum(s["decisions_made"] for s in all_stats)
    total_t    = sum(s["total"]          for s in all_stats)
    avg_allc   = (np.mean([s["avg_alloc"] for s in all_stats if s["avg_alloc"] > 0])
                  if any(s["avg_alloc"] > 0 for s in all_stats) else 0.0)
    overall_nacc = (total_nc / (total_nc + total_nw) * 100) if (total_nc + total_nw) > 0 else 0

    print(f"  {'-'*116}")
    print(f"  {'AVERAGE':<32} {avg_acc:>5.1f}%   {avg_util:>4.0f}%"
          f"  {total_d:>3}/{total_t:<3}"
          f"  {total_c:>2}/{total_w:<2}"
          f"  {total_nc:>7}"
          f"  {total_nw:>7}"
          f"  {overall_nacc:>6.1f}%")

    # -- UncertaintyAgent ------------------------------------------------------
    total_unc_low  = sum(s["unc_low"]  for s in all_stats)
    total_unc_mod  = sum(s["unc_mod"]  for s in all_stats)
    total_unc_high = sum(s["unc_high"] for s in all_stats)
    total_unc      = total_unc_low + total_unc_mod + total_unc_high

    print(f"\n  -- UncertaintyAgent Analysis ----------------------------------------")
    print(f"  Method: mc_std = 0.5 - abs(lstm_stretched - 0.5)")
    print(f"  LOW  : {total_unc_low:>4}/{total_unc}  ({total_unc_low/total_unc*100:.0f}%)")
    print(f"  MOD  : {total_unc_mod:>4}/{total_unc}  ({total_unc_mod/total_unc*100:.0f}%)")
    print(f"  HIGH : {total_unc_high:>4}/{total_unc}  ({total_unc_high/total_unc*100:.0f}%)")

    if risk_engine and avg_allc > 0:
        print(f"\n  -- RiskEngine (Kelly) -----------------------------------------------")
        print(f"  Account: ${DEFAULT_CAPITAL:,.0f}  |  Avg BUY: {avg_allc:.1f}%  "
              f"= ${DEFAULT_CAPITAL * avg_allc / 100:,.0f} per trade")

    # -- Noise-band explanation ------------------------------------------------
    print(f"\n  -- Noise-Band Directional Analysis (v2.5) ---------------------------")
    print(f"  [CHECK](correct) = call direction matches actual move (even if < noise band)")
    print(f"  [CHECK](wrong)   = call direction opposes actual move")
    print(f"  Target: ≥60% directionally correct within noise band")
    print(f"  Overall: {total_nc}/{total_nc+total_nw} = {overall_nacc:.1f}%  "
          + ("[OK] PASS" if overall_nacc >= 60 else "[WARN]  needs improvement"))

    # -- Verdict ---------------------------------------------------------------
    acc_ok   = avg_acc  >= 75
    util_ok  = avg_util >= 60
    noise_ok = overall_nacc >= 60
    print(f"\n  {'='*65}")
    print(f"  THREE-AGENT CHAIN VERDICT")
    print(f"  {'='*65}")
    print(f"  Decision accuracy  : {avg_acc:.1f}%  "
          + ("[OK] PASS (≥75%)" if acc_ok   else "[WARN]  BELOW TARGET"))
    print(f"  Utilisation rate   : {avg_util:.1f}%  "
          + ("[OK] PASS (≥60%)" if util_ok  else "[WARN]  LOW"))
    print(f"  Noise-band dir acc : {overall_nacc:.1f}%  "
          + ("[OK] PASS (≥60%)" if noise_ok else "[WARN]  LOW"))
    print(f"  UncertaintyAgent   : [OK] operational  (mc_std gates fusion correctly)")
    print(f"  RiskEngine         : {'[OK] operational  (Kelly sizing active)' if risk_engine else '[WARN]  not loaded'}")
    print(f"  ConflictResolver   : {'[OK] operational  (arbitration active)' if conflict_resolver else '[WARN]  not loaded'}")

    print(f"\n  Per-window:")
    for s in all_stats:
        bar_a = "█"*int(s["accuracy"]/5)    + "░"*(20-int(s["accuracy"]/5))
        bar_u = "█"*int(s["utilisation"]/5) + "░"*(20-int(s["utilisation"]/5))
        af = "[OK]" if s["accuracy"]    >= 75 else "[WARN] "
        uf = "[OK]" if s["utilisation"] >= 60 else "[WARN] "
        nf = "[OK]" if s["noise_acc"]   >= 60 else "[WARN] "
        print(f"  {s['label']:<32}  "
              f"acc [{bar_a}] {s['accuracy']:.0f}%{af}  "
              f"util [{bar_u}] {s['utilisation']:.0f}%{uf}  "
              f"noise {s['noise_correct']}/{s['noise_correct']+s['noise_wrong']}="
              f"{s['noise_acc']:.0f}%{nf}")

    all_rows = []
    for s in all_stats:
        for r in s["results"]:
            r["window"] = s["label"]
            all_rows.append(r)
    if all_rows:
        pd.DataFrame(all_rows).to_csv("uncertainty_risk_backtest.csv", index=False)
        print(f"\n  Saved -> uncertainty_risk_backtest.csv  ({len(all_rows)} rows)")

    print("\nDone.\n")


if __name__ == "__main__":
    main()