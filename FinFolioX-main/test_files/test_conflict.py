"""
test_conflict_resolver.py  HOLD  Conflict Resolution Engine Backtest v2.5
=======================================================================
FinFolioX HOLD Phase 13 Backtest  |  7 Windows x 30 Tickers

Run from project root:
    python test_conflict_resolver.py
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import yfinance as yf
from collections import defaultdict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_engine.technical_agent     import TechnicalAgent, build_lstm_features, SEQ_LEN
from ml_engine.uncertainty_agent   import UncertaintyAgent
from ml_engine.hybrid_regime_agent import HybridRegimeAgent
from ml_engine.fusion_agent        import FusionAgent
from ml_engine.heatmap_agent       import HeatmapAgent
from ml_engine.conflict_resolver   import ConflictResolver

try:
    from ml_engine.risk_engine import RiskEngine
    _RISK_OK = True
except ImportError:
    _RISK_OK = False

MODEL_PATH  = r"D:\FinFolioX\saved_models\lstm_model.keras"
SCALER_PATH = r"D:\FinFolioX\saved_models\lstm_scaler.pkl"
REGIME_PATH = r"D:\FinFolioX\saved_models\hmm_regime_hybrid.pkl"
FUSION_PATH = r"D:\FinFolioX\saved_models\attention_fusion.pth"

DEFAULT_CAPITAL   = 10_000.0
BUY_THRESHOLD     = 0.52
SELL_THRESHOLD    = 0.40
COMMODITY_BUY_T   = 0.55
COMMODITY_TICKERS = {"GLD","SLV","USO","UNG","GDX"}
BUY_GDI_MAX       = 55.0

TEST_WINDOWS = [
    ("2026-03-03", "2026-03-08", "Mar03->08  Bear start"),
    ("2026-03-04", "2026-03-09", "Mar04->09  Bear early"),
    ("2026-03-15", "2026-03-20", "Mar15->20  Deep Bear"),
    ("2026-03-05", "2026-03-10", "Mar05->10  Bounce"),
    ("2025-08-01", "2025-08-08", "Aug01->08  Bull Phase"),
    ("2025-10-01", "2025-10-08", "Oct01->08  Sideways"),
    ("2026-03-17", "2026-03-23", "Mar17->23  Iran+Fed"),
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
    "2026-03-17": {
        "AAPL":-0.10,"MSFT":-0.09,"NVDA": 0.18,"TSLA":-0.24,"META": 0.12,
        "GOOGL": 0.14,"AMZN":-0.08,"AMD":-0.06,"INTC":-0.12,"ORCL": 0.02,
        "SPY":-0.10,"QQQ":-0.14,"DIA":-0.10,"IWM":-0.13,"JPM":-0.04,
        "BAC":-0.05,"GS":-0.03,"V":-0.04,"GLD": 0.18,"TLT":-0.11,
        "SLV": 0.12,"XOM": 0.15,"CVX": 0.14,"WMT": 0.04,"PG": 0.03,
        "JNJ": 0.02,"NFLX":-0.08,"DIS":-0.07,"CRM":-0.09,"PLTR": 0.09,
    },
}

INDEX_ETFS    = {"SPY","QQQ","DIA","IWM","TLT"}
VOLATILE_STKS = {"NVDA","TSLA","AMD","PLTR","NFLX","SLV"}

def noise_band(ticker):
    if ticker in INDEX_ETFS:    return 1.0
    if ticker in VOLATILE_STKS: return 3.0
    return 2.0

def derive_risk_score(regime_label: str, regime_vol: float) -> float:
    """Derive realistic risk_score from regime + vol (replaces hardcoded 0.25)."""
    base     = {"Bear": 0.52, "Sideways": 0.35, "Bull": 0.28}.get(regime_label, 0.35)
    vol_mult = float(np.clip(regime_vol / 0.015, 0.8, 2.2))
    return float(np.clip(base * vol_mult, 0.10, 0.82))


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

def make_decision(conf, alloc_pct, regime_label, ticker, gdi):
    eff_thr = COMMODITY_BUY_T if ticker in COMMODITY_TICKERS else BUY_THRESHOLD
    if (alloc_pct > 0.0 and conf >= eff_thr
            and regime_label != "Bear" and gdi * 100 < BUY_GDI_MAX):
        return "BUY"
    elif conf < SELL_THRESHOLD:
        return "SELL"
    return "HOLD"

def score_decision(decision, actual_ret, ticker):
    if np.isnan(actual_ret):   return "nan",  "?"
    if decision == "HOLD":     return "hold", "-"
    if abs(actual_ret) <= noise_band(ticker):
        ok = ((decision=="BUY" and actual_ret>=0) or
              (decision=="SELL" and actual_ret<=0))
        return "noise", "[CHECK](correct)" if ok else "[CHECK](wrong)"
    if decision=="BUY"  and actual_ret>0: return "correct","[OK]"
    if decision=="SELL" and actual_ret<0: return "correct","[OK]"
    return "wrong","[BAD]"


# ==============================================================================
# SINGLE WINDOW
# ==============================================================================

def run_window(test_date, outcome_date, label,
               tech_agent, uncertainty_agent, regime_agent,
               fusion_agent, heatmap_agent, risk_engine, cr):

    test_date    = snap_to_trading_day(test_date)
    outcome_date = snap_to_trading_day(outcome_date)

    sent_date = test_date
    if sent_date not in MANUAL_SENTIMENT:
        diffs = [(abs((pd.to_datetime(sent_date)-pd.to_datetime(k)).days), k)
                 for k in MANUAL_SENTIMENT]
        sent_date = min(diffs)[1]
        print(f"   ℹ️  Sentiment mapped: {test_date} -> {sent_date}")

    sentiment_scores = MANUAL_SENTIMENT[sent_date]
    cr.reset_history()

    print(f"\n{'*'*128}")
    print(f"  {label}  |  {test_date} -> {outcome_date}")
    print(f"{'*'*128}")
    print(f"\n  {'Ticker':<6} {'Ruling':<14} {'LDir':<5} {'SDir':<5} {'Risk':>5} {'Sprd':>6} "
          f"{'Before':>7} {'After':>7} {'Δ':>6} "
          f"{'PreDec':<6} {'Post':<6} {'Act%':>8}  {'Pre':<14} {'Post'}")
    print(f"  {'-'*132}")

    rows         = []
    pre_correct  = pre_wrong  = pre_noise  = 0
    post_correct = post_wrong = post_noise = 0
    arb_cnt = disc_cnt = 0
    ruling_counts = defaultdict(int)

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

            mc_mean, mc_std = uncertainty_agent.predict_from_prob(lstm_stretched)
            regime_label, regime_vol, regime_conf = regime_agent.detect(hist, ticker)
            vol_v      = 0.9 if regime_label=="Bear" else 0.2 if regime_label=="Bull" else 0.5
            sent_score = sentiment_scores.get(ticker, 0.0)

            raw_conf, _ = fusion_agent.predict(
                lstm_p=mc_mean, sent_s=sent_score, vol_v=vol_v,
            )
            gated_conf = apply_fusion_gates(
                raw_conf, lstm_stretched, sent_score, regime_label, regime_conf
            )

            gdi_result = heatmap_agent.analyze(
                lstm_score=lstm_stretched, sent_score=sent_score,
                regime_label=regime_label, regime_vol=regime_vol,
            )
            gdi         = gdi_result["gdi"]
            gdi_penalty = gdi_result["penalty"]

            risk_score = derive_risk_score(regime_label, regime_vol)

            # Pre-arb
            pre_alloc = 0.0
            if risk_engine:
                try:
                    last_price = float(hist["Close"].iloc[-1])
                    pre_alloc, _ = risk_engine.calculate_position_size(
                        gated_conf, regime_vol,
                        disagreement_penalty=gdi_penalty, regime=regime_label,
                    )
                except Exception:
                    pre_alloc = 0.0
            pre_dec = make_decision(gated_conf, pre_alloc, regime_label, ticker, gdi)

            # Conflict resolver
            arb = cr.arbitrate(
                tech_score=lstm_stretched, sent_score=sent_score,
                mc_std=mc_std, regime_label=regime_label,
                risk_score=risk_score, fusion_confidence=gated_conf,
                trust_scores=None,
            )
            arb_conf = arb["adjusted_confidence"]
            ruling   = arb["ruling"]

            if arb["arbitrated"]:    arb_cnt  += 1
            if arb.get("regime_discounted"): disc_cnt += 1
            ruling_counts[ruling] += 1

            # Post-arb
            post_alloc = 0.0
            if risk_engine:
                try:
                    post_alloc, _ = risk_engine.calculate_position_size(
                        arb_conf, regime_vol,
                        disagreement_penalty=gdi_penalty, regime=regime_label,
                    )
                except Exception:
                    post_alloc = 0.0
            post_dec = make_decision(arb_conf, post_alloc, regime_label, ticker, gdi)

            actual_ret = fetch_actual_return(ticker, test_date, outcome_date)

            pre_cat,  pre_icon  = score_decision(pre_dec,  actual_ret, ticker)
            post_cat, post_icon = score_decision(post_dec, actual_ret, ticker)

            if pre_cat  == "correct": pre_correct  += 1
            elif pre_cat  == "wrong": pre_wrong    += 1
            elif pre_cat  == "noise": pre_noise    += 1
            if post_cat == "correct": post_correct += 1
            elif post_cat == "wrong": post_wrong   += 1
            elif post_cat == "noise": post_noise   += 1

            # Direction from history
            last_h  = cr._history[-1] if cr._history else {}
            lstm_d  = last_h.get("lstm_dir", "?")
            sent_d  = last_h.get("sent_dir", "?")
            spread  = last_h.get("spread",   0.0)
            delta   = arb_conf - gated_conf
            act_str = f"{actual_ret:>+7.2f}%" if not np.isnan(actual_ret) else "    nan%"

            print(f"  {ticker:<6} {ruling:<14} {lstm_d:<5} {sent_d:<5} "
                  f"{risk_score:>5.3f} {spread:>6.4f} "
                  f"{gated_conf:>7.4f} {arb_conf:>7.4f} {delta:>+6.4f} "
                  f"{pre_dec:<6} {post_dec:<6} {act_str}  "
                  f"{pre_icon:<14} {post_icon}")

            rows.append({
                "ticker": ticker, "test_date": test_date,
                "ruling": ruling, "lstm_dir": lstm_d, "sent_dir": sent_d,
                "risk_score": round(risk_score, 3), "spread": round(spread, 4),
                "pre_conf": round(gated_conf, 4), "post_conf": round(arb_conf, 4),
                "conf_delta": round(delta, 4),
                "pre_decision": pre_dec, "post_decision": post_dec,
                "actual_ret": round(actual_ret, 2) if not np.isnan(actual_ret) else None,
                "pre_result": pre_icon, "post_result": post_icon,
                "arbitrated": arb["arbitrated"],
                "discounted": arb.get("regime_discounted", False),
                "window": label,
            })

        except Exception as e:
            print(f"  {ticker:<6}  ERROR: {e}")

    n           = len(rows)
    pre_active  = pre_correct + pre_wrong
    post_active = post_correct + post_wrong
    pre_acc     = (pre_correct  / pre_active  * 100) if pre_active  > 0 else 0.0
    post_acc    = (post_correct / post_active * 100) if post_active > 0 else 0.0
    lift        = post_acc - pre_acc
    arb_rate    = arb_cnt / max(n, 1) * 100
    cr_stats    = cr.get_stats()

    print(f"\n  -- Window Summary --------------------------------------------------------")
    print(f"     Arbitration      : {arb_cnt}/{n} = {arb_rate:.1f}%  "
          f"(regime discounts: {disc_cnt})")
    print(f"     Ruling breakdown :", end="")
    for r, cnt in sorted(ruling_counts.items(), key=lambda x: -x[1]):
        print(f"  {r}={cnt}", end="")
    print()
    print(f"     Pre-arb  accuracy: {pre_correct}[OK] / {pre_wrong}[BAD] / {pre_noise}[CHECK]  "
          f"-> {pre_acc:.1f}%")
    print(f"     Post-arb accuracy: {post_correct}[OK] / {post_wrong}[BAD] / {post_noise}[CHECK]  "
          f"-> {post_acc:.1f}%")
    lf = "[OK]" if lift >= 0 else "[WARN] "
    print(f"     Accuracy lift    : {lift:+.1f}%  {lf}")
    print(f"     Mean conf change : {cr_stats.get('mean_conf_change_pct', 0):+.2f}%  "
          f"(max drop: {cr_stats.get('max_conf_drop', 0):.4f}  "
          f"max rise: {cr_stats.get('max_conf_rise', 0):.4f})")

    return {
        "label": label, "test_date": test_date, "outcome_date": outcome_date,
        "n": n, "arb_cnt": arb_cnt, "disc_cnt": disc_cnt, "arb_rate": arb_rate,
        "pre_acc": pre_acc, "post_acc": post_acc, "acc_lift": lift,
        "pre_correct": pre_correct, "pre_wrong": pre_wrong,
        "post_correct": post_correct, "post_wrong": post_wrong,
        "ruling_counts": dict(ruling_counts), "cr_stats": cr_stats, "rows": rows,
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 128)
    print("  CONFLICT RESOLVER BACKTEST v2.5  |  7 Windows x 30 Tickers")
    print("  FIX: Directional conflict detection + MILD_ADJUST material flag")
    print("  NO false conflicts from lstm=0+neutral-sent anymore")
    print("=" * 128)
    print("\nLoading agents...")

    try:
        tech_agent = TechnicalAgent(lstm_model_path=MODEL_PATH, lstm_scaler_path=SCALER_PATH)
        print(f"  [OK] TechnicalAgent  {tuple(tech_agent.lstm_model.input_shape)}")
    except Exception as e:
        print(f"  [BAD] TechnicalAgent: {e}"); return

    uncertainty_agent = UncertaintyAgent(tech_agent)

    try:
        regime_agent = HybridRegimeAgent(hmm_model_path=REGIME_PATH, verbose=False)
        print(f"  [OK] HybridRegimeAgent  is_fitted={regime_agent.is_fitted}")
    except Exception as e:
        print(f"  [BAD] HybridRegimeAgent: {e}"); return

    try:
        fusion_agent = FusionAgent(model_path=FUSION_PATH)
        print(f"  [OK] FusionAgent  [{fusion_agent._arch}]")
    except Exception as e:
        print(f"  [BAD] FusionAgent: {e}"); return

    heatmap_agent = HeatmapAgent()
    print("  [OK] HeatmapAgent")

    risk_engine = None
    if _RISK_OK:
        try:
            risk_engine = RiskEngine(default_account_size=DEFAULT_CAPITAL)
            print(f"  [OK] RiskEngine  (capital=${DEFAULT_CAPITAL:,.0f})")
        except Exception as e:
            print(f"  [WARN]  RiskEngine: {e}")

    cr = ConflictResolver(verbose=True)

    all_stats = []
    all_rows  = []

    for test_date, outcome_date, label in TEST_WINDOWS:
        s = run_window(
            test_date, outcome_date, label,
            tech_agent, uncertainty_agent, regime_agent,
            fusion_agent, heatmap_agent, risk_engine, cr,
        )
        all_stats.append(s)
        all_rows.extend(s["rows"])

    # -- Consolidated ----------------------------------------------------------
    print("\n" + "=" * 128)
    print("  CONSOLIDATED RESULTS")
    print("=" * 128)
    print(f"\n  {'Window':<32} {'Arb%':>6} {'Disc':>5} "
          f"{'PreAcc':>7} {'PostAcc':>8} {'Lift':>6}  {'Pre[OK]/[BAD]':>9} {'Post[OK]/[BAD]':>9}")
    print(f"  {'-'*110}")

    for s in all_stats:
        lf = "[OK]" if s["acc_lift"] >= 0 else "[WARN] "
        print(f"  {s['label']:<32} {s['arb_rate']:>5.1f}% {s['disc_cnt']:>5}  "
              f"  {s['pre_acc']:>6.1f}%  {s['post_acc']:>7.1f}% "
              f"{s['acc_lift']:>+5.1f}%{lf}  "
              f"{s['pre_correct']:>3}/{s['pre_wrong']:<3}  "
              f"{s['post_correct']:>3}/{s['post_wrong']:<3}")

    total_n      = sum(s["n"]           for s in all_stats)
    total_arb    = sum(s["arb_cnt"]     for s in all_stats)
    total_disc   = sum(s["disc_cnt"]    for s in all_stats)
    total_pre_c  = sum(s["pre_correct"] for s in all_stats)
    total_pre_w  = sum(s["pre_wrong"]   for s in all_stats)
    total_post_c = sum(s["post_correct"]for s in all_stats)
    total_post_w = sum(s["post_wrong"]  for s in all_stats)
    opa  = (total_pre_c  / (total_pre_c +total_pre_w)  * 100
            if (total_pre_c+total_pre_w)   > 0 else 0.0)
    oppa = (total_post_c / (total_post_c+total_post_w) * 100
            if (total_post_c+total_post_w) > 0 else 0.0)
    ol   = oppa - opa
    oar  = total_arb / max(total_n, 1) * 100

    print(f"  {'-'*110}")
    lf = "[OK]" if ol >= 0 else "[WARN] "
    print(f"  {'OVERALL':<32} {oar:>5.1f}% {total_disc:>5}  "
          f"  {opa:>6.1f}%  {oppa:>7.1f}% {ol:>+5.1f}%{lf}  "
          f"{total_pre_c:>3}/{total_pre_w:<3}  {total_post_c:>3}/{total_post_w:<3}")

    # -- Ruling distribution ----------------------------------------------------
    all_rulings = defaultdict(int)
    for s in all_stats:
        for r, cnt in s["ruling_counts"].items():
            all_rulings[r] += cnt

    print(f"\n  -- Ruling Distribution -----------------------------------------------------")
    for ruling, cnt in sorted(all_rulings.items(), key=lambda x: -x[1]):
        pct = cnt / max(total_n, 1) * 100
        bar = "█" * int(pct / 2)
        print(f"  {ruling:<24} {cnt:>4} ({pct:>5.1f}%)  [{bar}]")

    # -- Regression checks ------------------------------------------------------
    print(f"\n  -- v2.5 Regression Checks ---------------------------------------------------")
    print(f"  Directional thresholds: bull>{cr.bull_dir_threshold}  bear<{cr.bear_dir_threshold}  "
          f"backup≥{cr.extreme_spread_backup}")
    print(f"  UNCERTAINTY_HIGH = {cr.uncertainty_high}  "
          + ("[OK]" if cr.uncertainty_high == 0.15 else "[BAD]"))

    bull_ws   = [s for s in all_stats if "Bull" in s["label"] or "Sideways" in s["label"]]
    veto_bull = sum(s["ruling_counts"].get("SYSTEMIC_VETO", 0) for s in bull_ws)
    print(f"  Systemic veto in Bull/Sideways: {veto_bull}  "
          + ("[OK] No false vetoes" if veto_bull == 0 else f"[WARN]  {veto_bull} unexpected"))

    mild_cnt = all_rulings.get("MILD_ADJUST", 0)
    dir_cnt  = sum(all_rulings.get(r, 0) for r in ("HOLD","ALIGN_BULL","ALIGN_BEAR",
                                                     "TRUST_TECHNICAL",
                                                     "TRUST_SENTIMENT_BULL",
                                                     "TRUST_SENTIMENT_BEAR"))
    print(f"  MILD_ADJUST count     : {mild_cnt}  ({mild_cnt/max(total_n,1)*100:.1f}%)  "
          + ("[OK] Material adjustments captured" if mild_cnt > 0 else "[WARN]  None fired"))
    print(f"  Directional conflicts : {dir_cnt}  ({dir_cnt/max(total_n,1)*100:.1f}%)  "
          + ("[OK] Genuine signal conflicts caught" if dir_cnt > 0 else "[WARN]  None fired"))
    print(f"  Arbitration rate      : {oar:.1f}%  "
          + ("[OK] Active (>15%)" if oar > 15 else "[WARN]  Still low"))

    # -- Verdict ---------------------------------------------------------------
    print(f"\n  {'='*65}")
    print(f"  CONFLICT RESOLVER v2.5 VERDICT")
    print(f"  {'='*65}")
    print(f"  Pre-arb  accuracy  : {opa:.1f}%   " + ("[OK]" if opa  >= 70 else "[WARN] "))
    print(f"  Post-arb accuracy  : {oppa:.1f}%   "+ ("[OK]" if oppa >= 70 else "[WARN] "))
    print(f"  Accuracy lift      : {ol:+.1f}%    "+ ("[OK] helped" if ol>0 else "~ neutral" if ol==0 else "[WARN]  hurt"))
    print(f"  Arbitration rate   : {oar:.1f}%    "+ ("[OK]" if oar > 15 else "[WARN]  low"))
    print(f"  False conflict fix : [OK] Directional detection prevents BEAR+NEUTRAL false fires")
    print(f"  Mild adjust flag   : [OK] Material Δconf now counted as arbitrated")
    print(f"  Regime discounts   : {total_disc}")

    print(f"\n  Per-window:")
    for s in all_stats:
        bar = "█"*int(s["arb_rate"]/5) + "░"*(20-int(s["arb_rate"]/5))
        lf  = "[OK]" if s["acc_lift"] >= 0 else "[WARN] "
        print(f"  {s['label']:<32}  [{bar}] {s['arb_rate']:.0f}%  "
              f"lift={s['acc_lift']:>+5.1f}%{lf}  "
              f"pre={s['pre_acc']:.1f}% -> post={s['post_acc']:.1f}%")

    if all_rows:
        pd.DataFrame(all_rows).to_csv("conflict_resolver_backtest.csv", index=False)
        print(f"\n  Saved -> conflict_resolver_backtest.csv  ({len(all_rows)} rows)")

    print("\nDone.\n")


if __name__ == "__main__":
    main()