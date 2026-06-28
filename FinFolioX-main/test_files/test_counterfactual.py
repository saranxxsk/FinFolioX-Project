"""
test_counterfactual.py  HOLD  Counterfactual Decision Engine Backtest v2.1
===================================================================
FinFolioX HOLD Phase 15 Backtest  |  7 Windows x 30 Tickers

Tests CounterfactualEngine v2.0 on top of the full agent pipeline:
  Pipeline -> actual decision -> fetch T+5 price -> analyze() -> regret

Includes the new Mar17->23 window with Iran war / Fed hawkish sentiment.

Metrics reported per window:
  - Optimal match rate  : % of decisions that were already optimal
  - Mean regret         : average P&L left on the table
  - Confidence calib    : % where confidence direction matched outcome
  - Regret level dist   : NONE / LOW / MODERATE / HIGH / EXTREME
  - Opportunity cost    : total dollar regret at $10,000 capital

Run from project root:
    python test_counterfactual.py
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import yfinance as yf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_engine.technical_agent     import TechnicalAgent, build_lstm_features, SEQ_LEN
from ml_engine.uncertainty_agent   import UncertaintyAgent
from ml_engine.hybrid_regime_agent import HybridRegimeAgent
from ml_engine.fusion_agent        import FusionAgent
from ml_engine.heatmap_agent       import HeatmapAgent
from ml_engine.counterfactual_engine import CounterfactualEngine

try:
    from ml_engine.conflict_resolver import ConflictResolver
    _CONFLICT_OK = True
except ImportError:
    _CONFLICT_OK = False

try:
    from ml_engine.risk_engine import RiskEngine
    _RISK_OK = True
except ImportError:
    _RISK_OK = False

# -- Paths ---------------------------------------------------------------------
MODEL_PATH  = r"D:\FinFolioX\saved_models\lstm_model.keras"
SCALER_PATH = r"D:\FinFolioX\saved_models\lstm_scaler.pkl"
REGIME_PATH = r"D:\FinFolioX\saved_models\hmm_regime_hybrid.pkl"
FUSION_PATH = r"D:\FinFolioX\saved_models\attention_fusion.pth"

# -- Constants -----------------------------------------------------------------
DEFAULT_CAPITAL    = 10_000.0
BUY_THRESHOLD      = 0.52
SELL_THRESHOLD     = 0.40
STRONG_CONF        = 0.75   # v2.2: override Bear regime when LSTM overwhelmingly bullish
BEAR_LEAN_SELL     = 0.60   # v2.2: Bear regime + conf <= this -> SELL not HOLD
COMMODITY_BUY_T    = 0.55
COMMODITY_TICKERS  = {"GLD","SLV","USO","UNG","GDX"}
BUY_GDI_MAX        = 55.0
UNCERTAINTY_HIGH   = 0.15
UNCERTAINTY_MOD    = 0.05

# -- Test windows HOLD same 6 as test_fusion.py + new Mar17->23 -------------------
TEST_WINDOWS = [
    ("2026-03-03", "2026-03-08", "Mar03->08  Bear start"),
    ("2026-03-04", "2026-03-09", "Mar04->09  Bear early"),
    ("2026-03-15", "2026-03-20", "Mar15->20  Deep Bear"),
    ("2026-03-05", "2026-03-10", "Mar05->10  Bounce"),
    ("2025-08-01", "2025-08-08", "Aug01->08  Bull Phase"),
    ("2025-10-01", "2025-10-08", "Oct01->08  Sideways"),
    ("2026-03-17", "2026-03-23", "Mar17->23  Iran+Fed"),   # ← NEW
]

# -- 30 tickers ----------------------------------------------------------------
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

# -- Sentiment scores HOLD all 7 windows ------------------------------------------
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
    # ← NEW window: Iran war escalation + hawkish Fed (March 18 hold) + oil >$100
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

def fetch_price_at(ticker, date_str):
    """Fetch closing price at a specific date (or nearest prior trading day)."""
    import io, contextlib
    dt     = pd.to_datetime(date_str)
    start  = (dt - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    end    = (dt + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    with contextlib.redirect_stdout(io.StringIO()):
        df = yf.download(ticker, start=start, end=end,
                         auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty:
        return float("nan")
    try:
        return float(df["Close"].asof(dt))
    except Exception:
        return float(df["Close"].iloc[-1])

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
               fusion_agent, heatmap_agent, conflict_resolver,
               risk_engine, cf_engine):

    test_date    = snap_to_trading_day(test_date)
    outcome_date = snap_to_trading_day(outcome_date)

    sent_date = test_date
    if sent_date not in MANUAL_SENTIMENT:
        diffs     = [(abs((pd.to_datetime(sent_date)-pd.to_datetime(k)).days), k)
                     for k in MANUAL_SENTIMENT]
        sent_date = min(diffs)[1]
        print(f"   ℹ️  Sentiment mapped: {test_date} -> {sent_date}")

    sentiment_scores = MANUAL_SENTIMENT[sent_date]

    # Fetch TLT prices for hold_pnl proxy
    tlt_entry = fetch_price_at("TLT", test_date)
    tlt_exit  = fetch_price_at("TLT", outcome_date)

    print(f"\n{'*'*116}")
    print(f"  {label}  |  {test_date} -> {outcome_date}")
    print(f"{'*'*116}")
    print(f"\n  {'Ticker':<6} {'Decision':<8} {'ArbConf':>8} {'EntryP':>8} "
          f"{'ExitP':>8} {'Move%':>7} {'Optimal':<8} "
          f"{'Regret%':>8} {'Level':<10} {'CFcalib'}")
    print(f"  {'-'*118}")

    window_cf_results = []
    window_rows       = []

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
            regime_label_t, regime_vol, regime_conf = regime_agent.detect(hist, ticker)
            vol_v      = 0.9 if regime_label_t=="Bear" else 0.2 if regime_label_t=="Bull" else 0.5
            sent_score = sentiment_scores.get(ticker, 0.0)

            raw_conf, _ = fusion_agent.predict(
                lstm_p=mc_mean, sent_s=sent_score, vol_v=vol_v,
            )
            gated_conf = apply_fusion_gates(
                raw_conf, lstm_stretched, sent_score, regime_label_t, regime_conf
            )

            gdi_result  = heatmap_agent.analyze(
                lstm_score=lstm_stretched, sent_score=sent_score,
                regime_label=regime_label_t, regime_vol=regime_vol,
            )
            gdi         = gdi_result["gdi"]
            gdi_penalty = gdi_result["penalty"]

            arb_conf = gated_conf
            if conflict_resolver:
                try:
                    res = conflict_resolver.arbitrate(
                        tech_score=lstm_stretched, sent_score=sent_score,
                        mc_std=mc_std, regime_label=regime_label_t,
                        risk_score=0.2, fusion_confidence=gated_conf,
                        trust_scores=None,
                    )
                    arb_conf = res.get("adjusted_confidence", gated_conf)
                except Exception:
                    arb_conf = gated_conf

            alloc_pct = 0.0
            if risk_engine:
                try:
                    last_price = float(hist["Close"].iloc[-1])
                    alloc_pct, _ = risk_engine.calculate_position_size(
                        arb_conf, regime_vol,
                        disagreement_penalty=gdi_penalty, regime=regime_label_t,
                    )
                except Exception:
                    alloc_pct = 0.0

            eff_thr = COMMODITY_BUY_T if ticker in COMMODITY_TICKERS else BUY_THRESHOLD
            if (alloc_pct > 0.0 and arb_conf >= eff_thr
                    and regime_label_t != "Bear" and gdi * 100 < BUY_GDI_MAX):
                decision = "BUY"
            elif (alloc_pct > 0.0 and arb_conf >= STRONG_CONF
                    and regime_label_t == "Bear" and gdi * 100 < BUY_GDI_MAX):
                decision = "BUY"          # FIX A: high-confidence Bear override
            elif arb_conf <= SELL_THRESHOLD:
                decision = "SELL"         # boundary fix: <= not <
            elif regime_label_t == "Bear" and arb_conf <= BEAR_LEAN_SELL:
                decision = "SELL"         # FIX B: Bear lean-SELL
            else:
                decision = "HOLD"

            # -- Fetch entry / exit prices -------------------------------------
            entry_price = fetch_price_at(ticker, test_date)
            exit_price  = fetch_price_at(ticker, outcome_date)

            if (np.isnan(entry_price) or np.isnan(exit_price)
                    or entry_price <= 0):
                print(f"  {ticker:<6}  price data unavailable HOLD skipped")
                continue

            # -- Run counterfactual --------------------------------------------
            cf_result = cf_engine.analyze(
                actual_decision=decision,
                decision_price=entry_price,
                actual_price_t5=exit_price,
                confidence=arb_conf,
                ticker=ticker,
                tlt_price_start=tlt_entry if not np.isnan(tlt_entry) else None,
                tlt_price_end=tlt_exit   if not np.isnan(tlt_exit)  else None,
            )
            cf_engine.record_to_tracker(cf_result)
            window_cf_results.append(cf_result)

            # Calibration flag
            cal = cf_result.get("confidence_calibrated")
            cal_str = "[OK]" if cal is True else "[BAD]" if cal is False else "~"

            # Regret level icon
            icons = {"NONE":"[OK]","LOW":"🔵","MODERATE":"🟡","HIGH":"🟠","EXTREME":"🔴"}
            icon  = icons.get(cf_result["regret_level"], "❓")

            move_pct_disp = cf_result["move_pct"] * 100
            rgt_disp      = cf_result["regret_score"] * 100
            opt           = cf_result["optimal_decision"]
            match_marker  = "+" if decision == opt else " "

            print(f"  {ticker:<6} {decision:<8} {arb_conf:>8.4f} "
                  f"{entry_price:>8.2f} {exit_price:>8.2f} "
                  f"{move_pct_disp:>+6.2f}% {opt:<8}{match_marker}"
                  f"{rgt_disp:>7.2f}% {icon}{cf_result['regret_level']:<9} {cal_str}")

            window_rows.append({
                "ticker":        ticker,
                "test_date":     test_date,
                "outcome_date":  outcome_date,
                "decision":      decision,
                "arb_conf":      round(arb_conf, 4),
                "entry_price":   round(entry_price, 4),
                "exit_price":    round(exit_price, 4),
                "move_pct":      round(move_pct_disp, 3),
                "optimal":       opt,
                "regret_score":  round(rgt_disp, 3),
                "regret_level":  cf_result["regret_level"],
                "conf_calibrated": cal,
                "is_ambiguous":  cf_result["is_ambiguous"],
                "window":        label,
            })

        except Exception as e:
            print(f"  {ticker:<6}  ERROR: {e}")

    # -- Window summary ---------------------------------------------------------
    summary  = cf_engine.get_regret_summary(window_cf_results)
    miss_bd  = cf_engine.get_miss_type_breakdown(window_cf_results)
    opp_cost = cf_engine.opportunity_cost(window_cf_results, DEFAULT_CAPITAL)

    print(f"\n  -- Window Summary -------------------------------------------------")
    if summary:
        lc = summary["level_counts"]
        print(f"     Decisions      : {summary['n']}  ({summary['n_ambiguous']} ambiguous moves)")
        print(f"     Optimal match  : {summary['optimal_match_rate']:.1f}%  "
              f"(AI chose optimal in {int(summary['optimal_match_rate']*summary['n']/100)}/{summary['n']})")
        print(f"     Mean regret    : {summary['mean_regret_pct']:.3f}%  "
              f"(max: {summary['max_regret_pct']:.3f}%)")
        print(f"     Total regret   : {summary['total_regret_pct']:.3f}%  "
              f"= ${opp_cost:,.2f} opp. cost at ${DEFAULT_CAPITAL:,.0f}")
        print(f"     Conf. calib    : {summary['calib_rate_pct']:.1f}%"
              if summary['calib_rate_pct'] is not None else
              "     Conf. calib    : n/a")
        print(f"     Regret dist    : "
              f"NONE={lc['NONE']} LOW={lc['LOW']} MOD={lc['MODERATE']} "
              f"HIGH={lc['HIGH']} EXTREME={lc['EXTREME']}")
    if miss_bd:
        dom = miss_bd["dominant_issue"]
        dom_icon = "🟡" if dom == "HOLD_BIAS" else "🔴" if dom == "WRONG_DIR" else "[OK]"
        print(f"     Miss breakdown : "
              f"CORRECT={miss_bd['correct']}  "
              f"HOLD_BIAS={miss_bd['hold_bias']}({miss_bd['hold_bias_pct']:.0f}%)  "
              f"WRONG_DIR={miss_bd['wrong_dir']}({miss_bd['wrong_dir_pct']:.0f}%)  "
              f"-> dominant={dom_icon}{dom}")
        print(f"     Regret by type : "
              f"HOLD_BIAS=${cf_engine.opportunity_cost([r for r in window_cf_results if cf_engine.classify_miss_type(r)=='HOLD_BIAS'], DEFAULT_CAPITAL):,.0f}  "
              f"WRONG_DIR=${cf_engine.opportunity_cost([r for r in window_cf_results if cf_engine.classify_miss_type(r)=='WRONG_DIR'], DEFAULT_CAPITAL):,.0f}")
    tracker_s = cf_engine.tracker.get_summary()
    print(f"     Tracker status : {tracker_s['status']}  "
          f"(rolling mean={tracker_s['mean_regret']*100:.3f}%  "
          f"extreme_rate={tracker_s['extreme_rate']*100:.1f}%)")

    return {
        "label":        label,
        "test_date":    test_date,
        "outcome_date": outcome_date,
        "summary":      summary,
        "miss_bd":      miss_bd,
        "opp_cost":     opp_cost,
        "rows":         window_rows,
        "cf_results":   window_cf_results,
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 116)
    print("  COUNTERFACTUAL DECISION ENGINE BACKTEST  |  7 Windows x 30 Tickers")
    print("  Phase 15 v2.1 HOLD Fixed: conf<=0.40 boundary (was <0.40), outlier analysis")
    print("=" * 116)
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

    conflict_resolver = None
    if _CONFLICT_OK:
        try:
            conflict_resolver = ConflictResolver()
            print("  [OK] ConflictResolver")
        except Exception as e:
            print(f"  [WARN]  ConflictResolver: {e}")

    risk_engine = None
    if _RISK_OK:
        try:
            risk_engine = RiskEngine(default_account_size=DEFAULT_CAPITAL)
            print(f"  [OK] RiskEngine  (capital=${DEFAULT_CAPITAL:,.0f})")
        except Exception as e:
            print(f"  [WARN]  RiskEngine: {e}")

    cf_engine = CounterfactualEngine()

    # -- Run windows ------------------------------------------------------------
    all_stats   = []
    all_rows    = []
    all_cf      = []

    for test_date, outcome_date, label in TEST_WINDOWS:
        s = run_window(
            test_date, outcome_date, label,
            tech_agent, uncertainty_agent, regime_agent,
            fusion_agent, heatmap_agent, conflict_resolver,
            risk_engine, cf_engine,
        )
        all_stats.append(s)
        all_rows.extend(s["rows"])
        all_cf.extend(s["cf_results"])

    # -- Consolidated ----------------------------------------------------------
    print("\n" + "=" * 116)
    print("  CONSOLIDATED COUNTERFACTUAL RESULTS")
    print("=" * 116)
    print(f"\n  {'Window':<32} {'N':>4} {'OptMatch':>9} {'MeanRgt':>9} "
          f"{'MaxRgt':>8} {'OppCost':>10} {'CalibPct':>9}  {'NONE':>5} "
          f"{'LOW':>5} {'MOD':>5} {'HIGH':>5} {'EXT':>5}")
    print(f"  {'-'*116}")

    for s in all_stats:
        sm = s["summary"]
        if not sm:
            print(f"  {s['label']:<32}  (no data)")
            continue
        lc = sm["level_counts"]
        of = "[OK]" if sm["optimal_match_rate"] >= 50 else "[WARN] "
        print(f"  {s['label']:<32} {sm['n']:>4} "
              f"  {sm['optimal_match_rate']:>7.1f}%{of}"
              f"  {sm['mean_regret_pct']:>7.3f}%"
              f"  {sm['max_regret_pct']:>7.3f}%"
              f"  ${s['opp_cost']:>8,.2f}"
              f"  {(str(sm['calib_rate_pct'])+'%') if sm['calib_rate_pct'] else 'n/a':>8}"
              f"  {lc['NONE']:>5} {lc['LOW']:>5} {lc['MODERATE']:>5}"
              f"  {lc['HIGH']:>5} {lc['EXTREME']:>5}")

    # Overall aggregate
    all_summary = cf_engine.get_regret_summary(all_cf)
    total_opp   = cf_engine.opportunity_cost(all_cf, DEFAULT_CAPITAL)
    lc_all      = all_summary.get("level_counts", {})

    print(f"  {'-'*116}")
    print(f"  {'OVERALL':<32} {all_summary.get('n',0):>4} "
          f"  {all_summary.get('optimal_match_rate',0):>7.1f}%  "
          f"  {all_summary.get('mean_regret_pct',0):>7.3f}%"
          f"  {all_summary.get('max_regret_pct',0):>7.3f}%"
          f"  ${total_opp:>8,.2f}"
          f"  {(str(all_summary.get('calib_rate_pct'))+'%') if all_summary.get('calib_rate_pct') else 'n/a':>8}"
          f"  {lc_all.get('NONE',0):>5} {lc_all.get('LOW',0):>5}"
          f"  {lc_all.get('MODERATE',0):>5}  {lc_all.get('HIGH',0):>5}"
          f"  {lc_all.get('EXTREME',0):>5}")

    # -- Miss-type breakdown across all windows --------------------------------
    all_miss_bd = cf_engine.get_miss_type_breakdown(all_cf)

    # -- Regret level breakdown -------------------------------------------------
    print(f"\n  -- Regret Level Explanation -----------------------------------------")
    print(f"  [OK] NONE    : regret ≤ 0.2%  HOLD AI chose optimally")
    print(f"  🔵 LOW     : 0.2–1.0%       HOLD minor miss, acceptable")
    print(f"  🟡 MODERATE: 1.0–3.0%       HOLD noticeable miss, review signals")
    print(f"  🟠 HIGH    : 3.0–7.0%       HOLD significant miss, recalibration recommended")
    print(f"  🔴 EXTREME : > 7.0%         HOLD major miss, trust penalty applied")

    # -- Root-cause diagnosis --------------------------------------------------
    om   = all_summary.get("optimal_match_rate", 0)
    mr   = all_summary.get("mean_regret_pct", 99)
    cr   = all_summary.get("calib_rate_pct")
    om_ok = om >= 50
    mr_ok = mr < 3.0
    cr_ok = cr is not None and cr >= 60

    hb_total   = all_miss_bd.get("hold_bias", 0)
    wd_total   = all_miss_bd.get("wrong_dir", 0)
    hb_regret  = all_miss_bd.get("hold_bias_regret_pct", 0)
    wd_regret  = all_miss_bd.get("wrong_dir_regret_pct", 0)
    hb_dollar  = cf_engine.opportunity_cost(
        [r for r in all_cf if cf_engine.classify_miss_type(r) == "HOLD_BIAS"],
        DEFAULT_CAPITAL)
    wd_dollar  = cf_engine.opportunity_cost(
        [r for r in all_cf if cf_engine.classify_miss_type(r) == "WRONG_DIR"],
        DEFAULT_CAPITAL)

    print(f"\n  -- 🔬 Root-Cause Diagnosis ------------------------------------------")
    print(f"  Issue 1 HOLD WRONG_DIR  (LSTM predicted wrong direction on big moves):")
    print(f"    Count  : {wd_total} decisions  ({all_miss_bd.get('wrong_dir_pct',0):.1f}% of all)")
    print(f"    Cost   : ${wd_dollar:,.2f}  ({wd_regret:.2f}% of capital)")
    print(f"    Cause  : LSTM model not trained on extreme momentum events")
    print(f"    Action : Logged to RegretTracker -> MetaAgent recalibration")

    print(f"\n  Issue 2 HOLD HOLD_BIAS  (system defaulted to HOLD in big-move situations):")
    print(f"    Count  : {hb_total} decisions  ({all_miss_bd.get('hold_bias_pct',0):.1f}% of all)")
    print(f"    Cost   : ${hb_dollar:,.2f}  ({hb_regret:.2f}% of capital)")
    print(f"    Cause  : Bear-regime hard-block + neutral-zone fell into HOLD")
    print(f"    Fixes applied in v2.2:")
    print(f"      FIX A: conf >= 0.75 in Bear -> BUY  (LSTM overwhelmingly bullish)")
    print(f"      FIX B: Bear + conf <= 0.60 -> SELL  (medium conf in downtrend = lean bearish)")
    print(f"      FIX C: conf <= 0.40 boundary fix   (was `<`, now `<=`)")

    # Show top remaining HOLD_BIAS cases after fix
    hb_cases = sorted(
        [r for r in all_rows if cf_engine.classify_miss_type(
            {"actual_decision": r["decision"], "optimal_decision": r["optimal"]}
        ) == "HOLD_BIAS"],
        key=lambda x: x.get("regret_score", 0), reverse=True
    )
    if hb_cases[:4]:
        print(f"\n  Remaining HOLD_BIAS cases (top 4) HOLD need LSTM retraining or higher conf:")
        for r in hb_cases[:4]:
            print(f"    {r['ticker']:<6} {r['window']:<25}  "
                  f"conf={r['arb_conf']:.3f}  move={r['move_pct']:>+6.2f}%  "
                  f"regret={r['regret_score']:.2f}%")

    wd_cases = sorted(
        [r for r in all_rows if cf_engine.classify_miss_type(
            {"actual_decision": r["decision"], "optimal_decision": r["optimal"]}
        ) == "WRONG_DIR"],
        key=lambda x: x.get("regret_score", 0), reverse=True
    )
    if wd_cases[:4]:
        print(f"\n  Top WRONG_DIR cases HOLD inherent model limitations:")
        for r in wd_cases[:4]:
            print(f"    {r['ticker']:<6} {r['window']:<25}  "
                  f"dec={r['decision']:<5}  move={r['move_pct']:>+6.2f}%  "
                  f"regret={r['regret_score']:.2f}%  -> LSTM was wrong")

    # -- Final verdict ---------------------------------------------------------
    print(f"\n  {'='*65}")
    print(f"  COUNTERFACTUAL ENGINE VERDICT  (v2.2)")
    print(f"  {'='*65}")
    print(f"  Optimal match rate    : {om:.1f}%  "
          + ("[OK] PASS (≥50%)" if om_ok else "[WARN]  LOW"))
    print(f"  Mean regret           : {mr:.3f}%  "
          + ("[OK] PASS (<3%)" if mr_ok else "[WARN]  HIGH"))
    print(f"  Confidence calibration: {cr:.1f}%  "
          + ("[OK] PASS (≥60%)" if cr_ok else "[WARN]  LOW") if cr else "  n/a")
    print(f"  Total opportunity cost: ${total_opp:,.2f}")
    print(f"  HOLD_BIAS cost (Issue 2): ${hb_dollar:,.2f}  "
          + ("[OK] Fixed by FIX A/B/C" if hb_total < 10 else
             f"[WARN]  {hb_total} cases remain HOLD consider LSTM retraining"))
    print(f"  WRONG_DIR cost (Issue 1): ${wd_dollar:,.2f}  "
          + "[WARN]  Inherent HOLD logged to RegretTracker for MetaAgent")

    tracker_final = cf_engine.tracker.get_summary()
    print(f"\n  RegretTracker (rolling {tracker_final['n']} decisions):")
    print(f"    Mean regret   : {tracker_final['mean_regret']*100:.3f}%")
    print(f"    Std regret    : {tracker_final['std_regret']*100:.3f}%")
    print(f"    Extreme rate  : {tracker_final['extreme_rate']*100:.1f}%")
    print(f"    System status : {tracker_final['status']}")

    # -- Per-window bars --------------------------------------------------------
    print(f"\n  Per-window optimal match rate + miss breakdown:")
    for s in all_stats:
        sm = s.get("summary", {})
        mb = s.get("miss_bd", {})
        if not sm:
            continue
        rate = sm.get("optimal_match_rate", 0)
        bar  = "█" * int(rate / 5) + "░" * (20 - int(rate / 5))
        flag = "[OK]" if rate >= 50 else "[WARN] "
        hb   = mb.get("hold_bias", 0)
        wd   = mb.get("wrong_dir", 0)
        print(f"  {s['label']:<32} [{bar}] {rate:.1f}%{flag}  "
              f"HB={hb} WD={wd}  "
              f"regret={sm.get('mean_regret_pct',0):.2f}%  "
              f"opp=${s['opp_cost']:,.0f}")

    # -- Save -------------------------------------------------------------------
    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv("counterfactual_backtest.csv", index=False)
        print(f"\n  Saved -> counterfactual_backtest.csv  ({len(all_rows)} rows)")

    print("\nDone.\n")


if __name__ == "__main__":
    main()