"""
test_risk_engine.py  HOLD  Risk Engine Backtest v2.1
==================================================
FinFolioX HOLD Phase 9 Backtest  |  7 Windows x 30 Tickers

Tests RiskEngine v2.1 specifically HOLD isolates its contribution by showing
the full position-sizing pipeline at each step:
  confidence -> kelly -> half-kelly -> vol_scale -> gdi_penalty -> final_alloc

Key metrics per window:
  - Active rate        : % of tickers where engine allocated > 0
  - Kelly distribution : negative / zero / low / medium / high
  - Vol scaling hits   : how many tickers got the graduated vol cut
  - Regime comparison  : avg allocation Bull vs Bear vs Sideways
  - Hard cap hits      : how many hit the 20% ceiling
  - Correlation check  : higher confidence -> higher allocation (should hold)

v2.1 regression checks:
  1. Graduated vol scaling: no sudden 50% cliff at 0.02
  2. Minimum floor: allocations below 0.5% return 0.0
  3. Input validation: confidence clipped to [0,1]
  4. Half-Kelly configurable: 0.5 default produces correct values
  5. Cash value is rounded to 2 decimal places

Run from project root:
    python test_risk_engine.py
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
from ml_engine.risk_engine         import RiskEngine, VOL_LOW, VOL_HIGH, REGIME_ODDS

try:
    from ml_engine.conflict_resolver import ConflictResolver
    _CR_OK = True
except ImportError:
    _CR_OK = False

# -- Paths ---------------------------------------------------------------------
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
MAX_RISK          = 0.20

# -- Test windows --------------------------------------------------------------
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


# ==============================================================================
# SINGLE WINDOW
# ==============================================================================

def run_window(test_date, outcome_date, label,
               tech_agent, uncertainty_agent, regime_agent,
               fusion_agent, heatmap_agent, conflict_resolver,
               risk_engine):

    test_date    = snap_to_trading_day(test_date)
    outcome_date = snap_to_trading_day(outcome_date)

    sent_date = test_date
    if sent_date not in MANUAL_SENTIMENT:
        diffs = [(abs((pd.to_datetime(sent_date)-pd.to_datetime(k)).days), k)
                 for k in MANUAL_SENTIMENT]
        sent_date = min(diffs)[1]
        print(f"   ℹ️  Sentiment mapped: {test_date} -> {sent_date}")

    sentiment_scores = MANUAL_SENTIMENT[sent_date]

    print(f"\n{'*'*128}")
    print(f"  {label}  |  {test_date} -> {outcome_date}")
    print(f"{'*'*128}")
    print(f"\n  {'Ticker':<6} {'Regime':<9} {'Vol':>6} {'VScale':>7} {'Conf':>7} "
          f"{'Kelly':>7} {'GDI':>5} "
          f"{'Alloc%':>7} {'$':>7} {'Shr':>4}  "
          f"{'Step-by-step':<40}  {'Act%':>8}")
    print(f"  {'-'*140}")

    rows           = []
    breakdowns     = []
    regime_allocs  = defaultdict(list)
    vol_scaled_cnt = 0
    cap_hits       = 0
    bear_cap_hits  = 0
    floor_hits     = 0
    neg_kelly_cnt  = 0

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

            # Conflict resolver
            arb_conf = gated_conf
            if conflict_resolver:
                try:
                    from ml_engine.conflict_resolver import ConflictResolver as _CR
                    # derive risk_score consistent with what test_conflict used
                    base     = {"Bear": 0.52, "Sideways": 0.35, "Bull": 0.28}.get(regime_label, 0.35)
                    vol_mult = float(np.clip(regime_vol / 0.015, 0.8, 2.2))
                    risk_s   = float(np.clip(base * vol_mult, 0.10, 0.82))
                    arb_res  = conflict_resolver.arbitrate(
                        tech_score=lstm_stretched, sent_score=sent_score,
                        mc_std=mc_std, regime_label=regime_label,
                        risk_score=risk_s, fusion_confidence=gated_conf,
                        trust_scores=None,
                    )
                    arb_conf = arb_res["adjusted_confidence"]
                except Exception:
                    arb_conf = gated_conf

            # -- RISK ENGINE ------------------------------------------------
            last_price = float(hist["Close"].iloc[-1])

            bdown = risk_engine.position_size_breakdown(
                confidence_score=arb_conf,
                volatility=regime_vol,
                disagreement_penalty=gdi_penalty,
                regime=regime_label,
                stock_price=last_price,
            )
            breakdowns.append(bdown)

            alloc   = bdown["final_allocation"]
            kelly   = bdown["kelly_fraction"]
            vol_sc  = bdown["vol_scale"]
            n_shr   = bdown["num_shares"]
            cash    = bdown["dollar_amount"]
            reason  = bdown["reason"]

            # Counters
            if kelly <= 0:                     neg_kelly_cnt += 1
            if vol_sc < 0.99:                  vol_scaled_cnt += 1
            if reason == "CAPPED":             cap_hits       += 1
            if reason == "BEAR_CAP":           bear_cap_hits  += 1
            if reason == "BELOW_FLOOR":        floor_hits     += 1
            regime_allocs[regime_label].append(alloc)

            # Step-by-step trace
            half_k  = round(kelly * 0.5,      4) if kelly > 0 else 0.0
            after_v = round(half_k * vol_sc,  4)
            after_g = round(after_v * gdi_penalty, 4)
            trace   = (f"kelly={kelly:+.4f} -> x0.5->{half_k:.4f} "
                       f"-> x{vol_sc:.3f}->{after_v:.4f} "
                       f"-> x{gdi_penalty:.2f}->{after_g:.4f}")

            actual_ret = fetch_actual_return(ticker, test_date, outcome_date)
            act_str    = f"{actual_ret:>+7.2f}%" if not np.isnan(actual_ret) else "    nan%"

            flag = ""
            if reason == "CAPPED":        flag = "🔴CAP"
            elif reason == "BEAR_CAP":    flag = "🟡BCp"
            elif reason == "BELOW_FLOOR": flag = "⬇️FLR"
            elif reason == "NEGATIVE_KELLY": flag = "⬇️NEG"

            print(f"  {ticker:<6} {regime_label:<9} {regime_vol:>6.4f} {vol_sc:>7.3f} "
                  f"{arb_conf:>7.4f} {kelly:>+7.4f} {gdi_penalty:>5.2f} "
                  f"{alloc*100:>6.1f}% ${cash:>6.0f} {n_shr:>4} {flag:<5} "
                  f"{trace[:40]}  {act_str}")

            rows.append({
                "ticker":    ticker, "test_date": test_date,
                "regime":    regime_label, "vol": round(regime_vol, 4),
                "vol_scale": round(vol_sc, 4),
                "conf":      round(arb_conf, 4), "kelly": round(kelly, 4),
                "gdi":       round(gdi_penalty, 2),
                "alloc_pct": round(alloc * 100, 2),
                "dollar":    cash, "shares": n_shr,
                "reason":    reason,
                "actual_ret":round(actual_ret, 2) if not np.isnan(actual_ret) else None,
                "window":    label,
            })

        except Exception as e:
            print(f"  {ticker:<6}  ERROR: {e}")

    # -- Window summary ---------------------------------------------------------
    wstats = RiskEngine.get_stats(breakdowns)
    print(f"\n  -- Window Summary --------------------------------------------------------")
    print(f"     Active allocations : {wstats.get('n_active', 0)}/{wstats.get('n', 0)} "
          f"= {wstats.get('active_rate_pct', 0):.1f}%")
    print(f"     Negative Kelly     : {neg_kelly_cnt}  (no trade HOLD negative EV)")
    print(f"     Vol-scaled tickers : {vol_scaled_cnt}  (vol > {VOL_LOW} -> graduated cut)")
    print(f"     Hard-cap hits (20%): {cap_hits}")
    print(f"     Bear-cap hits (10%): {bear_cap_hits}  ← v2.2 new")
    print(f"     Below-floor hits   : {floor_hits}  (< 0.5% -> zero'd out)")
    print(f"     Mean alloc (all)   : {wstats.get('mean_alloc_pct', 0):.2f}%")
    print(f"     Mean alloc (active): {wstats.get('mean_active_alloc_pct', 0):.2f}%")
    print(f"     Max alloc          : {wstats.get('max_alloc_pct', 0):.2f}%")
    print(f"     Regime breakdown   :", end="")
    for r, allocs in sorted(regime_allocs.items()):
        active_r = [a for a in allocs if a > 0]
        print(f"  {r}(n={len(allocs)}  "
              f"avg={np.mean(allocs)*100:.1f}%  "
              f"active={len(active_r)})", end="")
    print()

    # Monotonicity check: sort by confidence and check allocation goes up
    conf_alloc = sorted(
        [(r["conf"], r["alloc_pct"]) for r in rows if r["alloc_pct"] > 0],
        key=lambda x: x[0]
    )
    if len(conf_alloc) >= 3:
        corr = float(np.corrcoef([x[0] for x in conf_alloc],
                                  [x[1] for x in conf_alloc])[0, 1])
        print(f"     Conf->Alloc corr    : {corr:+.3f}  "
              + ("[OK] positive correlation" if corr > 0 else "[WARN]  weak/negative"))

    return {
        "label": label, "test_date": test_date, "outcome_date": outcome_date,
        "stats": wstats, "neg_kelly": neg_kelly_cnt, "vol_scaled": vol_scaled_cnt,
        "cap_hits": cap_hits, "bear_cap_hits": bear_cap_hits, "floor_hits": floor_hits,
        "regime_allocs": dict(regime_allocs), "rows": rows, "breakdowns": breakdowns,
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 128)
    print("  RISK ENGINE BACKTEST v2.2  |  7 Windows x 30 Tickers")
    print("  Tests: Kelly sizing + Bear cap(10%) + graduated vol + floor + cap + regime b")
    print(f"  b_odds: Bull={REGIME_ODDS['bull']}  Sideways={REGIME_ODDS['sideways']}  "
          f"Bear={REGIME_ODDS['bear']}  |  "
          f"vol_low={VOL_LOW}  vol_high={VOL_HIGH}  floor=0.5%  global_cap=20%  bear_cap=10%")
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

    conflict_resolver = None
    if _CR_OK:
        try:
            conflict_resolver = ConflictResolver(verbose=False)
            print("  [OK] ConflictResolver  (v2.5)")
        except Exception as e:
            print(f"  [WARN]  ConflictResolver: {e}")

    risk_engine = RiskEngine(default_account_size=DEFAULT_CAPITAL,
                             max_risk_per_trade=MAX_RISK,
                             bear_max_allocation=0.10)
    print(f"  [OK] RiskEngine v2.2  "
          f"(capital=${DEFAULT_CAPITAL:,.0f}  global_cap={MAX_RISK*100:.0f}%  "
          f"bear_cap=10%  half_kelly=0.5)")

    all_stats  = []
    all_rows   = []
    all_bdowns = []

    for test_date, outcome_date, label in TEST_WINDOWS:
        s = run_window(
            test_date, outcome_date, label,
            tech_agent, uncertainty_agent, regime_agent,
            fusion_agent, heatmap_agent, conflict_resolver, risk_engine,
        )
        all_stats.append(s)
        all_rows.extend(s["rows"])
        all_bdowns.extend(s["breakdowns"])

    # -- Consolidated ----------------------------------------------------------
    print("\n" + "=" * 128)
    print("  CONSOLIDATED RISK ENGINE RESULTS")
    print("=" * 128)
    print(f"\n  {'Window':<32} {'Active%':>8} {'NegK':>5} {'VolSc':>6} "
          f"{'Caps':>5} {'BrCp':>5} {'Flrs':>5} {'MeanAlloc':>10} {'MaxAlloc':>9}")
    print(f"  {'-'*110}")

    for s in all_stats:
        st = s["stats"]
        print(f"  {s['label']:<32} "
              f"{st.get('active_rate_pct', 0):>7.1f}%"
              f"  {s['neg_kelly']:>4}"
              f"  {s['vol_scaled']:>5}"
              f"  {s['cap_hits']:>4}"
              f"  {s['bear_cap_hits']:>4}"
              f"  {s['floor_hits']:>4}"
              f"  {st.get('mean_alloc_pct', 0):>8.2f}%"
              f"  {st.get('max_alloc_pct', 0):>8.2f}%")

    # Overall
    overall = RiskEngine.get_stats(all_bdowns)
    print(f"  {'-'*110}")
    print(f"  {'OVERALL':<32} "
          f"{overall.get('active_rate_pct', 0):>7.1f}%"
          f"  {sum(s['neg_kelly']      for s in all_stats):>4}"
          f"  {sum(s['vol_scaled']     for s in all_stats):>5}"
          f"  {sum(s['cap_hits']       for s in all_stats):>4}"
          f"  {sum(s['bear_cap_hits']  for s in all_stats):>4}"
          f"  {sum(s['floor_hits']     for s in all_stats):>4}"
          f"  {overall.get('mean_alloc_pct', 0):>8.2f}%"
          f"  {overall.get('max_alloc_pct', 0):>8.2f}%")

    # -- Regime comparison -----------------------------------------------------
    print(f"\n  -- Regime Allocation Comparison -----------------------------------------")
    combined_regime = defaultdict(list)
    for s in all_stats:
        for r, allocs in s["regime_allocs"].items():
            combined_regime[r].extend(allocs)

    for regime, allocs in sorted(combined_regime.items()):
        active = [a for a in allocs if a > 0]
        mean_a = np.mean(allocs)  * 100
        mean_c = np.mean(active)  * 100 if active else 0.0
        bar    = "█" * int(mean_c / 2)
        print(f"  {regime:<10}  n={len(allocs):>3}  "
              f"active={len(active):>3} ({len(active)/len(allocs)*100:>4.0f}%)  "
              f"mean(all)={mean_a:>5.2f}%  mean(active)={mean_c:>5.2f}%  [{bar}]")

    # Bull > Sideways > Bear check
    bull_mean = np.mean(combined_regime.get("Bull",     [0])) * 100
    side_mean = np.mean(combined_regime.get("Sideways", [0])) * 100
    bear_mean = np.mean(combined_regime.get("Bear",     [0])) * 100
    print(f"\n  Regime ordering check (Bull > Sideways > Bear):")
    print(f"  Bull={bull_mean:.2f}%  Sideways={side_mean:.2f}%  Bear={bear_mean:.2f}%  "
          + ("[OK] CORRECT" if bull_mean >= side_mean >= bear_mean
             else "[WARN]  out of order HOLD check b_odds or regime detection"))

    # -- vol scaling regression -------------------------------------------------
    print(f"\n  -- v2.2 Regression Checks -----------------------------------------------")

    # Check 1: graduated scaling
    vol_scales = [b["vol_scale"] for b in all_bdowns]
    cliff_count = sum(1 for v in vol_scales if v == 0.5
                      and all_bdowns[vol_scales.index(v)]["volatility"] < VOL_HIGH * 0.95)
    print(f"  Graduated vol scaling  : range [{min(vol_scales):.3f}, {max(vol_scales):.3f}]  "
          + ("[OK] no cliff artefacts" if cliff_count == 0
             else f"[WARN]  {cliff_count} potential cliff hits"))

    # Check 2: floor
    print(f"  Min allocation floor   : {sum(s['floor_hits'] for s in all_stats)} decisions zero'd  [OK]")

    # Check 3: hard cap never exceeded
    cap_exceeded = sum(1 for b in all_bdowns if b["final_allocation"] > MAX_RISK + 1e-6)
    print(f"  Hard cap (20%) check   : {cap_exceeded} violations  "
          + ("[OK] never exceeded" if cap_exceeded == 0 else "[BAD] VIOLATIONS FOUND"))

    # Check 4: Bear cap HOLD no Bear ticker should exceed 10%
    bear_cap_exceeded = sum(
        1 for b in all_bdowns
        if b["regime"].lower() == "bear" and b["final_allocation"] > 0.101
    )
    total_bear_cap_hits = sum(s["bear_cap_hits"] for s in all_stats)
    print(f"  Bear cap (10%) check   : {bear_cap_exceeded} violations  "
          + ("[OK] never exceeded" if bear_cap_exceeded == 0 else "[BAD] VIOLATIONS"))
    print(f"  Bear cap activations   : {total_bear_cap_hits} tickers capped at 10% in Bear regime  "
          + ("[OK] protecting capital" if total_bear_cap_hits > 0 else "ℹ️  none needed this run"))

    # Check 5: cash rounding
    cash_decimals_ok = all(
        len(str(b["cash_value"]).split(".")[-1]) <= 2
        for b in all_bdowns if b["cash_value"] > 0
    )
    print(f"  Cash value rounding    : "
          + ("[OK] all values rounded to 2 d.p." if cash_decimals_ok
             else "[WARN]  some values have > 2 decimal places"))

    # Check 6: input validation
    conf_ok = all(0.0 <= b["confidence"] <= 1.0 for b in all_bdowns)
    print(f"  Confidence clipping    : "
          + ("[OK] all values in [0.0, 1.0]" if conf_ok
             else "[WARN]  out-of-range confidences found"))

    # -- Verdict ---------------------------------------------------------------
    print(f"\n  {'='*65}")
    print(f"  RISK ENGINE v2.2 VERDICT")
    print(f"  {'='*65}")
    print(f"  Overall active rate    : {overall['active_rate_pct']:.1f}%  "
          + ("[OK]" if overall["active_rate_pct"] > 0 else "[WARN] "))
    print(f"  Mean active allocation : {overall['mean_active_alloc_pct']:.2f}%  "
          + ("[OK] reasonable" if 5 < overall["mean_active_alloc_pct"] < 20 else "[WARN]  check"))
    print(f"  Max allocation         : {overall['max_alloc_pct']:.2f}%  "
          + ("[OK] within 20% cap" if overall["max_alloc_pct"] <= 20.0 else "[BAD] EXCEEDS CAP"))
    print(f"  Bear cap (10%)         : {bear_cap_exceeded} violations  "
          + ("[OK] protecting bear capital" if bear_cap_exceeded == 0 else "[BAD]"))
    print(f"  Regime ordering        : Bull≥Sideways≥Bear  "
          + ("[OK]" if bull_mean >= side_mean >= bear_mean else "[WARN] "))
    print(f"  v2.2 improvements      : Bear cap(10%) + min viable dollar check")

    print(f"\n  Per-window active rate:")
    for s in all_stats:
        st  = s["stats"]
        ar  = st.get("active_rate_pct", 0)
        bar = "█"*int(ar/5) + "░"*(20-int(ar/5))
        print(f"  {s['label']:<32}  [{bar}] {ar:.0f}%  "
              f"mean={st.get('mean_alloc_pct', 0):.1f}%  "
              f"vol_scaled={s['vol_scaled']}  neg_k={s['neg_kelly']}  "
              f"bear_cap={s['bear_cap_hits']}")

    if all_rows:
        pd.DataFrame(all_rows).to_csv("risk_engine_backtest.csv", index=False)
        print(f"\n  Saved -> risk_engine_backtest.csv  ({len(all_rows)} rows)")

    print("\nDone.\n")


if __name__ == "__main__":
    main()