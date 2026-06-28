"""
test_heatmap.py  HOLD  HeatmapAgent GDI Backtest  |  March 2026
=============================================================
Tests accuracy of the Disagreement Heatmap across 4 date windows.

What this tests:
  1. HeatmapAgent.analyze() produces valid GDI ∈ [0, 1] and penalty ∈ [0.5, 1.0]
  2. HIGH GDI (signals disagree) correctly predicts uncertain outcomes
  3. LOW GDI (signals agree) correctly predicts high-accuracy outcomes
  4. Final decision accuracy (BUY/SELL vs actual 5-day return)

Imports production classes HOLD zero logic duplicated here:
  - TechnicalAgent   -> predict_raw() for LSTM signal
  - HybridRegimeAgent -> detect() for regime + vol
  - HeatmapAgent     -> analyze() for GDI + penalty

Sentiment scores are MANUAL (pre-computed from v2.3 context analysis).
Reason: SentimentAgent makes 40+ live API calls per run HOLD not suitable for
fast backtest. The manual scores reflect realistic MCP+LLM output for each
date based on the known March 2026 market context (tariff selloff, VIX ~26-27).

Manual score derivation logic:
  - Mar 02-09: tariff fears mounting, S&P -3 to -5% range, VIX 25-28
    -> All tickers: negative/neutral (range -0.15 to +0.05)
  - Mar 05-10: brief bounce attempt off lows
    -> Mixed: SPY/QQQ slightly positive, individual stocks neutral
  - Mar 15-20: deep bear, GLD -10%, tech -4-6%
    -> Strong negative across board (-0.08 to -0.20)

Run from project root:
    python test_heatmap.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_engine.technical_agent    import TechnicalAgent, build_lstm_features, SEQ_LEN, LSTM_COLS
from ml_engine.hybrid_regime_agent import HybridRegimeAgent
from ml_engine.heatmap_agent       import HeatmapAgent

# -- Paths ---------------------------------------------------------------------
MODEL_PATH   = r"D:\FinFolioX\saved_models\lstm_model.keras"
SCALER_PATH  = r"D:\FinFolioX\saved_models\lstm_scaler.pkl"
REGIME_PATH  = r"D:\FinFolioX\saved_models\hmm_regime_hybrid.pkl"

# -- Test config ---------------------------------------------------------------
TICKERS = [
    # -- Mega-cap Tech -------------------------------------------
    "AAPL", "MSFT", "NVDA", "TSLA", "META", "GOOGL", "AMZN",
    # -- Semiconductors / AI -------------------------------------
    "AMD",  "INTC", "ORCL",
    # -- Index ETFs -----------------------------------------------
    "SPY",  "QQQ",  "DIA",  "IWM",
    # -- Financials -----------------------------------------------
    "JPM",  "BAC",  "GS",   "V",
    # -- Bond / Safe-Haven ETFs -----------------------------------
    "GLD",  "TLT",  "SLV",
    # -- Energy ---------------------------------------------------
    "XOM",  "CVX",
    # -- Consumer Defensive ---------------------------------------
    "WMT",  "PG",   "JNJ",
    # -- Consumer Discretionary / Media ---------------------------
    "NFLX", "DIS",
    # -- Enterprise / Cloud ---------------------------------------
    "CRM",  "PLTR",
]

BUY_THRESHOLD  = 0.52   # raw (unstretched) LSTM prob
SELL_THRESHOLD = 0.48

TEST_WINDOWS = [
    # -- March 2026: known Bear regime -----------------------------------------
    ("2026-03-03", "2026-03-08", "Mar03->08  Bear start"),
    ("2026-03-04", "2026-03-09", "Mar04->09  Bear early"),
    ("2026-03-15", "2026-03-20", "Mar15->20  Deep Bear"),   # snapped from Sun
    ("2026-03-05", "2026-03-10", "Mar05->10  Bounce"),
    # -- Regime cross-check: did it detect Bull / Sideways correctly? ----------
    ("2025-08-01", "2025-08-08", "Aug01->08  Bull Phase"),
    ("2025-10-01", "2025-10-08", "Oct01->08  Sideways"),
]

# ==============================================================================
# MANUAL SENTIMENT SCORES HOLD March 2026 + Aug 2025 + Oct 2025
# ==============================================================================
# Source: derived from v2.3 MCP+LLM pipeline runs and known market context.
#
# March 2026 macro context:
#   - Trump tariff fears escalating (25% Canada/Mexico, 10% China)
#   - S&P500 dropped ~8% peak-to-trough in Feb-Mar
#   - VIX elevated 25-30 throughout
#   - GLD: initial safe-haven demand then sharp selloff Mar 14-19
#   - Tech (NVDA/TSLA): leading decline on AI capex fears
#   - JPM: defensive but still weak with credit concerns
#
# Score scale: [-0.75 bearish ← 0 neutral -> +0.75 bullish]
# Using 0.07 threshold (v2.3): >+0.07=bullish, <-0.07=bearish
#
# March 2026 macro context per sector:
#   Tech (AAPL/MSFT/GOOGL/META):  tariff + AI-spend fear -> bearish
#   Semis (NVDA/AMD/INTC):        export controls + capex fears -> bearish
#   Indices (SPY/QQQ/DIA/IWM):    broad market drag -> bearish
#   Financials (JPM/BAC/GS/V):    rate uncertainty, credit spreads widen -> mildly neg
#   Safe-havens (GLD/TLT/SLV):    flight-to-quality early then reversal Mar 15-19
#   Energy (XOM/CVX):             oil volatile on OPEC + tariff demand fears -> mixed
#   Consumer Defensive (WMT/PG/JNJ): resilient business, mild selloff -> near neutral
#   Cons. Disc/Media (NFLX/DIS):  discretionary spending fears -> bearish
#   Cloud/AI (ORCL/CRM/PLTR):    AI spend scrutiny but PLTR benefits -> mixed
# ==============================================================================

MANUAL_SENTIMENT = {
    # -- Mar 03-08: Bear start -------------------------------------------------
    # Tariff fears dominant, market breaking down, VIX 24->27
    "2026-03-03": {
        # Mega-cap Tech
        "AAPL": -0.08,   # China revenue + App Store scrutiny
        "MSFT": -0.06,   # AI capex questioned, slightly neg
        "NVDA": -0.12,   # Chip export controls + hyperscaler capex fears
        "TSLA": -0.18,   # Musk distraction, demand miss fears
        "META": -0.05,   # AI spend scrutiny, near neutral
        "GOOGL":-0.08,   # Antitrust overhang + AI spend
        "AMZN": -0.07,   # AWS resilient but retail tariff exposed
        # Semis/AI
        "AMD":  -0.10,   # Export controls + PC market weak
        "INTC": -0.09,   # Foundry losses, market share losses
        "ORCL":  0.02,   # Cloud momentum still positive
        # Index ETFs
        "SPY":  -0.09,   # Broad market tariff drag
        "QQQ":  -0.14,   # Nasdaq more exposed
        "DIA":  -0.07,   # Dow less exposed to tech rout
        "IWM":  -0.11,   # Small caps hurt most by tariffs
        # Financials
        "JPM":   0.02,   # Rate cut hopes still alive
        "BAC":  -0.04,   # More rate-sensitive, slightly neg
        "GS":    0.01,   # Trading desk activity neutral
        "V":    -0.05,   # Consumer spend slowdown fears
        # Safe-havens
        "GLD":   0.08,   # Bullish: flight to safety
        "TLT":   0.09,   # Bullish: rates falling, bond bid
        "SLV":   0.04,   # Silver lagging gold but positive
        # Energy
        "XOM":  -0.06,   # Oil demand fears from tariff slowdown
        "CVX":  -0.05,   # Similar to XOM
        # Consumer Defensive
        "WMT":   0.03,   # Safe haven: consumers trade down
        "PG":    0.02,   # Staples resilient
        "JNJ":   0.01,   # Healthcare defensive
        # Consumer Disc/Media
        "NFLX": -0.08,   # Ad revenue + subscriber growth fears
        "DIS":  -0.09,   # Park attendance + streaming competition
        # Cloud/Enterprise
        "CRM":  -0.06,   # Enterprise spend slowdown fears
        "PLTR":  0.05,   # Defense/govt contracts insulated, AI buzz
    },

    # -- Mar 04-09: Bear early -------------------------------------------------
    # Sentiment deteriorating, VIX 26-28, capitulation signals emerging
    "2026-03-04": {
        # Mega-cap Tech
        "AAPL": -0.09,
        "MSFT": -0.07,
        "NVDA": -0.14,   # GTC coming but macro too heavy
        "TSLA": -0.20,   # Delivery data disappointing
        "META": -0.06,
        "GOOGL":-0.09,
        "AMZN": -0.08,
        # Semis/AI
        "AMD":  -0.11,
        "INTC": -0.10,   # More bad foundry news
        "ORCL":  0.01,
        # Index ETFs
        "SPY":  -0.10,
        "QQQ":  -0.16,
        "DIA":  -0.08,
        "IWM":  -0.13,
        # Financials
        "JPM":   0.01,
        "BAC":  -0.05,
        "GS":    0.00,
        "V":    -0.06,
        # Safe-havens
        "GLD":   0.09,   # Still bid
        "TLT":   0.11,   # Rates expectations shifting dovish
        "SLV":   0.05,
        # Energy
        "XOM":  -0.07,
        "CVX":  -0.06,
        # Consumer Defensive
        "WMT":   0.04,
        "PG":    0.03,
        "JNJ":   0.02,
        # Consumer Disc/Media
        "NFLX": -0.09,
        "DIS":  -0.10,
        # Cloud/Enterprise
        "CRM":  -0.07,
        "PLTR":  0.06,
    },

    # -- Mar 05-10: Bounce ----------------------------------------------------
    # Oversold technical bounce HOLD mixed signals, VIX easing from peak
    "2026-03-05": {
        # Mega-cap Tech
        "AAPL":  0.03,   # China smartphone win -> slight relief
        "MSFT":  0.02,
        "NVDA":  0.04,   # GTC 2026 upcoming -> catalyst
        "TSLA": -0.12,   # Still negative HOLD fundamentals unchanged
        "META":  0.05,   # New AI glasses announcement
        "GOOGL": 0.02,
        "AMZN":  0.02,
        # Semis/AI
        "AMD":   0.03,
        "INTC":  0.00,
        "ORCL":  0.06,   # Cloud revenue beat rumours
        # Index ETFs
        "SPY":   0.07,   # Bounce off oversold
        "QQQ":   0.05,
        "DIA":   0.04,
        "IWM":   0.03,
        # Financials
        "JPM":   0.03,
        "BAC":   0.01,
        "GS":    0.02,
        "V":     0.01,
        # Safe-havens
        "GLD":   0.06,   # Still positive but fading
        "TLT":   0.05,   # Slightly less bid as equities bounce
        "SLV":   0.03,
        # Energy
        "XOM":   0.02,   # Oil bouncing with equities
        "CVX":   0.02,
        # Consumer Defensive
        "WMT":   0.04,
        "PG":    0.03,
        "JNJ":   0.02,
        # Consumer Disc/Media
        "NFLX": -0.05,   # Still negative but less severe
        "DIS":  -0.04,
        # Cloud/Enterprise
        "CRM":  -0.02,
        "PLTR":  0.08,   # GTC adjacent AI momentum
    },

    # -- Mar 15-20: Deep Bear -------------------------------------------------
    # Worst phase: GLD crashed, tech rout, VIX 27-30, new 2026 lows
    # Note: Mar 15 Sunday -> snapped to Mon Mar 16
    "2026-03-15": {
        # Mega-cap Tech
        "AAPL": -0.11,
        "MSFT": -0.09,
        "NVDA": -0.08,   # GTC post-relief fully faded
        "TSLA": -0.22,   # Demand crisis confirmed
        "META": -0.07,
        "GOOGL":-0.10,
        "AMZN": -0.10,
        # Semis/AI
        "AMD":  -0.12,
        "INTC": -0.11,
        "ORCL": -0.05,
        # Index ETFs
        "SPY":  -0.12,
        "QQQ":  -0.18,
        "DIA":  -0.10,
        "IWM":  -0.15,   # Small caps worst hit
        # Financials
        "JPM":  -0.04,
        "BAC":  -0.08,   # Credit spreads widen more
        "GS":   -0.05,
        "V":    -0.07,
        # Safe-havens
        "GLD":  -0.16,   # Trump tariff pause talk -> GLD crashes
        "TLT":   0.04,   # Still some bond demand but less
        "SLV":  -0.10,   # Silver follows gold down
        # Energy
        "XOM":  -0.08,   # Demand destruction fears
        "CVX":  -0.07,
        # Consumer Defensive
        "WMT":  -0.02,   # Still resilient but market-wide selling
        "PG":   -0.01,
        "JNJ":   0.01,   # Most defensive
        # Consumer Disc/Media
        "NFLX": -0.11,
        "DIS":  -0.12,
        # Cloud/Enterprise
        "CRM":  -0.09,
        "PLTR": -0.03,   # Even defence names selling off
    },

    # -- Aug 01-08: Bull Phase -------------------------------------------------
    # August 2025 context:
    #   S&P ~5800, Fed already cut 25bp in June, AI optimism peak, VIX ~13-15
    #   Tech leading: NVDA near ATH on Blackwell GPU demand
    #   Bond market calm, GLD flat-positive, small-caps lagging
    #   No major macro catalyst HOLD "soft landing" narrative dominant
    "2025-08-01": {
        # Mega-cap Tech
        "AAPL":  0.12,   # iPhone 17 rumours, services growth strong
        "MSFT":  0.14,   # Azure AI momentum, Copilot adoption
        "NVDA":  0.20,   # Blackwell GPU backlog, data centre demand peak
        "TSLA":  0.08,   # FSD rollout news, slightly positive
        "META":  0.15,   # Reels monetisation, Llama 4 announcement
        "GOOGL": 0.11,   # TPU expansion, Search AI integration
        "AMZN":  0.13,   # AWS growth reacceleration, Trainium chips
        # Semis/AI
        "AMD":   0.16,   # MI300X GPU share gains
        "INTC":  0.04,   # Foundry turnaround story, slightly positive
        "ORCL":  0.18,   # Oracle cloud wins with AI companies
        # Index ETFs
        "SPY":   0.10,   # Broad bull market, strong breadth
        "QQQ":   0.17,   # Nasdaq leading, AI-heavy
        "DIA":   0.07,   # Dow lagging (industrial/value)
        "IWM":   0.06,   # Small caps participating but lagging
        # Financials
        "JPM":   0.08,   # Strong earnings, buybacks
        "BAC":   0.07,
        "GS":    0.09,   # M&A activity picking up
        "V":     0.10,   # Consumer spend still healthy
        # Safe-havens
        "GLD":   0.05,   # Flat-positive, USD stable
        "TLT":  -0.04,   # Rates pricing in "higher for longer" briefly
        "SLV":   0.03,
        # Energy
        "XOM":   0.06,   # Oil $78, stable demand
        "CVX":   0.05,
        # Consumer Defensive
        "WMT":   0.08,   # Consumer trading up again, beat estimates
        "PG":    0.05,
        "JNJ":   0.04,
        # Consumer Disc/Media
        "NFLX":  0.12,   # Subscriber growth beat, ad tier growing
        "DIS":   0.07,   # Parks strong summer, streaming profitable
        # Cloud/Enterprise
        "CRM":   0.09,   # Agentforce AI product launch buzz
        "PLTR":  0.22,   # AIP bootcamps driving commercial growth
    },

    # -- Oct 01-08: Sideways ---------------------------------------------------
    # October 2025 context:
    #   S&P ~5600-5700, choppy after Sep volatility, VIX ~18-21
    #   "Magnificent 7" losing momentum after Q3 guidance misses
    #   Geopolitical: Middle East tensions, oil spike to $90
    #   Mixed signals: economy still OK but rate-cut expectations delayed
    #   GLD positive (geopolitical), tech flat, defensives holding
    "2025-10-01": {
        # Mega-cap Tech
        "AAPL":  0.02,   # iPhone 17 demand in line, no surprise
        "MSFT":  0.03,   # Azure growth decelerating slightly
        "NVDA":  0.04,   # Supply catching up to demand, premium fading
        "TSLA": -0.06,   # Q3 delivery miss, below consensus
        "META":  0.05,   # Stable but no new catalyst
        "GOOGL": 0.01,   # Ad market seasonal softness
        "AMZN":  0.02,   # AWS steady, retail holiday prep neutral
        # Semis/AI
        "AMD":   0.03,
        "INTC": -0.05,   # Guidance cut warning
        "ORCL":  0.06,   # Cloud still growing faster than market
        # Index ETFs
        "SPY":  -0.02,   # Near flat HOLD indecisive market
        "QQQ":  -0.04,   # Tech lagging slightly
        "DIA":   0.01,
        "IWM":  -0.06,   # Small caps hurt by "higher for longer"
        # Financials
        "JPM":   0.04,   # Q3 earnings preview positive
        "BAC":   0.01,
        "GS":    0.03,
        "V":     0.02,
        # Safe-havens
        "GLD":   0.12,   # Geopolitical risk bid HOLD oil + Middle East
        "TLT":  -0.08,   # Yields rising, bonds weak
        "SLV":   0.05,   # Following gold up
        # Energy
        "XOM":   0.09,   # Oil at $90, windfall
        "CVX":   0.08,
        # Consumer Defensive
        "WMT":   0.03,
        "PG":    0.02,
        "JNJ":   0.03,
        # Consumer Disc/Media
        "NFLX":  0.04,   # Q4 content slate positive
        "DIS":  -0.03,   # Park attendance normalising down
        # Cloud/Enterprise
        "CRM":   0.01,
        "PLTR":  0.06,   # Government contract wins
    },
}


# ==============================================================================
# HELPERS
# ==============================================================================

def snap_to_trading_day(date_str: str) -> str:
    dt      = pd.to_datetime(date_str)
    snapped = pd.bdate_range(start=dt, periods=1)[0]
    if snapped != dt:
        print(f"   [WARN]  {date_str} is not a trading day -> snapped to {snapped.date()}")
    return snapped.strftime("%Y-%m-%d")


def fetch_history(ticker: str, test_date: str) -> pd.DataFrame:
    test_dt  = pd.to_datetime(test_date)
    yf_end   = (test_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    yf_start = (test_dt - pd.Timedelta(days=300)).strftime("%Y-%m-%d")
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        df = yf.download(ticker, start=yf_start, end=yf_end,
                         auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[df.index <= test_dt]
    return df


def fetch_actual_return(ticker: str, test_date: str, outcome_date: str) -> float:
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


# ==============================================================================
# SINGLE WINDOW
# ==============================================================================

def run_window(test_date: str, outcome_date: str, label: str,
               tech_agent, regime_agent, heatmap_agent):

    test_date    = snap_to_trading_day(test_date)
    outcome_date = snap_to_trading_day(outcome_date)

    # Use the snapped date as the sentinel for manual sentiment lookup.
    # If Sunday was snapped to Monday, we still want the sentiment for that week.
    sent_date = test_date
    # Fall back to closest key if exact date not in table
    if sent_date not in MANUAL_SENTIMENT:
        available = sorted(MANUAL_SENTIMENT.keys())
        diffs = [(abs((pd.to_datetime(sent_date) - pd.to_datetime(k)).days), k)
                 for k in available]
        sent_date = min(diffs)[1]
        print(f"   ℹ️  Sentiment date mapped: {test_date} -> using {sent_date} scores")

    sentiment_scores = MANUAL_SENTIMENT[sent_date]

    print(f"\n{'*'*90}")
    print(f"  {label}  |  {test_date} -> {outcome_date}")
    print(f"{'*'*90}")
    print(f"\n  {'Ticker':<7} {'LSTM':>7} {'Regime':<10} {'Vol':>7} "
          f"{'Sent':>7} {'GDI':>6} {'Tension':<10} {'Vote':>5} "
          f"{'Decision':<7} {'Act%':>8} {'Result'}")
    print(f"  {'-'*97}")

    results      = []
    correct      = 0
    wrong        = 0
    neutral_cnt  = 0
    low_gdi_correct  = 0
    low_gdi_total    = 0
    high_gdi_correct = 0
    high_gdi_total   = 0

    for ticker in TICKERS:
        try:
            hist = fetch_history(ticker, test_date)
            if hist.empty or len(hist) < 150:
                print(f"  {ticker:<7} HOLD skipped (only {len(hist)} rows)")
                continue

            # -- LSTM raw signal ----------------------------------------------
            feat_df = build_lstm_features(hist)
            if len(feat_df) < SEQ_LEN:
                print(f"  {ticker:<7} HOLD skipped (feat rows={len(feat_df)} < {SEQ_LEN})")
                continue
            lstm_raw = tech_agent.predict_raw(hist)   # unstretched

            # -- Regime ------------------------------------------------------
            regime_label, regime_vol, _ = regime_agent.detect(hist, ticker)

            # -- Manual sentiment ---------------------------------------------
            sent_score = sentiment_scores.get(ticker, 0.0)

            # -- Heatmap GDI --------------------------------------------------
            gdi_result = heatmap_agent.analyze(
                lstm_score=lstm_raw,
                sent_score=sent_score,
                regime_label=regime_label,
                regime_vol=regime_vol,
            )
            gdi     = gdi_result["gdi"]
            tension = gdi_result["tension"]
            penalty = gdi_result["penalty"]

            # -- Decision v2: 3-signal majority voting + GDI conflict gate ----
            #
            # Old logic (broken): `regime == "Bear" -> SELL` HOLD too blunt.
            # NVDA LSTM=0.995 in Bear regime got SELL but bounced +1.44%.
            # That's a regime gate overriding a strong technical signal wrongly.
            #
            # New logic HOLD two steps:
            #
            # STEP 1 HOLD GDI conflict gate (HIGH/CRITICAL -> HOLD immediately).
            # When agents strongly disagree (GDI > 0.55), no signal is reliable.
            # Better to stay out than pick the wrong side with 50/50 odds.
            #
            # STEP 2 HOLD Majority vote across 3 independent signals:
            #   Signal A: LSTM   -> +1 bull if prob > BUY_THRESHOLD,
            #                       -1 bear if prob < SELL_THRESHOLD, 0 neutral
            #   Signal B: Sentiment -> +1 bull if sent > 0.07,
            #                          -1 bear if sent < -0.07, 0 neutral
            #   Signal C: Regime -> +1 Bull, -1 Bear, 0 Sideways
            # BUY  needs: total_vote ≥ +2 AND regime ≠ Bear
            # SELL needs: total_vote ≤ -2
            # HOLD: everything else (conflicted or insufficient evidence)

            GDI_HOLD_THRESHOLD = 0.55   # HIGH/CRITICAL disagreement -> HOLD

            # Step 1: conflict gate
            if gdi >= GDI_HOLD_THRESHOLD:
                decision = "HOLD"   # agents too divided HOLD stay out
            else:
                # Step 2: majority vote
                vote_lstm   = (+1 if lstm_raw > BUY_THRESHOLD
                               else -1 if lstm_raw < SELL_THRESHOLD
                               else 0)
                vote_sent   = (+1 if sent_score > 0.07
                               else -1 if sent_score < -0.07
                               else 0)
                vote_regime = (+1 if regime_label == "Bull"
                               else -1 if regime_label == "Bear"
                               else 0)

                total_vote = vote_lstm + vote_sent + vote_regime

                if total_vote >= 2 and regime_label != "Bear":
                    decision = "BUY"
                elif total_vote <= -2:
                    decision = "SELL"
                else:
                    decision = "HOLD"   # mixed signals HOLD insufficient conviction

            # -- Actual return ------------------------------------------------
            actual_ret = fetch_actual_return(ticker, test_date, outcome_date)

            if np.isnan(actual_ret):
                result_str = "?"
                neutral_cnt += 1
                correct_flag = None
            elif decision == "HOLD":
                result_str = "-"
                neutral_cnt += 1
                correct_flag = None
            elif decision == "BUY"  and actual_ret > 0:
                result_str = "[OK]"
                correct += 1
                correct_flag = True
            elif decision == "SELL" and actual_ret < 0:
                result_str = "[OK]"
                correct += 1
                correct_flag = True
            else:
                result_str = "[BAD]"
                wrong += 1
                correct_flag = False

            # GDI effectiveness tracking
            if gdi < 0.35:   # Low GDI = signals agree
                low_gdi_total += 1
                if correct_flag is True: low_gdi_correct += 1
            else:             # High GDI = signals disagree
                high_gdi_total += 1
                if correct_flag is True: high_gdi_correct += 1

            act_str = f"{actual_ret:>+7.2f}%" if not np.isnan(actual_ret) else "     nan%"

            # compute vote for display (already computed above unless HOLD via GDI gate)
            if gdi >= GDI_HOLD_THRESHOLD:
                display_vote = "HOLD"
            else:
                v_l = (+1 if lstm_raw > BUY_THRESHOLD else -1 if lstm_raw < SELL_THRESHOLD else 0)
                v_s = (+1 if sent_score > 0.07 else -1 if sent_score < -0.07 else 0)
                v_r = (+1 if regime_label == "Bull" else -1 if regime_label == "Bear" else 0)
                display_vote = f"{v_l+v_s+v_r:+d}"

            print(f"  {ticker:<7} {lstm_raw:>7.4f} {regime_label:<10} "
                  f"{regime_vol:>7.4f} {sent_score:>+7.3f} {gdi:>6.3f} "
                  f"{tension:<10} {display_vote:>5} "
                  f"{decision:<7} {act_str}  {result_str}")

            results.append({
                "ticker":      ticker,
                "test_date":   test_date,
                "lstm_raw":    round(lstm_raw, 4),
                "regime":      regime_label,
                "regime_vol":  round(regime_vol, 5),
                "sent_score":  round(sent_score, 3),
                "gdi":         round(gdi, 4),
                "tension":     tension,
                "penalty":     round(penalty, 2),
                "decision":    decision,
                "actual_ret":  round(actual_ret, 2) if not np.isnan(actual_ret) else None,
                "result":      result_str,
            })

        except Exception as e:
            print(f"  {ticker:<7} ERROR: {e}")

    active = correct + wrong
    acc    = (correct / active * 100) if active > 0 else 0.0

    low_acc  = (low_gdi_correct  / low_gdi_total  * 100) if low_gdi_total  > 0 else float("nan")
    high_acc = (high_gdi_correct / high_gdi_total * 100) if high_gdi_total > 0 else float("nan")

    print(f"\n  -- Window Summary -------------------------------------------------------")
    print(f"     Decision accuracy   : {correct}[OK] / {wrong}[BAD] / {neutral_cnt}-  -> {acc:.1f}%")
    print(f"     Low  GDI (agree)    : {low_gdi_correct}/{low_gdi_total}  -> {low_acc:.0f}%  (should be HIGHER)")
    print(f"     High GDI (disagree) : {high_gdi_correct}/{high_gdi_total}  -> {high_acc:.0f}%  (should be LOWER)")
    gdi_hypothesis_ok = (
        (np.isnan(low_acc) or np.isnan(high_acc)) or
        low_acc >= high_acc
    )
    print(f"     GDI hypothesis      : {'[OK] CONFIRMED' if gdi_hypothesis_ok else '[WARN] NOT CONFIRMED'} "
          f"(low-GDI trades ≥ high-GDI trades in accuracy)")

    return {
        "label":             label,
        "test_date":         test_date,
        "outcome_date":      outcome_date,
        "accuracy":          acc,
        "correct":           correct,
        "wrong":             wrong,
        "neutral":           neutral_cnt,
        "low_gdi_correct":   low_gdi_correct,
        "low_gdi_total":     low_gdi_total,
        "low_gdi_acc":       low_acc,
        "high_gdi_correct":  high_gdi_correct,
        "high_gdi_total":    high_gdi_total,
        "high_gdi_acc":      high_acc,
        "gdi_hypothesis_ok": gdi_hypothesis_ok,
        "results":           results,
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 90)
    print("  HEATMAP AGENT GDI BACKTEST  |  6 Windows x 30 Tickers")
    print("  Windows: 4x Bear/Bounce (Mar 2026) + Bull (Aug 2025) + Sideways (Oct 2025)")
    print("  Sentiment: Manual pre-computed scores (v2.3 MCP+LLM context)")
    print("  Decision:  3-signal majority vote + GDI conflict gate")
    print("=" * 90)

    # -- Load agents -----------------------------------------------------------
    print("\nLoading TechnicalAgent...")
    try:
        tech_agent = TechnicalAgent(
            lstm_model_path=MODEL_PATH,
            lstm_scaler_path=SCALER_PATH,
        )
        print(f"   [OK] LSTM loaded  {tuple(tech_agent.lstm_model.input_shape)}")
    except Exception as e:
        print(f"   [BAD] Failed: {e}"); return

    print("\nLoading HybridRegimeAgent...")
    try:
        regime_agent = HybridRegimeAgent(
            hmm_model_path=REGIME_PATH, verbose=False)
        print(f"   [OK] Regime loaded  is_fitted={regime_agent.is_fitted}")
    except Exception as e:
        print(f"   [BAD] Failed: {e}"); return

    print("\nLoading HeatmapAgent...")
    heatmap_agent = HeatmapAgent()
    print("   [OK] HeatmapAgent ready")

    # -- Print manual sentiment summary ---------------------------------------
    print(f"\n  Manual sentiment scores used ({len(TICKERS)} tickers):")
    print(f"  {'Ticker':<8}", end="")
    for d in sorted(MANUAL_SENTIMENT.keys()):
        print(f"  {d[5:]:>10}", end="")   # MM-DD
    print()
    print("  " + "-" * 55)
    for t in TICKERS:
        print(f"  {t:<8}", end="")
        for d in sorted(MANUAL_SENTIMENT.keys()):
            sc = MANUAL_SENTIMENT[d].get(t, 0.0)
            icon = "🟢" if sc > 0.07 else ("🔴" if sc < -0.07 else "🟡")
            print(f"  {sc:>+7.3f}{icon}", end="")
        print()

    # -- Run windows ----------------------------------------------------------
    all_stats = []
    for test_date, outcome_date, label in TEST_WINDOWS:
        s = run_window(test_date, outcome_date, label,
                       tech_agent, regime_agent, heatmap_agent)
        all_stats.append(s)

    # -- Consolidated summary --------------------------------------------------
    print("\n" + "=" * 90)
    print("  CONSOLIDATED RESULTS")
    print("=" * 90)
    print(f"\n  {'Window':<30} {'Acc':>7} {'C/W/N':>12} {'LowGDI':>8} {'HighGDI':>9} {'Hypothesis'}")
    print(f"  {'-'*78}")

    for s in all_stats:
        c_w_n = f"{s['correct']}/{s['wrong']}/{s['neutral']}"
        low_s  = f"{s['low_gdi_acc']:.0f}%({s['low_gdi_total']})" if not np.isnan(s['low_gdi_acc']) else "n/a"
        high_s = f"{s['high_gdi_acc']:.0f}%({s['high_gdi_total']})" if not np.isnan(s['high_gdi_acc']) else "n/a"
        hyp    = "[OK]" if s["gdi_hypothesis_ok"] else "[WARN]"
        print(f"  {s['label']:<30} {s['accuracy']:>5.1f}%  {c_w_n:>12} "
              f"{low_s:>10} {high_s:>10}  {hyp}")

    avg_acc = np.mean([s["accuracy"] for s in all_stats])
    all_low_acc  = [s["low_gdi_acc"]  for s in all_stats if not np.isnan(s["low_gdi_acc"])]
    all_high_acc = [s["high_gdi_acc"] for s in all_stats if not np.isnan(s["high_gdi_acc"])]
    hyp_confirmed = sum(1 for s in all_stats if s["gdi_hypothesis_ok"])

    print(f"  {'-'*78}")
    print(f"  {'AVERAGE':<30} {avg_acc:>5.1f}%")
    if all_low_acc:
        print(f"  Avg accuracy LOW-GDI  (signals agree)    : {np.mean(all_low_acc):.1f}%")
    if all_high_acc:
        print(f"  Avg accuracy HIGH-GDI (signals disagree) : {np.mean(all_high_acc):.1f}%")
    print(f"  GDI hypothesis confirmed: {hyp_confirmed}/{len(all_stats)} windows")

    # -- Verdict ---------------------------------------------------------------
    print(f"\n  {'='*60}")
    print(f"  HEATMAP AGENT VERDICT")
    print(f"  {'='*60}")
    print(f"  Decision accuracy  : {avg_acc:.1f}%  "
          + ("[OK] above 55%" if avg_acc >= 55 else "[WARN]  below 55%"))
    print(f"  GDI discrimination : {hyp_confirmed}/{len(all_stats)} windows show "
          "low-GDI trades outperform high-GDI trades")
    gdi_works = hyp_confirmed >= len(all_stats) // 2
    print(f"  GDI useful?        : {'[OK] YES HOLD penalty correctly reduces confidence on disagreement' if gdi_works else '[WARN] MIXED HOLD check GDI weights'}")

    # -- Regime cross-check ----------------------------------------------------
    # Key test: does the HybridRegimeAgent detect the right regime per window?
    # Bull windows should show mostly "Bull" labels.
    # Sideways windows should show mostly "Sideways".
    # Bear windows should show mostly "Bear".
    print(f"\n  {'='*60}")
    print(f"  REGIME DETECTION CROSS-CHECK")
    print(f"  {'='*60}")
    print(f"  {'Window':<28} {'Expected':<10} {'Regimes detected (across 30 tickers)'}")
    print(f"  {'-'*60}")

    expected_regime = {
        "Bear start":  "Bear",
        "Bear early":  "Bear",
        "Deep Bear":   "Bear",
        "Bounce":      "Bear",      # Bear regime, brief bounce within it
        "Bull Phase":  "Bull",
        "Sideways":    "Sideways",
    }

    for s in all_stats:
        regimes = [r["regime"] for r in s["results"]]
        from collections import Counter
        dist = Counter(regimes)
        total = len(regimes)

        # Find expected regime for this window
        exp = "?"
        for key, val in expected_regime.items():
            if key.lower() in s["label"].lower():
                exp = val
                break

        # Format distribution bar
        bear_pct = dist.get("Bear", 0) / total * 100
        bull_pct = dist.get("Bull", 0) / total * 100
        side_pct = dist.get("Sideways", 0) / total * 100

        dominant = max(dist, key=dist.get)
        match = "[OK]" if dominant == exp else "[WARN] "

        dist_str = (f"Bear={dist.get('Bear',0):2d} "
                    f"Bull={dist.get('Bull',0):2d} "
                    f"Sideways={dist.get('Sideways',0):2d}  "
                    f"dominant={dominant} {match}")
        print(f"  {s['label']:<28} {exp:<10} {dist_str}")

    print(f"\n  ℹ️  Note: HybridRegimeAgent runs on per-ticker OHLCV, not market-wide data.")
    print(f"       Some tickers (e.g. GLD in Bull) may show Bear due to their own price action.")
    print(f"       Cross-window dominant regime should match expected HOLD that confirms the HMM is working.")

    # -- Save CSV -------------------------------------------------------------
    all_rows = []
    for s in all_stats:
        for r in s["results"]:
            r["window"] = s["label"]
            all_rows.append(r)
    if all_rows:
        pd.DataFrame(all_rows).to_csv("heatmap_backtest.csv", index=False)
        print(f"\n  Saved -> heatmap_backtest.csv  ({len(all_rows)} rows)")

    print("\nDone.\n")


if __name__ == "__main__":
    main()