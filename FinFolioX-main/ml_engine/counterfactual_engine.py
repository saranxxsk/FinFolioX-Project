"""
ml_engine/counterfactual_engine.py  HOLD  Counterfactual Decision Engine v2.0
===========================================================================
PHASE 15: COUNTERFACTUAL DECISION ENGINE (The "What-If" Simulator)

CHANGELOG v2.0:
  - Added RegretTracker: rolling window of regret scores per agent type
  - Added noise-band awareness: tiny moves flagged as AMBIGUOUS
  - Added opportunity_cost() for portfolio-level dollar regret
  - Added confidence calibration check (was direction right?)
  - Improved _classify_regret with finer granularity (6 levels)
  - Added get_regret_summary() for batch/window-level reporting
  - Improved TLT hold_pnl: annualised 5-day risk-free if TLT absent
  - print_regret_audit() extended with table + calibration note
  - Added regret_weighted_signal() for ASC / MetaAgent integration
  - SELL P&L correctly modelled as short: profit = -price_change - commission

Three Components:
  A. Regret Matrix       - Simulates BUY/SELL/HOLD universes, finds optimal.
  B. Ledger/Batch        - Aggregates regret across windows.
  C. LLM Retrospective   - Generates "Trader's Diary" entries via Groq.
"""

import os
import sys
import numpy as np
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==============================================================================
# CONFIGURATION
# ==============================================================================
HIGH_REGRET_THRESHOLD  = 0.05
TRADE_COMMISSION       = 0.001   # 0.1% round-trip
ANNUAL_RISK_FREE_RATE  = 0.05    # 5% annualised
TRADING_DAYS_PER_YEAR  = 252


# ==============================================================================
# ROLLING REGRET TRACKER
# ==============================================================================
class RegretTracker:
    """
    Rolling window of regret scores.
    Lets MetaAgent / ASC detect if the system is systematically
    missing moves (high mean) or making random errors (high std).
    """

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self._scores: deque = deque(maxlen=window_size)
        self._levels: deque = deque(maxlen=window_size)

    def record(self, regret_score: float, regret_level: str):
        self._scores.append(float(regret_score))
        self._levels.append(regret_level)

    @property
    def mean_regret(self) -> float:
        return float(np.mean(self._scores)) if self._scores else 0.0

    @property
    def std_regret(self) -> float:
        return float(np.std(self._scores)) if len(self._scores) > 1 else 0.0

    @property
    def extreme_rate(self) -> float:
        if not self._levels:
            return 0.0
        bad = sum(1 for l in self._levels if l in ("HIGH", "EXTREME"))
        return bad / len(self._levels)

    def get_summary(self) -> dict:
        return {
            "n":            len(self._scores),
            "mean_regret":  round(self.mean_regret, 6),
            "std_regret":   round(self.std_regret, 6),
            "extreme_rate": round(self.extreme_rate, 4),
            "status":       ("HEALTHY"  if self.mean_regret < 0.03 else
                             "ELEVATED" if self.mean_regret < 0.07 else
                             "CRITICAL"),
        }


# ==============================================================================
# COUNTERFACTUAL ENGINE v2.0
# ==============================================================================
class CounterfactualEngine:
    """
    The Multiverse Simulator.

    analyze() is the core method HOLD it simulates three parallel universes
    (BUY / SELL / HOLD) and computes regret against the optimal outcome.

    Parameters for analyze():
        actual_decision  : "BUY" | "SELL" | "HOLD"
        decision_price   : entry price at decision time
        actual_price_t5  : exit price T+5 trading days later
        confidence       : fusion confidence ∈ [0, 1]
        ticker           : ticker string (for noise-band sizing)
        tlt_price_start  : TLT price at decision (optional, for hold_pnl)
        tlt_price_end    : TLT price at T+5 (optional)

    Returns dict with:
        hypothetical_buy_pnl / sell_pnl / hold_pnl
        optimal_decision, optimal_pnl
        actual_decision, actual_pnl
        regret_score, regret_level
        move_pct, is_ambiguous, confidence_calibrated
    """

    _INDEX_ETFS    = {"SPY", "QQQ", "DIA", "IWM", "TLT"}
    _VOLATILE_STKS = {"NVDA", "TSLA", "AMD", "PLTR", "NFLX", "SLV"}

    def __init__(self):
        self.tracker = RegretTracker(window_size=30)
        self.llm     = None
        self._init_llm()
        print("   [OK] Phase 15: Counterfactual Engine v2.0 Initialized.")

    def _init_llm(self):
        try:
            from finfolio_x.settings import GROQ_API_KEY, LLM_MODEL_NAME, LLM_TEMPERATURE
            if GROQ_API_KEY:
                from langchain_groq import ChatGroq
                self.llm = ChatGroq(
                    groq_api_key=GROQ_API_KEY,
                    model_name=LLM_MODEL_NAME,
                    temperature=LLM_TEMPERATURE,
                )
                print("      - LLM Retrospective: ONLINE (Groq)")
            else:
                print("      - LLM Retrospective: OFFLINE (no API key)")
        except Exception as e:
            print(f"      - LLM Retrospective: OFFLINE ({e})")

    # --------------------------------------------------------------------------
    # A. REGRET MATRIX HOLD core method
    # --------------------------------------------------------------------------
    def analyze(self,
                actual_decision:  str,
                decision_price:   float,
                actual_price_t5:  float,
                confidence:       float = 0.5,
                ticker:           str   = "",
                tlt_price_start:  float = None,
                tlt_price_end:    float = None) -> dict:
        """
        Simulate BUY / SELL / HOLD universes and compute regret.
        Does NOT mutate state HOLD call record_to_tracker() to update rolling stats.
        """
        if decision_price <= 0:
            raise ValueError(f"decision_price must be > 0, got {decision_price}")

        move_pct     = (actual_price_t5 - decision_price) / decision_price
        hold_pnl_val = self._hold_pnl(tlt_price_start, tlt_price_end)
        nb           = self._noise_band(ticker)
        is_ambiguous = abs(move_pct) < nb

        # Universe P&Ls
        # BUY:  long -> profit from price rise, lose on fall
        # SELL: short -> profit from price fall, lose on rise
        # HOLD: risk-free proxy (cash / TLT-scaled)
        buy_pnl  = float(move_pct  - TRADE_COMMISSION)
        sell_pnl = float(-move_pct - TRADE_COMMISSION)

        universes = {
            "BUY":  round(buy_pnl,      6),
            "SELL": round(sell_pnl,     6),
            "HOLD": round(hold_pnl_val, 6),
        }

        optimal_decision = max(universes, key=universes.get)
        optimal_pnl      = universes[optimal_decision]
        actual_clean     = self._clean_decision(actual_decision)
        actual_pnl       = universes.get(actual_clean, hold_pnl_val)
        regret_score     = float(max(0.0, optimal_pnl - actual_pnl))
        regret_level     = self._classify_regret(regret_score)

        # Confidence calibration HOLD did direction match outcome?
        outcome_up  = move_pct > 0
        if confidence >= 0.52:
            conf_calibrated = outcome_up        # bullish conf -> expect up
        elif confidence < 0.40:
            conf_calibrated = not outcome_up    # bearish conf -> expect down
        else:
            conf_calibrated = None              # neutral zone HOLD unscoreable

        return {
            "hypothetical_buy_pnl":  universes["BUY"],
            "hypothetical_sell_pnl": universes["SELL"],
            "hypothetical_hold_pnl": universes["HOLD"],
            "optimal_decision":      optimal_decision,
            "optimal_pnl":           round(optimal_pnl,  6),
            "actual_decision":       actual_clean,
            "actual_pnl":            round(actual_pnl,   6),
            "regret_score":          round(regret_score, 6),
            "regret_level":          regret_level,
            "move_pct":              round(float(move_pct), 6),
            "is_ambiguous":          is_ambiguous,
            "confidence_calibrated": conf_calibrated,
        }

    def record_to_tracker(self, cf_result: dict):
        """Push result into rolling RegretTracker."""
        self.tracker.record(cf_result["regret_score"], cf_result["regret_level"])

    # --------------------------------------------------------------------------
    # B. BATCH / LEDGER HELPERS
    # --------------------------------------------------------------------------
    def get_regret_summary(self, results: list) -> dict:
        """
        Aggregate counterfactual stats across a list of analyze() results.
        Used by test_counterfactual.py window + consolidated summaries.
        """
        if not results:
            return {}

        regrets  = [r["regret_score"] for r in results]
        levels   = [r["regret_level"] for r in results]
        ambig    = [r for r in results if r["is_ambiguous"]]

        optimal_match = sum(
            1 for r in results
            if r["actual_decision"] == r["optimal_decision"]
        )

        calib_vals = [r["confidence_calibrated"] for r in results
                      if r["confidence_calibrated"] is not None]
        calib_rate = (sum(calib_vals) / len(calib_vals)) if calib_vals else None

        level_counts = {
            l: levels.count(l)
            for l in ("NONE", "LOW", "MODERATE", "HIGH", "EXTREME")
        }

        return {
            "n":                  len(results),
            "n_ambiguous":        len(ambig),
            "mean_regret_pct":    round(float(np.mean(regrets)) * 100, 3),
            "max_regret_pct":     round(float(np.max(regrets))  * 100, 3),
            "optimal_match_rate": round(optimal_match / len(results) * 100, 1),
            "calib_rate_pct":     round(calib_rate * 100, 1) if calib_rate is not None else None,
            "level_counts":       level_counts,
            "total_regret_pct":   round(float(sum(regrets)) * 100, 3),
        }

    def opportunity_cost(self, results: list, capital: float = 10_000.0) -> float:
        """Dollar cost of not taking optimal actions across all decisions."""
        return round(sum(r["regret_score"] for r in results) * capital, 2)

    def regret_weighted_signal(self, cf_result: dict,
                               fusion_confidence: float) -> float:
        """
        Adjusts fusion confidence downward based on regret level.
        For MetaAgent / ASC trust recalibration after a bad call.
        """
        penalty_map = {
            "NONE":     0.00,
            "LOW":      0.05,
            "MODERATE": 0.12,
            "HIGH":     0.22,
            "EXTREME":  0.35,
        }
        penalty = penalty_map.get(cf_result.get("regret_level", "NONE"), 0.0)
        return float(np.clip(fusion_confidence - penalty, 0.0, 1.0))

    # --------------------------------------------------------------------------
    # C. LLM RETROSPECTIVE
    # --------------------------------------------------------------------------
    def generate_retrospective(self,
                               ticker:        str,
                               decision_date: str,
                               cf_result:     dict,
                               regime_label:  str   = "Unknown",
                               confidence:    float = 0.5) -> str:
        context = (
            f"Ticker: {ticker}  |  Date: {decision_date}  |  Regime: {regime_label}\n"
            f"Confidence: {confidence:.4f}  |  Decision: {cf_result['actual_decision']}\n"
            f"Actual P&L: {cf_result['actual_pnl']*100:+.2f}%  |  "
            f"Move: {cf_result['move_pct']*100:+.2f}%\n"
            f"BUY={cf_result['hypothetical_buy_pnl']*100:+.2f}%  "
            f"SELL={cf_result['hypothetical_sell_pnl']*100:+.2f}%  "
            f"HOLD={cf_result['hypothetical_hold_pnl']*100:+.2f}%\n"
            f"Optimal: {cf_result['optimal_decision']} "
            f"({cf_result['optimal_pnl']*100:+.2f}%)  |  "
            f"Regret: {cf_result['regret_score']*100:.2f}% ({cf_result['regret_level']})"
        )

        if not self.llm:
            return self._fallback_retrospective(cf_result, ticker, decision_date)

        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            sys_msg = SystemMessage(content=(
                "You are Chief Performance Auditor for FinFolio-X. "
                "Write a 3-sentence Trader's Diary entry. Cover: "
                "(1) what the AI did and if it was right or wrong, "
                "(2) what the optimal action was and P&L missed, "
                "(3) what the system should learn. "
                "Be analytical, third person, cite specific numbers."
            ))
            response = self.llm.invoke([sys_msg, HumanMessage(content=context)])
            return response.content.strip()
        except Exception as e:
            print(f"      [!] LLM Retrospective failed: {e}")
            return self._fallback_retrospective(cf_result, ticker, decision_date)

    def _fallback_retrospective(self, cf_result: dict,
                                ticker: str, decision_date: str) -> str:
        a  = cf_result["actual_decision"]
        o  = cf_result["optimal_decision"]
        rg = cf_result["regret_score"] * 100
        ap = cf_result["actual_pnl"]   * 100
        op = cf_result["optimal_pnl"]  * 100
        if a == o:
            return (f"[{decision_date}] {ticker}: AI correctly chose {a} "
                    f"(P&L: {ap:+.2f}%). Optimal HOLD no regret.")
        return (f"[{decision_date}] {ticker}: AI chose {a} ({ap:+.2f}%) "
                f"but optimal was {o} ({op:+.2f}%). "
                f"Regret: {rg:.2f}%. Recalibrate agent weights for similar setups.")

    # --------------------------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------------------------
    @staticmethod
    def print_regret_audit(cf_result: dict, retrospective: str = ""):
        ambig_note = "  [WARN]  AMBIGUOUS (inside noise band)" if cf_result["is_ambiguous"] else ""
        print(f"\n      -- Counterfactual Regret Audit --------------------------")
        print(f"      Move: {cf_result['move_pct']*100:+.2f}%{ambig_note}")
        print("      ┌----------┬------------┬---------------------┐")
        print("      │  Action  │    P&L     │  Note               │")
        print("      ├----------┼------------┼---------------------┤")

        for action in ["BUY", "SELL", "HOLD"]:
            pnl  = cf_result[f"hypothetical_{action.lower()}_pnl"]
            note = ""
            if (action == cf_result["actual_decision"]
                    and action == cf_result["optimal_decision"]):
                note = "← ACTUAL + OPTIMAL [OK]"
            elif action == cf_result["actual_decision"]:
                note = "← ACTUAL"
            elif action == cf_result["optimal_decision"]:
                note = "← OPTIMAL"
            print(f"      │  {action:6s}  │ {pnl*100:+8.2f}%  │ {note:<19} │")

        print("      └----------┴------------┴---------------------┘")

        icons = {"NONE":"[OK]","LOW":"🔵","MODERATE":"🟡","HIGH":"🟠","EXTREME":"🔴"}
        icon  = icons.get(cf_result["regret_level"], "❓")
        print(f"      Regret: {cf_result['regret_score']*100:.2f}%  "
              f"{icon} {cf_result['regret_level']}")

        cal = cf_result.get("confidence_calibrated")
        if cal is not None:
            tag = "[OK] calibrated" if cal else "[WARN]  miscalibrated"
            print(f"      Confidence direction: {tag}")

        if retrospective:
            print(f"\n      Diary: {retrospective}")
        print("      " + "-" * 58)

    # --------------------------------------------------------------------------
    # PRIVATE HELPERS
    # --------------------------------------------------------------------------
    @classmethod
    def _noise_band(cls, ticker: str) -> float:
        t = ticker.upper()
        if t in cls._INDEX_ETFS:    return 0.010
        if t in cls._VOLATILE_STKS: return 0.030
        return 0.020

    @staticmethod
    def _hold_pnl(tlt_start: float, tlt_end: float) -> float:
        if tlt_start and tlt_end and tlt_start > 0:
            return float((tlt_end - tlt_start) / tlt_start * 0.20)
        days_fraction = 5 / TRADING_DAYS_PER_YEAR
        return float((1 + ANNUAL_RISK_FREE_RATE) ** days_fraction - 1)

    @staticmethod
    def _clean_decision(d: str) -> str:
        s = str(d).upper().strip()
        if "BUY"  in s: return "BUY"
        if "SELL" in s: return "SELL"
        return "HOLD"

    @staticmethod
    def _classify_regret(rs: float) -> str:
        if rs <= 0.002: return "NONE"
        if rs <= 0.010: return "LOW"
        if rs <= 0.030: return "MODERATE"
        if rs <= 0.070: return "HIGH"
        return "EXTREME"

    # --------------------------------------------------------------------------
    # MISS-TYPE CLASSIFIER  (v2.1 HOLD addresses Issue 1 + Issue 2)
    # --------------------------------------------------------------------------
    @staticmethod
    def classify_miss_type(cf_result: dict) -> str:
        """
        Classifies WHY the AI missed the optimal decision.

        Returns one of:
          "CORRECT"      HOLD actual == optimal, no miss
          "HOLD_BIAS"    HOLD actual=HOLD but optimal=BUY/SELL (Issue 2)
                           -> System too conservative; should have traded
          "WRONG_DIR"    HOLD actual was BUY but market fell, or SELL but rose
                           -> LSTM predicted the wrong direction (Issue 1)
          "SUBOPTIMAL"   HOLD traded but chose wrong instrument
                           (e.g. BUY when SELL was better)
        """
        actual  = cf_result["actual_decision"]
        optimal = cf_result["optimal_decision"]

        if actual == optimal:
            return "CORRECT"
        if actual == "HOLD" and optimal in ("BUY", "SELL"):
            return "HOLD_BIAS"
        if actual in ("BUY", "SELL") and optimal in ("BUY", "SELL") and actual != optimal:
            return "WRONG_DIR"
        return "SUBOPTIMAL"

    def get_miss_type_breakdown(self, results: list) -> dict:
        """
        Aggregate miss-type counts across a list of analyze() results.
        Returns counts + pct for each type.
        Useful for diagnosing Issue 1 (WRONG_DIR) vs Issue 2 (HOLD_BIAS).
        """
        if not results:
            return {}
        counts = {"CORRECT": 0, "HOLD_BIAS": 0, "WRONG_DIR": 0, "SUBOPTIMAL": 0}
        for r in results:
            mt = self.classify_miss_type(r)
            counts[mt] = counts.get(mt, 0) + 1

        n = len(results)
        hold_bias_regret = sum(
            r["regret_score"] for r in results
            if self.classify_miss_type(r) == "HOLD_BIAS"
        )
        wrong_dir_regret = sum(
            r["regret_score"] for r in results
            if self.classify_miss_type(r) == "WRONG_DIR"
        )
        return {
            "n":                   n,
            "correct":             counts["CORRECT"],
            "hold_bias":           counts["HOLD_BIAS"],
            "wrong_dir":           counts["WRONG_DIR"],
            "suboptimal":          counts["SUBOPTIMAL"],
            "hold_bias_pct":       round(counts["HOLD_BIAS"] / n * 100, 1),
            "wrong_dir_pct":       round(counts["WRONG_DIR"] / n * 100, 1),
            "hold_bias_regret_pct":round(hold_bias_regret * 100, 3),
            "wrong_dir_regret_pct":round(wrong_dir_regret * 100, 3),
            "dominant_issue":      (
                "HOLD_BIAS"  if counts["HOLD_BIAS"] > counts["WRONG_DIR"]
                else "WRONG_DIR" if counts["WRONG_DIR"] > counts["HOLD_BIAS"]
                else "BALANCED"
            ),
        }

    # --------------------------------------------------------------------------
    # LEGACY COMPAT
    # --------------------------------------------------------------------------
    def get_regret_penalty(self, regret_score: float) -> float:
        """Legacy MetaAgent trust penalty multiplier."""
        if regret_score <= 0.01:  return 0.0
        if regret_score <= 0.05:  return -0.15
        if regret_score <= 0.15:  return -0.30
        return -0.50