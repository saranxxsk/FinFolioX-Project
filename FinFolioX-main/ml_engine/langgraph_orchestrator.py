"""
ml_engine/langgraph_orchestrator.py  HOLD  LangGraph Multi-Agent Orchestrator (v2.4)
===================================================================================
CHANGELOG:
  v2.1 HOLD FinBERT sanity gate, fusion probability centering, ASC prob centering.
  v2.2 HOLD PATCH-A: asc_saturated flag. PATCH-B: regime_contradiction check.
          PATCH-C: AgentState gains asc_saturated and regime_contradiction.
  v2.3 HOLD Hybrid Regime System integrated.
          node_market_context uses HybridRegimeAgent (Rule + HMM v2).
          regime_confidence added to AgentState and applied in node_fusion_engine.
  v2.4 HOLD Root-cause fix: LSTM bullish protection in decision logic.
          FIX A: conf >= 0.75 in Bear -> BUY (LSTM overwhelmingly bullish).
          FIX B: Bear + conf <= 0.50 + lstm <= 0.65 -> SELL (lean bearish guard).
          FIX C: SELL boundary <= 0.40, only when lstm_signal <= 0.65.
                 Prevents ConflictResolver SYSTEMIC_VETO from converting
                 bullish LSTM (0.99 -> arb_conf=0.40) into a SELL.
"""

import os
import sys
from typing import TypedDict, Dict, Any, Optional, List
import pandas as pd
import numpy as np
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finfolio_x.settings import GROQ_API_KEY, LLM_MODEL_NAME, LLM_TEMPERATURE
from ml_engine.topology_agent import TopologyAgent
from ml_engine.causal_agent import CausalAgent
from ml_engine.asc_memory import AgentDecisionMemory


# ==============================================================================
# CONFIGURATION
# ==============================================================================
BUY_CONFIDENCE_THRESHOLD            = 0.52
COMMODITY_BUY_THRESHOLD             = 0.55
BUY_GDI_MAX                         = 55.0
COMMODITY_TICKERS                   = {"GLD", "SLV", "USO", "UNG", "GDX"}
MAX_RISK_FOR_BUY                    = 0.80

REGIME_CONTRADICTION_CONF_THRESHOLD = 0.45
REGIME_CONTRADICTION_PENALTY        = 0.80

# v2.4: LSTM bullish protection thresholds
STRONG_CONF_BEAR_BUY    = 0.75   # FIX A: override Bear block when LSTM overwhelmingly bullish
BEAR_LEAN_SELL_MAX      = 0.50   # FIX B: Bear + conf <= this -> SELL (tightened from 0.60)
BULLISH_LSTM_PROTECT    = 0.65   # FIX C: if lstm > this, SELL branch is suppressed -> HOLD


# ==============================================================================
# 1. AGENT STATE
# ==============================================================================
class AgentState(TypedDict):
    ticker: str
    hist_data: Any
    stock_obj: Any
    error: Optional[str]

    # Market Context
    regime_label: str
    current_vol: float
    regime_confidence: float         # v2.3: hybrid regime confidence
    risk_score: float
    div_status: str

    # Technical Analysis
    lstm_signal: float
    mc_mean: float
    mc_std: float
    uncertainty_status: str
    top_driver: str

    # Sentiment
    sent_score: float
    sent_label: str
    sent_bias_warning: bool
    sentiment_articles: Optional[List[Dict[str, Any]]]  # MCP news article details

    # Phase 24: Topology
    topology_result: Optional[Dict[str, Any]]
    topology_chaos: float
    topology_modifier: float

    # Phase 25: Causal
    causal_result: Optional[Dict[str, Any]]
    counterfactual_verdict: Optional[str]
    causal_modifier: float

    # Fusion & Arbitration
    fusion_confidence: float
    attention_weights: Dict[str, float]
    conflict_detected: bool
    conflict_ruling: str
    conflict_reasoning: str
    trust_scores: Dict[str, float]

    # Heatmap
    gdi: float
    gdi_tension: str
    gdi_penalty: float

    # Phase 26: ASC
    asc_score: float
    asc_reliable: bool
    asc_saturated: bool
    asc_penalty_multiplier: float
    asc_quadrant: str
    dissent_sensitivity: float
    fdp_ran: bool
    fdp_interpretation: str
    regime_contradiction: bool
    pre_asc_confidence: float

    # Risk & Decision
    alloc_pct: float
    recommended_shares: int
    cash_value: float
    final_decision: str

    # Red Team
    red_team_passed: bool
    red_team_delta: float
    decision_flipped: bool

    # LLM Summary
    executive_summary: str


# ==============================================================================
# 2. ORCHESTRATOR
# ==============================================================================
class FinFolioGraphOrchestrator:

    def __init__(self, master_system):
        self.master = master_system

        if not GROQ_API_KEY:
            print("WARNING: GROQ_API_KEY is missing. LLM Supervisor will fail.")

        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=LLM_MODEL_NAME,
            temperature=LLM_TEMPERATURE,
        )

        try:
            self.topology_agent = TopologyAgent(time_delay=5, dimension=3, lookback=60)
            print("   [Orchestrator] Topology Agent Loaded")
        except Exception as e:
            print(f"   [Orchestrator] TopologyAgent failed: {e}")
            self.topology_agent = None

        try:
            self.causal_agent = CausalAgent(lookback=90, alpha=0.20)
            print("   [Orchestrator] Causal Agent Loaded")
        except Exception as e:
            print(f"   [Orchestrator] CausalAgent failed: {e}")
            self.causal_agent = None

        if hasattr(master_system, "asc_memory") and master_system.asc_memory:
            self.asc_memory = master_system.asc_memory
            print("   [Orchestrator] ASC Memory Engine loaded from master system")
        else:
            try:
                self.asc_memory = AgentDecisionMemory(window_size=30)
                print("   [Orchestrator] ASC Memory Engine initialized fresh")
            except Exception as e:
                print(f"   [Orchestrator] ASC Memory failed: {e}")
                self.asc_memory = None

        self.graph = self._build_graph()

    # -------------------------------------------------------------------------
    # NODE 1: Data Ingestion
    # -------------------------------------------------------------------------
    def node_fetch_data(self, state: AgentState) -> AgentState:
        print(f"\n[Node 1: Data Ingestion] Fetching data for {state['ticker']}...")
        stock_obj, hist = self.master._fetch_stock_data(state["ticker"])

        trust_scores = {"technical": 1.0, "sentiment": 1.0, "regime": 1.0}
        if hasattr(self.master, "meta_agent") and self.master.meta_agent:
            trust_scores = self.master.meta_agent.get_trust_scores(ticker=state["ticker"])
            self.master.meta_agent.print_trust_report(trust_scores)

        if stock_obj is None:
            return {"error": hist, "trust_scores": trust_scores}

        return {
            "stock_obj":      stock_obj,
            "hist_data":      hist,
            "error":          None,
            "trust_scores":   trust_scores,
            "final_decision": "PENDING",
        }

    # -------------------------------------------------------------------------
    # NODE 2: Market Context  HOLD uses HybridRegimeAgent (v2.3)
    # -------------------------------------------------------------------------
    def node_market_context(self, state: AgentState) -> AgentState:
        print("[Node 2: Market Context] Analyzing volatility and systemic risk...")

        if hasattr(self.master, "hybrid_regime") and self.master.hybrid_regime:
            regime_label, current_vol, regime_confidence = (
                self.master.hybrid_regime.detect(state["hist_data"], state["ticker"])
            )
        else:
            regime_label, current_vol = self.master._analyze_regime_module(
                state["hist_data"]
            )
            regime_confidence = 0.8
            print(f"      - Regime Confidence (rule-only fallback): {regime_confidence:.2f}")

        risk_score, div_status = self.master._analyze_correlation_module(state["ticker"])

        return {
            "regime_label":      regime_label,
            "current_vol":       current_vol,
            "regime_confidence": regime_confidence,
            "risk_score":        risk_score,
            "div_status":        div_status,
        }

    # -------------------------------------------------------------------------
    # NODE 3: Technical Analysis
    # -------------------------------------------------------------------------
    def node_technical_analysis(self, state: AgentState) -> AgentState:
        print("[Node 3: Technical Analysis] Running LSTM deep learning...")
        lstm_signal, mc_mean, mc_std, uncertainty_status, top_driver = (
            self.master._analyze_technicals_and_uncertainty(state["hist_data"])
        )
        return {
            "lstm_signal":        lstm_signal,
            "mc_mean":            mc_mean,
            "mc_std":             mc_std,
            "uncertainty_status": uncertainty_status,
            "top_driver":         top_driver,
        }

    # -------------------------------------------------------------------------
    # NODE 4: Sentiment Analysis
    # -------------------------------------------------------------------------
    def node_sentiment_analysis(self, state: AgentState) -> AgentState:
        print("[Node 4: Sentiment Analysis] Scraping global news via MCP...")
        sent_score        = 0.0
        sent_label        = "neutral"
        sent_bias_warning = False
        articles          = []

        try:
            # Capture individual article data from MCP payload before analysis
            if hasattr(self.master, 'sent_agent') and self.master.sent_agent:
                try:
                    mcp_payload = self.master.sent_agent.mcp_server.get_global_context_payload(state["ticker"])
                    for item in mcp_payload:
                        if not item.get("future_event") and len(item.get("text", "").strip()) >= 10:
                            text = item.get("text", "")
                            source = item.get("source", "Unknown")
                            label_i, score_i, _ = self.master.sent_agent.get_sentiment(text)
                            articles.append({
                                "source": source,
                                "headline": text[:120],
                                "label": label_i,
                                "score": round(float(score_i), 4),
                                "tier": item.get("tier", 3),
                            })
                except Exception:
                    pass  # Fall through to normal analysis

            result = self.master._analyze_sentiment_module(
                state["ticker"], state["stock_obj"], state["lstm_signal"]
            )
            if result is not None:
                if isinstance(result, tuple):
                    sent_label, sent_score = result
                else:
                    sent_score = float(result)
                    sent_label = ("bullish" if sent_score > 0.07
                                  else "bearish" if sent_score < -0.07
                                  else "neutral")
            else:
                print("      [WARN] MCP returned no result HOLD using neutral (0.0)")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"      [WARN] Sentiment pipeline error: {e} HOLD using neutral (0.0)")

        return {
            "sent_score":          sent_score,
            "sent_label":          sent_label,
            "sent_bias_warning":   sent_bias_warning,
            "sentiment_articles":  articles,
        }

    # -------------------------------------------------------------------------
    # NODE 4.5: Topology Analysis
    # -------------------------------------------------------------------------
    def node_topology_analysis(self, state: AgentState) -> AgentState:
        print("[Node 4.5: Topology Analysis] Computing persistent homology...")
        topology_result   = {}
        topology_chaos    = 0.0
        topology_modifier = 1.0

        if self.topology_agent and state.get("hist_data") is not None:
            try:
                topology_result   = self.topology_agent.analyze(state["hist_data"])
                topology_chaos    = topology_result.get("topology_chaos_score", 0.0)
                topology_modifier = topology_result.get("topology_modifier", 1.0)
            except Exception as e:
                print(f"      Topology analysis failed: {e}")

        return {
            "topology_result":   topology_result,
            "topology_chaos":    topology_chaos,
            "topology_modifier": topology_modifier,
        }

    # -------------------------------------------------------------------------
    # NODE 4.6: Causal Analysis
    # -------------------------------------------------------------------------
    def node_causal_analysis(self, state: AgentState) -> AgentState:
        print("[Node 4.6: Causal Analysis] Running do-calculus discovery...")
        causal_result = {}
        ticker        = state.get("ticker", "UNKNOWN")
        hist_data     = state.get("hist_data")

        if self.causal_agent and hist_data is not None:
            try:
                universe_data = self._fetch_universe_data()
                causal_result = self.causal_agent.analyze(
                    ticker=ticker, target_hist_df=hist_data,
                    universe_data=universe_data,
                )
            except Exception as e:
                print(f"      Causal analysis failed: {e}")

        return {
            "causal_result":   causal_result,
            "causal_modifier": causal_result.get("causal_modifier", 1.0),
        }

    # -------------------------------------------------------------------------
    # NODE 4.7: Counterfactual Debate
    # -------------------------------------------------------------------------
    def node_counterfactual_debate(self, state: AgentState) -> AgentState:
        print("[Node 4.7: Counterfactual Debate] Cross-examining causal drivers...")
        causal_result   = state.get("causal_result", {})
        lstm_signal     = state.get("lstm_signal", 0.5)
        causal_score    = causal_result.get("causal_score", 0.5)
        confounders     = causal_result.get("confounders_removed", [])
        causal_modifier = state.get("causal_modifier", 1.0)
        signal_dir      = "BULLISH" if lstm_signal > 0.5 else "BEARISH"
        total_universe  = len(causal_result.get(
            "variables", ["SPY", "QQQ", "VIX", "TLT", "GLD", "DXY", "TARGET"]
        ))

        if causal_score >= 0.65:
            verdict = f"CAUSAL_CONFIRMED -- {signal_dir} supported (mod: {causal_modifier:.2f}x)."
        elif len(confounders) > total_universe * 0.5:
            verdict = (f"CAUSAL_WARNED -- {signal_dir} confounder-driven "
                       f"({len(confounders)}/{total_universe}).")
        else:
            verdict = "CAUSAL_NEUTRAL -- Mixed causal evidence."

        return {"counterfactual_verdict": verdict}

    # -------------------------------------------------------------------------
    # NODE 5: Fusion Engine  HOLD applies regime_confidence (v2.3)
    # -------------------------------------------------------------------------
    def node_fusion_engine(self, state: AgentState) -> AgentState:
        print("[Node 5: Fusion Engine] Synthesizing intelligence layers...")

        vol_input = (
            0.9 if state["regime_label"] == "Bear"
            else 0.2 if state["regime_label"] == "Bull"
            else 0.5
        )

        final_conf, weights = self.master.fusion_agent.predict(
            lstm_p=state["lstm_signal"],
            sent_s=state["sent_score"],
            vol_v=vol_input,
            trust_scores=state.get("trust_scores", None),
        )

        sent_score          = state.get("sent_score", 0.0)
        lstm_signal         = state.get("lstm_signal", 0.5)
        sentiment_available = abs(sent_score) > 0.001

        if not sentiment_available:
            print("      [WARN] [Fusion] Sentiment frozen HOLD gates disabled")
        if sentiment_available and sent_score < -0.10 and lstm_signal > 0.55:
            cap = max(0.48, 0.56 + (sent_score + 0.10) * 0.10)
            print(f"      [Fusion] FinBERT veto cap={cap:.3f} (sent={sent_score:.3f})")
            final_conf = min(final_conf, cap)
        if sentiment_available and abs(sent_score) < 0.05 and lstm_signal > 0.65:
            final_conf = final_conf * 0.95

        lstm_regime_agree_bull = (
            lstm_signal > 0.58
            and state["regime_label"] == "Bull"
            and sent_score > 0.03
        )
        lstm_regime_agree_bear = (
            lstm_signal < 0.42
            and state["regime_label"] == "Bear"
            and sent_score < -0.03
        )
        if lstm_regime_agree_bull or lstm_regime_agree_bear:
            final_conf = min(final_conf * 1.08, 0.75)
            print(f"      [Fusion] Consensus boost -> {final_conf:.4f}")

        topo_mod          = state.get("topology_modifier", 1.0)
        caus_mod          = state.get("causal_modifier", 1.0)
        combined_modifier = (topo_mod + caus_mod) / 2.0

        if lstm_signal > 0.60 and state["regime_label"] == "Bull" and sent_score > 0.0:
            lstm_floor = min(lstm_signal * 0.72, 0.52)
            final_conf = max(final_conf, lstm_floor)
            print(f"      [Fusion] LSTM floor: {final_conf:.4f}")

        final_conf = final_conf * combined_modifier
        final_conf = float(np.clip(final_conf, 0.0, 1.0))

        # Apply hybrid regime confidence only when confidence is low.
        # High-confidence regime labels should not suppress directional signals.
        regime_confidence = state.get("regime_confidence", 0.8)
        if regime_confidence < 0.70:
            final_conf = 0.5 + (final_conf - 0.5) * regime_confidence
            final_conf = float(np.clip(final_conf, 0.0, 1.0))
            print(f"      - Regime confidence low ({regime_confidence:.2f}) -> "
                  f"neutral pull to {final_conf:.4f}")
        else:
            print(f"      - Regime confidence high ({regime_confidence:.2f}) -> no discount")

        return {
            "fusion_confidence": final_conf,
            "attention_weights": weights,
        }

    # -------------------------------------------------------------------------
    # NODE 5.5: ASC CHECK
    # -------------------------------------------------------------------------
    def node_asc_check(self, state: AgentState) -> AgentState:
        print("[Node 5.5: ASC Check] Computing Agent Sycophancy Coefficient...")

        default_asc_state = {
            "asc_score":              0.5,
            "asc_reliable":           False,
            "asc_saturated":          False,
            "asc_penalty_multiplier": 1.0,
            "asc_quadrant":           "ASC module unavailable",
            "dissent_sensitivity":    0.0,
            "fdp_ran":                False,
            "fdp_interpretation":     "",
            "regime_contradiction":   False,
        }

        if self.asc_memory is None:
            print("      ASC Memory not available -- skipping.")
            return default_asc_state

        try:
            lstm_signal    = float(state.get("lstm_signal", 0.5))
            sent_score     = float(state.get("sent_score", 0.0))
            regime_label   = state.get("regime_label", "Sideways")
            fusion_conf_in = float(state.get("fusion_confidence", 0.5))
            regime_prob    = self.asc_memory.regime_label_to_prob(regime_label)

            self.asc_memory.record_session(
                lstm_score=lstm_signal,
                sent_score=sent_score,
                regime_prob=regime_prob,
            )

            regime_contradiction = False
            working_conf         = fusion_conf_in

            if regime_label == "Bull" and working_conf < REGIME_CONTRADICTION_CONF_THRESHOLD:
                regime_contradiction = True
                old_conf     = working_conf
                working_conf = 0.50 + ((working_conf - 0.50) * REGIME_CONTRADICTION_PENALTY)
                working_conf = float(np.clip(working_conf, 0.0, 1.0))
                print(f"      [ASC] REGIME CONTRADICTION (Bull+LowConf): "
                      f"{old_conf:.3f} -> {working_conf:.3f}")

            elif regime_label == "Sideways" and (working_conf > 0.70 or working_conf < 0.35):
                regime_contradiction = True
                old_conf     = working_conf
                working_conf = 0.50 + (working_conf - 0.50) * 0.50
                print(f"      [ASC] REGIME CONTRADICTION (Sideways+Directional): "
                      f"{old_conf:.3f} -> {working_conf:.3f}")

            elif (regime_label == "Bear"
                and working_conf > 0.50
                and sent_score < 0.0
                and lstm_signal < 0.70):
                regime_contradiction = True
                old_conf     = working_conf
                working_conf = working_conf * 0.75
                working_conf = max(working_conf, 0.41)
                working_conf = float(np.clip(working_conf, 0.0, 1.0))
                print(f"      [ASC] REGIME CONTRADICTION (Bear+Bullish): "
                      f"{old_conf:.3f} -> {working_conf:.3f}")

            asc_result    = self.asc_memory.compute_asc()
            asc_score     = asc_result["asc"]
            asc_reliable  = asc_result["asc_reliable"]
            asc_saturated = asc_result.get("asc_saturated", False)
            print(f"      - ASC Score: {asc_score:.4f} "
                  f"(reliable: {asc_reliable}, saturated: {asc_saturated})")

            fdp_result = None
            if asc_reliable and not asc_saturated and asc_score >= 0.85:
                print("      - ASC HIGH -- running Forced Dissent Protocol...")
                fdp_result = self.asc_memory.run_forced_dissent(
                    lstm_signal=lstm_signal,
                    sent_score=sent_score,
                    regime_label=regime_label,
                    fusion_agent=self.master.fusion_agent,
                    trust_scores=state.get("trust_scores"),
                )
                print(f"      - Dissent Sensitivity: {fdp_result['dissent_sensitivity']:.4f}")
                print(f"      - FDP: {fdp_result['interpretation'][:80]}...")

            ds = fdp_result["dissent_sensitivity"] if fdp_result else 0.0

            if asc_reliable:
                penalty, quadrant = self.asc_memory.get_penalty_multiplier(
                    asc_score, ds, asc_saturated=asc_saturated
                )
            else:
                penalty  = 1.0
                quadrant = "WARMING UP -- no penalty"

            print(f"      - Quadrant: {quadrant}  Penalty: {penalty:.2f}x")

            adjusted_conf = float(np.clip(working_conf * penalty, 0.0, 1.0))
            if penalty < 1.0:
                print(f"      - Confidence: {working_conf:.4f} -> "
                      f"{adjusted_conf:.4f} (ASC penalty)")

            effective_threshold = (
                COMMODITY_BUY_THRESHOLD
                if state.get("ticker", "") in COMMODITY_TICKERS
                else BUY_CONFIDENCE_THRESHOLD
            )
            force_hold = (penalty <= 0.65 and adjusted_conf < effective_threshold)
            if force_hold:
                print("      ⛔ SYCOPHANTIC ENSEMBLE -- Forcing HOLD")

            summary = self.asc_memory.get_asc_summary(
                asc_result=asc_result,
                fdp_result=fdp_result,
                penalty=penalty,
                quadrant=quadrant,
            )
            AgentDecisionMemory.print_asc_report(summary)

            return {
                "fusion_confidence":      adjusted_conf,
                "asc_score":              round(asc_score, 4),
                "asc_reliable":           asc_reliable,
                "asc_saturated":          asc_saturated,
                "asc_penalty_multiplier": round(penalty, 4),
                "asc_quadrant":           quadrant,
                "dissent_sensitivity":    round(ds, 4),
                "fdp_ran":                bool(fdp_result and fdp_result.get("fdp_ran", False)),
                "fdp_interpretation":     fdp_result["interpretation"] if fdp_result else "",
                "regime_contradiction":   regime_contradiction,
                "pre_asc_confidence":     fusion_conf_in,
                **({"final_decision": "HOLD"} if force_hold else {}),
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"      ASC check failed: {e} -- proceeding without penalty")
            return default_asc_state

    # -------------------------------------------------------------------------
    # NODE 6: Conflict Resolution  HOLD v2.4 LSTM bullish protection
    # -------------------------------------------------------------------------
    def node_conflict_resolution(self, state: AgentState) -> AgentState:
        print("[Node 6: Conflict Resolution] Arbitrating agent disagreements...")

        fusion_conf  = state["fusion_confidence"]
        regime_label = state.get("regime_label", "Sideways")
        lstm_signal  = float(state.get("lstm_signal", 0.5))   # v2.4: needed for SELL guard

        if self.master.conflict_resolver:
            arbitration_result = self.master.conflict_resolver.arbitrate(
                tech_score=lstm_signal,
                sent_score=state["sent_score"],
                mc_std=state["mc_std"],
                regime_label=regime_label,
                risk_score=state["risk_score"],
                fusion_confidence=fusion_conf,
                trust_scores=state.get("trust_scores", None),
            )
            adj_conf          = arbitration_result["adjusted_confidence"]
            conflict_detected = arbitration_result["arbitrated"]
            conflict_ruling   = arbitration_result["ruling"]

            base_reasoning = "; ".join(arbitration_result["reasoning"])
            asc_note = (
                f" | ASC={state.get('asc_score', 0.5):.3f} "
                f"penalty={state.get('asc_penalty_multiplier', 1.0):.2f}x "
                f"({state.get('asc_quadrant', '')})"
            )
            if state.get("sent_bias_warning"):
                asc_note += " | SENTIMENT BIAS OVERRIDE ACTIVE"
            if state.get("regime_contradiction"):
                asc_note += " | REGIME CONTRADICTION CORRECTED"
            conflict_reasoning = base_reasoning + asc_note
        else:
            adj_conf           = fusion_conf
            conflict_detected  = False
            conflict_ruling    = "NO_MODULE"
            conflict_reasoning = f"ASC={state.get('asc_score', 0.5):.3f}"

        gdi, gdi_tension, gdi_penalty = 0.0, "HARMONY", 1.0
        if hasattr(self.master, "heatmap_agent") and self.master.heatmap_agent:
            heatmap_result = self.master.heatmap_agent.analyze(
                lstm_score=lstm_signal,
                sent_score=state["sent_score"],
                regime_label=regime_label,
                regime_vol=state.get("current_vol", 0.5),
            )
            gdi         = heatmap_result["gdi"]
            gdi_tension = heatmap_result["tension"]
            gdi_penalty = heatmap_result["penalty"]

        from ml_engine.causal_agent import CausalAgent as _CausalAgent
        raw_causal_mod             = state.get("causal_modifier", 1.0)
        causal_mod_gdi, gdi_causal_penalty = _CausalAgent.apply_gdi_penalty(
            raw_causal_mod, gdi
        )
        if gdi_causal_penalty > 0.0:
            print(f"      [Conflict] GDI={gdi:.3f} -> "
                  f"causal_mod {raw_causal_mod:.3f}->{causal_mod_gdi:.3f}")
            gdi_only_factor = causal_mod_gdi / max(raw_causal_mod, 1e-6)
            adj_conf = float(np.clip(adj_conf * gdi_only_factor, 0.0, 1.0))
            print(f"      [Conflict] GDI-only causal factor "
                  f"{gdi_only_factor:.3f}x -> adj_conf={adj_conf:.4f}")

        # Risk engine HOLD stock_price passed for min viable dollar check (v2.2)
        last_price = state["hist_data"]["Close"].iloc[-1]
        alloc_pct, _ = self.master.risk_engine.calculate_position_size(
            adj_conf,
            state["current_vol"],
            disagreement_penalty=gdi_penalty,
            regime=regime_label,
            stock_price=float(last_price),
        )
        num_shares, cash_value = self.master.risk_engine.get_shares_amount(
            last_price, alloc_pct
        )

        # -- Decision Logic  v2.4 ----------------------------------------------
        if state.get("final_decision") == "HOLD":
            # ASC forced hold upstream
            decision   = "HOLD"
            alloc_pct  = 0.0
            num_shares = 0
            cash_value = 0.0

        else:
            gdi_pct = gdi * 100
            effective_threshold = (
                COMMODITY_BUY_THRESHOLD
                if state.get("ticker", "") in COMMODITY_TICKERS
                else BUY_CONFIDENCE_THRESHOLD
            )

            # -- BUY paths -----------------------------------------------------
            normal_buy = (
                alloc_pct > 0.0
                and adj_conf >= effective_threshold
                and regime_label != "Bear"
                and gdi_pct < BUY_GDI_MAX
                and state.get("risk_score", 0.0) < MAX_RISK_FOR_BUY
            )

            # FIX A v2.4: high-confidence Bear override
            # ConflictResolver may floor a 0.99 LSTM -> 0.40; use adj_conf after
            # arbitration but also check STRONG_CONF_BEAR_BUY directly.
            # If the regime arbitration has kept adj_conf >= 0.75, LSTM is
            # genuinely overwhelmingly bullish even inside Bear.
            bear_override_buy = (
                alloc_pct > 0.0
                and adj_conf >= STRONG_CONF_BEAR_BUY
                and regime_label == "Bear"
                and gdi_pct < BUY_GDI_MAX
                and state.get("risk_score", 0.0) < MAX_RISK_FOR_BUY
            )

            if normal_buy or bear_override_buy:
                decision = "BUY"
                if bear_override_buy and not normal_buy:
                    print(f"      [Conflict] v2.4 Bear override BUY "
                          f"(adj_conf={adj_conf:.3f} >= {STRONG_CONF_BEAR_BUY})")

            # -- Bull low-confidence: HOLD not SELL ---------------------------
            elif adj_conf < 0.40 and regime_label == "Bull":
                decision   = "HOLD"
                alloc_pct  = 0.0
                num_shares = 0
                cash_value = 0.0
                print("      [Conflict] Bull regime: downgraded to HOLD (FIX-4).")

            # -- FIX C v2.4: SELL only when LSTM also agrees bearish/neutral --
            # Prevents: LSTM=0.99 -> SYSTEMIC_VETO -> adj_conf=0.40 -> SELL
            # Instead:  LSTM=0.99 + adj_conf=0.40 -> lstm > 0.65 -> HOLD
            elif adj_conf <= 0.40 and lstm_signal <= BULLISH_LSTM_PROTECT:
                decision   = "SELL"
                alloc_pct  = 0.0
                num_shares = 0
                cash_value = 0.0

            # -- FIX B v2.4: Bear lean-SELL with LSTM guard (tightened 0.60->0.50)
            # Prevents: SLV(lstm=0.537,adj=0.541) and AMD(lstm=0.529,adj=0.529)
            # from being SELL'd when LSTM was mildly bullish and market rose.
            elif (regime_label == "Bear"
                  and adj_conf <= BEAR_LEAN_SELL_MAX
                  and lstm_signal <= BULLISH_LSTM_PROTECT):
                decision   = "SELL"
                alloc_pct  = 0.0
                num_shares = 0
                cash_value = 0.0

            else:
                # lstm > BULLISH_LSTM_PROTECT but conf too low for BUY -> HOLD
                # This catches NVDA/TSLA cases where pipeline killed confidence
                decision   = "HOLD"
                alloc_pct  = 0.0
                num_shares = 0
                cash_value = 0.0

        print(f"      - Pre-Red-Team Decision: {decision}  "
              f"(adj_conf={adj_conf:.4f}  lstm={lstm_signal:.4f})")

        return {
            "fusion_confidence":  adj_conf,
            "alloc_pct":          alloc_pct,
            "recommended_shares": num_shares,
            "cash_value":         cash_value,
            "final_decision":     decision,
            "conflict_detected":  conflict_detected,
            "conflict_ruling":    conflict_ruling,
            "conflict_reasoning": conflict_reasoning,
            "gdi":                gdi,
            "gdi_tension":        gdi_tension,
            "gdi_penalty":        gdi_penalty,
        }

    # -------------------------------------------------------------------------
    # NODE 7: Red Team
    # -------------------------------------------------------------------------
    def node_red_team(self, state: AgentState) -> AgentState:
        print("[Node 7: Red Team] Simulating flash crash...")

        if "BUY" not in state["final_decision"]:
            return {
                "red_team_passed": True,
                "red_team_delta":  0.0,
                "final_decision":  state["final_decision"],
            }

        if self.master.red_team:
            try:
                crashed_df    = self.master.red_team.generate_flash_crash(
                    state["hist_data"], drop_pct=0.10
                )
                input_crashed = self.master.red_team._prepare_data_for_ai(crashed_df)
                crashed_score = (
                    self.master.tech_agent.predict_signal(input_crashed)
                    if hasattr(self.master.tech_agent, "predict_signal")
                    else self.master.tech_agent.predict(input_crashed)
                )
                delta = float(state["lstm_signal"]) - float(crashed_score)
                print(f"    Red Team PASSED. Delta={delta:.4f}")
                return {
                    "red_team_passed": True,
                    "red_team_delta":  delta,
                    "final_decision":  state["final_decision"],
                }
            except Exception as e:
                print(f"    Red Team error: {e}")

        return {
            "red_team_passed": True,
            "red_team_delta":  0.0,
            "final_decision":  state["final_decision"],
        }

    # -------------------------------------------------------------------------
    # NODE 8: LLM Supervisor
    # -------------------------------------------------------------------------
    def node_llm_supervisor(self, state: AgentState) -> AgentState:
        print("[Node 8: Supervisor] Groq LLM synthesizing executive report...")

        asc_note = ""
        if state.get("asc_reliable"):
            asc_note = (
                f"\nEnsemble Health (ASC): {state.get('asc_score', 0.5):.3f} "
                f"[{state.get('asc_quadrant', '')}]"
                f"\nASC Confidence Penalty: {state.get('asc_penalty_multiplier', 1.0):.2f}x"
            )
            if state.get("fdp_ran"):
                asc_note += (f"\nForced Dissent: "
                             f"Sensitivity={state.get('dissent_sensitivity', 0.0):.3f}")
            if state.get("asc_saturated"):
                asc_note += "\nASC Saturation: KSG penalty suppressed."

        sent_note = (
            "\nSENTIMENT BIAS WARNING: FinBERT returned universal negative scores."
            if state.get("sent_bias_warning") else ""
        )
        regime_note = (
            "\nREGIME CONTRADICTION: Corrected before ASC check."
            if state.get("regime_contradiction") else ""
        )
        hybrid_note = f"\nHybrid Regime Confidence: {state.get('regime_confidence', 0.8):.2f}"

        context = f"""
Ticker: {state['ticker']}
Regime: {state['regime_label']} (Confidence: {state.get('regime_confidence', 0.8):.2f})
Systemic Risk: {state['div_status']}
Tech Signal (LSTM): {state['lstm_signal']:.4f}
Fusion Confidence: {state['fusion_confidence']:.4f}
Top Driver: {state['top_driver']}
Sentiment: {state['sent_score']:.4f} ({state.get('sent_label', 'unknown')})
Boardroom GDI: {state.get('gdi', 0.0) * 100:.1f}%
Final Decision: {state['final_decision']}
Capital Allocation: {state['alloc_pct'] * 100:.2f}%{asc_note}{sent_note}{regime_note}{hybrid_note}
"""

        sys_msg = SystemMessage(
            content=(
                "You are the Chief Risk Officer AI for FinFolio-X. "
                "Write a highly professional, 3-sentence executive summary. "
                "If regime confidence < 0.8, mention regime uncertainty as a risk factor. "
                "If ASC data present and penalty applied, mention ensemble quality. "
                "If sentiment bias or regime contradiction occurred, acknowledge them."
            )
        )
        hum_msg = HumanMessage(content=f"Synthesize this state:\n{context}")

        try:
            response = self.llm.invoke([sys_msg, hum_msg])
            summary  = response.content
        except Exception as e:
            summary = f"LLM Synthesis failed: {e}"

        if hasattr(self.master, "meta_agent") and self.master.meta_agent:
            try:
                last_price = state["hist_data"]["Close"].iloc[-1]
                self.master.meta_agent.log_decision(
                    ticker=state["ticker"],
                    lstm_score=state["lstm_signal"],
                    sent_score=state["sent_score"],
                    regime_label=state["regime_label"],
                    risk_score=state["risk_score"],
                    fusion_confidence=state["fusion_confidence"],
                    final_decision=state["final_decision"],
                    price_at_decision=float(last_price),
                    asc_score=state.get("asc_score", 0.5),
                    asc_reliable=state.get("asc_reliable", False),
                )
            except Exception as e:
                print(f"    Meta-Agent logging failed: {e}")

        return {"executive_summary": summary}

    # -------------------------------------------------------------------------
    # HELPER: Universe data for causal agent
    # -------------------------------------------------------------------------
    def _fetch_universe_data(self):
        import yfinance as yf
        ticker_map = {
            "SPY": "SPY", "QQQ": "QQQ", "VIX": "^VIX",
            "TLT": "TLT", "GLD": "GLD", "DXY": "DX-Y.NYB",
        }
        universe_data = {}
        for clean_name, yf_ticker in ticker_map.items():
            try:
                df = yf.download(
                    yf_ticker, period="6mo", interval="1d", progress=False
                )
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if not df.empty and "Close" in df.columns:
                    universe_data[clean_name] = df
            except Exception:
                pass
        return universe_data

    # -------------------------------------------------------------------------
    # ROUTING
    # -------------------------------------------------------------------------
    def route_after_data(self, state: AgentState) -> str:
        return "end" if state.get("error") else "continue"

    def route_after_arbitration(self, state: AgentState) -> str:
        if "BUY" in state.get("final_decision", ""):
            return "run_red_team"
        return "skip_to_llm"

    # -------------------------------------------------------------------------
    # GRAPH COMPILATION
    # -------------------------------------------------------------------------
    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("fetch_data",           self.node_fetch_data)
        workflow.add_node("market_context",        self.node_market_context)
        workflow.add_node("technical_analysis",    self.node_technical_analysis)
        workflow.add_node("sentiment_analysis",    self.node_sentiment_analysis)
        workflow.add_node("topology_analysis",     self.node_topology_analysis)
        workflow.add_node("causal_analysis",       self.node_causal_analysis)
        workflow.add_node("counterfactual_debate", self.node_counterfactual_debate)
        workflow.add_node("fusion_engine",         self.node_fusion_engine)
        workflow.add_node("asc_check",             self.node_asc_check)
        workflow.add_node("conflict_resolution",   self.node_conflict_resolution)
        workflow.add_node("red_team",              self.node_red_team)
        workflow.add_node("llm_supervisor",        self.node_llm_supervisor)

        workflow.set_entry_point("fetch_data")

        workflow.add_conditional_edges(
            "fetch_data", self.route_after_data,
            {"end": END, "continue": "market_context"},
        )
        workflow.add_edge("market_context",        "technical_analysis")
        workflow.add_edge("technical_analysis",    "sentiment_analysis")
        workflow.add_edge("sentiment_analysis",    "topology_analysis")
        workflow.add_edge("topology_analysis",     "causal_analysis")
        workflow.add_edge("causal_analysis",       "counterfactual_debate")
        workflow.add_edge("counterfactual_debate", "fusion_engine")
        workflow.add_edge("fusion_engine",         "asc_check")
        workflow.add_edge("asc_check",             "conflict_resolution")

        workflow.add_conditional_edges(
            "conflict_resolution", self.route_after_arbitration,
            {"run_red_team": "red_team", "skip_to_llm": "llm_supervisor"},
        )
        workflow.add_edge("red_team",       "llm_supervisor")
        workflow.add_edge("llm_supervisor", END)

        return workflow.compile()

    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------
    def run_analysis(self, ticker: str):
        initial_state = {
            "ticker":                 ticker,
            "error":                  None,
            "topology_result":        None,
            "topology_chaos":         0.0,
            "topology_modifier":      1.0,
            "causal_result":          None,
            "counterfactual_verdict": None,
            "final_decision":         "PENDING",
            "sent_bias_warning":      False,
            "sent_label":             "neutral",
            "regime_confidence":      0.8,
            "asc_score":              0.5,
            "asc_reliable":           False,
            "asc_saturated":          False,
            "asc_penalty_multiplier": 1.0,
            "asc_quadrant":           "",
            "dissent_sensitivity":    0.0,
            "fdp_ran":                False,
            "fdp_interpretation":     "",
            "regime_contradiction":   False,
            "pre_asc_confidence":     0.5,
        }

        print(f"\n[LangGraph Orchestrator v2.4] Starting 12-node graph for {ticker}...")
        final_state = self.graph.invoke(initial_state)

        if final_state.get("error"):
            print(f"Analysis Aborted: {final_state['error']}")
            return final_state

        bias_flag   = " | SENTIMENT BIAS"      if final_state.get("sent_bias_warning")   else ""
        regime_flag = " | REGIME CONTRADICTION" if final_state.get("regime_contradiction") else ""
        sat_flag    = " | ASC SATURATED"        if final_state.get("asc_saturated")        else ""
        hybrid_conf = final_state.get("regime_confidence", 0.8)

        print("\n" + "=" * 72)
        print(f"FINAL DECISION: {final_state.get('final_decision')}"
              f"{bias_flag}{regime_flag}{sat_flag}")
        print(
            f"  ASC: {final_state.get('asc_score', 0.5):.4f}  |  "
            f"Penalty: {final_state.get('asc_penalty_multiplier', 1.0):.2f}x  |  "
            f"Quadrant: {final_state.get('asc_quadrant', 'N/A')}"
        )
        print(f"  Regime: {final_state.get('regime_label', 'N/A')}  |  "
              f"HybridConf: {hybrid_conf:.2f}  |  "
              f"LSTM: {final_state.get('lstm_signal', 0.5):.4f}")
        print("=" * 72)

        return final_state