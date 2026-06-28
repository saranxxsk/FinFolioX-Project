import os
import sys
import time
import yfinance as yf
import numpy as np
import pandas as pd
import joblib
import random
import requests
import re
import xml.etree.ElementTree as ET
from datetime import datetime

# ==============================================================================
# PROJECT CONFIGURATION & PATH SETUP
# ==============================================================================
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.views import BASE_DIR
from ml_engine.technical_agent import TechnicalAgent
from ml_engine.sentiment_agent import SentimentAgent
from ml_engine.fusion_agent import FusionAgent
from ml_engine.regime_agent import RegimeAgent
from ml_engine.risk_engine import RiskEngine
from ml_engine.correlation_agent import CorrelationDivergenceDetector
from ml_engine.uncertainty_agent import UncertaintyAgent
from ml_engine.explainability_agent import ExplainabilityAgent
from ml_engine.topology_agent import TopologyAgent
from ml_engine.causal_agent import CausalAgent
from ml_engine.asc_memory import AgentDecisionMemory
from ml_engine.hybrid_regime_agent import HybridRegimeAgent   # ← NEW

# ==============================================================================
# SYSTEM CONSTANTS
# ==============================================================================
SYSTEM_VERSION = "27.0 (Hybrid Regime + ASC Sycophancy Detection)"
DEFAULT_CAPITAL = 10_000.0
MAX_RISK_PER_TRADE = 0.20
NEWS_LOOKBACK_ITEMS = 5
UNCERTAINTY_THRESHOLD_HIGH = 0.15
UNCERTAINTY_THRESHOLD_MODERATE = 0.05
DIVERGENCE_THRESHOLD_CRITICAL = 0.70
DIVERGENCE_THRESHOLD_MINOR = 0.40
BUY_GDI_MAX = 55.0

COMMODITY_MAP = {
    "GOLD": "GLD",
    "SILVER": "SLV",
    "OIL": "USO",
    "NATGAS": "UNG",
}

COMMODITY_TICKERS = {"GLD", "SLV", "USO", "UNG", "GDX"}
BUY_CONFIDENCE_THRESHOLD = 0.52
COMMODITY_BUY_THRESHOLD  = 0.55

# ==============================================================================
# PHASE 11 IMPORT
# ==============================================================================
try:
    from adversarial_tester import AdversarialTester
    print("   [OK] Phase 11 (Red Team) Loaded via direct import.")
except ImportError:
    try:
        from ml_engine.adversarial_tester import AdversarialTester
        print("   [OK] Phase 11 (Red Team) Loaded via package import.")
    except ImportError:
        print("   [WARN] Phase 11 Module missing. Red Team Disabled.")
        AdversarialTester = None

# ==============================================================================
# PHASE 13 IMPORT
# ==============================================================================
try:
    from conflict_resolver import ConflictResolver
    print("   [OK] Phase 13 (Conflict Arbitrator) Loaded via direct import.")
except ImportError:
    try:
        from ml_engine.conflict_resolver import ConflictResolver
        print("   [OK] Phase 13 (Conflict Arbitrator) Loaded via package import.")
    except ImportError:
        print("   [WARN] Phase 13 Module missing. Conflict Resolution Disabled.")
        ConflictResolver = None

# ==============================================================================
# PHASE 14 IMPORT
# ==============================================================================
try:
    from meta_agent import MetaAgent
    print("   [+] Phase 14 (Meta-Agent) Loaded via direct import.")
except ImportError:
    try:
        from ml_engine.meta_agent import MetaAgent
        print("   [+] Phase 14 (Meta-Agent) Loaded via package import.")
    except ImportError:
        print("   [!] Phase 14 Module missing. Meta-Agent Disabled.")
        MetaAgent = None

# ==============================================================================
# PHASE 16 IMPORT
# ==============================================================================
try:
    from heatmap_agent import HeatmapAgent
    print("   [+] Phase 16 (Heatmap Agent) Loaded via direct import.")
except ImportError:
    try:
        from ml_engine.heatmap_agent import HeatmapAgent
        print("   [+] Phase 16 (Heatmap Agent) Loaded via package import.")
    except ImportError:
        print("   [!] Phase 16 Module missing. Heatmap Agent Disabled.")
        HeatmapAgent = None


# ==============================================================================
# FINFOLIO-X MASTER SYSTEM CLASS
# ==============================================================================
class FinFolioSystem:
    """
    The Master Orchestrator for FinFolio-X AI Trading System.

    Architecture:
    1.  Technical Agent   (LSTM)         : Analyzes price trends and patterns.
    2.  Sentiment Agent   (FinBERT)      : Analyzes global news via MCP.
    3.  Regime Agent      (HMM)          : Detects hidden market states.
    3b. Hybrid Regime     (Rule + HMM)   : Fuses rule-based + HMM for accuracy.
    4.  Correlation Agent (Graph)        : Detects systemic risk and anomalies.
    5.  Uncertainty Agent (Bayesian)     : Quantifies model confidence.
    6.  Explainability    (SHAP)         : Explains WHY the model decided.
    7.  Topology Agent    (TDA)          : Phase 24 Geometric Market Shape.
    8.  Fusion Agent      (Attention)    : Weighs all inputs to make a decision.
    9.  Risk Engine       (Kelly)        : Calculates optimal position sizing.
    10. Conflict Resolver (Phase 13)     : Arbitrates agent disagreements.
    """

    def __init__(self):
        self._print_startup_banner()

        MODELS_DIR = r"D:/FinFolioX/saved_models"

        # 1. Technical Agent
        print("\n   🔹 [1/11] Loading Technical Agent (LSTM)...")
        try:
            self.tech_agent = TechnicalAgent(
    lstm_model_path=os.path.join(MODELS_DIR, "lstm_model.keras"),
    lstm_scaler_path=os.path.join(MODELS_DIR, "lstm_scaler.pkl"),
)
            print("      [OK] LSTM Brain Online.")
        except Exception as e:
            print(f"      [BAD] Critical Error loading Technical Agent: {e}")
            sys.exit(1)

        # 2. Sentiment Agent
        print("   🔹 [2/11] Loading Sentiment Agent (FinBERT)...")
        try:
            self.sent_agent = SentimentAgent()
            print("      [OK] FinBERT Model Loaded Successfully.")
        except Exception as e:
            print(f"      Warning: Sentiment Agent failed ({e}). Using fallback.")
            self.sent_agent = None

        # 3. Regime Agent (HMM HOLD kept for backward compatibility)
        print("   🔹 [3/11] Loading Regime Agent (HMM Market Detector)...")
        try:
            self.regime_agent = RegimeAgent(
                model_path=os.path.join(MODELS_DIR, "hmm_regime.pkl")
            )
            print("      [OK] Hidden Markov Model Loaded Successfully.")
        except Exception as e:
            print(f"      [WARN] Warning: Regime Agent failed ({e}).")
            self.regime_agent = None

        # 3b. Hybrid Regime Agent (Rule + HMM v2)  ← NEW
        print("   🔹 [3b/11] Loading Hybrid Regime Agent (Rule + HMM v2)...")
        try:
            self.hybrid_regime = HybridRegimeAgent(
        hmm_model_path=os.path.join(MODELS_DIR, "hmm_regime_hybrid.pkl"),
        verbose=True,
    )
            print("      [OK] Hybrid Regime System Online.")
        except Exception as e:
            print(f"      [WARN] Hybrid Regime Agent failed ({e}). Using rule-only fallback.")
            self.hybrid_regime = None

        # 4. Correlation Agent
        print("   🔹 [4/11] Loading Correlation Agent (Statistical Graph)...")
        try:
            self.corr_agent = CorrelationDivergenceDetector()
            print("      [OK] Market Graph Engine Initialized.")
        except Exception as e:
            print(f"      [WARN] Warning: Correlation Agent failed ({e}).")
            self.corr_agent = None

        # 5. Uncertainty Agent
        print("   🔹 [5/11] Loading Uncertainty Agent (Bayesian Wrapper)...")
        try:
            self.uncertainty_agent = UncertaintyAgent(self.tech_agent)
            print("      [OK] Uncertainty Engine Initialized.")
        except Exception as e:
            print(f"      [WARN] Warning: Uncertainty Agent failed ({e}).")
            self.uncertainty_agent = None

        # 6. Fusion Agent
        print("   🔹 [6/11] Loading Fusion Agent (Multi-Head Attention)...")
        try:
            self.fusion_agent = FusionAgent(
                model_path=os.path.join(MODELS_DIR, "attention_fusion.pth")
            )
            print("      [OK] Attention Mechanism Loaded Successfully.")
        except Exception as e:
            print(f"      [BAD] Critical Error loading Fusion Agent: {e}")
            sys.exit(1)

        # 7. Risk Engine
        print("   🔹 [7/11] Loading Risk Engine (Kelly Criterion)...")
        self.risk_engine = RiskEngine(
    default_account_size=DEFAULT_CAPITAL,
    bear_max_allocation=0.10,   # ← v2.2: cap Bear allocations at 10%
)
        print(f"      [OK] Risk Manager Online (Account: ${DEFAULT_CAPITAL:,.2f}).")

        # 8. Explainability Agent (lazy init)
        print("   🔹 [8/11] Preparing Explainability Agent (Perturbation)...")
        self.explainability_agent = None

        # 9. Topological Shape Agent (Phase 24)
        print("   🔹 [9/11] Loading Topological Shape Agent (Ripser)...")
        try:
            self.topology_agent = TopologyAgent(time_delay=5, dimension=3, lookback=60)
        except Exception:
            self.topology_agent = None

        # 10. Causal Discovery Agent (Phase 25)
        print("   🔹 [10/11] Loading Causal Discovery Agent (PC Algorithm)...")
        try:
            self.causal_agent = CausalAgent(lookback=90, alpha=0.20)
            print("      [OK] Causal Discovery Engine Online.")
        except Exception as e:
            print(f"      [WARN] Warning: Causal Agent failed ({e}).")
            self.causal_agent = None

        # 11. ASC Memory Engine (Phase 26)
        print("   🔹 [11/11] Loading ASC Memory Engine (Sycophancy Detection)...")
        try:
            self.asc_memory = AgentDecisionMemory(window_size=30)
            print("      [OK] ASC Memory Engine Online.")
        except Exception as e:
            print(f"      [WARN] Warning: ASC Memory Engine failed ({e}).")
            self.asc_memory = None

        # Regime Scaler (kept for backward compat)
        self.regime_scaler_path = os.path.join(MODELS_DIR, "regime_scaler.pkl")
        if os.path.exists(self.regime_scaler_path):
            self.regime_scaler = joblib.load(self.regime_scaler_path)
        else:
            self.regime_scaler = None
            print("      [WARN] Warning: Regime Scaler not found.")

        print("\n[OK] SYSTEM INITIALIZATION COMPLETE. ALL ENGINES ONLINE.\n")

        # Phase hooks
        self.red_team         = AdversarialTester(self) if AdversarialTester else None
        self.conflict_resolver = ConflictResolver()     if ConflictResolver  else None
        self.meta_agent       = MetaAgent()             if MetaAgent         else None
        self.heatmap_agent    = HeatmapAgent()          if HeatmapAgent      else None

    def _print_startup_banner(self):
        print("\n" + "█" * 72)
        print("🚀 INITIALIZING FINFOLIO-X: EXPLAINABLE AI TRADING SYSTEM")
        print("█" * 72)
        print(f"   • Version: {SYSTEM_VERSION}")
        print("   • Mode: Live Inference (Real-Time Data)")
        print("   • Architecture: Multi-Agent Mixture of Experts (MoE) + XAI")
        print("   • Copyright © 2026 FinFolio Team")
        print("-" * 72)

    # ==========================================================================
    # HELPER: TECHNICAL INDICATORS
    # ==========================================================================
    def _calculate_rsi(self, prices, window=14):
        delta = prices.diff()
        gain  = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs    = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices):
        ema_12 = prices.ewm(span=12, adjust=False).mean()
        ema_26 = prices.ewm(span=26, adjust=False).mean()
        return ema_12 - ema_26

    # ==========================================================================
    # MODULAR ANALYSIS METHODS
    # ==========================================================================
    def _fetch_stock_data(self, ticker):
        ticker = COMMODITY_MAP.get(ticker.upper(), ticker)
        try:
            print("   ⏳ Fetching historical data from Yahoo Finance...")
            stock = yf.Ticker(ticker)
            hist  = stock.history(period="2y")
            if len(hist) < 200:
                return None, "[BAD] Not enough historical data (Need > 200 days)."
            hist["SMA_50"]  = hist["Close"].rolling(window=50).mean()
            hist["SMA_200"] = hist["Close"].rolling(window=200).mean()
            hist["RSI"]     = self._calculate_rsi(hist["Close"])
            hist["MACD"]    = self._calculate_macd(hist["Close"])
            hist.dropna(inplace=True)
            if len(hist) < 60:
                return None, "[BAD] Not enough data after processing indicators."
            return stock, hist
        except Exception as e:
            return None, f"[BAD] Data Connection Error: {e}"

    def _analyze_technicals_and_uncertainty(self, hist):
        print("\n   📈 [Technical Analysis] Reading Charts (LSTM)...")
        lstm_signal = self.tech_agent.predict(hist)          # ← already correct, untouched
        print(f"      - LSTM Signal: {lstm_signal:.4f}")

        from ml_engine.technical_agent import build_lstm_features
        feature_df    = build_lstm_features(hist)
        last_100_days = feature_df.tail(100)

        if self.explainability_agent is None:
            self.explainability_agent = ExplainabilityAgent(self.tech_agent, feature_df)

        print("   [CHECK] [Explainability] Running Perturbation Analysis...")
        shap_scores, top_driver = self.explainability_agent.explain_prediction(last_100_days)
        if shap_scores:
            impact_val   = shap_scores.get(top_driver, 0.0)
            print(f"      - Top Driver: {top_driver} (Impact: {impact_val:.4f})")
            sorted_feats = sorted(shap_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            print(f"      - Key Factors: {', '.join([f'{k}={v:.3f}' for k, v in sorted_feats])}")

        print("   🎲 [Uncertainty Agent] Computing confidence distance...")
        mc_mean, mc_std = self.uncertainty_agent.predict_from_prob(lstm_signal)  # ← ONLY CHANGE

        uncertainty_status = "[OK] High Certainty"
        if mc_std > UNCERTAINTY_THRESHOLD_MODERATE:
            uncertainty_status = "[WARN] Moderate Uncertainty"
        if mc_std > UNCERTAINTY_THRESHOLD_HIGH:
            uncertainty_status = "🚨 HIGH UNCERTAINTY (Guessing)"

        print(f"      - MC Mean: {mc_mean:.4f}  Uncertainty: {mc_std:.4f} ({uncertainty_status})")
        return lstm_signal, mc_mean, mc_std, uncertainty_status, top_driver

    def _analyze_sentiment_module(self, ticker, stock_obj, lstm_signal):
        print("\n   📰 [Sentiment Analysis] Initiating MCP Protocol...")
        if self.sent_agent is None:
            print("      [!] Sentiment Agent unavailable. Using neutral score.")
            return 0.0
        try:
            result = self.sent_agent.analyze_with_mcp(ticker)
            if not result:
                print("      [WARN] MCP failed to return valid data. Defaulting to neutral.")
                return 0.0
            sent_label, sent_score = result
            print(f"      - Sentiment Score: {sent_score:.4f} ({sent_label})")
            return sent_score
        except Exception as e:
            print(f"      [WARN] MCP/FinBERT Pipeline Error: {e}. Defaulting to neutral.")
            return 0.0

    def _analyze_regime_module(self, hist, ticker=""):
        """
        Rule-based regime detection v3.
        Kept as fallback when HybridRegimeAgent is unavailable.
        """
        print("\n   ⛈️  [Regime Detection] Detecting Market State (Rule-Based)...")

        current_vol = hist["Close"].pct_change().rolling(10).std().iloc[-1]
        if pd.isna(current_vol):
            current_vol = 0.015

        sma_50  = float(hist["SMA_50"].iloc[-1])
        sma_200 = float(hist["SMA_200"].iloc[-1])
        ret_5d  = (hist["Close"].iloc[-1] / hist["Close"].iloc[-5] - 1.0) if len(hist) >= 5 else 0.0
        rsi_now = float(hist["RSI"].iloc[-1]) if "RSI" in hist.columns else 50.0

        if sma_50 > sma_200 and current_vol < 0.025:
            regime_label = "Bull"
        elif sma_50 < sma_200 and current_vol > 0.015:
            regime_label = "Bear"
        else:
            regime_label = "Sideways"

        if regime_label == "Bull" and len(hist) >= 6:
            sma_50_prev = float(hist["SMA_50"].iloc[-6])
            sma_slope   = (sma_50 - sma_50_prev) / sma_50_prev
            if ret_5d < -0.015 or rsi_now < 45:
                regime_label = "Sideways"
                print(f"      [WARN] Bull->Sideways: strong breakdown (5d={ret_5d:.2%}, RSI={rsi_now:.1f})")
            elif ret_5d < 0 and rsi_now < 55 and sma_slope < 0:
                regime_label = "Sideways"
                print(f"      [WARN] Bull->Sideways: exhaustion (5d={ret_5d:.2%}, RSI={rsi_now:.1f})")

        _COMMODITY = {"GLD", "SLV", "USO", "UNG", "GDX", "DJP", "PDBC"}
        if ticker.upper() in _COMMODITY:
            ret_10d = (hist["Close"].iloc[-1] / hist["Close"].iloc[-10] - 1.0) if len(hist) >= 10 else 0.0
            if rsi_now > 65 and ret_10d > 0.05:
                regime_label = "Bull"
                print(f"      ⚡ Commodity momentum: ->Bull (RSI={rsi_now:.1f}, 10d={ret_10d:.2%})")
            elif rsi_now > 60 and ret_10d > 0.03 and regime_label == "Bear":
                regime_label = "Sideways"
                print(f"      ⚡ Commodity momentum: Bear->Sideways")

        if regime_label == "Bear" and rsi_now < 35:
            regime_label = "Sideways"
            print(f"      [WARN] Bear->Sideways: oversold bounce risk (RSI={rsi_now:.1f})")
        if regime_label == "Bear" and rsi_now > 60:
            regime_label = "Sideways"
            print(f"      [WARN] Bear->Sideways: RSI contradicts Bear (RSI={rsi_now:.1f})")

        print(f"      - Vol={current_vol:.4f}  RSI={rsi_now:.1f}  5d={ret_5d:+.2%}  -> {regime_label}")
        return regime_label, current_vol

    def _analyze_correlation_module(self, ticker):
        print("\n   🕸️  [Systemic Risk] Analyzing Cross-Asset Divergence...")
        risk_score, _ = self.corr_agent.get_market_context(ticker)
        div_status = "[OK] Synced"
        if risk_score > DIVERGENCE_THRESHOLD_MINOR:
            div_status = "[WARN] Minor Divergence"
        if risk_score > DIVERGENCE_THRESHOLD_CRITICAL:
            div_status = "🚨 CRITICAL DIVERGENCE (Anomaly)"
        print(f"      - Divergence Score: {risk_score:.4f}  Status: {div_status}")
        return risk_score, div_status

    def _fetch_universe_data(self):
        UNIVERSE_TICKERS = ["SPY", "QQQ", "VIX", "TLT", "GLD", "DXY"]
        universe_data = {}
        print("\n   📊 [Causal Setup] Fetching macro universe data...")
        for sym in UNIVERSE_TICKERS:
            try:
                universe_data[sym] = yf.download(sym, period="6mo", interval="1d", progress=False)
            except Exception as e:
                print(f"      [WARN] Failed to fetch {sym}: {e}")
        return universe_data

    # ==========================================================================
    # MAIN ANALYZER ORCHESTRATOR
    # ==========================================================================
    def analyze_stock(self, ticker="AAPL"):
        print(f"📊 STARTING DEEP DIVE ANALYSIS FOR: {ticker}")

        stock_obj, hist = self._fetch_stock_data(ticker)
        if stock_obj is None:
            return hist

        last_price = hist["Close"].iloc[-1]

        trust_scores = None
        if self.meta_agent:
            trust_scores = self.meta_agent.get_trust_scores(ticker=ticker)
            self.meta_agent.print_trust_report(trust_scores)

        lstm_signal, mc_mean, mc_std, uncertainty_status, top_driver = (
            self._analyze_technicals_and_uncertainty(hist)
        )
        sent_score = self._analyze_sentiment_module(ticker, stock_obj, lstm_signal)

        # -- Hybrid Regime Detection ← NEW -------------------------------------
        print("\n   ⛈️  [Regime Detection] Running Hybrid Regime System...")
        if self.hybrid_regime:
            regime_label, current_vol, regime_confidence = (
                self.hybrid_regime.detect(hist, ticker)
            )
        else:
            regime_label, current_vol = self._analyze_regime_module(hist, ticker)
            regime_confidence = 0.8
            print(f"      - Regime Confidence (fallback): {regime_confidence:.2f}")

        risk_score, div_status = self._analyze_correlation_module(ticker)

        # Re-fetch trust scores with regime context for stronger influence
        if self.meta_agent:
            trust_scores = self.meta_agent.get_trust_scores(ticker=ticker, regime=regime_label)

        # -- Phase 24: Topological Analysis -----------------------------------
        topo_modifier  = 1.0
        topo_signal    = "UNKNOWN"
        topology_result = {}
        if hasattr(self, "topology_agent") and self.topology_agent:
            print("\n   🌀 [Phase 24] Computing Persistent Homology...")
            topology_result = self.topology_agent.analyze(hist)
            topo_modifier   = topology_result.get("topology_modifier", 1.0)
            topo_signal     = topology_result.get("market_shape_signal", "UNKNOWN")

        # -- Phase 25: Causal Discovery ----------------------------------------
        causal_modifier = 1.0
        causal_score    = 0.5
        causal_result   = {}
        if hasattr(self, "causal_agent") and self.causal_agent:
            print("\n   🔗 [Phase 25] Running Causal Discovery (PC Algorithm)...")
            try:
                universe_data   = self._fetch_universe_data()
                causal_result   = self.causal_agent.analyze(
                    ticker=ticker, target_hist_df=hist, universe_data=universe_data,
                )
                causal_modifier = causal_result.get("causal_modifier", 1.0)
                causal_score    = causal_result.get("causal_score", 0.5)
                print(f"      - Causal Score: {causal_score:.4f} (Modifier: {causal_modifier:.3f}x)")
            except Exception as e:
                print(f"      [WARN] Causal analysis failed: {e}")

        # -- Phase 11: Red Team ------------------------------------------------
        robustness_penalty = 0.0
        if self.red_team:
            print("\n   🛡️  [Red Team] Running Live Robustness Check...")
            try:
                crashed_df    = self.red_team.generate_flash_crash(hist, drop_pct=0.20)
                input_crashed = self.red_team._prepare_data_for_ai(crashed_df)
                crashed_score = (
                    self.tech_agent.predict_signal(input_crashed)
                    if hasattr(self.tech_agent, "predict_signal")
                    else self.tech_agent.predict(input_crashed)
                )
                robustness_delta = lstm_signal - crashed_score
                if robustness_delta < 0.02:
                    print(f"      [BAD] Model is stubborn! (Delta: {robustness_delta:.4f})")
                    robustness_penalty = 0.2
                else:
                    print(f"      [OK] PASS: Model detected the crash. (Delta: {robustness_delta:.4f})")
            except Exception as e:
                print(f"      [WARN] Red Team check failed: {e}")

        # -- Fusion ------------------------------------------------------------
        print("\n   🧠 [Fusion Engine] Synthesizing Intelligence Layers...")
        vol_input = 0.9 if regime_label == "Bear" else 0.2 if regime_label == "Bull" else 0.5

        combined_modifier = (topo_modifier + causal_modifier) / 2.0
        combined_modifier = max(combined_modifier, 0.93)

        final_conf, weights = self.fusion_agent.predict(
            lstm_p=mc_mean,
            sent_s=sent_score,
            vol_v=vol_input,
            trust_scores=trust_scores,
        )

        if lstm_signal > 0.60 and regime_label == "Bull" and sent_score > 0.0:
            lstm_floor = min(lstm_signal * 0.72, 0.52)
            final_conf = max(final_conf, lstm_floor)

        final_conf = float(np.clip(final_conf * combined_modifier, 0.0, 1.0))

        # Apply hybrid regime confidence ← NEW
        final_conf = float(np.clip(final_conf * regime_confidence, 0.0, 1.0))
        print(f"      - Regime confidence: {regime_confidence:.2f}x -> {final_conf:.4f}")

        # -- FinBERT Gates -----------------------------------------------------
        sentiment_available = abs(sent_score) > 0.001
        if not sentiment_available:
            print("      [WARN] [Fusion] Sentiment frozen at 0.0 HOLD gates disabled")
        if sentiment_available and sent_score < -0.05 and lstm_signal > 0.55:
            print(f"      [Fusion] FinBERT veto: negative sentiment ({sent_score:.3f})")
            final_conf = min(final_conf, 0.54)
        if sentiment_available and abs(sent_score) < 0.05 and lstm_signal > 0.65:
            final_conf = final_conf * 0.95

        lstm_regime_agree_bull = (lstm_signal > 0.58 and regime_label == "Bull" and sent_score > 0.03)
        lstm_regime_agree_bear = (lstm_signal < 0.42 and regime_label == "Bear" and sent_score < -0.03)
        if lstm_regime_agree_bull or lstm_regime_agree_bear:
            final_conf = min(final_conf * 1.08, 0.75)
            print(f"      [Fusion] Consensus boost -> {final_conf:.4f}")

        print(f"      - Raw Fusion Confidence: {final_conf:.4f}")

        # -- Phase 13: Conflict Resolution -------------------------------------
        if self.conflict_resolver:
            arbitration_result = self.conflict_resolver.arbitrate(
                tech_score=lstm_signal, sent_score=sent_score, mc_std=mc_std,
                regime_label=regime_label, risk_score=risk_score,
                fusion_confidence=final_conf, trust_scores=trust_scores,
            )
            final_conf = arbitration_result["adjusted_confidence"]
            self.conflict_resolver.print_report(arbitration_result)
        else:
            if risk_score > DIVERGENCE_THRESHOLD_CRITICAL:
                final_conf *= 0.5
            if mc_std > 0.10:
                final_conf *= 0.8

        # -- Phase 16: Disagreement Heatmap ------------------------------------
        gdi_penalty = 1.0
        gdi_value   = 0.0
        if self.heatmap_agent:
            heatmap_result = self.heatmap_agent.analyze(
                lstm_score=lstm_signal, sent_score=sent_score,
                regime_label=regime_label, regime_vol=current_vol,
            )
            self.heatmap_agent.print_heatmap(heatmap_result)
            gdi_penalty = heatmap_result["penalty"]
            gdi_value   = heatmap_result["gdi"] * 100

        # -- Risk Management ---------------------------------------------------
        print("\n   [Risk Engine] Calculating Position Sizing (Kelly)...")
        alloc_pct, kelly_debug = self.risk_engine.calculate_position_size(
    final_conf, current_vol, disagreement_penalty=gdi_penalty,
    regime=regime_label, stock_price=last_price,   # ← v2.2: min $50 check
)
        num_shares, cash_value = self.risk_engine.get_shares_amount(last_price, alloc_pct)

        # -- Final Report ------------------------------------------------------
        print("\n" + "█" * 72)
        print(f"🏆 FINFOLIO-X INTELLIGENCE REPORT: {ticker}")
        print("█" * 72)
        print(f"   📊 AI Confidence Score  : {final_conf:.4f} (Scale: 0.0 - 1.0)")
        print(f"   🎲 Model Uncertainty    : {mc_std:.4f} ({uncertainty_status})")
        print(f"   ⛈️  Market Regime        : {regime_label} (Vol: {current_vol:.4f}, HybridConf: {regime_confidence:.2f})")
        print(f"   🕸️  Systemic Risk        : {risk_score:.4f} ({div_status})")
        print(f"   🌀 Topological Shape    : {topo_signal} (Mod: {topo_modifier:.2f}x)")
        print(f"   🔗 Causal Dynamics      : Score={causal_score:.4f} (Conf: {len(causal_result.get('confounders_removed', []))})")
        print(f"   [CHECK] Primary SHAP Driver  : {top_driver}")
        print("-" * 72)

        effective_threshold = COMMODITY_BUY_THRESHOLD if ticker in COMMODITY_TICKERS else BUY_CONFIDENCE_THRESHOLD
        decision = "HOLD"
        if alloc_pct > 0.0 and final_conf >= effective_threshold and regime_label != "Bear" and gdi_value < BUY_GDI_MAX:
            decision = "BUY 🟢"
        elif final_conf < 0.40 and regime_label != "Bull":
            decision = "SELL 🔴"
        elif final_conf < 0.40 and regime_label == "Bull":
            decision = "HOLD"

        print(f"   🚀 STRATEGY SIGNAL      : {decision}")
        if decision == "BUY 🟢":
            print(f"   💰 RECOMMENDED SIZE     : ${cash_value:.2f}")
            print(f"   📉 PORTFOLIO WEIGHT     : {alloc_pct * 100:.1f}%")
            print(f"   📦 ORDER QUANTITY       : {num_shares} Shares (@ ${last_price:.2f})")
            print(f"   🧮 KELLY EDGE           : {kelly_debug:.4f}")
        else:
            print("   ⛔ RISK ADVICE          : Stay Cash / Do Not Enter Trade.")

        w_lstm = weights.get("LSTM_Focus", 0)
        w_sent = weights.get("Sentiment_Focus", 0)
        w_vol  = weights.get("Volatility_Focus", 0)
        print("-" * 72)
        print("   [CHECK] AI REASONING (ATTENTION WEIGHTS):")
        print(f"      • Technicals (Chart) : {w_lstm:.2f}")
        print(f"      • Sentiment (News)   : {w_sent:.2f}")
        print(f"      • Risk (Volatility)  : {w_vol:.2f}")
        max_focus = max(w_lstm, w_sent, w_vol)
        if max_focus == w_lstm:
            focus_msg = "The AI is prioritizing the Price Trend."
        elif max_focus == w_sent:
            focus_msg = "The AI is prioritizing News/Sentiment."
        else:
            focus_msg = "The AI is prioritizing Risk Management (Defensive)."
        print(f"      👉 Insight: {focus_msg}")
        print("█" * 72)
        print("\n   Disclaimer: This tool is for educational purposes only.")
        print("   It does not constitute financial advice. Trading involves risk.")
        print("   (c) FinFolio-X Team 2026")

        if self.meta_agent:
            try:
                self.meta_agent.log_decision(
                    ticker=ticker, lstm_score=lstm_signal, sent_score=sent_score,
                    regime_label=regime_label, risk_score=risk_score,
                    fusion_confidence=final_conf, final_decision=decision,
                    price_at_decision=last_price,
                )
            except Exception as e:
                print(f"   [!] Meta-Agent logging failed: {e}")

    def run_stress_test(self, ticker="AAPL"):
        if self.red_team:
            self.red_team.run_robustness_test(ticker)
        else:
            print("[BAD] Cannot run stress test: Phase 11 module not loaded.")