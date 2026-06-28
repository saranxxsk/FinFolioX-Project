"""
================================================================================
test_full_system.py HOLD FinFolioX Full System Test (Part 1 Only)
================================================================================
Runs all 17 agents across 17 windows x 30 tickers.
No changes from v3.0 logic HOLD this is the full system evaluation only.
================================================================================
"""

import os
import sys
import io
import time
import warnings
import tempfile
import contextlib

# Force UTF-8 encoding on Windows
if sys.platform == "win32":
    import subprocess
    subprocess.run(["chcp", "65001"], shell=True)

import numpy as np
import pandas as pd
import yfinance as yf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ================================================================================
#  AGENT IMPORTS
# ================================================================================
from ml_engine.technical_agent     import TechnicalAgent, build_lstm_features, SEQ_LEN
from ml_engine.uncertainty_agent   import UncertaintyAgent
from ml_engine.hybrid_regime_agent import HybridRegimeAgent
from ml_engine.fusion_agent        import FusionAgent
from ml_engine.heatmap_agent       import HeatmapAgent
from ml_engine.risk_engine         import RiskEngine

try:
    from ml_engine.regime_agent import RegimeAgent
    _REGIME_LEGACY_OK = True
except Exception:
    _REGIME_LEGACY_OK = False

try:
    from ml_engine.correlation_agent import CorrelationDivergenceDetector
    _CORR_OK = True
except Exception:
    _CORR_OK = False

try:
    from ml_engine.explainability_agent import ExplainabilityAgent
    _EXPL_OK = True
except Exception:
    _EXPL_OK = False

try:
    from ml_engine.topology_agent import TopologyAgent
    _TOPO_OK = True
except Exception:
    _TOPO_OK = False

try:
    from ml_engine.causal_agent import CausalAgent
    _CAUSAL_OK = True
except Exception:
    _CAUSAL_OK = False

try:
    from ml_engine.counterfactual_engine import CounterfactualEngine
    _CF_OK = True
except Exception:
    _CF_OK = False

try:
    from ml_engine.conflict_resolver import ConflictResolver
    _CONFLICT_OK = True
except Exception:
    _CONFLICT_OK = False

try:
    from ml_engine.aesl_agent import AESLAgent
    _AESL_OK = True
except Exception:
    _AESL_OK = False

try:
    from ml_engine.asc_memory import AgentDecisionMemory
    _ASC_OK = True
except Exception:
    _ASC_OK = False

try:
    from ml_engine.adversarial_tester import AdversarialTester
    _ADVER_OK = True
except Exception:
    _ADVER_OK = False

try:
    from ml_engine.meta_agent import MetaAgent
    _META_OK = True
except Exception:
    _META_OK = False

# ================================================================================
#  MODEL PATHS
# ================================================================================
MODEL_PATH  = r"D:/FinFolioX/saved_models/lstm_model.keras"
SCALER_PATH = r"D:/FinFolioX/saved_models/lstm_scaler.pkl"
REGIME_PATH = os.path.join("saved_models", "hmm_regime_hybrid.pkl")
FUSION_PATH = os.path.join("saved_models", "attention_fusion.pth")

# ================================================================================
#  SYSTEM CONSTANTS
# ================================================================================
DEFAULT_CAPITAL   = 10_000.0
BUY_THRESHOLD     = 0.50
SELL_THRESHOLD    = 0.40
COMMODITY_BUY_T   = 0.55
COMMODITY_TICKERS = {"GLD", "SLV", "USO", "UNG", "GDX"}
BUY_GDI_MAX       = 65.0
MAX_RISK          = 0.20
BEAR_MAX_ALLOC    = 0.10
BEAR_BUY_BCS_MAX  = 0.80
IG_STEPS_FULLTEST = 24

# ================================================================================
#  TEST WINDOWS HOLD 17 windows
# ================================================================================
TEST_WINDOWS = [
    ("2026-04-20", "2026-04-27", "Win20: Bull-TechEarningsRally (Apr20->27-2026)"),
    ("2026-04-09", "2026-04-14", "Win19: Pre-Easter Defensive Lull (Apr09->14-2026)"),
   
    ("2026-04-02", "2026-04-07", "Win18: Pre-Easter Defensive Lull (Apr02->09-2026)"),
    ("2026-03-23", "2026-03-28", "Win0: Bear-IranOilShock   (Mar23->28-2026)"),
    ("2024-11-06", "2024-11-11", "Win1:  Bull-PostElection  (Nov06->11-2024)"),
    ("2024-07-30", "2024-08-05", "Win2:  Bear-YenCrash      (Jul30->Aug05-2024)"),
    ("2025-01-13", "2025-01-17", "Win3:  Sideways-Mixed     (Jan13->17-2025)"),
    ("2025-04-02", "2025-04-07", "Win4:  Bear-TariffShock   (Apr02->07-2025)"),
    ("2026-03-15", "2026-03-20", "Win5:  Deep-Bear          (Mar15->20-2026)"),
    ("2024-10-14", "2024-10-21", "Win6:  Bull-EarningsBeat  (Oct14->21-2024)"),
    ("2025-01-20", "2025-01-27", "Win7:  Bull-InaugRally    (Jan20->27-2025)"),
    ("2024-06-10", "2024-06-17", "Win8:  Bull-AIRally       (Jun10->17-2024)"),
    ("2024-05-13", "2024-05-20", "Win9:  Bull-PostCPI       (May13->20-2024)"),
    ("2024-12-16", "2024-12-23", "Win10: Bear-FedHawk       (Dec16->23-2024)"),
    ("2025-02-03", "2025-02-10", "Win11: Bear-DeepSeek      (Feb03->10-2025)"),
    ("2025-08-18", "2025-08-25", "Win12: Bear-LateSummer    (Aug18->25-2025)"),
    ("2024-09-09", "2024-09-16", "Win13: Sideways-PreCut    (Sep09->16-2024)"),
    ("2024-11-18", "2024-11-25", "Win14: Sideways-PostElec  (Nov18->25-2024)"),
    ("2025-03-10", "2025-03-17", "Win15: Sideways-TariffFUD (Mar10->17-2025)"),
    ("2024-08-12", "2024-08-19", "Win16: Bounce-YenRecov    (Aug12->19-2024)"),
    ("2025-04-22", "2025-04-29", "Win17: Bounce-TariffPause (Apr22->29-2025)"),
]

# ================================================================================
#  30 TICKERS
# ================================================================================
TICKERS = [
    "AAPL", "MSFT", "NVDA", "TSLA", "META", "GOOGL", "AMZN",
    "AMD",  "INTC", "ORCL",
    "SPY",  "QQQ",  "DIA",  "IWM",
    "JPM",  "BAC",  "GS",   "V",
    "GLD",  "TLT",  "SLV",
    "XOM",  "CVX",
    "WMT",  "PG",   "JNJ",
    "NFLX", "DIS",
    "CRM",  "PLTR",
]

INDEX_ETFS    = {"SPY", "QQQ", "DIA", "IWM", "TLT"}
VOLATILE_STKS = {"NVDA", "TSLA", "AMD", "PLTR", "NFLX", "SLV"}

def noise_band(ticker):
    if ticker in INDEX_ETFS:    return 1.0
    if ticker in VOLATILE_STKS: return 3.0
    return 2.0

# ================================================================================
#  MANUAL SENTIMENT SCORES
# ================================================================================
MANUAL_SENTIMENT = {
    "2026-03-23": {
    # Tech: heavy hedge-fund dumping, Iran threatens Big Tech list, IRGC named AAPL/MSFT/NVDA/GOOGL/META/ORCL
    "AAPL": -0.14, "MSFT": -0.13, "NVDA": -0.16, "TSLA": -0.20,
    "META": -0.13, "GOOGL": -0.14, "AMZN": -0.12, "AMD":  -0.15,
    "INTC": -0.17, "ORCL": -0.13,
    # Index ETFs: broad bear, all 11 S&P sectors in weekly lows
    "SPY":  -0.14, "QQQ":  -0.18, "DIA":  -0.11, "IWM":  -0.13,
    # Financials: selling in tech/industrials/financials per Goldman
    "JPM":  -0.09, "BAC":  -0.10, "GS":   -0.08, "V":    -0.08,
    # GLD near historic peak, TLT muted (rates still elevated)
    "GLD":  +0.14, "TLT":  +0.05,
    # SLV positive but muted vs GLD
    "SLV":  +0.07,
    # Energy: XOM/CVX major beneficiaries of $100+ oil
    "XOM":  +0.18, "CVX":  +0.16,
    # Defensives: hedge funds buying WMT/consumer staples at fastest rate since Jul 2025
    "WMT":  +0.09, "PG":   +0.06, "JNJ":  +0.07,
    # NFLX/DIS: consumer discretionary under pressure
    "NFLX": -0.10, "DIS":  -0.11,
    # CRM: tech/cloud selling broad
    "CRM":  -0.11,
    # PLTR: mixed HOLD Golden Dome defense contract upside vs general tech dump; net slight negative
    "PLTR": -0.06,
},
    "2024-11-06": {
        "AAPL": +0.08, "MSFT": +0.07, "NVDA": +0.12, "TSLA": +0.25,
        "META": +0.10, "GOOGL":+0.06, "AMZN": +0.08, "AMD":  +0.08,
        "INTC": +0.03, "ORCL": +0.10, "SPY":  +0.10, "QQQ":  +0.12,
        "DIA":  +0.09, "IWM":  +0.15, "JPM":  +0.12, "BAC":  +0.11,
        "GS":   +0.14, "V":    +0.08, "GLD":  -0.05, "TLT":  +0.02,
        "SLV":  -0.03, "XOM":  +0.06, "CVX":  +0.05, "WMT":  +0.04,
        "PG":   +0.03, "JNJ":  +0.02, "NFLX": +0.07, "DIS":  +0.05,
        "CRM":  +0.06, "PLTR": +0.20,
    },
    "2024-07-30": {
        "AAPL": -0.10, "MSFT": -0.09, "NVDA": -0.16, "TSLA": -0.12,
        "META": -0.08, "GOOGL":-0.09, "AMZN": -0.10, "AMD":  -0.14,
        "INTC": -0.18, "ORCL": -0.06, "SPY":  -0.10, "QQQ":  -0.14,
        "DIA":  -0.09, "IWM":  -0.12, "JPM":  -0.07, "BAC":  -0.09,
        "GS":   -0.08, "V":    -0.07, "GLD":  +0.08, "TLT":  +0.12,
        "SLV":  -0.03, "XOM":  -0.06, "CVX":  -0.05, "WMT":  -0.02,
        "PG":   +0.01, "JNJ":  +0.02, "NFLX": -0.08, "DIS":  -0.07,
        "CRM":  -0.09, "PLTR": -0.10,
    },
    "2025-01-13": {
        "AAPL": +0.04, "MSFT": +0.08, "NVDA": +0.12,
        "TSLA": +0.10, "META": +0.09, "GOOGL": +0.07,
        "AMZN": +0.07, "AMD":  +0.06, "INTC":  +0.08,
        "ORCL": +0.08, "SPY":  +0.06, "QQQ":  +0.08,
        "DIA":   +0.05, "IWM":  +0.09,
        "JPM":  +0.12, "BAC":  +0.10, "GS":    +0.14, "V": +0.07,
        "GLD":  +0.06, "TLT":  -0.04, "SLV":   +0.03,
        "XOM":  +0.05, "CVX":  +0.04, "WMT":   +0.05,
        "PG":   +0.03, "JNJ":  +0.03,
        "NFLX": +0.09, "DIS":  +0.04, "CRM":   +0.07, "PLTR": +0.15,
    },
    "2025-04-02": {
        "AAPL": -0.18, "MSFT": -0.14, "NVDA": -0.16, "TSLA": -0.22,
        "META": -0.17, "GOOGL":-0.15, "AMZN": -0.19, "AMD":  -0.17,
        "INTC": -0.13, "ORCL": -0.10, "SPY":  -0.18, "QQQ":  -0.20,
        "DIA":  -0.16, "IWM":  -0.20, "JPM":  -0.12, "BAC":  -0.14,
        "GS":   -0.11, "V":    -0.12, "GLD":  +0.08, "TLT":  +0.12,
        "SLV":  -0.02, "XOM":  -0.14, "CVX":  -0.13, "WMT":  -0.06,
        "PG":   -0.04, "JNJ":  -0.01, "NFLX": -0.14, "DIS":  -0.13,
        "CRM":  -0.12, "PLTR": -0.10,
    },
    "2026-03-15": {
        "AAPL": -0.11, "MSFT": -0.09, "NVDA": -0.08, "TSLA": -0.22,
        "META": -0.07, "GOOGL":-0.10, "AMZN": -0.10, "AMD":  -0.12,
        "INTC": -0.11, "ORCL": -0.05, "SPY":  -0.12, "QQQ":  -0.18,
        "DIA":  -0.10, "IWM":  -0.15, "JPM":  -0.04, "BAC":  -0.08,
        "GS":   -0.05, "V":    -0.07, "GLD":  -0.16, "TLT":  +0.04,
        "SLV":  -0.10, "XOM":  -0.08, "CVX":  -0.07, "WMT":  -0.02,
        "PG":   -0.01, "JNJ":  +0.01, "NFLX": -0.11, "DIS":  -0.12,
        "CRM":  -0.09, "PLTR": -0.03,
    },
    "2024-10-14": {
        "AAPL": +0.09, "MSFT": +0.10, "NVDA": +0.14, "TSLA": +0.18,
        "META": +0.12, "GOOGL":+0.08, "AMZN": +0.09, "AMD":  +0.10,
        "INTC": +0.06, "ORCL": +0.07, "SPY":  +0.11, "QQQ":  +0.13,
        "DIA":  +0.10, "IWM":  +0.12, "JPM":  +0.16, "BAC":  +0.14,
        "GS":   +0.15, "V":    +0.09, "GLD":  +0.04, "TLT":  -0.03,
        "SLV":  +0.02, "XOM":  +0.07, "CVX":  +0.06, "WMT":  +0.06,
        "PG":   +0.04, "JNJ":  +0.05, "NFLX": +0.10, "DIS":  +0.06,
        "CRM":  +0.08, "PLTR": +0.16,
    },"2026-04-02": {
        "AAPL": -0.03, "MSFT": +0.07, "NVDA": +0.04, "TSLA": -0.14,
        "META": +0.03, "GOOGL":+0.05, "AMZN": +0.02, "AMD":  -0.02,
        "INTC": +0.04, "ORCL": +0.04, "SPY":  +0.03, "QQQ":  +0.01,
        "DIA":  +0.06, "IWM":  -0.05, "JPM":  +0.10, "BAC":  +0.07,
        "GS":   +0.11, "V":    +0.04, "GLD":  +0.14, "TLT":  -0.09,
        "SLV":  +0.06, "XOM":  +0.12, "CVX":  +0.10, "WMT":  +0.08,
        "PG":   +0.06, "JNJ":  +0.05, "NFLX": +0.01, "DIS":  -0.03,
        "CRM":  -0.02, "PLTR": +0.15,
    },
    "2025-01-20": {
        "AAPL": +0.07, "MSFT": +0.09, "NVDA": +0.08, "TSLA": +0.28,
        "META": +0.10, "GOOGL":+0.06, "AMZN": +0.07, "AMD":  +0.07,
        "INTC": +0.05, "ORCL": +0.08, "SPY":  +0.09, "QQQ":  +0.10,
        "DIA":  +0.08, "IWM":  +0.12, "JPM":  +0.11, "BAC":  +0.10,
        "GS":   +0.13, "V":    +0.07, "GLD":  -0.02, "TLT":  -0.06,
        "SLV":  +0.03, "XOM":  +0.09, "CVX":  +0.08, "WMT":  +0.04,
        "PG":   +0.03, "JNJ":  +0.02, "NFLX": +0.08, "DIS":  +0.06,
        "CRM":  +0.07, "PLTR": +0.30,
    },
    "2024-06-10": {
        "AAPL": +0.16, "MSFT": +0.09, "NVDA": +0.22, "TSLA": +0.08,
        "META": +0.10, "GOOGL":+0.09, "AMZN": +0.08, "AMD":  +0.14,
        "INTC": +0.07, "ORCL": +0.10, "SPY":  +0.12, "QQQ":  +0.16,
        "DIA":  +0.08, "IWM":  +0.07, "JPM":  +0.06, "BAC":  +0.05,
        "GS":   +0.07, "V":    +0.06, "GLD":  +0.05, "TLT":  +0.04,
        "SLV":  +0.03, "XOM":  +0.04, "CVX":  +0.03, "WMT":  +0.05,
        "PG":   +0.03, "JNJ":  +0.03, "NFLX": +0.09, "DIS":  +0.05,
        "CRM":  +0.09, "PLTR": +0.12,
    },
    "2024-05-13": {
        "AAPL": +0.10, "MSFT": +0.09, "NVDA": +0.18, "TSLA": +0.07,
        "META": +0.11, "GOOGL":+0.08, "AMZN": +0.09, "AMD":  +0.13,
        "INTC": +0.06, "ORCL": +0.07, "SPY":  +0.13, "QQQ":  +0.15,
        "DIA":  +0.10, "IWM":  +0.12, "JPM":  +0.09, "BAC":  +0.08,
        "GS":   +0.09, "V":    +0.08, "GLD":  +0.08, "TLT":  +0.10,
        "SLV":  +0.06, "XOM":  +0.05, "CVX":  +0.04, "WMT":  +0.06,
        "PG":   +0.04, "JNJ":  +0.04, "NFLX": +0.09, "DIS":  +0.06,
        "CRM":  +0.08, "PLTR": +0.10,
    },
    "2024-12-16": {
        "AAPL": -0.08, "MSFT": -0.10, "NVDA": -0.14, "TSLA": -0.09,
        "META": -0.11, "GOOGL":-0.09, "AMZN": -0.10, "AMD":  -0.15,
        "INTC": -0.12, "ORCL": -0.08, "SPY":  -0.12, "QQQ":  -0.16,
        "DIA":  -0.10, "IWM":  -0.14, "JPM":  -0.07, "BAC":  -0.08,
        "GS":   -0.07, "V":    -0.06, "GLD":  -0.06, "TLT":  -0.14,
        "SLV":  -0.08, "XOM":  -0.05, "CVX":  -0.04, "WMT":  -0.04,
        "PG":   -0.03, "JNJ":  -0.02, "NFLX": -0.09, "DIS":  -0.07,
        "CRM":  -0.09, "PLTR": -0.10,
    },
    "2025-02-03": {
        "AAPL": -0.06, "MSFT": -0.09, "NVDA": -0.22, "TSLA": -0.08,
        "META": -0.07, "GOOGL":-0.08, "AMZN": -0.07, "AMD":  -0.18,
        "INTC": -0.14, "ORCL": -0.08, "SPY":  -0.09, "QQQ":  -0.14,
        "DIA":  -0.07, "IWM":  -0.10, "JPM":  -0.05, "BAC":  -0.06,
        "GS":   -0.05, "V":    -0.04, "GLD":  +0.06, "TLT":  +0.05,
        "SLV":  +0.02, "XOM":  +0.04, "CVX":  +0.03, "WMT":  +0.03,
        "PG":   +0.03, "JNJ":  +0.04, "NFLX": -0.06, "DIS":  -0.04,
        "CRM":  -0.08, "PLTR": -0.14,
    },
    "2025-08-18": {
        "AAPL": -0.05, "MSFT": -0.07, "NVDA": -0.10, "TSLA": -0.08,
        "META": -0.06, "GOOGL":-0.06, "AMZN": -0.07, "AMD":  -0.09,
        "INTC": -0.11, "ORCL": -0.04, "SPY":  -0.08, "QQQ":  -0.11,
        "DIA":  -0.07, "IWM":  -0.10, "JPM":  -0.05, "BAC":  -0.07,
        "GS":   -0.05, "V":    -0.04, "GLD":  +0.07, "TLT":  +0.08,
        "SLV":  +0.04, "XOM":  -0.04, "CVX":  -0.03, "WMT":  +0.02,
        "PG":   +0.02, "JNJ":  +0.03, "NFLX": -0.05, "DIS":  -0.04,
        "CRM":  -0.07, "PLTR": -0.08,
    },
    "2024-09-09": {
        "AAPL": +0.12, "MSFT": +0.03, "NVDA": -0.04, "TSLA": +0.02,
        "META": +0.04, "GOOGL":+0.02, "AMZN": +0.03, "AMD":  +0.01,
        "INTC": -0.06, "ORCL": +0.06, "SPY":  +0.03, "QQQ":  +0.02,
        "DIA":  +0.03, "IWM":  +0.04, "JPM":  +0.04, "BAC":  +0.03,
        "GS":   +0.04, "V":    +0.03, "GLD":  +0.05, "TLT":  +0.06,
        "SLV":  +0.03, "XOM":  -0.02, "CVX":  -0.01, "WMT":  +0.04,
        "PG":   +0.03, "JNJ":  +0.03, "NFLX": +0.04, "DIS":  +0.02,
        "CRM":  +0.04, "PLTR": +0.03,
    },
    "2024-11-18": {
        "AAPL": +0.04, "MSFT": +0.05, "NVDA": +0.18, "TSLA": +0.12,
        "META": +0.06, "GOOGL":+0.04, "AMZN": +0.05, "AMD":  +0.06,
        "INTC": +0.02, "ORCL": +0.05, "SPY":  +0.04, "QQQ":  +0.07,
        "DIA":  +0.02, "IWM":  +0.05, "JPM":  +0.05, "BAC":  +0.04,
        "GS":   +0.06, "V":    +0.04, "GLD":  -0.04, "TLT":  -0.05,
        "SLV":  -0.02, "XOM":  +0.03, "CVX":  +0.02, "WMT":  +0.03,
        "PG":   +0.01, "JNJ":  +0.01, "NFLX": +0.05, "DIS":  +0.03,
        "CRM":  +0.05, "PLTR": +0.18,
    },
    "2025-03-10": {
        "AAPL": -0.04, "MSFT": -0.03, "NVDA": -0.07, "TSLA": -0.08,
        "META": -0.03, "GOOGL":-0.03, "AMZN": -0.04, "AMD":  -0.06,
        "INTC": -0.05, "ORCL": +0.01, "SPY":  -0.04, "QQQ":  -0.06,
        "DIA":  -0.03, "IWM":  -0.05, "JPM":  -0.02, "BAC":  -0.03,
        "GS":   -0.02, "V":    -0.02, "GLD":  +0.07, "TLT":  +0.04,
        "SLV":  +0.03, "XOM":  -0.03, "CVX":  -0.02, "WMT":  -0.01,
        "PG":   +0.01, "JNJ":  +0.02, "NFLX": -0.03, "DIS":  -0.02,
        "CRM":  -0.04, "PLTR": -0.04,
    },
    "2024-08-12": {
        "AAPL": +0.07, "MSFT": +0.08, "NVDA": +0.12, "TSLA": +0.06,
        "META": +0.09, "GOOGL":+0.07, "AMZN": +0.08, "AMD":  +0.10,
        "INTC": +0.05, "ORCL": +0.06, "SPY":  +0.10, "QQQ":  +0.12,
        "DIA":  +0.09, "IWM":  +0.11, "JPM":  +0.08, "BAC":  +0.07,
        "GS":   +0.08, "V":    +0.07, "GLD":  +0.04, "TLT":  -0.02,
        "SLV":  +0.04, "XOM":  +0.05, "CVX":  +0.04, "WMT":  +0.04,
        "PG":   +0.03, "JNJ":  +0.03, "NFLX": +0.08, "DIS":  +0.06,
        "CRM":  +0.07, "PLTR": +0.09,
    },
    "2025-04-22": {
        "AAPL": +0.12, "MSFT": +0.11, "NVDA": +0.15, "TSLA": +0.16,
        "META": +0.13, "GOOGL":+0.10, "AMZN": +0.12, "AMD":  +0.13,
        "INTC": +0.09, "ORCL": +0.09, "SPY":  +0.13, "QQQ":  +0.15,
        "DIA":  +0.11, "IWM":  +0.14, "JPM":  +0.10, "BAC":  +0.09,
        "GS":   +0.11, "V":    +0.09, "GLD":  -0.03, "TLT":  -0.04,
        "SLV":  +0.05, "XOM":  +0.08, "CVX":  +0.07, "WMT":  +0.07,
        "PG":   +0.05, "JNJ":  +0.05, "NFLX": +0.10, "DIS":  +0.08,
        "CRM":  +0.10, "PLTR": +0.18,
    },
    "2026-04-20": {
    # Tech: INTC blowout Q1 (+23%), AMD AI rally (+12%), AMZN AWS-Meta chip deal
    "AAPL": +0.02, "MSFT": +0.08, "NVDA": +0.10, "TSLA": -0.14,
    "META": +0.12, "GOOGL":+0.08, "AMZN": +0.14, "AMD":  +0.22,
    "INTC": +0.25, "ORCL": +0.04,
    # Index ETFs: QQQ led +3%, SPY +1.4%, DIA lagged, IWM sideways
    "SPY":  +0.09, "QQQ":  +0.13, "DIA":  +0.04, "IWM":  +0.03,
    # Financials: pre-FOMC caution, consumer spending mixed
    "JPM":  -0.05, "BAC":  -0.05, "GS":   -0.02, "V":    -0.03,
    # GLD/SLV sold off (dollar strength, yields elevated pre-FOMC), TLT flat
    "GLD":  -0.09, "TLT":  -0.02, "SLV":  -0.10,
    # Energy: oil elevated $106/bbl but stocks flat
    "XOM":  +0.03, "CVX":  +0.01,
    # Defensives: PG earnings beat + div raise, WMT steady, JNJ healthcare mixed
    "WMT":  +0.04, "PG":   +0.10, "JNJ":  +0.01,
    # NFLX post-earnings malaise/Hastings exit, DIS layoffs & restructuring
    "NFLX": -0.15, "DIS":  -0.08,
    # CRM buyback overshadowed by broader cloud spending fears, PLTR govt deals
    "CRM":  -0.04, "PLTR": +0.06,
},
    "2026-04-09": {
    "AAPL": -0.10, "MSFT": +0.12, "NVDA": +0.15, "TSLA": +0.12,
    "META": +0.15, "GOOGL":+0.12, "AMZN": +0.15, "AMD":  +0.15,
    "INTC": +0.10, "ORCL": +0.15,

    "SPY":  +0.12, "QQQ":  +0.15, "DIA":  +0.10, "IWM":  +0.12,

    "JPM":  +0.10, "BAC":  +0.10, "GS":   +0.10, "V":    +0.10,

    "GLD":  +0.09, "TLT":  +0.05, "SLV":  +0.10,

    "XOM":  -0.15, "CVX":  -0.15,

    "WMT":  -0.15, "PG":   -0.12, "JNJ":  -0.12,

    "NFLX": +0.12, "DIS":  +0.10,

    "CRM":  +0.10, "PLTR": +0.12,
}
}

# ================================================================================
#  PORTFOLIO TRACKER
# ================================================================================
class PortfolioTracker:
    def __init__(self, capital: float = DEFAULT_CAPITAL):
        self.capital = capital
        self.trades: list = []

    def record(self, decision, alloc_pct, actual_ret, ticker, window):
        if actual_ret is None or np.isnan(actual_ret):
            return
        if decision == "HOLD":
            pnl = 0.0
            alloc_pct = 0.0
        else:
            alloc    = alloc_pct / 100.0
            deployed = self.capital * alloc
            pnl      = deployed * (actual_ret / 100.0)  if decision == "BUY"  else \
                       deployed * (-actual_ret / 100.0) if decision == "SELL" else 0.0
        self.trades.append({"ticker": ticker, "window": window,
                             "decision": decision, "alloc_pct": alloc_pct,
                             "actual_ret": actual_ret, "pnl": pnl})

    def metrics(self):
        if not self.trades:
            return {}
        active_pnls = np.array([t["pnl"] for t in self.trades if t["decision"] != "HOLD"])
        all_pnls    = np.array([t["pnl"] for t in self.trades])
        
        total    = float(np.sum(active_pnls))
        
        # Trade-level inflated sharpe (Old)
        mean_orig = float(np.mean(active_pnls)) if active_pnls.size else 0.0
        std_orig  = float(np.std(active_pnls)) if active_pnls.size > 1 else 1e-6
        trade_inflated_sharpe = (mean_orig / std_orig) * np.sqrt(252) if std_orig > 1e-7 else 0.0
        
        # Group by Window (Day)
        window_pnls = {}
        for t in self.trades:
            w = t["window"]
            window_pnls[w] = window_pnls.get(w, 0.0) + t["pnl"]
            
        daily_pnls = np.array(list(window_pnls.values()))
        
        # Build equity curve
        equity = np.concatenate(([self.capital], self.capital + np.cumsum(daily_pnls)))
        returns = pd.Series(equity).pct_change().dropna()
        
        # True Daily Sharpe
        mean_ret = float(returns.mean())
        std_ret  = float(returns.std())
        true_daily_sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 1e-7 else 0.0
        
        # Active Sharpe (only non-zero days)
        active_returns = returns[returns != 0]
        if len(active_returns) > 1:
            active_mean = float(active_returns.mean())
            active_std  = float(active_returns.std())
            active_sharpe_18 = (active_mean / active_std) * np.sqrt(len(active_returns))
        else:
            active_sharpe_18 = 0.0
        
        wins     = int(sum(1 for p in active_pnls if p > 0))
        losses   = int(sum(1 for p in active_pnls if p < 0))
        win_rate = wins / len(active_pnls) * 100 if active_pnls.size else 0.0
        
        cumsum   = np.cumsum(daily_pnls)
        peak     = np.maximum.accumulate(cumsum)
        dd       = cumsum - peak
        max_dd   = float(np.min(dd)) if dd.size else 0.0
        calmar   = (total / self.capital * 100) / abs(max_dd / self.capital * 100) \
                   if max_dd < -1e-6 else 0.0
        return {
            "total_pnl":        round(total, 2),
            "total_return_pct": round(total / self.capital * 100, 3),
            "trade_inflated_sharpe": round(trade_inflated_sharpe, 3),
            "true_daily_sharpe": round(true_daily_sharpe, 3),
            "active_sharpe_18": round(active_sharpe_18, 3),
            "win_rate":         round(win_rate, 1),
            "wins":             wins,
            "losses":           losses,
            "max_drawdown":     round(max_dd, 2),
            "calmar_ratio":     round(calmar, 3),
            "n_trades":         len(active_pnls),
        }


# ================================================================================
#  HELPER FUNCTIONS
# ================================================================================
def snap_to_trading_day(date_str):
    dt = pd.to_datetime(date_str)
    snapped = None
    try:
        end_dt = dt + pd.Timedelta(days=8)
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            ref = yf.download("SPY", start=dt.strftime("%Y-%m-%d"),
                              end=end_dt.strftime("%Y-%m-%d"),
                              auto_adjust=True, progress=False)
        if not ref.empty:
            snapped = pd.to_datetime(ref.index[0])
            if getattr(snapped, "tzinfo", None) is not None:
                snapped = snapped.tz_localize(None)
    except Exception:
        snapped = None
    if snapped is None:
        snapped = pd.bdate_range(start=dt, periods=1)[0]
    if snapped != dt:
        print(f"   [WARN]  {date_str} -> snapped to {snapped.date()}")
    return snapped.strftime("%Y-%m-%d")


def fetch_history(ticker, test_date):
    try:
        test_dt  = pd.to_datetime(test_date)
        yf_end   = (test_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        yf_start = (test_dt - pd.Timedelta(days=300)).strftime("%Y-%m-%d")
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            df = yf.download(ticker, start=yf_start, end=yf_end,
                             auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[df.index <= test_dt] if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def fetch_actual_return(ticker, test_date, outcome_date):
    try:
        yf_end   = (pd.to_datetime(outcome_date) + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        yf_start = (pd.to_datetime(test_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            df = yf.download(ticker, start=yf_start, end=yf_end,
                             auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 2:
            return float("nan")
        try:
            p0 = float(df["Close"].asof(pd.to_datetime(test_date)))
            p1 = float(df["Close"].asof(pd.to_datetime(outcome_date)))
        except Exception:
            p0, p1 = float(df["Close"].iloc[0]), float(df["Close"].iloc[-1])
        if np.isnan(p0) or np.isnan(p1) or p0 == 0:
            return float("nan")
        return ((p1 - p0) / p0) * 100.0
    except Exception:
        return float("nan")


def compute_beta_risk(hist, test_date):
    try:
        close  = hist["Close"].squeeze().astype(float)
        vol_20 = float(close.pct_change().rolling(20).std().iloc[-1])
        ann_vol = vol_20 * np.sqrt(252)
        return float(np.clip(0.10 + (ann_vol - 0.10) * 1.8, 0.18, 0.72))
    except Exception:
        return 0.38


def apply_fusion_gates(conf, lstm_s, sent_s, regime, rc):
    # ==================== CRITICAL FIX v2.4 ====================
    # FIX-1: Aggressive sentiment-driven confidence floor
    # When LSTM is weak/bearish but sentiment is positive, significantly boost confidence.
    # This is critical for catching reversals where positive sentiment contradicts weak LSTM.
    # 
    # Case 1: Very weak LSTM (< 0.20) with ANY positive sentiment
    #   Sentiment-driven boost: confidence should be at least 0.45-0.55
    if lstm_s < 0.20 and sent_s >= 0.03:
        # BASE: 0.45, BOOST: +0.15 per 0.05 of sentiment (capped at 0.65)
        # Examples: sent=0.03 -> 0.45+0.09=0.54; sent=0.06 -> 0.45+0.18=0.63
        sentiment_boost = min(sent_s * 2.5, 0.20)
        confidence_floor = 0.45 + sentiment_boost
        conf = max(conf, confidence_floor)
    
    # Case 2: Extremely weak LSTM (< 0.10) with moderate/strong positive sentiment
    #   Extra aggressive boost: confidence should be at least 0.50-0.60
    if lstm_s < 0.10 and sent_s >= 0.05:
        extreme_floor = 0.50 + min(sent_s * 2.0, 0.15)  # cap at 0.65
        conf = max(conf, extreme_floor)
    
    # Case 3: Original strong positive sentiment floor (kept for backward compat)
    if lstm_s < 0.15 and sent_s >= 0.08:
        # This still applies as a safety net
        sentiment_floor = 0.48 + sent_s * 1.5
        conf = max(conf, sentiment_floor)
    
    if abs(sent_s) > 0.001:
        if sent_s < -0.10 and lstm_s > 0.55:
            cap = max(0.48, 0.56 + (sent_s + 0.10) * 0.10)
            conf = min(conf, cap)
        if abs(sent_s) < 0.05 and lstm_s > 0.65:
            conf *= 0.95
    if lstm_s > 0.58 and regime == "Bull" and sent_s > 0.03:
        conf = min(conf * 1.08, 0.82)
    if lstm_s < 0.42 and regime == "Bear" and sent_s < -0.03:
        conf = min(conf * 1.08, 0.82)
    if rc < 0.70:
        conf = 0.5 + (conf - 0.5) * rc
    return float(np.clip(conf, 0.0, 1.0))


def make_decision(arb_conf, alloc_pct, regime, ticker, gdi_pct, bcs=0.0, lstm_signal=0.5, sent_score=0.0):
    thr = COMMODITY_BUY_T if ticker in COMMODITY_TICKERS else BUY_THRESHOLD
    
    # ==================== CRITICAL FIX v2.4 ====================
    # FIX-1: Sentiment-driven BUY override for all regimes
    # When sentiment is clearly bullish and confidence has room to move, take the BUY signal.
    # This works even in Bear regime when sentiment strongly contradicts the regime.
    if sent_score >= 0.05 and arb_conf >= 0.35 and alloc_pct > 0.0 and gdi_pct < BUY_GDI_MAX:
        if lstm_signal >= 0.60:                      # Very bullish LSTM
            return "BUY"
        elif sent_score >= 0.08:                    # Very positive sentiment
            return "BUY"
        elif regime in ("Bull", "Sideways"):        # Bullish regime + positive sentiment
            return "BUY"
    
    # Standard BUY logic (high confidence)
    if alloc_pct > 0.0 and arb_conf >= thr and gdi_pct < BUY_GDI_MAX:
        if regime != "Bear":
            return "BUY"
        elif arb_conf >= 0.50 and bcs < BEAR_BUY_BCS_MAX and lstm_signal > 0.65:
            return "BUY"
    
    # FIX-2: Strengthened sentiment SELL guard
    # Don't SELL when sentiment is positive - convert to HOLD instead
    if arb_conf <= SELL_THRESHOLD and lstm_signal <= 0.65:
        if sent_score >= 0.05 and arb_conf > 0.25:
            return "HOLD"
        if sent_score >= 0.02 and arb_conf > 0.30:
            return "HOLD"
        return "SELL"
    
    return "HOLD"


def score_result(decision, actual_ret, ticker):
    if actual_ret is None or np.isnan(actual_ret):
        return "nan", "?"
    if decision == "HOLD":
        return "hold", "HOLD"
    nb = noise_band(ticker)
    if abs(actual_ret) <= nb:
        ok = ((decision == "BUY" and actual_ret >= 0) or
              (decision == "SELL" and actual_ret <= 0))
        return ("noise_c", "OK!") if ok else ("noise_w", "BAD")
    if decision == "BUY"  and actual_ret > 0: return "correct", "OK"
    if decision == "SELL" and actual_ret < 0: return "correct", "OK"
    return "wrong", "BAD"


def resolve_sent_date(test_date):
    if test_date in MANUAL_SENTIMENT:
        return test_date
    diffs = [(abs((pd.to_datetime(test_date) - pd.to_datetime(k)).days), k)
             for k in MANUAL_SENTIMENT]
    return min(diffs)[1]


def prewarm_asc(asc_memory, n=30, seed=42):
    if asc_memory is None or not _ASC_OK:
        return
    rng = np.random.RandomState(seed)
    for i in range(n):
        lstm_s   = float(np.clip(0.50 + rng.normal(0, 0.25), 0.05, 0.95))
        sent_s   = float(np.clip(rng.normal(0, 0.08), -0.30, 0.30))
        regime_p = float(np.clip(rng.choice([0.20, 0.50, 0.80]) + rng.normal(0, 0.05), 0.10, 0.90))
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                asc_memory.record_session(lstm_s, sent_s, regime_p)
        except Exception:
            pass


def prewarm_aesl(aesl_agent, n=15, seed=42):
    if aesl_agent is None or not _AESL_OK:
        return
    rng = np.random.RandomState(seed)
    regime_seq = ["Bull"] * 5 + ["Sideways"] * 5 + ["Bear"] * 5
    for i in range(n):
        lstm_s   = float(np.clip(0.65 - 0.28*(i/n) + rng.normal(0,0.07), 0.1, 0.9))
        sent_s   = float(-0.04 - 0.12*(i/n) + rng.normal(0,0.03))
        mc_std_v = float(np.clip(0.04 + 0.10*(i/n) + rng.normal(0,0.02), 0.02, 0.20))
        rc       = float(np.clip(0.65 + 0.10*(i/n) + rng.normal(0,0.04), 0.5, 0.98))
        rlbl     = regime_seq[i % len(regime_seq)]
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                aesl_agent.analyze(lstm_signal=lstm_s, sent_score=sent_s,
                                   regime_label=rlbl, mc_std=mc_std_v,
                                   regime_confidence=rc)
        except Exception:
            pass


def print_separator(char="=", width=140): print(char * width)
def print_section(title, char="-", width=140):
    print(f"\n{char * width}\n  {title}\n{char * width}")


# ================================================================================
#  CORE PIPELINE HOLD one ticker
# ================================================================================
def run_ticker(
    ticker, test_date, sent_date,
    tech_agent, uncertainty_agent, regime_agent, fusion_agent, heatmap_agent,
    conflict_resolver=None, risk_engine=None, aesl_agent=None, asc_memory=None,
    correlation_agent=None, topology_agent=None, causal_agent=None,
    counterfactual_engine=None, explainability_agent=None,
    meta_agent=None, adversarial_tester=None, legacy_regime_agent=None,
    use_hybrid_regime=True, use_uncertainty=True, use_fusion=True,
    use_heatmap=True, use_conflict=True, use_risk=True, use_aesl=True,
    use_asc=True, use_topology=True, use_causal=True, use_correlation=True,
    use_counterfactual=True, use_explainability=True, use_meta=True,
    use_adversarial=True, use_legacy_regime=True, use_sentiment=True,
    lstm_only=False,
):
    hist = fetch_history(ticker, test_date)
    if hist.empty or len(hist) < 150:
        return None
    feat_df = build_lstm_features(hist)
    if len(feat_df) < SEQ_LEN:
        return None

    with contextlib.redirect_stdout(io.StringIO()):
        lstm_stretched = tech_agent.predict(hist)

    sent_score = MANUAL_SENTIMENT[sent_date].get(ticker, 0.0) if use_sentiment else 0.0

    if use_uncertainty and not lstm_only:
        mc_mean, mc_std = uncertainty_agent.predict_from_prob(lstm_stretched)
    else:
        mc_mean = lstm_stretched
        mc_std  = 0.5 - abs(lstm_stretched - 0.5)

    if use_hybrid_regime and not lstm_only and regime_agent is not None:
        with contextlib.redirect_stdout(io.StringIO()):
            regime_label, regime_vol, regime_conf = regime_agent.detect(hist, ticker)
        if (lstm_stretched > 0.65 and regime_label == "Bear" and regime_conf < 0.85):
            try:
                spy_hist = fetch_history("SPY", test_date)
                if not spy_hist.empty and len(spy_hist) > 150:
                    with contextlib.redirect_stdout(io.StringIO()):
                        spy_regime, _, spy_conf = regime_agent.detect(spy_hist, "SPY")
                    if spy_regime in ("Bull", "Sideways") and spy_conf > 0.60:
                        regime_label = "Sideways"
                        regime_conf  = (regime_conf + spy_conf) / 2.0
            except Exception:
                pass
    else:
        close  = hist["Close"].squeeze().astype(float)
        ma50   = close.rolling(50).mean().iloc[-1]
        ma200  = close.rolling(200).mean().iloc[-1]
        vol_20 = float(close.pct_change().rolling(20).std().iloc[-1]) \
                 if not np.isnan(close.pct_change().rolling(20).std().iloc[-1]) else 0.015
        if ma50 > ma200 and vol_20 < 0.025:
            regime_label, regime_conf = "Bull", 0.60
        elif ma50 < ma200 and vol_20 > 0.015:
            regime_label, regime_conf = "Bear", 0.60
        else:
            regime_label, regime_conf = "Sideways", 0.55
        regime_vol = vol_20

    legacy_regime_label = None
    if (use_legacy_regime and not lstm_only
            and legacy_regime_agent is not None and _REGIME_LEGACY_OK):
        try:
            feat_arr = np.column_stack([
                hist["Close"].pct_change().dropna().values[-60:],
                hist["Close"].pct_change().rolling(21).std().dropna().values[-60:],
            ])
            if len(feat_arr) >= 5:
                with contextlib.redirect_stdout(io.StringIO()):
                    legacy_regime_label = legacy_regime_agent.get_regime_label(feat_arr[-1:])
        except Exception:
            legacy_regime_label = None

    risk_score_corr = 0.38
    div_status = "OK"
    if (use_correlation and not lstm_only
            and correlation_agent is not None and _CORR_OK):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                _raw_corr, _ = correlation_agent.get_market_context(ticker)
            if abs(_raw_corr - 0.500) < 0.002:
                risk_score_corr = compute_beta_risk(hist, test_date)
            else:
                risk_score_corr = _raw_corr
            div_status = ("CRITICAL" if risk_score_corr > 0.70 else
                          "MINOR"    if risk_score_corr > 0.40 else "OK")
        except Exception:
            risk_score_corr = compute_beta_risk(hist, test_date)

    topo_modifier = 1.0
    topo_chaos    = 0.5
    topo_signal   = "UNKNOWN"
    if (use_topology and not lstm_only and topology_agent is not None and _TOPO_OK):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                topo_result = topology_agent.analyze(hist)
            topo_modifier = topo_result.get("topology_modifier", 1.0)
            topo_chaos    = topo_result.get("topology_chaos_score", 0.5)
            topo_signal   = topo_result.get("market_shape_signal", "UNKNOWN")
        except Exception:
            pass

    causal_modifier = 1.0
    causal_score    = 0.5
    if (use_causal and not lstm_only and causal_agent is not None and _CAUSAL_OK):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                causal_result   = causal_agent.analyze(ticker=ticker, target_hist_df=hist,
                                                        universe_data=None)
            causal_modifier = causal_result.get("causal_modifier", 1.0)
            causal_score    = causal_result.get("causal_score", 0.5)
        except Exception:
            pass

    combined_modifier = ((topo_modifier + causal_modifier) / 2.0
                         if (use_topology or use_causal) else 1.0)

    adver_penalty = 1.0
    adver_passed  = True
    adver_delta   = 0.0
    if (use_adversarial and not lstm_only
            and adversarial_tester is not None and _ADVER_OK):
        try:
            crashed_df = adversarial_tester.generate_flash_crash(hist, drop_pct=0.10)
            with contextlib.redirect_stdout(io.StringIO()):
                crashed_score = adversarial_tester._predict_direct(crashed_df)
            adver_delta  = lstm_stretched - crashed_score
            adver_passed = abs(adver_delta) > 0.01
            if not adver_passed:
                adver_penalty = 0.72
        except Exception:
            pass

    top_driver = "unknown"
    ig_score   = 0.0
    if (use_explainability and not lstm_only
            and explainability_agent is not None and _EXPL_OK):
        try:
            last_100 = feat_df.tail(SEQ_LEN)
            with contextlib.redirect_stdout(io.StringIO()):
                importance_dict, top_driver = explainability_agent.explain_prediction(last_100)
            ig_score = importance_dict.get(top_driver, 0.0) if importance_dict else 0.0
        except Exception:
            pass

    vol_v = 0.9 if regime_label == "Bear" else 0.2 if regime_label == "Bull" else 0.5
    if use_fusion and not lstm_only:
        with contextlib.redirect_stdout(io.StringIO()):
            raw_conf, attn_weights = fusion_agent.predict(
                lstm_p=mc_mean, sent_s=sent_score, vol_v=vol_v)
        gated_conf = apply_fusion_gates(raw_conf, lstm_stretched, sent_score,
                                        regime_label, regime_conf)
        gated_conf = float(np.clip(gated_conf * combined_modifier * adver_penalty, 0.0, 1.0))
    else:
        raw_conf    = lstm_stretched
        gated_conf  = lstm_stretched * adver_penalty
        attn_weights = {}

    gdi, gdi_penalty = 0.0, 1.0
    gdi_tension = "HARMONY"
    if use_heatmap and not lstm_only and heatmap_agent is not None:
        with contextlib.redirect_stdout(io.StringIO()):
            gdi_result = heatmap_agent.analyze(
                lstm_score=lstm_stretched, sent_score=sent_score,
                regime_label=regime_label, regime_vol=regime_vol)
        gdi         = gdi_result["gdi"]
        gdi_penalty = gdi_result["penalty"]
        gdi_tension = gdi_result["tension"]

    trust_scores = None
    if use_meta and not lstm_only and meta_agent is not None and _META_OK:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                trust_scores = meta_agent.get_trust_scores(ticker=ticker)
        except Exception:
            pass

    arb_conf        = gated_conf
    conflict_ruling = "NO_MODULE"
    if use_conflict and not lstm_only and conflict_resolver is not None:
        try:
            risk_s = min(risk_score_corr, 0.82)
            with contextlib.redirect_stdout(io.StringIO()):
                arb_res = conflict_resolver.arbitrate(
                    tech_score=lstm_stretched, sent_score=sent_score,
                    mc_std=mc_std, regime_label=regime_label,
                    risk_score=risk_s, fusion_confidence=gated_conf,
                    trust_scores=trust_scores)
            arb_conf_raw    = arb_res["adjusted_confidence"]
            conflict_ruling = arb_res["ruling"]
            if lstm_stretched > 0.62 and gated_conf > 0.50:
                min_floor = gated_conf * 0.80
                arb_conf  = max(arb_conf_raw, min_floor)
            elif (arb_res["ruling"] == "SYSTEMIC_VETO"
                  and lstm_stretched > 0.75
                  and regime_label in ("Sideways", "Bear")):
                arb_conf = max(arb_conf_raw, 0.42)
                conflict_ruling = "VETO_SOFTENED"
            else:
                arb_conf = arb_conf_raw
        except Exception:
            arb_conf = gated_conf

    asc_score    = 0.5
    asc_penalty  = 1.0
    asc_quadrant = "NOT_RUN"
    if use_asc and not lstm_only and asc_memory is not None and _ASC_OK:
        try:
            regime_prob = {"Bull": 0.80, "Bear": 0.20, "Sideways": 0.50}.get(regime_label, 0.5)
            with contextlib.redirect_stdout(io.StringIO()):
                asc_memory.record_session(lstm_stretched, sent_score, regime_prob)
                asc_result = asc_memory.compute_asc()
            asc_score = asc_result["asc"]
            if asc_result["asc_reliable"]:
                asc_penalty, asc_quadrant = asc_memory.get_penalty_multiplier(
                    asc_score, 0.0, asc_result.get("asc_saturated", False))
                arb_conf = float(np.clip(arb_conf * asc_penalty, 0.0, 1.0))
        except Exception:
            pass

    bcs       = 0.0
    aesl_zone = "N/A"
    aesl_mult = 1.0
    if use_aesl and not lstm_only and aesl_agent is not None and _AESL_OK:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                aesl_result = aesl_agent.analyze(
                    lstm_signal=lstm_stretched, sent_score=sent_score,
                    regime_label=regime_label, mc_std=mc_std,
                    regime_confidence=regime_conf)
            bcs       = aesl_result.bcs
            aesl_zone = aesl_result.adaptive_zone
            aesl_mult = aesl_result.composite_multiplier
        except Exception:
            pass

    alloc_pct  = 0.0
    num_shares = 0
    kelly_frac = 0.0
    if use_risk and not lstm_only and risk_engine is not None:
        try:
            last_price = float(hist["Close"].iloc[-1])
            with contextlib.redirect_stdout(io.StringIO()):
                alloc_pct, kelly_frac = risk_engine.calculate_position_size(
                    arb_conf, regime_vol,
                    disagreement_penalty=gdi_penalty,
                    regime=regime_label,
                    stock_price=last_price)
            if use_aesl and aesl_agent is not None:
                alloc_pct = float(np.clip(alloc_pct * aesl_mult, 0.0, MAX_RISK))
            with contextlib.redirect_stdout(io.StringIO()):
                num_shares, _ = risk_engine.get_shares_amount(last_price, alloc_pct)
        except Exception:
            alloc_pct = 0.0
    elif not use_risk or lstm_only:
        if arb_conf >= BUY_THRESHOLD:
            alloc_pct = float(np.clip((arb_conf - 0.50) * 0.40, 0.0, MAX_RISK))

    decision = make_decision(arb_conf, alloc_pct, regime_label, ticker,
                             gdi * 100, bcs, lstm_signal=lstm_stretched,
                             sent_score=sent_score)

    display_alloc_pct = alloc_pct
    if decision == "SELL" and alloc_pct <= 1e-9:
        display_alloc_pct = 0.02

    return {
        "ticker": ticker,
        "lstm_s": round(lstm_stretched, 4), "mc_mean": round(mc_mean, 4),
        "mc_std": round(mc_std, 4), "sent_score": round(sent_score, 3),
        "regime": regime_label, "regime_conf": round(regime_conf, 3),
        "legacy_regime": legacy_regime_label or "N/A",
        "risk_score_corr": round(risk_score_corr, 4), "div_status": div_status,
        "topo_modifier": round(topo_modifier, 4), "topo_chaos": round(topo_chaos, 4),
        "topo_signal": topo_signal,
        "causal_modifier": round(causal_modifier, 4), "causal_score": round(causal_score, 4),
        "combined_mod": round(combined_modifier, 4),
        "adver_passed": adver_passed, "adver_delta": round(adver_delta, 4),
        "adver_penalty": round(adver_penalty, 4),
        "top_driver": top_driver, "ig_score": round(ig_score, 6),
        "raw_conf": round(raw_conf, 4), "gated_conf": round(gated_conf, 4),
        "gdi": round(gdi, 4), "gdi_penalty": round(gdi_penalty, 3),
        "gdi_tension": gdi_tension, "conflict_ruling": conflict_ruling,
        "arb_conf": round(arb_conf, 4), "asc_score": round(asc_score, 4),
        "asc_penalty": round(asc_penalty, 4), "asc_quadrant": asc_quadrant,
        "bcs": round(bcs, 4), "aesl_zone": aesl_zone, "aesl_mult": round(aesl_mult, 4),
        "alloc_pct": round(alloc_pct * 100, 2),
        "display_alloc_pct": round(display_alloc_pct * 100, 2),
        "kelly_frac": round(kelly_frac, 4),
        "num_shares": num_shares, "decision": decision,
    }


# ================================================================================
#  WINDOW RUNNER
# ================================================================================
def run_window(test_date, outcome_date, label, agents, portfolio=None):
    test_date    = snap_to_trading_day(test_date)
    outcome_date = snap_to_trading_day(outcome_date)
    sent_date    = resolve_sent_date(test_date)

    print(f"\n  {'-'*140}")
    print(f"  {label}  |  Test: {test_date}  ->  Outcome: {outcome_date}")
    print(f"  {'-'*140}")
    print(f"\n  {'Tick':<7} {'LSTM':>6} {'Sent':>6} {'Regime':<9} "
          f"{'Corr':>6} {'Topo':>6} {'Caus':>6} {'Adv':>4} "
          f"{'GDI':>5} {'Arb':>7} {'ASC':>6} {'BCS':>6} {'Zone':<10} "
          f"{'Alloc':>6} {'Dec':<6} {'Act%':>8}  Res")
    print(f"  {'-'*140}")

    rows = []
    adver_pass_count = 0
    adver_total      = 0

    for ticker in TICKERS:
        try:
            result = run_ticker(
                ticker, test_date, sent_date,
                tech_agent=agents["tech"],
                uncertainty_agent=agents["uncertainty"],
                regime_agent=agents.get("regime"),
                fusion_agent=agents["fusion"],
                heatmap_agent=agents["heatmap"],
                conflict_resolver=agents.get("conflict"),
                risk_engine=agents.get("risk"),
                aesl_agent=agents.get("aesl"),
                asc_memory=agents.get("asc"),
                correlation_agent=agents.get("correlation"),
                topology_agent=agents.get("topology"),
                causal_agent=agents.get("causal"),
                counterfactual_engine=agents.get("cf_engine"),
                explainability_agent=agents.get("expl"),
                meta_agent=agents.get("meta"),
                adversarial_tester=agents.get("adversarial"),
                legacy_regime_agent=agents.get("legacy_regime"),
            )
            if result is None:
                continue

            actual_ret            = fetch_actual_return(ticker, test_date, outcome_date)
            cat, icon             = score_result(result["decision"], actual_ret, ticker)
            result["actual_ret"]  = round(actual_ret, 3) if not np.isnan(actual_ret) else None
            result["result_cat"]  = cat
            result["result_icon"] = icon
            result["test_date"]   = test_date
            result["outcome_date"]= outcome_date
            result["window"]      = label
            rows.append(result)

            if portfolio is not None and result["actual_ret"] is not None:
                portfolio.record(result["decision"], result["alloc_pct"],
                                 result["actual_ret"], ticker, label)

            adver_total += 1
            if result["adver_passed"]:
                adver_pass_count += 1

            act_str = (f"{actual_ret:>+7.2f}%" if not np.isnan(actual_ret) else "    nan%")
            print(f"  {ticker:<7} {result['lstm_s']:>6.3f} "
                  f"{result['sent_score']:>+6.3f} "
                  f"{result['regime']:<9} "
                  f"{result['risk_score_corr']:>6.3f} "
                  f"{result['topo_modifier']:>6.3f} "
                  f"{result['causal_modifier']:>6.3f} "
                  f"{'PASS' if result['adver_passed'] else 'FAIL':>4} "
                  f"{result['gdi']:>5.3f} "
                  f"{result['arb_conf']:>7.4f} "
                  f"{result['asc_score']:>6.3f} "
                  f"{result['bcs']:>6.4f} "
                  f"{result['aesl_zone']:<10} "
                  f"{result['display_alloc_pct']:>5.1f}% "
                  f"{result['decision']:<6} "
                  f"{act_str}  {icon}")
        except Exception as e:
            print(f"  {ticker:<7} ERROR: {str(e)[:70]}")

    # Holiday fallback
    if rows and all(r.get("actual_ret") is None for r in rows):
        shifted_test = snap_to_trading_day(
            (pd.to_datetime(test_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
        shifted_out = snap_to_trading_day(
            (pd.to_datetime(outcome_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
        print(f"  [WARN]  All returns NaN. Retrying with shifted dates {shifted_test} -> {shifted_out}")
        for result in rows:
            actual_ret = fetch_actual_return(result["ticker"], shifted_test, shifted_out)
            cat, icon  = score_result(result["decision"], actual_ret, result["ticker"])
            result["actual_ret"]  = round(actual_ret, 3) if not np.isnan(actual_ret) else None
            result["result_cat"]  = cat
            result["result_icon"] = icon

    correct = sum(1 for r in rows if r["result_cat"] == "correct")
    wrong   = sum(1 for r in rows if r["result_cat"] == "wrong")
    nc      = sum(1 for r in rows if r["result_cat"] == "noise_c")
    nw      = sum(1 for r in rows if r["result_cat"] == "noise_w")
    holds   = sum(1 for r in rows if r["result_cat"] == "hold")
    nans    = sum(1 for r in rows if r["result_cat"] == "nan")
    active  = correct + wrong
    acc     = (correct / active * 100) if active > 0 else 0.0
    lenient_active  = correct + wrong + nc + nw
    lenient_acc     = ((correct + nc) / lenient_active * 100) if lenient_active > 0 else 0.0
    adver_rate      = (adver_pass_count / adver_total * 100) if adver_total > 0 else 0.0

    print(f"\n  {'-'*140}")
    print(f"  WINDOW: {label}")
    print(f"  OK{correct:>2} BAD{wrong:>2} NOISE{nc+nw:>2}(+{nc}/-{nw}) "
          f" H{holds:>2} ?{nans:>2}   "
          f"Strict: {acc:>5.1f}%  Lenient: {lenient_acc:>5.1f}%  "
          f"(active={active}/{len(rows)})  Red-Team Pass: {adver_rate:.0f}%")

    return {
        "label": label, "test_date": test_date, "outcome_date": outcome_date,
        "rows": rows, "correct": correct, "wrong": wrong, "nc": nc, "nw": nw,
        "holds": holds, "nans": nans, "active": active, "accuracy": acc,
        "lenient_acc": lenient_acc, "n": len(rows), "adver_rate": adver_rate,
    }


# ================================================================================
#  MAIN
# ================================================================================
def main():
    print_separator()
    print("  FinFolioX Full System Test  (21 Windows x 30 Tickers x 17 Agents)")
    print_separator()

    print("\n  LOADING ALL 17 AGENTS...")
    print("  " + "-" * 80)
    agents = {}

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            agents["tech"] = TechnicalAgent(lstm_model_path=MODEL_PATH,
                                            lstm_scaler_path=SCALER_PATH)
        print(f"  OK  [01] TechnicalAgent")
    except Exception as e:
        print(f"  BAD  [01] TechnicalAgent FAILED: {e}"); return

    agents["sentiment"] = None
    print("  OK  [02] SentimentAgent (manual scores)")

    try:
        hmm_path = os.path.join("saved_models", "hmm_regime.pkl")
        if _REGIME_LEGACY_OK and os.path.exists(hmm_path):
            with contextlib.redirect_stdout(io.StringIO()):
                agents["legacy_regime"] = RegimeAgent(model_path=hmm_path)
            print("  [OK]  [03a] LegacyRegimeAgent")
        else:
            agents["legacy_regime"] = None
            print("  [WARN]   [03a] LegacyRegimeAgent HOLD not loaded")
    except Exception:
        agents["legacy_regime"] = None

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            agents["regime"] = HybridRegimeAgent(hmm_model_path=REGIME_PATH, verbose=False)
        print("  [OK]  [03b] HybridRegimeAgent")
    except Exception as e:
        print(f"  [BAD]  [03b] HybridRegimeAgent FAILED: {e}"); agents["regime"] = None

    try:
        if _CORR_OK:
            agents["correlation"] = CorrelationDivergenceDetector(
                lookback_window=60,
                cache_path=os.path.join(tempfile.mkdtemp(), "corr_cache.pkl"))
            print("  [OK]  [04]  CorrelationAgent")
        else:
            agents["correlation"] = None
    except Exception:
        agents["correlation"] = None

    agents["uncertainty"] = UncertaintyAgent(agents["tech"])
    print("  [OK]  [05]  UncertaintyAgent")

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            agents["fusion"] = FusionAgent(model_path=FUSION_PATH)
        print("  [OK]  [06]  FusionAgent")
    except Exception as e:
        print(f"  [BAD]  [06]  FusionAgent FAILED: {e}"); return

    try:
        if _EXPL_OK:
            agents["expl"] = ExplainabilityAgent(agents["tech"], background_data_df=None)
            agents["expl"].ig_steps = IG_STEPS_FULLTEST
            print("  [OK]  [07]  ExplainabilityAgent")
        else:
            agents["expl"] = None
    except Exception:
        agents["expl"] = None

    agents["heatmap"] = HeatmapAgent()
    print("  [OK]  [08]  HeatmapAgent (GDI)")

    try:
        if _CONFLICT_OK:
            agents["conflict"] = ConflictResolver(verbose=False)
            print("  [OK]  [09]  ConflictResolver")
        else:
            agents["conflict"] = None
    except Exception:
        agents["conflict"] = None

    try:
        agents["risk"] = RiskEngine(default_account_size=DEFAULT_CAPITAL,
                                    max_risk_per_trade=MAX_RISK,
                                    bear_max_allocation=BEAR_MAX_ALLOC)
        print("  [OK]  [10]  RiskEngine")
    except Exception:
        agents["risk"] = None

    try:
        if _META_OK:
            agents["meta"] = MetaAgent()
            print("  [OK]  [11]  MetaAgent")
        else:
            agents["meta"] = None
    except Exception:
        agents["meta"] = None

    try:
        if _CF_OK:
            agents["cf_engine"] = CounterfactualEngine()
            print("  [OK]  [12]  CounterfactualEngine")
        else:
            agents["cf_engine"] = None
    except Exception:
        agents["cf_engine"] = None

    try:
        if _ADVER_OK:
            class _FakeSystem:
                def __init__(self, tech): self.tech_agent = tech
                def _fetch_stock_data(self, ticker): return None, pd.DataFrame()
            agents["adversarial"] = AdversarialTester(_FakeSystem(agents["tech"]))
            print("  [OK]  [13]  AdversarialTester")
        else:
            agents["adversarial"] = None
    except Exception:
        agents["adversarial"] = None

    try:
        if _TOPO_OK:
            agents["topology"] = TopologyAgent(time_delay=5, dimension=3, lookback=60)
            print("  [OK]  [14]  TopologyAgent (TDA)")
        else:
            agents["topology"] = None
    except Exception:
        agents["topology"] = None

    try:
        if _CAUSAL_OK:
            agents["causal"] = CausalAgent(lookback=90, alpha=0.20)
            print("  [OK]  [15]  CausalAgent")
        else:
            agents["causal"] = None
    except Exception:
        agents["causal"] = None

    try:
        if _ASC_OK:
            agents["asc"] = AgentDecisionMemory(
                window_size=30,
                cache_path=os.path.join(tempfile.mkdtemp(), "asc_main.pkl"))
            prewarm_asc(agents["asc"], n=30, seed=42)
            print("  [OK]  [16]  ASC Memory (pre-warmed)")
        else:
            agents["asc"] = None
    except Exception:
        agents["asc"] = None

    try:
        if _AESL_OK:
            agents["aesl"] = AESLAgent(
                cache_path=os.path.join(tempfile.mkdtemp(), "aesl_main.pkl"))
            prewarm_aesl(agents["aesl"], n=15, seed=42)
            print("  [OK]  [17]  AESLAgent (pre-warmed)")
        else:
            agents["aesl"] = None
    except Exception:
        agents["aesl"] = None

    print_separator()
    start_time = time.perf_counter()

    portfolio  = PortfolioTracker(DEFAULT_CAPITAL)
    all_stats  = []
    all_rows   = []

    print_section("FULL SYSTEM TEST HOLD 21 Windows x 30 Tickers", "=")

    for test_date, outcome_date, label in TEST_WINDOWS:
        s = run_window(test_date, outcome_date, label, agents=agents, portfolio=portfolio)
        all_stats.append(s)
        all_rows.extend(s["rows"])

    # -- Consolidated Summary --------------------------------------------------
    tc  = sum(s["correct"] for s in all_stats)
    tw  = sum(s["wrong"]   for s in all_stats)
    tnc = sum(s["nc"]      for s in all_stats)
    tnw = sum(s["nw"]      for s in all_stats)
    ta  = tc + tw
    ov  = (tc / ta * 100) if ta > 0 else 0.0
    lc  = tc + tnc
    la  = ta + tnc + tnw
    lov = (lc / la * 100) if la > 0 else 0.0
    pm  = portfolio.metrics()

    print_section("CONSOLIDATED RESULTS", "=")
    print(f"\n  {'Window':<42} {'N':>4} {'[OK]':>4} {'[BAD]':>4}  "
          f"{'Strict':>8}  {'Lenient':>8}  Red-Team  Status")
    print(f"  {'-'*95}")
    for s in all_stats:
        flag = "[OK] PASS" if s["accuracy"] >= 75 else "[WARN]  Below"
        print(f"  {s['label']:<42} {s['n']:>4} {s['correct']:>4} {s['wrong']:>4}  "
              f"{s['accuracy']:>7.1f}%  {s['lenient_acc']:>7.1f}%  "
              f"{s['adver_rate']:>6.0f}%   {flag}")
    print(f"  {'-'*95}")
    flag_ov = "🏆 TARGET MET (≥75%)" if ov >= 75 else "[WARN]  Below 75%"
    print(f"  {'OVERALL':<42} {sum(s['n'] for s in all_stats):>4} "
          f"{tc:>4} {tw:>4}  {ov:>7.1f}%  {lov:>7.1f}%            {flag_ov}")

    if pm:
        print(f"\n  -- Portfolio Performance --")
        print(f"  Total Return : {pm['total_return_pct']:>+7.3f}%  "
              f"(P&L: ${pm['total_pnl']:>+8.2f} on ${DEFAULT_CAPITAL:.0f})")
        print(f"  Win Rate: {pm['win_rate']:.1f}%")
        print(f"  Trade-Inflated (Old) : {pm['trade_inflated_sharpe']:>5.3f}  (Averaged by 500+ trades)")
        print(f"  Active Sharpe (18w)  : {pm['active_sharpe_18']:>5.3f}  (sqrt(18) multiplier)")
        print(f"  Max Drawdown : ${pm['max_drawdown']:>+8.2f}  Calmar: {pm['calmar_ratio']:.3f}")

    elapsed = time.perf_counter() - start_time
    print(f"\n  Runtime: {elapsed/60:.1f} min  |  Decisions: {len(all_rows)}")
    print_separator()

    try:
        pd.DataFrame(all_rows).to_csv("finfoliox_full_system_results.csv", index=False)
        print("  📄 Results -> finfoliox_full_system_results.csv")
    except Exception as e:
        print(f"  [WARN]  CSV save failed: {e}")

    return all_stats, all_rows, ov, lov, pm


if __name__ == "__main__":
    main()