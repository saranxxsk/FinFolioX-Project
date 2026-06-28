"""
api/views.py  HOLD  Django REST Framework API Views (Phase 19 + Phase 26)
=======================================================================
Exposes the FinFolio-X AI engine as 8 REST endpoints. Phase 26 extends
the /api/analyze/ response with a new "ensemble_health" JSON block that
contains the Agent Sycophancy Coefficient (asc_score), Forced Dissent
Protocol results (fdp_ran, dissent_sensitivity, fdp_interpretation), the
confidence penalty multiplier applied, and the ensemble quadrant label.
The _state_to_json() helper is updated to extract these four new AgentState
fields so they flow from the LangGraph execution graph to the React frontend
without any additional API endpoints or schema migrations.
"""

import os
import sys
import json
import traceback
import numpy as np
import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.views import View

from ml_engine.topology_agent import TopologyAgent
from ml_engine.causal_agent import CausalAgent

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


# ==============================================================================
# NUMPY-SAFE JSON ENCODER
# ==============================================================================

class NumpySafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _topology_to_dict(topology_result: dict) -> dict:
    if not topology_result:
        return {}
    return {
        "betti0":               topology_result.get("betti0", 0.5),
        "betti1":               topology_result.get("betti1", 0.5),
        "persistence_entropy":  topology_result.get("persistence_entropy", 0.5),
        "topology_chaos_score": topology_result.get("topology_chaos_score", 0.5),
        "dominant_structure":   topology_result.get("dominant_structure", "UNKNOWN"),
        "market_shape_signal":  topology_result.get("market_shape_signal", "UNKNOWN"),
        "topology_modifier":    topology_result.get("topology_modifier", 1.0),
        "h0_bars":              topology_result.get("h0_bars", []),
        "h1_bars":              topology_result.get("h1_bars", []),
        "status":               topology_result.get("status", "unknown"),
    }


# ==============================================================================
# SINGLETON LOADERS
# ==============================================================================

_system_instance  = None
_topology_agent   = None
_causal_agent     = None


def _get_system():
    global _system_instance
    if _system_instance is None:
        from ml_engine.master_system import FinFolioSystem
        _system_instance = FinFolioSystem()
    return _system_instance


def _get_topology_agent():
    global _topology_agent
    if _topology_agent is None:
        try:
            _topology_agent = TopologyAgent(time_delay=5, dimension=3, lookback=60)
        except Exception as e:
            print(f"   [WARN] TopologyAgent initialization failed: {e}")
    return _topology_agent


def _get_causal_agent():
    global _causal_agent
    if _causal_agent is None:
        try:
            _causal_agent = CausalAgent(lookback=90, alpha=0.05)
        except Exception as e:
            print(f"   [WARN] CausalAgent initialization failed: {e}")
    return _causal_agent


# ==============================================================================
# 1. POST /api/analyze/
# ==============================================================================

class AnalyzeView(APIView):
    """
    Accepts {"ticker": "AAPL"} and runs the full LangGraph 11-node pipeline.
    Returns AgentState as JSON including Phase 26 ensemble_health block.
    """

    def post(self, request):
        ticker = request.data.get("ticker", "").strip().upper()
        if not ticker:
            return Response(
                {"error": "Missing 'ticker' field. Send {'ticker': 'AAPL'}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            system = _get_system()
            print(f"\n🌐 API REQUEST: LangGraph Orchestrator for {ticker}...")

            from ml_engine.langgraph_orchestrator import FinFolioGraphOrchestrator
            orchestrator = FinFolioGraphOrchestrator(system)
            final_state  = orchestrator.run_analysis(ticker)

            if final_state.get("error"):
                return Response(
                    {"error": final_state["error"]},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

            result = _state_to_json(final_state, ticker)
            return Response(result, status=status.HTTP_200_OK)

        except Exception as e:
            traceback.print_exc()
            return Response(
                {"error": str(e), "traceback": traceback.format_exc()},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


# ==============================================================================
# 2. GET /api/history/
# ==============================================================================

class HistoryView(APIView):
    def get(self, request):
        ledger_path = os.path.join(BASE_DIR, "data", "meta", "decision_ledger.csv")
        if not os.path.exists(ledger_path):
            return Response(
                {"error": "Decision ledger not found. Run an analysis first."},
                status=status.HTTP_404_NOT_FOUND,
            )
        try:
            df = pd.read_csv(ledger_path, encoding="utf-8", on_bad_lines="skip")
            records = json.loads(df.to_json(orient="records"))
            return Response({"count": len(records), "decisions": records}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": f"Failed to read ledger: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ==============================================================================
# 3. GET /api/trust-scores/
# ==============================================================================

class TrustScoresView(APIView):
    def get(self, request):
        trust_path = os.path.join(BASE_DIR, "data", "meta", "trust_scores.json")
        if not os.path.exists(trust_path):
            return Response(
                {"error": "Trust scores not found. Run an analysis first."},
                status=status.HTTP_404_NOT_FOUND,
            )
        try:
            with open(trust_path, "r", encoding="utf-8") as f:
                scores = json.load(f)
            for agent in ["technical", "sentiment", "regime"]:
                val = scores.get(agent, 1.0)
                if val > 1.05:
                    scores[f"{agent}_status"] = "BOOSTED"
                elif val < 0.95:
                    scores[f"{agent}_status"] = "PENALIZED"
                else:
                    scores[f"{agent}_status"] = "NORMAL"
            return Response(scores, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": f"Failed to read trust scores: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ==============================================================================
# 4. POST /api/evaluate/
# ==============================================================================

class EvaluateView(APIView):
    def post(self, request):
        try:
            from ml_engine.meta_agent import MetaAgent
            meta   = MetaAgent()
            before = meta.get_trust_scores()
            meta.evaluate_past_decisions()
            after  = meta.get_trust_scores()

            ledger_path = os.path.join(BASE_DIR, "data", "meta", "decision_ledger.csv")
            evaluated   = []
            if os.path.exists(ledger_path):
                df = pd.read_csv(ledger_path, encoding="utf-8", on_bad_lines="skip")
                evaluated_df = df[df["evaluated"] == "YES"]
                evaluated = json.loads(evaluated_df.to_json(orient="records"))

            return Response({
                "message":              "Hindsight evaluation complete.",
                "trust_before":         before,
                "trust_after":          after,
                "evaluated_decisions":  evaluated,
                "total_evaluated":      len(evaluated),
            }, status=status.HTTP_200_OK)

        except Exception as e:
            traceback.print_exc()
            return Response({"error": f"Evaluation failed: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ==============================================================================
# 5. POST /api/simulate/
# ==============================================================================

class SimulateView(APIView):
    def post(self, request):
        ticker           = request.data.get("ticker", "").strip().upper()
        start_date       = request.data.get("start_date", "2024-01-01")
        end_date         = request.data.get("end_date", "2024-12-31")
        starting_capital = float(request.data.get("starting_capital", 10000))
        decision_interval = int(request.data.get("decision_interval", 5))
        scenarios        = request.data.get("scenarios", [])
        data_mode        = request.data.get("data_mode", "historical")
        gbm_params       = request.data.get("gbm_params", None)

        if not ticker:
            return Response({"error": "Missing 'ticker' field."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            system = _get_system()
            from ml_engine.simulation_engine import DigitalTwinSimulator
            twin = DigitalTwinSimulator(system=system)
            results = twin.run_simulation(
                ticker=ticker, start_date=start_date, end_date=end_date,
                starting_capital=starting_capital, decision_interval=decision_interval,
                scenarios=scenarios, data_mode=data_mode, gbm_params=gbm_params,
            )

            def _sanitize_nan(obj):
                if isinstance(obj, float) and (obj != obj or obj == float("inf") or obj == float("-inf")):
                    return None
                if isinstance(obj, dict):
                    return {k: _sanitize_nan(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_sanitize_nan(v) for v in obj]
                return obj

            return Response(_sanitize_nan(results), status=status.HTTP_200_OK)

        except Exception as e:
            traceback.print_exc()
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ==============================================================================
# 6. GET /api/topology/<ticker>/
# ==============================================================================

@method_decorator(csrf_exempt, name="dispatch")
class TopologyView(View):
    def get(self, request, ticker: str):
        try:
            import yfinance as yf
            hist_df = yf.download(ticker.upper(), period="6mo", interval="1d", progress=False)
            if hist_df.empty:
                return JsonResponse({"error": f"No data found for ticker '{ticker}'"}, status=404)

            topology_agent = _get_topology_agent()
            if topology_agent is None:
                return JsonResponse({"error": "Topology Agent not available"}, status=500)

            result  = topology_agent.analyze(hist_df)
            cloud   = result.get("point_cloud")
            cloud_3d = []
            if cloud is not None:
                pts = cloud[:200] if len(cloud) > 200 else cloud
                cloud_3d = pts.tolist() if hasattr(pts, "tolist") else pts

            payload = _topology_to_dict(result)
            payload["ticker"]       = ticker.upper()
            payload["point_cloud_3d"] = cloud_3d
            return JsonResponse(payload, encoder=NumpySafeEncoder)

        except Exception as exc:
            return JsonResponse({"error": str(exc), "status": "error"}, status=500)


# ==============================================================================
# 7. GET /api/causal/<ticker>/
# ==============================================================================

@method_decorator(csrf_exempt, name="dispatch")
class CausalAnalysisView(View):
    def get(self, request, ticker: str):
        try:
            import yfinance as yf
            ticker = ticker.upper()
            hist_df = yf.download(ticker, period="6mo", interval="1d", progress=False)
            if hist_df.empty:
                return JsonResponse({"error": f"No data for '{ticker}'"}, status=404)

            universe_syms = ["SPY", "QQQ", "VIX", "TLT", "GLD", "DXY"]
            universe_data = {}
            for sym in universe_syms:
                try:
                    df = yf.download(sym, period="6mo", interval="1d", progress=False)
                    if not df.empty:
                        universe_data[sym] = df
                except Exception:
                    pass

            causal_agent = _get_causal_agent()
            if causal_agent is None:
                return JsonResponse({"error": "Causal Agent not available"}, status=500)

            result = causal_agent.analyze(
                ticker=ticker, target_hist_df=hist_df,
                universe_data=universe_data if universe_data else None,
            )
            safe_result         = {k: v for k, v in result.items() if k != "hist_df"}
            safe_result["ticker"] = ticker
            safe_result["status"] = "ok"
            return JsonResponse(safe_result, encoder=NumpySafeEncoder)

        except Exception as exc:
            return JsonResponse({"error": str(exc), "status": "error"}, status=500)


# ==============================================================================
# 8. POST /api/causal/counterfactual/
# ==============================================================================

@method_decorator(csrf_exempt, name="dispatch")
class CounterfactualQueryView(View):
    def post(self, request):
        try:
            import yfinance as yf
            body     = json.loads(request.body)
            ticker   = body.get("ticker", "AAPL").upper()
            variable = body.get("variable", "VIX").upper()
            sigma    = float(body.get("sigma", -1.0))

            hist_df  = yf.download(ticker, period="6mo", interval="1d", progress=False)
            var_df   = yf.download(variable, period="6mo", interval="1d", progress=False)
            if hist_df.empty:
                return JsonResponse({"error": f"No data for '{ticker}'"}, status=404)

            causal_agent = _get_causal_agent()
            if causal_agent is None:
                return JsonResponse({"error": "Causal Agent not available"}, status=500)

            universe_data  = {variable: var_df} if not var_df.empty else None
            causal_result  = causal_agent.analyze(ticker=ticker, target_hist_df=hist_df, universe_data=universe_data)

            causal_effect = 0.0
            for driver in causal_result.get("true_causal_drivers", []):
                if driver["variable"] == variable:
                    causal_effect = driver["causal_effect"]
                    break

            if not var_df.empty:
                var_returns = np.log(var_df["Close"].values[1:] / var_df["Close"].values[:-1])
                var_std  = float(np.std(var_returns))
                var_last = float(var_returns[-1]) if len(var_returns) > 0 else 0.0
                var_mean = float(np.mean(var_returns))
                hypothetical = var_mean + sigma * var_std
            else:
                var_std, var_last, var_mean = 0.01, 0.0, 0.0
                hypothetical = sigma * var_std

            tgt_returns = np.log(hist_df["Close"].values[1:] / hist_df["Close"].values[:-1])
            factual_ret = float(tgt_returns[-1]) if len(tgt_returns) > 0 else 0.0
            cf_return   = factual_ret - causal_effect * (var_last - hypothetical)
            delta       = cf_return - factual_ret
            direction   = "ABOVE" if sigma > 0 else "BELOW"
            magnitude   = abs(sigma)

            narrative = (
                f"If {variable} had been {magnitude:.1f} standard deviations "
                f"{direction} its historical mean (do({variable}={hypothetical:.4f}) "
                f"instead of observed {var_last:.4f}), {ticker} would have returned "
                f"{cf_return * 100:+.3f}% instead of the factual {factual_ret * 100:+.3f}%. "
                f"Δ = {delta * 100:+.3f}%. "
                f"Causal effect used: β_do({variable}->{ticker}) = {causal_effect:.5f}."
            )

            return JsonResponse({
                "ticker": ticker, "variable": variable, "sigma": sigma,
                "query": f"What if {variable} had been {sigma:+.1f}σ from its mean?",
                "factual_return": round(factual_ret, 6),
                "counterfactual_return": round(cf_return, 6),
                "delta": round(delta, 6),
                "narrative": narrative,
                "causal_effect_used": round(causal_effect, 6),
                "status": "ok",
            }, encoder=NumpySafeEncoder)

        except Exception as exc:
            return JsonResponse({"error": str(exc), "status": "error"}, status=500)


# ==============================================================================
# HELPER: Convert AgentState -> JSON (updated for Phase 26)
# ==============================================================================

def _state_to_json(state, ticker):
    return {
        "ticker":           ticker,
        "system_version":   "26.0 (ASC + Sycophancy Detection)",

        "regime": {
            "label":      state.get("regime_label", "Unknown"),
            "volatility": _safe_float(state.get("current_vol", 0)),
        },
        "technical": {
            "lstm_signal":      _safe_float(state.get("lstm_signal", 0)),
            "mc_mean":          _safe_float(state.get("mc_mean", 0)),
            "mc_std":           _safe_float(state.get("mc_std", 0)),
            "uncertainty_status": state.get("uncertainty_status", "Unknown"),
            "top_driver":       state.get("top_driver", "Unknown"),
        },
        "sentiment": {
            "score": _safe_float(state.get("sent_score", 0)),
            "label": state.get("sent_label", "neutral"),
            "bias_warning": bool(state.get("sent_bias_warning", False)),
            "articles": state.get("sentiment_articles", []),
        },
        "systemic_risk": {
            "risk_score": _safe_float(state.get("risk_score", 0)),
            "div_status": state.get("div_status", "Unknown"),
        },
        "fusion": {
            "confidence":       _safe_float(state.get("fusion_confidence", 0)),
            "attention_weights": _safe_dict(state.get("attention_weights", {})),
        },
        "decision": {
            "action":              state.get("final_decision", "UNKNOWN"),
            "allocation_pct":      _safe_float(state.get("alloc_pct", 0)) * 100,
            "recommended_shares":  state.get("recommended_shares", 0),
            "cash_value":          _safe_float(state.get("cash_value", 0)),
        },
        "conflict": {
            "detected":  state.get("conflict_detected", False),
            "ruling":    state.get("conflict_ruling", "N/A"),
            "reasoning": state.get("conflict_reasoning", ""),
        },
        "trust_scores": state.get("trust_scores", {}),
        "disagreement": {
            "gdi":          _safe_float(state.get("gdi", 0)) * 100,
            "tension":      state.get("gdi_tension", "N/A"),
            "kelly_penalty": _safe_float(state.get("gdi_penalty", 1.0)),
        },
        "topology": _topology_to_dict(state.get("topology_result", {})),
        "red_team": {
            "passed": state.get("red_team_passed", True),
            "delta":  _safe_float(state.get("red_team_delta", 0)),
        },

        # Counterfactual & Causal Intelligence
        "counterfactual_verdict": state.get("counterfactual_verdict", ""),

        # -- Phase 26: Ensemble Health (ASC) --------------------------------
        "ensemble_health": {
            "asc_score":             _safe_float(state.get("asc_score", 0.5)),
            "asc_reliable":          bool(state.get("asc_reliable", False)),
            "asc_penalty_multiplier": _safe_float(state.get("asc_penalty_multiplier", 1.0)),
            "asc_quadrant":          state.get("asc_quadrant", ""),
            "dissent_sensitivity":   _safe_float(state.get("dissent_sensitivity", 0.0)),
            "fdp_ran":               bool(state.get("fdp_ran", False)),
            "fdp_interpretation":    state.get("fdp_interpretation", ""),
        },

        "executive_summary": state.get("executive_summary", ""),
    }


def _safe_float(val):
    try:
        if isinstance(val, (np.floating, np.integer)):
            return float(val)
    except ImportError:
        pass
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _safe_dict(d):
    if not isinstance(d, dict):
        return {}
    return {k: _safe_float(v) for k, v in d.items()}