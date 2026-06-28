from django.urls import path
from . import views

urlpatterns = [
    # POST /api/analyze/  HOLD Run the full LangGraph AI pipeline
    path("analyze/", views.AnalyzeView.as_view(), name="analyze"),

    # GET  /api/history/  HOLD Fetch decision ledger as JSON
    path("history/", views.HistoryView.as_view(), name="history"),

    # GET  /api/trust-scores/  HOLD Fetch current trust multipliers
    path("trust-scores/", views.TrustScoresView.as_view(), name="trust-scores"),

    # POST /api/evaluate/  HOLD Trigger T+5 hindsight evaluation
    path("evaluate/", views.EvaluateView.as_view(), name="evaluate"),

    # POST /api/simulate/  HOLD Phase 21: Digital Twin Simulation
    path("simulate/", views.SimulateView.as_view(), name="simulate"),

    # GET  /api/topology/<ticker>/  HOLD Phase 24: Topological Shape Agent Analysis
    path("topology/<str:ticker>/", views.TopologyView.as_view(), name="topology"),

    # GET  /api/causal/<ticker>/  HOLD Phase 25: Causal Discovery Agent Analysis
    path("causal/<str:ticker>/", views.CausalAnalysisView.as_view(), name="causal-analysis"),

    # POST /api/causal/counterfactual/  HOLD Phase 25: On-Demand Counterfactual Query
    path("causal/counterfactual/", views.CounterfactualQueryView.as_view(), name="counterfactual-query"),
]
