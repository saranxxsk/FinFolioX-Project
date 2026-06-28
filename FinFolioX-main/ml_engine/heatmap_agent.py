"""
ml_engine/heatmap_agent.py  HOLD  Disagreement Heatmap Agent (v2.1)
=================================================================
Measures how much the three signal layers disagree:
  - LSTM technical signal
  - Sentiment score (FinBERT / MCP)
  - Hybrid Regime label + volatility

High disagreement -> high GDI -> confidence penalty applied before final decision.
Low disagreement  -> signals are aligned -> no penalty, conviction boosted.

Interface (consumed by finfolio_system.py and langgraph_orchestrator.py):
    result = agent.analyze(
        lstm_score=float,     # stretched LSTM prob in [0, 1]
        sent_score=float,     # FinBERT score in [-0.75, +0.75]
        regime_label=str,     # "Bull" | "Bear" | "Sideways"
        regime_vol=float,     # 21-day decimal vol (e.g. 0.012)
    )
    # result keys: gdi, tension, penalty, detail

    agent.print_heatmap(result)  # pretty-print to console
"""

import numpy as np


# ==============================================================================
# TENSION BANDS
# ==============================================================================
TENSION_BANDS = [
    (0.20, "HARMONY",  1.00),   # all agents agree
    (0.35, "MILD",     0.95),   # minor disagreement
    (0.50, "MODERATE", 0.85),   # signals mixed
    (0.65, "HIGH",     0.70),   # strong disagreement
    (1.01, "CRITICAL", 0.50),   # complete contradiction
]

# ==============================================================================
# SIGNAL WEIGHTS
# How much each pairwise disagreement matters.
# LSTM↔Regime is most important (macro vs price action).
# Sentiment↔LSTM is second (news vs chart).
# Sentiment↔Regime is third (news vs macro context).
# Vol spike adds extra disagreement cost.
# ==============================================================================
W_LSTM_REGIME = 0.45   # technical vs macro HOLD most critical conflict
W_SENT_LSTM   = 0.30   # news vs chart signal
W_SENT_REGIME = 0.15   # news vs macro context
W_VOL_SPIKE   = 0.10   # elevated vol worsens all disagreements


class HeatmapAgent:
    """
    Group Disagreement Index (GDI) calculator.

    GDI ∈ [0, 1]:
      0.00 -> all agents perfectly aligned
      1.00 -> agents completely contradict each other

    Penalty ∈ [0.50, 1.00]:
      1.00 -> no penalty (harmony)
      0.50 -> 50% confidence reduction (critical disagreement)
    """

    def __init__(self):
        self._last_result = None

    # ==========================================================================
    # MAIN METHOD
    # ==========================================================================

    def analyze(self, lstm_score: float,
                sent_score: float,
                regime_label: str,
                regime_vol: float) -> dict:
        """
        Computes the Group Disagreement Index (GDI) across all signal layers.

        Parameters
        ----------
        lstm_score   : stretched LSTM probability ∈ [0, 1]
                       0.0 = strong SELL signal, 1.0 = strong BUY signal
        sent_score   : FinBERT-blended score ∈ [-0.75, +0.75]
                       negative = bearish, positive = bullish
        regime_label : "Bull" | "Bear" | "Sideways"
        regime_vol   : 21-day decimal daily volatility (e.g. 0.012 = 1.2%/day)

        Returns
        -------
        dict with keys:
          gdi       : float ∈ [0, 1]
          tension   : str   HOLD "HARMONY" | "MILD" | "MODERATE" | "HIGH" | "CRITICAL"
          penalty   : float ∈ [0.5, 1.0]  HOLD multiplier applied to fusion confidence
          detail    : dict  HOLD per-component disagreement scores (for logging)
        """
        # -- Convert all signals to [-1, +1] directional scale -----------------
        #
        # LSTM: 0.5 is neutral, 1.0 is max bullish, 0.0 is max bearish
        #   -> direction_lstm ∈ [-1, +1]
        direction_lstm = (lstm_score - 0.5) * 2.0

        # Sentiment: already ∈ [-0.75, +0.75] -> rescale to [-1, +1]
        direction_sent = float(np.clip(sent_score / 0.75, -1.0, 1.0))

        # Regime: map label to directional score
        regime_dir = {"Bull": +1.0, "Sideways": 0.0, "Bear": -1.0}.get(
            regime_label, 0.0)

        # -- Per-component disagreement -----------------------------------------
        # Disagreement = |difference| / 2 (normalised to [0, 1])
        # Two signals pointing in exactly opposite directions -> disagreement = 1.0
        # Two signals pointing in the same direction -> disagreement = 0.0

        d_lstm_regime = abs(direction_lstm - regime_dir) / 2.0
        if regime_label == "Sideways":
            d_lstm_regime *= 0.60
        d_sent_lstm   = abs(direction_sent - direction_lstm) / 2.0
        d_sent_regime = abs(direction_sent - regime_dir) / 2.0

        # Vol spike component: elevated vol amplifies the cost of disagreement.
        # Threshold: 2% daily (~32% annualised) is the "high vol" boundary.
        vol_spike = float(np.clip((regime_vol - 0.02) / 0.03, 0.0, 1.0))

        # -- Weighted GDI ------------------------------------------------------
        gdi_raw = (
            W_LSTM_REGIME * d_lstm_regime
            + W_SENT_LSTM   * d_sent_lstm
            + W_SENT_REGIME * d_sent_regime
            + W_VOL_SPIKE   * vol_spike
        )
        gdi = float(np.clip(gdi_raw, 0.0, 1.0))

        # -- Tension band lookup -----------------------------------------------
        tension = "CRITICAL"
        penalty = 0.50
        for threshold, band_name, band_penalty in TENSION_BANDS:
            if gdi < threshold:
                tension = band_name
                penalty = band_penalty
                break

        # -- Store detail for logging ------------------------------------------
        detail = {
            "direction_lstm":    round(direction_lstm, 3),
            "direction_sent":    round(direction_sent, 3),
            "direction_regime":  round(regime_dir, 3),
            "d_lstm_regime":     round(d_lstm_regime, 3),
            "d_sent_lstm":       round(d_sent_lstm, 3),
            "d_sent_regime":     round(d_sent_regime, 3),
            "vol_spike":         round(vol_spike, 3),
            "regime_vol":        round(regime_vol, 4),
        }

        result = {
            "gdi":     round(gdi, 4),
            "tension": tension,
            "penalty": round(penalty, 4),
            "detail":  detail,
        }

        self._last_result = result
        return result

    # ==========================================================================
    # PRETTY PRINTER
    # ==========================================================================

    def print_heatmap(self, result: dict):
        """
        Prints a formatted heatmap report to console.
        Called by finfolio_system.py after every analyze() call.
        """
        gdi     = result["gdi"]
        tension = result["tension"]
        penalty = result["penalty"]
        d       = result.get("detail", {})

        # Tension colour emoji
        icon = {"HARMONY": "🟢", "MILD": "🟡", "MODERATE": "🟠",
                "HIGH": "🔴", "CRITICAL": "🚨"}.get(tension, "⬜")

        bar_len  = int(gdi * 20)
        bar      = "█" * bar_len + "░" * (20 - bar_len)

        print(f"\n   -- Disagreement Heatmap -----------------------------")
        print(f"      GDI    : {gdi:.4f}  [{bar}]  {icon} {tension}")
        print(f"      Penalty: {penalty:.2f}x  (confidence multiplier)")
        print(f"      -- Component breakdown --------------------------")
        print(f"         LSTM↔Regime  : {d.get('d_lstm_regime', 0):.3f}  "
              f"(LSTM={d.get('direction_lstm', 0):+.2f}  "
              f"Regime={d.get('direction_regime', 0):+.2f})")
        print(f"         Sent↔LSTM    : {d.get('d_sent_lstm', 0):.3f}  "
              f"(Sent={d.get('direction_sent', 0):+.2f})")
        print(f"         Sent↔Regime  : {d.get('d_sent_regime', 0):.3f}")
        print(f"         Vol spike    : {d.get('vol_spike', 0):.3f}  "
              f"(daily_vol={d.get('regime_vol', 0):.4f})")
        print(f"   ----------------------------------------------------")