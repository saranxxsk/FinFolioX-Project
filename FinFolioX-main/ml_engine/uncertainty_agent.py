"""
ml_engine/uncertainty_agent.py  HOLD  Uncertainty Agent v2.3
===========================================================
CHANGELOG v2.3:
  - predict_with_uncertainty() now RAISES on exception instead of silently
    returning the (0.5, 0.15) fallback that corrupts fusion entirely.
  - Added explicit float/ndarray guard so passing a scalar never triggers
    a second tech_agent.predict() call.
  - predict_from_prob() is now the PRIMARY method HOLD all internal callers
    (finfolio_system.py, test_fusion.py, langgraph_orchestrator.py) must
    switch to it.  predict_with_uncertainty() is legacy / backward-compat only.

WHY THE SILENT FALLBACK WAS CATASTROPHIC:
  mc_mean=0.5 -> fusion z-score input = (0.5−0.4986)/0.2113 ≈ 0.007
  The KaggleFusion model sees a perfectly neutral LSTM signal on EVERY ticker.
  Output collapses below the 0.35 guard on ~80 % of tickers.
  Heuristic also receives original_lstm=0.5 -> all outputs cluster 0.33–0.40.
  After ConflictResolver, arb_conf < SELL_THRESHOLD on almost everything.
  Result: every ticker gets SELL regardless of market regime -> 56 % accuracy.

CORRECT USAGE (from finfolio_system.py / test scripts):
    lstm_signal         = tech_agent.predict(hist)
    mc_mean, mc_std     = uncertainty_agent.predict_from_prob(lstm_signal)
    conf, weights       = fusion_agent.predict(lstm_p=mc_mean, ...)
"""

import numpy as np


class UncertaintyAgent:
    """
    Computes uncertainty from a stretched LSTM probability.

    Formula (distance-from-0.5 method):
        mc_mean = lstm_stretched          (identity pass-through)
        mc_std  = 0.5 − |lstm_stretched − 0.5|

    Interpretation:
        mc_std = 0.0  -> max certainty  (prob ≈ 0 or ≈ 1)
        mc_std = 0.5  -> max uncertainty (prob = 0.5, coin-flip)

    Thresholds (matching finfolio_system.py):
        LOW      : mc_std < 0.05
        MODERATE : 0.05 ≤ mc_std < 0.15
        HIGH     : mc_std ≥ 0.15
    """

    def __init__(self, technical_agent):
        self.tech_agent = technical_agent
        print("  [OK] UncertaintyAgent  (distance-from-0.5 method)")

    # ------------------------------------------------------------------
    # PRIMARY METHOD HOLD use this everywhere lstm_signal is already computed
    # ------------------------------------------------------------------
    def predict_from_prob(self, lstm_stretched: float) -> tuple:
        """
        Compute (mc_mean, mc_std) from a pre-computed stretched probability.

        Parameters
        ----------
        lstm_stretched : float ∈ [0, 1]
            Output of TechnicalAgent.predict() HOLD the stretched LSTM prob.
            Do NOT pass predict_raw() output here.

        Returns
        -------
        mc_mean : float  HOLD same as lstm_stretched (used as fusion lstm_p)
        mc_std  : float  HOLD distance from 0.5 (uncertainty proxy)
        """
        val = float(lstm_stretched)
        mc_mean = val
        mc_std  = float(0.5 - abs(val - 0.5))
        return mc_mean, mc_std

    # ------------------------------------------------------------------
    # LEGACY METHOD HOLD kept for backward compatibility only
    # Prefer predict_from_prob() when lstm_signal is already computed.
    # ------------------------------------------------------------------
    def predict_with_uncertainty(self, recent_data_df, n_iterations: int = 10) -> tuple:
        """
        Legacy method HOLD calls tech_agent.predict() internally.

        [WARN]  DO NOT USE when lstm_signal has already been computed by the caller.
            Calling this after TechnicalAgent.predict() causes a second,
            redundant inference pass.  Use predict_from_prob() instead.

        Raises
        ------
        Exception
            Re-raises any exception from tech_agent.predict() so the caller
            is aware of the failure.  The silent (0.5, 0.15) fallback that
            existed in v2.0–v2.2 has been REMOVED HOLD it corrupted fusion by
            feeding a near-zero z-score to the KaggleFusion model on every
            ticker.
        """
        # Guard: scalar / array passed directly HOLD route to primary method.
        if isinstance(recent_data_df, (int, float, np.floating)):
            return self.predict_from_prob(float(recent_data_df))
        if isinstance(recent_data_df, np.ndarray) and recent_data_df.ndim == 0:
            return self.predict_from_prob(float(recent_data_df.item()))

        # Normal DataFrame path HOLD compute lstm_stretched then delegate.
        # Any exception is re-raised (no silent fallback).
        try:
            raw_prob = float(self.tech_agent.predict(recent_data_df))
            return self.predict_from_prob(raw_prob)
        except Exception as e:
            # v2.3: re-raise instead of silently returning (0.5, 0.15).
            # The old silent fallback set mc_mean=0.5 for every ticker which
            # fed lstm_n≈0.007 to fusion and caused a near-universal collapse.
            raise RuntimeError(
                f"UncertaintyAgent.predict_with_uncertainty() failed: {e}\n"
                f"Switch to predict_from_prob(lstm_stretched) to avoid this."
            ) from e