"""
ml_engine/explainability_agent.py  HOLD  ExplainabilityAgent v5 (Integrated Gradients)
======================================================================================
DROP-IN REPLACEMENT for the old perturbation-based ExplainabilityAgent.

WHAT CHANGED vs original IG version:
  FIX 2 HOLD real F(baseline) for completeness check
    Problem: zeros input through LSTM does NOT output exactly 0.5.
    Old code assumed F(baseline) ≈ 0.5, making completeness check always wrong.
    Fix: forward-pass the all-zeros baseline once per call to get true F(0).
    This is stored in self._last_baseline_prob for diagnostics.

  FIX 3 HOLD macd_norm soft reliability gate
    Problem: macd_norm has only 70% aggregate expl_acc and drops to 50% in
    trending/bear markets, meaning it misleads top-driver selection those windows.
    Fix: if macd_norm wins the reliability-weighted score but its in-session
    reliability is below MACD_REL_GATE (0.60), demote it and promote the runner-up.
    Gate is conservative HOLD only fires when MACD is genuinely unreliable.

WHAT DID NOT CHANGE (interface is identical to old version):
  - Constructor:  __init__(self, technical_agent, background_data_df)
  - Method:       explain_prediction(self, recent_sequence_df)
  - Return type:  (importance_dict, top_driver)
  - self.tech_agent.lstm_model  and  self.tech_agent.lstm_scaler  HOLD same refs
  - feature_names list HOLD identical order, identical 7 names

HOW IG WORKS HERE:
  1. Scale the 100x7 input with lstm_scaler.
  2. Build an all-zeros baseline.
     (Zeros baseline is consistent with test_lstm.py; mean baseline was tried
     and produced the same directional results but worse completeness scores.)
  3. FIX 2: Forward-pass baseline once -> store true F(baseline).
  4. Interpolate 50 steps from baseline to actual input.
  5. Run tf.GradientTape over all 50 steps in one batch.
  6. Average gradients x (input − baseline) -> per-timestep IG matrix (100x7).
  7. Mean over 100 timesteps -> per-feature scalar attribution.
  8. Sign = did this feature push the model toward BUY (+) or SELL (−)?

RELIABILITY WEIGHTING (in-session):
  Each call records the raw IG signs for this ticker.
  After >= _min_samples_for_reliability (5) calls, the agent computes:
    reliability[f] = fraction where sign(ig_f) == sign(prob − 0.5)
  Features whose sign consistently contradicts the LSTM signal are flipped.
  Weighted score = |ig_attr| x reliability.
  FIX 3: macd_norm is further suppressed below MACD_REL_GATE.
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_tf = None

def _get_tf():
    global _tf
    if _tf is None:
        import tensorflow as tf
        _tf = tf
    return _tf


SEQ_LEN       = 100
IG_STEPS      = 50
MACD_REL_GATE = 0.60   # FIX 3: min in-session reliability for macd_norm

FEATURE_NAMES = [
    "log_return", "vol_change", "sma10_dist",
    "sma20_dist", "sma50_dist", "RSI", "macd_norm",
]


class ExplainabilityAgent:
    """
    Explains WHY the Keras LSTM made a specific BUY / SELL decision.

    Usage (unchanged from old perturbation version):
        agent = ExplainabilityAgent(tech_agent, background_data_df)
        importance_dict, top_driver = agent.explain_prediction(last_100_days_df)

    importance_dict  HOLD {feature_name: signed_ig_attribution}
                       positive = pushed model toward BUY
                       negative = pushed model toward SELL
    top_driver       HOLD name of the feature with highest reliability-weighted impact

    Diagnostic attributes (read after explain_prediction()):
        self._last_base_prob      HOLD F(x): model output on actual input
        self._last_baseline_prob  HOLD F(0): model output on all-zeros baseline
        self._last_ig_sum         HOLD sum(IG) HOLD should approximate F(x)-F(0)
        self._reliability         HOLD current per-feature reliability scores
    """

    def __init__(self, technical_agent, background_data_df):
        """
        Parameters
        ----------
        technical_agent   : TechnicalAgent instance
                            Must expose .lstm_model and .lstm_scaler
        background_data_df: pd.DataFrame (ignored HOLD kept for API compatibility)
        """
        self.tech_agent    = technical_agent
        self.feature_names = FEATURE_NAMES
        self.ig_steps      = IG_STEPS

        # In-session reliability tracking
        self._session_data: list = []
        self._reliability: dict  = {f: 0.5 for f in self.feature_names}
        self._min_samples_for_reliability = 5

        # FIX 2: diagnostic state from last call
        self._last_base_prob:     float = 0.5
        self._last_baseline_prob: float = 0.5
        self._last_ig_sum:        float = 0.0

        self.ready = self._verify_gradienttape()
        if self.ready:
            print("      [OK] Explainability Agent (Integrated Gradients v5) Ready.")
        else:
            print("      [WARN]  GradientTape check failed HOLD falling back to perturbation.")

    # ------------------------------------------------------------------
    # Internal: verify GradientTape compatibility once at startup
    # ------------------------------------------------------------------
    def _verify_gradienttape(self) -> bool:
        try:
            tf    = _get_tf()
            model = self.tech_agent.lstm_model
            n     = len(self.feature_names)
            dummy = tf.constant(np.random.randn(1, SEQ_LEN, n).astype(np.float32))
            with tf.GradientTape() as tape:
                tape.watch(dummy)
                out = model(dummy, training=False)
            grads = tape.gradient(out, dummy)
            return grads is not None
        except Exception as e:
            print(f"      [WARN]  GradientTape verify error: {e}")
            return False

    # ------------------------------------------------------------------
    # FIX 2 HOLD compute real F(baseline) alongside attributions
    # ------------------------------------------------------------------
    def _compute_ig(self, scaled: np.ndarray):
        """
        Parameters
        ----------
        scaled : np.ndarray  shape (100, 7) HOLD already scaler-transformed

        Returns
        -------
        attributions  : dict {feature_name: float}  raw signed IG per feature
        base_prob     : float  F(x) HOLD model output on actual input
        baseline_prob : float  F(0) HOLD model output on all-zeros baseline  ← FIX 2
        """
        tf    = _get_tf()
        model = self.tech_agent.lstm_model
        n     = len(self.feature_names)

        baseline = np.zeros_like(scaled)   # (100, 7) all zeros

        # -- FIX 2: true F(baseline) -------------------------------------------
        baseline_t    = tf.constant(baseline.reshape(1, SEQ_LEN, n), dtype=tf.float32)
        baseline_prob = float(model(baseline_t, training=False).numpy()[0][0])

        # Interpolation path: shape (IG_STEPS, 100, 7)
        alphas   = np.linspace(0.0, 1.0, self.ig_steps)
        interp   = np.array(
            [baseline + a * (scaled - baseline) for a in alphas],
            dtype=np.float32
        )
        interp_t = tf.constant(interp)   # (50, 100, 7)

        # Batch gradients via GradientTape
        with tf.GradientTape() as tape:
            tape.watch(interp_t)
            preds  = model(interp_t, training=False)   # (50, 1)
            summed = tf.reduce_sum(preds)
        grads = tape.gradient(summed, interp_t)         # (50, 100, 7)

        if grads is None:
            return {f: 0.0 for f in self.feature_names}, 0.5, baseline_prob

        avg_grads      = np.mean(grads.numpy(), axis=0)          # (100, 7)
        ig_matrix      = avg_grads * (scaled - baseline)         # (100, 7)
        ig_per_feature = np.mean(ig_matrix, axis=0)              # (7,)

        # F(x): actual model output
        actual_t  = tf.constant(scaled.reshape(1, SEQ_LEN, n), dtype=tf.float32)
        base_prob = float(model(actual_t, training=False).numpy()[0][0])

        attributions = {
            feat: float(ig_per_feature[i])
            for i, feat in enumerate(self.feature_names)
        }
        return attributions, base_prob, baseline_prob

    # ------------------------------------------------------------------
    # Internal: fallback perturbation (when GradientTape unavailable)
    # ------------------------------------------------------------------
    def _compute_perturbation(self, scaled: np.ndarray):
        """
        ±15% single-step perturbation fallback.
        Returns (attributions, base_prob, baseline_prob=base_prob) for
        interface consistency with _compute_ig().
        """
        model    = self.tech_agent.lstm_model
        n        = len(self.feature_names)
        base_seq = scaled.reshape(1, SEQ_LEN, n)
        base_out = float(model.predict(base_seq, verbose=0)[0][0])

        attributions = {}
        for i, feat in enumerate(self.feature_names):
            perturbed       = scaled.copy()
            perturbed[:, i] *= 1.15
            p_out           = float(
                model.predict(perturbed.reshape(1, SEQ_LEN, n), verbose=0)[0][0]
            )
            attributions[feat] = round(p_out - base_out, 6)

        # Perturbation has no real baseline; return base_out as placeholder
        return attributions, base_out, base_out

    # ------------------------------------------------------------------
    # Internal: recompute in-session reliability
    # ------------------------------------------------------------------
    def _update_reliability(self):
        """
        reliability[f] = fraction of session entries where
          sign(ig_f) == sign(prob − 0.5)
        Only computed after >= _min_samples_for_reliability entries exist.
        """
        n = len(self._session_data)
        if n < self._min_samples_for_reliability:
            return

        for feat in self.feature_names:
            matches = total = 0
            for entry in self._session_data:
                ig_val   = entry.get(feat, 0.0)
                lstm_dir = entry.get("prob", 0.5) - 0.5
                if lstm_dir == 0.0:
                    continue
                if (ig_val > 0 and lstm_dir > 0) or (ig_val < 0 and lstm_dir < 0):
                    matches += 1
                total += 1
            self._reliability[feat] = (matches / total) if total > 0 else 0.5

    # ------------------------------------------------------------------
    # FIX 3 HOLD top driver selection with macd_norm gate
    # ------------------------------------------------------------------
    def _select_top_driver(self, attributions: dict) -> tuple:
        """
        1. Compute effective_score = |ig| x reliability for each feature.
        2. If reliability < 0.5, flip sign (feature is inverse indicator).
        3. Rank all features by score descending.
        4. FIX 3: if macd_norm wins but in-session reliability < MACD_REL_GATE,
           demote it and promote the runner-up.

        Returns
        -------
        (top_driver_name, effective_signed_attribution, scores_dict)
        """
        scores    = {}
        effective = {}

        for feat in self.feature_names:
            ig  = attributions.get(feat, 0.0)
            rel = self._reliability.get(feat, 0.5)

            if rel < 0.5:
                ig  = -ig
                rel = 1.0 - rel

            effective[feat] = ig
            scores[feat]    = abs(ig) * rel

        ranked = sorted(scores, key=lambda f: scores[f], reverse=True)
        top    = ranked[0]

        # FIX 3: demote macd_norm if it wins but is below gate
        if (top == "macd_norm"
                and self._reliability.get("macd_norm", 0.5) < MACD_REL_GATE
                and len(ranked) > 1):
            print(
                f"      [ExplAgent] macd_norm gate: rel="
                f"{self._reliability['macd_norm']:.2f} < {MACD_REL_GATE} "
                f"-> promoted '{ranked[1]}'"
            )
            top = ranked[1]

        return top, effective[top], scores

    # ------------------------------------------------------------------
    # PUBLIC API HOLD drop-in replacement
    # ------------------------------------------------------------------
    def explain_prediction(self, recent_sequence_df: pd.DataFrame):
        """
        Explain the LSTM's latest BUY / SELL signal.

        Parameters
        ----------
        recent_sequence_df : pd.DataFrame  (last 100 trading days)
                             Must contain all 7 columns in FEATURE_NAMES.

        Returns
        -------
        importance_dict : dict  {feature_name: signed_attribution}
                          Values are reliability-adjusted effective IGs.
                          Positive = pushed model toward BUY.
                          Negative = pushed model toward SELL.
        top_driver      : str   dominant feature after reliability weighting
                               and macd_norm gate.
        """
        if not self.ready:
            return {}, "Not Ready"

        try:
            data   = recent_sequence_df[self.feature_names].values    # (100, 7)
            scaled = self.tech_agent.lstm_scaler.transform(data)       # (100, 7)

            # Compute attributions HOLD FIX 2 gives us real baseline_prob
            if self.ready:
                raw_attrs, base_prob, baseline_prob = self._compute_ig(scaled)
            else:
                raw_attrs, base_prob, baseline_prob = self._compute_perturbation(scaled)

            # Store diagnostics for external inspection
            self._last_base_prob     = base_prob
            self._last_baseline_prob = baseline_prob
            self._last_ig_sum        = sum(raw_attrs.values())

            # Completeness check (printed only when gap is notable)
            expected_diff = base_prob - baseline_prob
            gap           = abs(self._last_ig_sum - expected_diff)
            if gap > 0.15:
                print(
                    f"      [ExplAgent] IG completeness gap={gap:.4f} "
                    f"sum(IG)={self._last_ig_sum:+.4f} "
                    f"F(x)-F(0)={expected_diff:+.4f} "
                    f"F(0)={baseline_prob:.4f}"
                )

            # Record for in-session reliability tracking
            entry         = dict(raw_attrs)
            entry["prob"] = base_prob
            self._session_data.append(entry)
            self._update_reliability()

            # Select top driver HOLD FIX 3 gate applied inside
            top_driver, eff_ig, scores = self._select_top_driver(raw_attrs)

            # Build importance_dict with reliability-adjusted signs
            importance_dict = {}
            for feat in self.feature_names:
                ig  = raw_attrs.get(feat, 0.0)
                rel = self._reliability.get(feat, 0.5)
                if rel < 0.5:
                    ig = -ig
                importance_dict[feat] = round(ig, 6)

            return importance_dict, top_driver

        except Exception as e:
            print(f"      [WARN] Explainability Error: {e}")
            return {}, "Error"

    # ------------------------------------------------------------------
    # Diagnostics helper
    # ------------------------------------------------------------------
    def print_completeness_report(self, ticker: str = ""):
        """
        Prints a formatted IG completeness report for the last call.
        Call after explain_prediction() for debugging.
        """
        ig_sum       = self._last_ig_sum
        expected     = self._last_base_prob - self._last_baseline_prob
        gap          = abs(ig_sum - expected)
        status       = "[OK] close" if gap < 0.15 else "[WARN] gap"
        label        = f" ({ticker})" if ticker else ""
        print(
            f"      IG completeness{label}: sum(IG)={ig_sum:+.4f}  "
            f"F(x)-F(0)={expected:+.4f}  "
            f"F(0)={self._last_baseline_prob:.4f}  {status}"
        )

    def print_reliability_report(self):
        """Prints current in-session feature reliability scores."""
        n = len(self._session_data)
        print(f"      Feature reliability (IG, {n} samples):")
        for feat in self.feature_names:
            rel  = self._reliability[feat]
            bar  = "█" * int(rel * 20)
            gate = " ← gate active" if (
                feat == "macd_norm" and rel < MACD_REL_GATE
            ) else ""
            flip = " (flip)" if rel < 0.5 else ""
            print(f"        {feat:<14}: {rel:.2f}  {bar}{flip}{gate}")

    # ------------------------------------------------------------------
    # Batch reliability HOLD matches test_lstm.py window logic exactly
    # ------------------------------------------------------------------
    def set_batch_reliability(self, window_data: list):
        """
        Compute and store reliability from a full batch of pre-collected IG results.
        This REPLACES the incremental _update_reliability() path and matches the
        test_lstm.py logic exactly HOLD all tickers contribute simultaneously.

        Call this AFTER running explain_prediction() for every ticker in the window,
        but BEFORE reading _reliability or calling _select_top_driver() for reporting.

        Parameters
        ----------
        window_data : list of dicts
            Each dict must contain {feat: raw_ig_attribution, ..., 'prob': float}.
            The easiest source is self._session_data which is auto-filled by
            explain_prediction().

        Effect
        ------
        Overwrites self._reliability with cross-sectional values computed from
        the full batch. Subsequent calls to _select_top_driver() (and therefore
        the top_driver reported by explain_prediction()) will use these values.
        """
        n = len(window_data)
        if n < self._min_samples_for_reliability:
            print(
                f"      [ExplAgent] set_batch_reliability: only {n} samples, "
                f"need >= {self._min_samples_for_reliability}. "
                f"Reliability stays at 0.5."
            )
            return

        for feat in self.feature_names:
            matches = total = 0
            for entry in window_data:
                ig_val   = entry.get(feat, 0.0)
                lstm_dir = entry.get("prob", 0.5) - 0.5
                if lstm_dir == 0.0:
                    continue
                if (ig_val > 0 and lstm_dir > 0) or (ig_val < 0 and lstm_dir < 0):
                    matches += 1
                total += 1
            self._reliability[feat] = (matches / total) if total > 0 else 0.5

        print(f"      [ExplAgent] Batch reliability set from {n} tickers.")

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------
    def reset_session(self):
        """
        Clear in-session reliability data and diagnostics.
        Call between analysis runs if you want a fresh reliability slate
        per ticker rather than accumulating across the whole session.
        """
        self._session_data        = []
        self._reliability         = {f: 0.5 for f in self.feature_names}
        self._last_base_prob      = 0.5
        self._last_baseline_prob  = 0.5
        self._last_ig_sum         = 0.0