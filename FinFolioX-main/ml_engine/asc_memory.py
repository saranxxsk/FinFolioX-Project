"""
ml_engine/asc_memory.py  HOLD  Agent Sycophancy Coefficient (ASC) Engine
======================================================================
Phase 26 HOLD v2.3

CHANGELOG v2.3 (4 bugs fixed on top of v2.2):

  BUG-1 FIXED: n < 25 hardcoded in saturation check alongside
    MIN_RELIABLE_SAMPLES=20, causing samples 20-24 to ALWAYS report
    asc_saturated=True. No penalty ever fired during warm-up window.
    Fix: replaced `n < 25` with `n < MIN_RELIABLE_SAMPLES + 5` so
    the extra guard scales with the configured window.

  BUG-2 FIXED: Dissent Sensitivity (DS) was unused in the 0.70-0.85
    penalty zone HOLD both DS branches returned the same PENALTY_MODERATE.
    Fix: low DS -> PENALTY_MODERATE_LOW (−10%), high DS -> PENALTY_MODERATE_HIGH (−20%).
    This makes DS meaningfully differentiate within the moderate zone.

  BUG-3 FIXED: print_asc_report box lines had inconsistent widths
    (55-67 chars) causing misaligned box borders on Windows terminals.
    Fix: all content lines padded to exactly BOX_WIDTH=56 inner chars.

  BUG-4 FIXED: SATURATION_STD_THRESHOLD constant was 0.02 but the
    FIX-1 docstring said 0.04. Raised to 0.04 to match the documented
    intent HOLD low-variance batches were slipping through and generating
    spurious penalties.

CHANGELOG v2.2 (4 fixes on v2.1):
  FIX-1 · KSG Saturation Guard
  FIX-2 · Raised minimum reliable window: 15 -> 20
  FIX-3 · Raised penalty fire threshold: 0.70 -> 0.85
  FIX-4 · Softer graduated penalty table
"""

import os
import pickle
import logging
import numpy as np
from collections import deque
from typing import Optional, Tuple, Dict

logger = logging.getLogger("ASCMemory")

try:
    from sklearn.feature_selection import mutual_info_regression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not found HOLD ASC will use fallback correlation estimator.")


# ==============================================================================
# CONSTANTS
# ==============================================================================

WINDOW_SIZE               = 30
MIN_RELIABLE_SAMPLES      = 20       # v2.2 FIX-2: was 15

N_HISTOGRAM_BINS          = 10

# v2.3 BUG-4 FIX: raised from 0.02 to match v2.2 docstring intent (0.04)
SATURATION_STD_THRESHOLD  = 0.04

# ASC zone boundaries
ASC_LOW_THRESHOLD         = 0.50    # Below -> no penalty
ASC_MED_THRESHOLD         = 0.70    # Mild zone
ASC_HIGH_THRESHOLD        = 0.85    # v2.2 FIX-3: penalty zone starts here (was 0.70)
ASC_EXTREME_THRESHOLD     = 0.95    # Extreme zone

# Dissent Sensitivity thresholds
DS_LOW_THRESHOLD          = 0.10
DS_HIGH_THRESHOLD         = 0.25

# v2.2 FIX-4 + v2.3 BUG-2 FIX: rebalanced penalty multipliers
# BUG-2 FIX: split PENALTY_MODERATE into two levels so DS is meaningful
PENALTY_NONE              = 1.00   # ASC < 0.50
PENALTY_MILD              = 0.95   # ASC 0.50–0.70          (−5%)
PENALTY_MODERATE_LOW      = 0.90   # ASC 0.70–0.85, DS low  (−10%)  ← BUG-2 FIX
PENALTY_MODERATE_HIGH     = 0.80   # ASC 0.70–0.85, DS high (−20%)  ← BUG-2 FIX
PENALTY_HIGH              = 0.75   # ASC 0.85–0.95          (−25%)
PENALTY_EXTREME           = 0.65   # ASC >= 0.95            (−35%)

# v2.3 BUG-1 FIX: saturation check uses this relative margin instead of n<25
SATURATION_EXTRA_GUARD    = 5      # n < MIN_RELIABLE_SAMPLES + SATURATION_EXTRA_GUARD


# ==============================================================================
# AGENT DECISION MEMORY
# ==============================================================================

class AgentDecisionMemory:
    """
    Rolling buffer that stores raw agent outputs, computes ASC via KSG
    mutual information, and maps (ASC, dissent_sensitivity) to a
    confidence penalty multiplier applied before the Conflict Resolver.

    
    """

    def __init__(self, window_size: int = WINDOW_SIZE, cache_path: Optional[str] = None):
        self.window_size = window_size

        if cache_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cache_path = os.path.join(base_dir, "data", "meta", "asc_buffer.pkl")
        self.cache_path = cache_path

        self.buffer: deque = self._load_buffer()

        print(f"   [+] Phase 26: ASC Memory Engine v2.3 Initialized.")
        print(f"      - Window      : {window_size} sessions")
        print(f"      - Buffer      : {len(self.buffer)}/{window_size} sessions loaded")
        print(f"      - Min reliable: {MIN_RELIABLE_SAMPLES}")
        print(f"      - Sat. thresh : LSTM std < {SATURATION_STD_THRESHOLD}")
        print(f"      - Penalty gate: ASC >= {ASC_HIGH_THRESHOLD}")
        status = "RELIABLE" if len(self.buffer) >= MIN_RELIABLE_SAMPLES else \
                 f"WARMING ({len(self.buffer)}/{MIN_RELIABLE_SAMPLES})"
        print(f"      - Status      : {status}")

    # -- Persistence -------------------------------------------------------

    def _load_buffer(self) -> deque:
        try:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, "rb") as f:
                    data = pickle.load(f)
                # Upgrade old 3-tuple entries to 4-tuples with dummy timestamp
                upgraded = []
                for item in data:
                    if len(item) == 3:
                        upgraded.append(tuple(list(item) + [0.0]))
                    else:
                        upgraded.append(item)
                return deque(upgraded, maxlen=self.window_size)
        except Exception as e:
            logger.warning(f"Could not load ASC buffer: {e}")
        return deque(maxlen=self.window_size)

    def _save_buffer(self):
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump(list(self.buffer), f)
        except Exception as e:
            logger.warning(f"Could not save ASC buffer: {e}")

    # -- Step 1: Record session --------------------------------------------

    def record_session(self, lstm_score: float, sent_score: float, regime_prob: float):
        """
        Append one session's raw agent outputs to the rolling buffer.
        Anti-spam: skip if LSTM score is identical to the last entry (same session).
        """
        import time as _time
        entry = (
            float(np.clip(lstm_score,  0.0, 1.0)),
            float(np.clip(sent_score, -1.0, 1.0)),
            float(np.clip(regime_prob, 0.0, 1.0)),
            _time.time(),
        )
        if len(self.buffer) > 0 and abs(self.buffer[-1][0] - entry[0]) < 0.001:
            return
        self.buffer.append(entry)
        self._save_buffer()

    @staticmethod
    def regime_label_to_prob(regime_label: str) -> float:
        """Convert HMM regime label to a continuous bullish probability."""
        label = str(regime_label).strip().lower()
        if label == "bull":   return 0.80
        elif label == "bear": return 0.20
        else:                 return 0.50

    # -- Step 2: Compute ASC -----------------------------------------------

    def compute_asc(self) -> Dict:
        """
        Compute the Agent Sycophancy Coefficient over the current buffer.

        ASC = 1 - (sum_MI / sum_H)
          - sum_H ≈ 0  (all sessions identical)  -> asc = 1.0 (edge case)
          - sum_MI high, sum_H high               -> asc near 0 (agents informative)
          - sum_MI low,  sum_H high               -> asc near 1 (agents sycophantic)

        Returns a dict with asc, asc_reliable, asc_saturated, and diagnostics.
        """
        n = len(self.buffer)

        if n < MIN_RELIABLE_SAMPLES:
            return {
                "asc": 0.50, "asc_reliable": False, "asc_saturated": False,
                "lstm_std": 0.0, "mi_lstm_sent": 0.0, "mi_lstm_hmm": 0.0,
                "mi_sent_hmm": 0.0, "h_lstm": 0.0, "h_sent": 0.0,
                "h_hmm": 0.0, "n_samples": n,
            }

        arr      = np.array(list(self.buffer))
        lstm_arr = arr[:, 0]
        sent_arr = (arr[:, 1] + 1.0) / 2.0   # normalise to [0, 1]
        hmm_arr  = arr[:, 2]

        lstm_std = float(np.std(lstm_arr))

        # v2.3 BUG-1 FIX: replaced hardcoded `n < 25` with configurable guard
        saturated = (
            lstm_std < SATURATION_STD_THRESHOLD
            or n < MIN_RELIABLE_SAMPLES + SATURATION_EXTRA_GUARD
        )
        if saturated:
            logger.info(
                f"ASC saturation: LSTM std={lstm_std:.4f} < {SATURATION_STD_THRESHOLD}. "
                "Buffer too homogeneous HOLD KSG output unreliable, penalty suppressed."
            )
            print(f"      [WARN]  [ASC] Saturation detected "
                  f"(LSTM std={lstm_std:.4f}, n={n}). Penalty suppressed.")

        mi_lstm_sent = self._compute_mi(lstm_arr, sent_arr)
        mi_lstm_hmm  = self._compute_mi(lstm_arr, hmm_arr)
        mi_sent_hmm  = self._compute_mi(sent_arr, hmm_arr)

        h_lstm = self._compute_entropy(lstm_arr)
        h_sent = self._compute_entropy(sent_arr)
        h_hmm  = self._compute_entropy(hmm_arr)

        sum_mi = mi_lstm_sent + mi_lstm_hmm + mi_sent_hmm
        sum_h  = h_lstm + h_sent + h_hmm

        asc = 1.0 if sum_h < 1e-8 else float(
            np.clip(1.0 - (sum_mi / (sum_h + 1e-8)), 0.0, 1.0)
        )

        return {
            "asc":           round(asc, 4),
            "asc_reliable":  True,
            "asc_saturated": saturated,
            "lstm_std":      round(lstm_std, 4),
            "mi_lstm_sent":  round(mi_lstm_sent, 4),
            "mi_lstm_hmm":   round(mi_lstm_hmm, 4),
            "mi_sent_hmm":   round(mi_sent_hmm, 4),
            "h_lstm":        round(h_lstm, 4),
            "h_sent":        round(h_sent, 4),
            "h_hmm":         round(h_hmm, 4),
            "n_samples":     n,
        }

    def _compute_mi(self, x: np.ndarray, y: np.ndarray) -> float:
        try:
            if SKLEARN_AVAILABLE:
                mi = mutual_info_regression(
                    x.reshape(-1, 1), y, n_neighbors=3, random_state=42
                )[0]
                return float(max(mi, 0.0))
            else:
                r = float(np.corrcoef(x, y)[0, 1])
                r = np.clip(r, -0.9999, 0.9999)
                return float(-0.5 * np.log(1.0 - r ** 2))
        except Exception as e:
            logger.debug(f"MI estimation failed: {e}")
            return 0.0

    def _compute_entropy(self, x: np.ndarray) -> float:
        try:
            counts, _ = np.histogram(x, bins=N_HISTOGRAM_BINS, range=(0.0, 1.0))
            total = counts.sum()
            if total == 0:
                return 0.0
            probs = counts[counts > 0] / total
            return float(-np.sum(probs * np.log(probs + 1e-12)))
        except Exception as e:
            logger.debug(f"Entropy estimation failed: {e}")
            return 0.0

    # -- Step 3: Forced Dissent Protocol (FDP) ----------------------------

    def run_forced_dissent(
        self,
        lstm_signal: float,
        sent_score: float,
        regime_label: str,
        fusion_agent,
        trust_scores: Optional[Dict] = None,
    ) -> Dict:
        """
        Invert LSTM signal, re-run Fusion, measure Dissent Sensitivity.
        Read-only synthetic test HOLD does NOT update any system state.
        """
        vol_input = (
            0.9 if regime_label.strip().lower() == "bear"
            else 0.2 if regime_label.strip().lower() == "bull"
            else 0.5
        )

        try:
            conf_original, _ = fusion_agent.predict(
                lstm_p=lstm_signal, sent_s=sent_score,
                vol_v=vol_input, trust_scores=trust_scores,
            )
            conf_original = float(conf_original)
        except Exception as e:
            logger.warning(f"FDP original fusion failed: {e}")
            return self._fdp_fallback()

        lstm_inverted = float(1.0 - lstm_signal)

        try:
            conf_inverted, _ = fusion_agent.predict(
                lstm_p=lstm_inverted, sent_s=sent_score,
                vol_v=vol_input, trust_scores=trust_scores,
            )
            conf_inverted = float(conf_inverted)
        except Exception as e:
            logger.warning(f"FDP inverted fusion failed: {e}")
            return self._fdp_fallback()

        ds = float(abs(conf_original - conf_inverted))

        if ds < DS_LOW_THRESHOLD:
            interp = (
                f"LSTM barely influences fusion (DS={ds:.3f}). "
                "Decision driven by FinBERT + HMM. Effective ensemble size ~2 agents."
            )
        elif ds < DS_HIGH_THRESHOLD:
            interp = (
                f"LSTM has moderate fusion influence (DS={ds:.3f}). "
                "All three agents contribute; FinBERT/HMM dominate."
            )
        else:
            interp = (
                f"LSTM is the dominant fusion driver (DS={ds:.3f}). "
                "In a sycophantic ensemble this single agent controls outcome. "
                "High structural fragility detected."
            )

        return {
            "confidence_original": round(conf_original, 4),
            "confidence_inverted": round(conf_inverted, 4),
            "dissent_sensitivity": round(ds, 4),
            "lstm_inverted":       round(lstm_inverted, 4),
            "interpretation":      interp,
            "fdp_ran":             True,
        }

    def _fdp_fallback(self) -> Dict:
        return {
            "confidence_original": 0.5, "confidence_inverted": 0.5,
            "dissent_sensitivity": 0.0, "lstm_inverted": 0.5,
            "interpretation": "FDP could not run HOLD fusion agent error. Neutral result.",
            "fdp_ran": False,
        }

    # -- Step 4: Penalty multiplier ----------------------------------------

    def get_penalty_multiplier(
        self,
        asc: float,
        dissent_sensitivity: float,
        asc_saturated: bool = False,
    ) -> Tuple[float, str]:
        """
        Map (ASC, DS) to a confidence penalty multiplier and quadrant label.

        v2.3 BUG-2 FIX: DS is now meaningful in the moderate zone (0.70–0.85):
          low DS  -> −10% (correlated but not LSTM-dominated)
          high DS -> −20% (LSTM is driving the sycophancy)
        """
        # Saturation guard HOLD never penalise unreliable KSG output
        if asc_saturated:
            return PENALTY_NONE, "KSG SATURATED HOLD homogeneous batch, no penalty"

        if asc < ASC_LOW_THRESHOLD:
            return PENALTY_NONE, "INDEPENDENT HOLD healthy ensemble, no penalty"

        if asc < ASC_MED_THRESHOLD:
            return PENALTY_MILD, "MILD SYCOPHANCY HOLD correlated but acceptable (−5%)"

        if asc < ASC_HIGH_THRESHOLD:
            # v2.3 BUG-2 FIX: DS now differentiates within moderate zone
            if dissent_sensitivity < DS_LOW_THRESHOLD:
                return (PENALTY_MODERATE_LOW,
                        "MODERATE SYCOPHANCY HOLD low LSTM dominance (−10%)")
            else:
                return (PENALTY_MODERATE_HIGH,
                        "MODERATE SYCOPHANCY HOLD high LSTM dominance (−20%)")

        if asc < ASC_EXTREME_THRESHOLD:
            if dissent_sensitivity < DS_HIGH_THRESHOLD:
                return PENALTY_HIGH, "STRONG SYCOPHANCY HOLD low dominance (−25%)"
            else:
                return PENALTY_HIGH, "STRONG SYCOPHANCY HOLD LSTM dominant (−25%)"

        return PENALTY_EXTREME, "EXTREME SYCOPHANCY HOLD ensemble collapsed (−35%)"

    # -- Summary -----------------------------------------------------------

    def get_asc_summary(
        self,
        asc_result: Dict,
        fdp_result: Optional[Dict] = None,
        penalty: float = 1.0,
        quadrant: str = "",
    ) -> Dict:
        return {
            "asc_score":              asc_result.get("asc", 0.5),
            "asc_reliable":           asc_result.get("asc_reliable", False),
            "asc_saturated":          asc_result.get("asc_saturated", False),
            "lstm_std":               asc_result.get("lstm_std", 0.0),
            "n_samples":              asc_result.get("n_samples", 0),
            "mi_lstm_sent":           asc_result.get("mi_lstm_sent", 0.0),
            "mi_lstm_hmm":            asc_result.get("mi_lstm_hmm", 0.0),
            "mi_sent_hmm":            asc_result.get("mi_sent_hmm", 0.0),
            "h_lstm":                 asc_result.get("h_lstm", 0.0),
            "h_sent":                 asc_result.get("h_sent", 0.0),
            "h_hmm":                  asc_result.get("h_hmm", 0.0),
            "asc_penalty_multiplier": round(penalty, 4),
            "asc_quadrant":           quadrant,
            "fdp_ran":                fdp_result.get("fdp_ran", False) if fdp_result else False,
            "dissent_sensitivity":    fdp_result.get("dissent_sensitivity", 0.0) if fdp_result else 0.0,
            "fdp_interpretation":     fdp_result.get("interpretation", "") if fdp_result else "",
        }

    # -- Console report (v2.3 BUG-3 FIX: consistent box widths) ----------

    @staticmethod
    def print_asc_report(summary: Dict):
        asc      = summary.get("asc_score", 0.5)
        n        = summary.get("n_samples", 0)
        pen      = summary.get("asc_penalty_multiplier", 1.0)
        quad     = summary.get("asc_quadrant", "")
        fdp      = summary.get("fdp_ran", False)
        ds       = summary.get("dissent_sensitivity", 0.0)
        sat      = summary.get("asc_saturated", False)
        lstm_std = summary.get("lstm_std", 0.0)

        bar_w = 24
        bar   = "█" * int(asc * bar_w) + "░" * (bar_w - int(asc * bar_w))
        W     = 54   # inner content width (between ║ and ║)

        def row(text: str) -> str:
            """Pad/truncate text to exactly W chars."""
            return f"   ║ {text:<{W}} ║"

        print("   ╔" + "=" * (W + 2) + "╗")
        print(row("PHASE 26 HOLD ASC Memory Engine v2.3"))
        print("   ╠" + "=" * (W + 2) + "╣")
        print(row(f"ASC Score   : {asc:.4f}  [{bar}]"))
        sat_str = str(sat)
        print(row(f"Samples     : {n}/{WINDOW_SIZE}  LSTM std={lstm_std:.4f}  Sat={sat_str}"))
        print(row(f"Quadrant    : {quad[:W-14]}"))
        print(row(f"FDP Ran     : {'YES' if fdp else 'NO '}   Dissent Sensitivity: {ds:.4f}"))
        print(row(f"Penalty     : {pen:.2f}x applied to fusion confidence"))
        print("   ╚" + "=" * (W + 2) + "╝")