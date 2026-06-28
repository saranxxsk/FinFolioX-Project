"""
ml_engine/conflict_resolver.py  HOLD  Conflict Resolution Engine v2.5
====================================================================
PHASE 13: NEURO-SYMBOLIC ARBITRATOR

CHANGELOG v2.5 (production-activation fixes):

  ROOT CAUSE 1 FIXED HOLD Raw spread was catching false conflicts:
    lstm=0.0 (bearish) + sent=-0.08 -> sent_norm=0.46 (neutral) -> spread=0.46
    Old code: spread > 0.35 -> CONFLICT -> Bear regime -> HOLD
    WRONG: that converts a correct SELL into HOLD, hurting accuracy.
    FIX: Replace raw spread with DIRECTIONAL conflict detection.
    Conflict only fires when LSTM and sentiment point in OPPOSING directions:
      • LSTM > 0.55 (bullish) AND sent_norm < 0.45 (bearish) -> CONFLICT
      • LSTM < 0.45 (bearish) AND sent_norm > 0.55 (bullish) -> CONFLICT
      • Either is NEUTRAL (0.45–0.55) -> NO conflict (let fusion decide)
    Backup: spread ≥ 0.55 still fires as extreme-case safety net.

  ROOT CAUSE 2 FIXED HOLD Material mild adjustments not counted as arbitrated:
    In no-conflict path, GLD conf 0.83->0.71 (Δ-0.12), BAC 0.93->0.79 (Δ-0.14)
    were happening but arbitrated=False -> 1.4% rate even with real conf changes.
    FIX: Any |conf_change| > MATERIAL_CHANGE_THRESHOLD (0.02) in mild path
    sets arbitrated=True with ruling="MILD_ADJUST".

  UNCERTAINTY_HIGH = 0.15   (v2.3 fix HOLD kept)
  Systemic veto: only fires when regime != "Bull" AND tech bearish  (v2.3 kept)
  Mild penalty floor: 0.85x max  (v2.3 kept)
  Regime discount thresholds: Bull > 0.40, Sideways > 0.55  (v2.4)

Expected arbitration rate after these fixes:
  MILD_ADJUST path  : ~10–20% (risk/uncertainty penalties on agreed signals)
  Directional conflict : ~5–15% (genuine BULL vs BEAR disagreements)
  Combined           : ~15–30% per window (vs 1.4% before)

Tie-Breaker priority (unchanged):
  C. Systemic Veto      HOLD blocks if macro toxic AND bearish AND not Bull
  A. Bayesian Certainty HOLD favours lower-uncertainty agent
  B. Regime Context     HOLD aligns with prevailing market regime
     B.5 Trust Scores   HOLD Phase 14 MetaAgent if trust gap ≥ 0.10
"""

import numpy as np
import logging
from collections import defaultdict

logger = logging.getLogger("ConflictResolver")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Directional thresholds HOLD calibrated for real FinBERT scale (±0.10 to ±0.24)
# FinBERT -0.12 -> sent_norm 0.44 -> BEAR.  FinBERT +0.12 -> sent_norm 0.56 -> BULL.
BULL_DIR_THRESHOLD       = 0.55   # sent_norm or tech_score above this = BULL direction
BEAR_DIR_THRESHOLD       = 0.45   # sent_norm or tech_score below this = BEAR direction
EXTREME_SPREAD_BACKUP    = 0.65   # tighter backup for truly extreme mismatch only

UNCERTAINTY_HIGH         = 0.15   # aligned with all other modules
SYSTEMIC_VETO_THRESHOLD  = 0.70
HOLD_CONFIDENCE          = 0.51
MATERIAL_CHANGE_THRESHOLD= 0.02   # min |Δconf| to flag mild path as arbitrated

# Regime-aware risk discount (v2.4 calibrated thresholds)
REGIME_RISK_DISCOUNT = {
    "Bull":     (0.40, 0.50),   # risk > 0.40 in Bull -> halved
    "Sideways": (0.55, 0.75),   # risk > 0.55 in Sideways -> 25% cut
    "Bear":     (1.01, 1.00),   # Bear -> no discount
}

VALID_RULINGS = frozenset({
    "NO_CONFLICT", "MILD_ADJUST", "SYSTEMIC_VETO",
    "ALIGN_BULL",  "ALIGN_BEAR",  "HOLD",
    "TRUST_TECHNICAL", "TRUST_SENTIMENT_BULL", "TRUST_SENTIMENT_BEAR",
})

_UNC_PENALTY_FLOOR = 0.85   # max 15% cut for moderate mc_std


class ConflictResolver:
    """
    Neuro-Symbolic Arbitrator v2.5.

    Uses DIRECTIONAL conflict detection HOLD only fires when LSTM and
    sentiment point in genuinely opposing directions.  This prevents
    the false-conflict problem where lstm=0 + slightly-negative sentiment
    (both bearish) was being detected as a conflict and converting correct
    SELL decisions to HOLD.

    arbitrate() inputs:
        tech_score        : LSTM prob ∈ [0, 1]
        sent_score        : FinBERT ∈ [-1, +1]  (normalised internally)
        mc_std            : Monte Carlo uncertainty ≥ 0
        regime_label      : "Bull" | "Bear" | "Sideways"
        risk_score        : Correlation divergence ∈ [0, 1]
        fusion_confidence : Raw fusion output ∈ [0, 1]
        trust_scores      : optional {"technical": float, "sentiment": float}
    """

    def __init__(self,
                 bull_dir_threshold:      float = BULL_DIR_THRESHOLD,
                 bear_dir_threshold:      float = BEAR_DIR_THRESHOLD,
                 extreme_spread_backup:   float = EXTREME_SPREAD_BACKUP,
                 uncertainty_high:        float = UNCERTAINTY_HIGH,
                 systemic_veto_threshold: float = SYSTEMIC_VETO_THRESHOLD,
                 verbose:                 bool  = False):

        self.bull_dir_threshold      = bull_dir_threshold
        self.bear_dir_threshold      = bear_dir_threshold
        self.extreme_spread_backup   = extreme_spread_backup
        self.uncertainty_high        = uncertainty_high
        self.systemic_veto_threshold = systemic_veto_threshold
        self.verbose                 = verbose
        self._history: list          = []

        print("   [OK] Phase 13: Conflict Resolution Engine v2.5 (Arbitrator) Initialized.")
        print(f"      Directional: bull>{bull_dir_threshold}  bear<{bear_dir_threshold}  "
              f"spread_backup≥{extreme_spread_backup}")
        print(f"      unc_high={uncertainty_high}  veto={systemic_veto_threshold}  "
              f"material_Δ>{MATERIAL_CHANGE_THRESHOLD}")

    # --------------------------------------------------------------------------
    # MAIN ENTRY POINT
    # --------------------------------------------------------------------------
    def arbitrate(self,
                  tech_score:        float,
                  sent_score:        float,
                  mc_std:            float,
                  regime_label:      str,
                  risk_score:        float,
                  fusion_confidence: float,
                  trust_scores:      dict = None) -> dict:

        reasoning         = []
        adjusted_conf     = fusion_confidence
        ruling            = "NO_CONFLICT"
        arbitrated        = False
        regime_discounted = False
        self._trust       = trust_scores or {}

        # -- Step 0: Regime-aware risk discount --------------------------------
        risk_score, regime_discounted, discount_note = self._apply_regime_risk_discount(
            risk_score, regime_label
        )
        if regime_discounted:
            reasoning.append(discount_note)
            arbitrated = True

        # -- Step 1: Normalise sentiment to [0, 1] -----------------------------
        sent_norm = float(np.clip((sent_score + 1.0) / 2.0, 0.0, 1.0))

        # -- Step 2: Directional conflict detection (v2.5 core fix) -----------
        lstm_dir = self._classify_direction(tech_score)
        sent_dir = self._classify_direction(sent_norm)
        spread   = abs(tech_score - sent_norm)

        # Genuine conflict = OPPOSING directions (neutral = no conflict)
        directional_conflict = (
            (lstm_dir == "BULL" and sent_dir == "BEAR") or
            (lstm_dir == "BEAR" and sent_dir == "BULL")
        )
        # Extreme spread backup HOLD catches very high spread even without clear direction
        extreme_spread = (
            spread >= self.extreme_spread_backup
            and abs(sent_score) > 0.15
        )

        conflict_triggered = directional_conflict or extreme_spread

        reasoning.append(
            f"lstm_dir={lstm_dir}({tech_score:.4f})  "
            f"sent_dir={sent_dir}({sent_norm:.4f}|raw={sent_score:.4f})  "
            f"spread={spread:.4f}  "
            f"directional={directional_conflict}  extreme={extreme_spread}"
        )

        # ======================================================================
        # TIE-BREAKER C: SYSTEMIC VETO (highest priority)
        # Only fires: risk high AND tech bearish AND not Bull regime
        # ======================================================================
        tech_is_bearish = tech_score < 0.45
        veto_eligible   = (risk_score > self.systemic_veto_threshold
                           and tech_is_bearish
                           and regime_label != "Bull")
        if veto_eligible:
            reasoning.append(
                f"⛔ SYSTEMIC VETO: risk={risk_score:.4f} > {self.systemic_veto_threshold}  "
                f"tech_bearish={tech_is_bearish}  regime={regime_label}"
            )
            adjusted_conf = min(fusion_confidence * 0.30, 0.35)
            ruling        = "SYSTEMIC_VETO"
            arbitrated    = True
            return self._build_and_record(
                arbitrated, fusion_confidence, adjusted_conf,
                ruling, reasoning, regime_discounted,
                lstm_dir, sent_dir, spread
            )

        # ======================================================================
        # NO CONFLICT PATH (mild adjustments only)
        # ======================================================================
        if not conflict_triggered:
            reasoning.append(
                f"[OK] No directional conflict HOLD applying mild adjustments."
            )
            adj_before    = fusion_confidence
            adjusted_conf = self._apply_mild_adjustments(
                fusion_confidence, risk_score, mc_std, reasoning
            )
            delta = abs(adjusted_conf - adj_before)
            # v2.5 FIX: flag material mild adjustments as arbitrated
            if delta >= MATERIAL_CHANGE_THRESHOLD:
                arbitrated = True
                ruling     = "MILD_ADJUST"
                reasoning.append(
                    f"ℹ️  Material adjustment Δ={adjusted_conf-adj_before:+.4f} "
                    f"-> flagged MILD_ADJUST."
                )
            return self._build_and_record(
                arbitrated, fusion_confidence, adjusted_conf,
                ruling, reasoning, regime_discounted,
                lstm_dir, sent_dir, spread
            )

        # ======================================================================
        # CONFLICT DETECTED (directional or extreme spread)
        # ======================================================================
        conflict_type = "DIRECTIONAL" if directional_conflict else "EXTREME_SPREAD"
        reasoning.append(
            f"🚨 CONFLICT ({conflict_type}): "
            f"{lstm_dir} LSTM vs {sent_dir} Sentiment -> Arbitration."
        )
        arbitrated = True

        # -- TIE-BREAKER A: Bayesian Certainty ---------------------------------
        if mc_std > self.uncertainty_high:
            reasoning.append(
                f"🎲 Bayesian: mc_std={mc_std:.4f} > {self.uncertainty_high} "
                f"-> Technical UNCERTAIN. Deferring to Sentiment."
            )
            if sent_norm < 0.40:
                adjusted_conf = min(fusion_confidence, 0.35)
                ruling        = "ALIGN_BEAR"
                reasoning.append("   Sentiment bearish + uncertain -> capped 0.35.")
            else:
                adjusted_conf = max(fusion_confidence, sent_norm)
                ruling        = "ALIGN_BULL"
                reasoning.append(f"   Sentiment bullish + uncertain -> raised to {adjusted_conf:.4f}.")
        else:
            reasoning.append(
                f"🎲 Bayesian: mc_std={mc_std:.4f} ≤ {self.uncertainty_high} -> Both confident."
            )
            adjusted_conf, ruling = self._trust_or_regime_tiebreak(
                tech_score, sent_norm, regime_label, fusion_confidence, reasoning
            )

        return self._build_and_record(
            arbitrated, fusion_confidence, adjusted_conf,
            ruling, reasoning, regime_discounted,
            lstm_dir, sent_dir, spread
        )

    # --------------------------------------------------------------------------
    # DIRECTIONAL CLASSIFIER (v2.5 new)
    # --------------------------------------------------------------------------
    def _classify_direction(self, score: float) -> str:
        """
        Classify a 0–1 probability as BULL, BEAR, or NEUTRAL.
        Calibrated for FinBERT range: ±0.10–0.24 -> sent_norm ∈ [0.38, 0.62].
        """
        if score > self.bull_dir_threshold:  return "BULL"
        if score < self.bear_dir_threshold:  return "BEAR"
        return "NEUTRAL"

    # --------------------------------------------------------------------------
    # REGIME-AWARE RISK DISCOUNT
    # --------------------------------------------------------------------------
    def _apply_regime_risk_discount(self, risk_score: float,
                                    regime_label: str) -> tuple:
        threshold, factor = REGIME_RISK_DISCOUNT.get(regime_label, (1.01, 1.00))
        if risk_score > threshold:
            discounted = risk_score * factor
            note = (
                f"Regime={regime_label}: risk {risk_score:.3f}->{discounted:.3f} "
                f"(x{factor:.2f} discount)"
            )
            return discounted, True, note
        return risk_score, False, ""

    # --------------------------------------------------------------------------
    # TRUST SCORES + REGIME TIE-BREAKER
    # --------------------------------------------------------------------------
    def _trust_or_regime_tiebreak(self, tech_score, sent_norm, regime_label,
                                  fusion_confidence, reasoning) -> tuple:
        tech_trust = self._trust.get("technical", 1.0)
        sent_trust = self._trust.get("sentiment", 1.0)
        trust_gap  = abs(tech_trust - sent_trust)

        if self._trust and trust_gap >= 0.10:
            if tech_trust > sent_trust:
                reasoning.append(
                    f"📊 Trust: Technical ({tech_trust:.2f}) > Sentiment ({sent_trust:.2f})."
                )
                return max(fusion_confidence, tech_score * 0.90), "TRUST_TECHNICAL"
            else:
                reasoning.append(
                    f"📊 Trust: Sentiment ({sent_trust:.2f}) > Technical ({tech_trust:.2f})."
                )
                if sent_norm < 0.40:
                    return min(fusion_confidence, 0.40), "TRUST_SENTIMENT_BEAR"
                return max(fusion_confidence, sent_norm * 0.90), "TRUST_SENTIMENT_BULL"
        else:
            if self._trust:
                reasoning.append(f"📊 Trust gap ({trust_gap:.2f}) < 0.10 -> Regime.")
            else:
                reasoning.append("📊 No trust data -> Regime.")
            return self._regime_tiebreak(
                tech_score, sent_norm, regime_label, fusion_confidence, reasoning
            )

    def _regime_tiebreak(self, tech_score, sent_norm, regime_label,
                         fusion_confidence, reasoning) -> tuple:
        if regime_label == "Bear":
            reasoning.append(
                "⛈️  Regime=Bear -> pessimism wins. Forcing HOLD (don't trade conflict)."
            )
            return HOLD_CONFIDENCE * 0.80, "HOLD"
        elif regime_label == "Bull":
            bullish_val = max(tech_score, sent_norm)
            reasoning.append(
                f"☀️  Regime=Bull -> optimism wins -> aligning to {bullish_val:.4f}."
            )
            return max(fusion_confidence, bullish_val * 0.90), "ALIGN_BULL"
        else:
            reasoning.append(
                "🌥️  Regime=Sideways -> preserve directional lean (soft HOLD)."
            )
            blend = max(fusion_confidence * 0.85, 0.48)
            return blend, "HOLD"

    # --------------------------------------------------------------------------
    # MILD ADJUSTMENTS (risk + uncertainty penalties)
    # --------------------------------------------------------------------------
    def _apply_mild_adjustments(self, confidence: float, risk_score: float,
                                mc_std: float, reasoning: list) -> float:
        adj = confidence

        if risk_score > 0.40:
            raw_penalty = 1.0 - (risk_score - 0.40) * 0.50
            penalty     = max(raw_penalty, 0.60)
            adj        *= penalty
            if self.verbose:
                reasoning.append(f"[WARN]  Risk penalty: x{penalty:.2f} -> {adj:.4f}")

        if mc_std > 0.05:
            raw_penalty = 1.0 - (mc_std - 0.05) * 2.0
            penalty     = max(raw_penalty, _UNC_PENALTY_FLOOR)
            adj        *= penalty
            if self.verbose:
                reasoning.append(f"[WARN]  Unc penalty: x{penalty:.2f} -> {adj:.4f}")

        return float(adj)

    # --------------------------------------------------------------------------
    # RESULT BUILDER + HISTORY LOG
    # --------------------------------------------------------------------------
    def _build_and_record(self, arbitrated, original, adjusted,
                          ruling, reasoning, regime_discounted,
                          lstm_dir="?", sent_dir="?", spread=0.0) -> dict:
        result = {
            "arbitrated":          arbitrated,
            "original_confidence": round(float(original),  4),
            "adjusted_confidence": round(float(adjusted),  4),
            "ruling":              ruling,
            "reasoning":           reasoning,
            "regime_discounted":   regime_discounted,
        }
        self._history.append({
            "arbitrated":  arbitrated,
            "original":    round(float(original), 4),
            "adjusted":    round(float(adjusted), 4),
            "ruling":      ruling,
            "conf_change": round(float(adjusted) - float(original), 4),
            "discounted":  regime_discounted,
            "lstm_dir":    lstm_dir,
            "sent_dir":    sent_dir,
            "spread":      round(spread, 4),
        })
        return result

    # --------------------------------------------------------------------------
    # BATCH STATS
    # --------------------------------------------------------------------------
    def get_stats(self) -> dict:
        if not self._history:
            return {}
        n          = len(self._history)
        arb_count  = sum(1 for h in self._history if h["arbitrated"])
        disc_count = sum(1 for h in self._history if h["discounted"])
        changes    = [h["conf_change"] for h in self._history]
        rulings    = [h["ruling"]      for h in self._history]
        rc = defaultdict(int)
        for r in rulings:
            rc[r] += 1
        # Direction breakdown
        lstm_dirs = defaultdict(int)
        sent_dirs = defaultdict(int)
        for h in self._history:
            lstm_dirs[h.get("lstm_dir", "?")] += 1
            sent_dirs[h.get("sent_dir", "?")] += 1
        return {
            "n":                    n,
            "arbitrated_count":     arb_count,
            "arbitration_rate":     round(arb_count / n * 100, 1),
            "discounted_count":     disc_count,
            "mean_conf_change_pct": round(float(np.mean(changes)) * 100, 2),
            "max_conf_drop":        round(float(min(changes)), 4),
            "max_conf_rise":        round(float(max(changes)), 4),
            "ruling_counts":        dict(rc),
            "lstm_dir_dist":        dict(lstm_dirs),
            "sent_dir_dist":        dict(sent_dirs),
        }

    def reset_history(self):
        self._history.clear()

    # --------------------------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------------------------
    @staticmethod
    def print_report(result: dict):
        print("\n   ⚖️  [Conflict Arbitrator v2.5] Decision Audit")
        print("   " + "-" * 60)
        tag = "[OK] NO_CONFLICT" if result["ruling"] == "NO_CONFLICT" else f"🚨 {result['ruling']}"
        print(f"      Ruling   : {tag}")
        if result.get("regime_discounted"):
            print(f"      🎯 Regime risk discount applied")
        delta = result["adjusted_confidence"] - result["original_confidence"]
        print(f"      Before   : {result['original_confidence']:.4f}")
        print(f"      After    : {result['adjusted_confidence']:.4f}  (Δ{delta:+.4f})")
        print(f"      Reasoning:")
        for i, r in enumerate(result["reasoning"], 1):
            print(f"        {i}. {r}")
        print("   " + "-" * 60)

    # --------------------------------------------------------------------------
    # LEGACY COMPAT
    # --------------------------------------------------------------------------
    def get_regret_penalty(self, regret_score: float) -> float:
        if regret_score <= 0.01: return 0.0
        if regret_score <= 0.05: return -0.15
        if regret_score <= 0.15: return -0.30
        return -0.50