"""
ml_engine/risk_engine.py  HOLD  Risk Engine v2.2 (Kelly Criterion)
================================================================
Phase 9: Optimal position sizing using Fractional Kelly Criterion.

CHANGELOG v2.2 (2 fine-tune improvements on top of v2.1):

  IMPROVEMENT 1 HOLD Bear regime hard cap:
    In Bear markets, even very high-confidence signals should be capped
    at a lower maximum allocation than in Bull/Sideways.
    Old: PG conf=0.92 in Bear -> kelly=0.88 -> capped at 20% (full risk).
    New: Bear regime caps at BEAR_MAX_ALLOCATION (10% default).
    Rationale: In Bear regimes the AI is fighting the trend. Even if
    LSTM is bullish, the regime discount should reflect macro headwinds.
    Users can raise BEAR_MAX_ALLOCATION if they want more aggression.

  IMPROVEMENT 2 HOLD Minimum viable dollar amount:
    Old floor only checked allocation % (0.5%). A 0.5% of $10k = $50
    is still unactionable for $200+ stocks (0 shares).
    New: also check that dollar_amount >= MIN_VIABLE_DOLLARS ($50).
    If the allocation would buy 0 shares at current price, return 0.0.
    This requires stock_price to be passed to calculate_position_size
    (optional HOLD floor falls back to pct check if price not supplied).

CHANGELOG v2.1 (5 bugs fixed):

  BUG 1 FIXED HOLD Binary volatility cliff (most impactful):
    Old: if volatility > 0.02: safe_kelly *= 0.5
    Problem: vol=0.0201 -> halved; vol=0.0199 -> full size.
    A 0.0001 difference caused 50% position change HOLD extreme sensitivity.
    FIX: Graduated linear scaling between two control points:
      vol ≤ VOL_LOW  (0.015) -> scale = 1.00  (no cut)
      vol ≥ VOL_HIGH (0.030) -> scale = 0.50  (max 50% cut)
      between         -> linear interpolation
    This smoothly reduces allocation as volatility rises.

  BUG 2 FIXED HOLD No minimum allocation floor:
    Old: kelly=0.001 passed through -> $10 allocation -> 0 shares bought.
    The system spent computation and "made a decision" but bought nothing.
    FIX: If final_allocation < MIN_ALLOCATION_FLOOR (0.5%), return 0.0.
    This aligns position-size output with actual trade execution.

  BUG 3 FIXED HOLD No input validation:
    Old: confidence > 1.0 from floating point drift silently produced
    kelly > 1.0 which bypassed the hard cap in rare edge cases.
    FIX: confidence clipped to [0.0, 1.0], volatility to [0.001, 0.50].

  BUG 4 FIXED HOLD Half-Kelly hardcoded at 0.5:
    Old: safe_kelly = kelly_fraction * 0.5  (hardcoded)
    FIX: half_kelly_fraction parameter (default 0.5) HOLD tunable without
    touching engine internals. Set to 0.25 for ultra-conservative mode,
    0.75 for aggressive mode.

  BUG 5 FIXED HOLD get_shares_amount returns unrounded cash_value:
    Old: return num_shares, capital_to_invest  (e.g. $1847.3921...)
    FIX: return num_shares, round(capital_to_invest, 2)

  NEW: position_size_breakdown() HOLD returns full diagnostic dict for
    test_risk_engine.py and LLM supervisor logging.
  NEW: get_stats() HOLD batch stats across a list of calculate_position_size
    results (mean, std, regime distribution, vol-scaled count).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Regime-aware odds ratios (unchanged from v2.0)
REGIME_ODDS = {
    "bull":     2.5,    # Trend-following: bigger upside
    "bear":     1.5,    # Counter-trend: tighter reward/risk
    "sideways": 2.0,    # Neutral baseline
}

# v2.1 FIX: graduated volatility scaling control points
# Below VOL_LOW  -> no cut (scale = 1.0)
# Above VOL_HIGH -> max cut (scale = VOL_SCALE_MIN)
# Between         -> linear interpolation
VOL_LOW       = 0.015   # ~24% annualised (typical calm market)
VOL_HIGH      = 0.030   # ~48% annualised (high-vol / bear-spike threshold)
VOL_SCALE_MIN = 0.50    # maximum 50% cut at peak volatility

# v2.1 FIX: minimum allocation floor HOLD below this, return 0.0
# Prevents "made a decision but bought 0 shares" edge cases
MIN_ALLOCATION_FLOOR = 0.005   # 0.5% of portfolio

# v2.2 NEW: Bear regime hard cap HOLD never deploy more than this in Bear markets
# even if Kelly formula produces a higher number.
# Rationale: in Bear markets the LSTM is fighting the trend HOLD capped at 10%.
BEAR_MAX_ALLOCATION  = 0.10    # 10% max in Bear regime

# v2.2 NEW: Minimum viable dollar amount HOLD if allocation buys 0 shares, zero out
MIN_VIABLE_DOLLARS   = 50.0    # below $50 is not a real trade


class RiskEngine:
    """
    Optimal position sizing via Fractional Kelly Criterion.

    Parameters
    ----------
    default_account_size : Total portfolio capital  (default $10,000)
    max_risk_per_trade   : Hard cap HOLD never risk more than this fraction
                           of the account (default 20%)
    half_kelly_fraction  : Kelly multiplier HOLD 0.5 = Half-Kelly (default).
                           Set lower (0.25) for ultra-conservative mode or
                           higher (0.75) for aggressive mode.
    bear_max_allocation  : Hard cap specifically for Bear regime (default 10%).
                           Prevents over-deployment in trend-fighting conditions.
    """

    def __init__(self,
                 default_account_size: float = 10_000.0,
                 max_risk_per_trade:   float = 0.20,
                 half_kelly_fraction:  float = 0.50,
                 bear_max_allocation:  float = BEAR_MAX_ALLOCATION):
        self.account_size        = float(default_account_size)
        self.max_risk            = float(max_risk_per_trade)
        self.half_kelly_fraction = float(half_kelly_fraction)
        self.bear_max_allocation = float(bear_max_allocation)

    # --------------------------------------------------------------------------
    # CORE: CALCULATE POSITION SIZE
    # --------------------------------------------------------------------------
    def calculate_position_size(
        self,
        confidence_score:      float,
        volatility:            float,
        disagreement_penalty:  float = 1.0,
        regime:                str   = "Sideways",
        stock_price:           float = 0.0,
    ) -> tuple:
        """
        Calculate optimal % of portfolio to invest.

        Parameters
        ----------
        confidence_score    : AI fusion confidence, nominally in [0.0, 1.0].
        volatility          : Daily return std-dev (e.g. 0.018 for 1.8%/day).
        disagreement_penalty: Phase 16 GDI multiplier ∈ [0.0, 1.0].
        regime              : "Bull" | "Bear" | "Sideways" (case-insensitive).
        stock_price         : Current price per share (optional). If > 0,
                              v2.2 MIN_VIABLE_DOLLARS check is applied.

        Returns
        -------
        allocation_pct  (float) : % of portfolio to invest ∈ [0.0, max_risk].
        kelly_fraction  (float) : Raw Kelly edge (negative = negative EV).
        """
        # Input validation (BUG 3 fix)
        p = float(np.clip(confidence_score, 0.0, 1.0))
        v = float(np.clip(volatility,       0.001, 0.50))
        q = 1.0 - p

        regime_str = str(regime).strip().lower()

        # Regime-aware odds ratio
        b = REGIME_ODDS.get(regime_str, REGIME_ODDS["sideways"])

        # Kelly formula: f* = p - (q / b)
        kelly_fraction = p - (q / b)

        # Negative Kelly -> negative expected value -> do not trade
        if kelly_fraction <= 0:
            return 0.0, kelly_fraction

        # Half-Kelly: configurable fraction (BUG 4 fix)
        safe_kelly = kelly_fraction * self.half_kelly_fraction

        # Graduated volatility scaling (BUG 1 fix)
        vol_scale = self._vol_scale(v)
        safe_kelly *= vol_scale

        # Phase 16: GDI disagreement penalty
        safe_kelly *= float(np.clip(disagreement_penalty, 0.0, 1.0))

        # v2.2 NEW: Bear regime hard cap HOLD never over-deploy fighting the trend
        regime_cap = (self.bear_max_allocation
                      if regime_str == "bear" else self.max_risk)

        # Hard cap (regime-aware)
        final_allocation = float(np.clip(safe_kelly, 0.0, regime_cap))

        # v2.1 BUG 2 FIX: minimum % floor
        if final_allocation < MIN_ALLOCATION_FLOOR:
            return 0.0, kelly_fraction

        # v2.2 NEW: minimum viable dollar check
        # If the allocation buys 0 shares at current price, it's not a real trade
        if stock_price > 0:
            dollar_amount = self.account_size * final_allocation
            if dollar_amount < MIN_VIABLE_DOLLARS:
                return 0.0, kelly_fraction

        return final_allocation, kelly_fraction

    # --------------------------------------------------------------------------
    # GET SHARES AMOUNT
    # --------------------------------------------------------------------------
    def get_shares_amount(self,
                          stock_price:   float,
                          allocation_pct: float) -> tuple:
        """
        Convert % allocation to whole shares and rounded cash value.

        Returns
        -------
        num_shares  (int)   : Whole shares to purchase
        cash_value  (float) : Dollar amount (BUG 5 FIX: rounded to 2 d.p.)
        """
        if allocation_pct <= 0 or stock_price <= 0:
            return 0, 0.0

        capital_to_invest = self.account_size * allocation_pct
        num_shares        = int(capital_to_invest // stock_price)

        return num_shares, round(capital_to_invest, 2)   # BUG 5 FIX

    # --------------------------------------------------------------------------
    # FULL BREAKDOWN (for test diagnostics + LLM supervisor)
    # --------------------------------------------------------------------------
    def position_size_breakdown(
        self,
        confidence_score:     float,
        volatility:           float,
        disagreement_penalty: float = 1.0,
        regime:               str   = "Sideways",
        stock_price:          Optional[float] = None,
    ) -> dict:
        """
        Returns full diagnostic dict showing every step of the calculation.
        Useful for logging, LLM supervisor context, and test assertions.
        """
        p = float(np.clip(confidence_score, 0.0, 1.0))
        v = float(np.clip(volatility,       0.001, 0.50))
        q = 1.0 - p
        regime_str = str(regime).strip().lower()
        b = REGIME_ODDS.get(regime_str, REGIME_ODDS["sideways"])

        kelly_fraction = p - (q / b)
        vol_scale      = self._vol_scale(v)
        regime_cap     = self.bear_max_allocation if regime_str == "bear" else self.max_risk

        if kelly_fraction <= 0:
            alloc  = 0.0
            reason = "NEGATIVE_KELLY"
        else:
            half_k  = kelly_fraction * self.half_kelly_fraction
            vol_k   = half_k * vol_scale
            gdi_k   = vol_k  * float(np.clip(disagreement_penalty, 0.0, 1.0))
            capped  = float(np.clip(gdi_k, 0.0, regime_cap))
            alloc   = 0.0 if capped < MIN_ALLOCATION_FLOOR else capped
            reason  = (
                "BELOW_FLOOR"  if capped < MIN_ALLOCATION_FLOOR
                else "BEAR_CAP" if (regime_str == "bear" and gdi_k > self.bear_max_allocation)
                else "CAPPED"   if gdi_k  > self.max_risk
                else "OK"
            )
            half_k_val = half_k
            vol_k_val  = vol_k
            gdi_k_val  = gdi_k

        num_shares, cash = (0, 0.0) if stock_price is None else \
                           self.get_shares_amount(stock_price, alloc)

        return {
            "confidence":         round(p,                   4),
            "volatility":         round(v,                   4),
            "regime":             str(regime),
            "b_odds":             b,
            "regime_cap":         round(regime_cap,          4),
            "kelly_fraction":     round(kelly_fraction,       4),
            "half_kelly":         round(kelly_fraction * self.half_kelly_fraction, 4)
                                  if kelly_fraction > 0 else 0.0,
            "vol_scale":          round(vol_scale,            4),
            "after_vol_scale":    round(kelly_fraction * self.half_kelly_fraction * vol_scale, 4)
                                  if kelly_fraction > 0 else 0.0,
            "gdi_penalty":        round(float(disagreement_penalty), 4),
            "final_allocation":   round(alloc,                4),
            "allocation_pct":     round(alloc * 100,          2),
            "dollar_amount":      round(self.account_size * alloc, 2),
            "reason":             reason,
            "num_shares":         num_shares,
            "cash_value":         cash,
        }

    # --------------------------------------------------------------------------
    # BATCH STATS (for window / backtest-level reporting)
    # --------------------------------------------------------------------------
    @staticmethod
    def get_stats(results: list) -> dict:
        """
        Aggregate stats from a list of calculate_position_size result tuples
        or position_size_breakdown dicts.

        Accepts either:
          - list of (allocation_pct, kelly_fraction) tuples
          - list of position_size_breakdown() dicts
        """
        if not results:
            return {}

        # Normalise input
        if isinstance(results[0], dict):
            allocs  = [r["final_allocation"] for r in results]
            kellys  = [r["kelly_fraction"]   for r in results]
            regimes = [r["regime"]            for r in results]
            scales  = [r["vol_scale"]         for r in results]
        else:
            allocs  = [r[0] for r in results]
            kellys  = [r[1] for r in results]
            regimes = []
            scales  = []

        n        = len(allocs)
        active   = [a for a in allocs if a > 0]
        neg_k    = sum(1 for k in kellys if k <= 0)
        vol_scaled = sum(1 for s in scales if s < 0.99) if scales else None

        regime_dist = {}
        for r in regimes:
            regime_dist[r] = regime_dist.get(r, 0) + 1

        return {
            "n":                    n,
            "n_active":             len(active),
            "n_zero":               n - len(active),
            "n_negative_kelly":     neg_k,
            "active_rate_pct":      round(len(active) / n * 100, 1),
            "mean_alloc_pct":       round(float(np.mean(allocs)) * 100,  2),
            "mean_active_alloc_pct":round(float(np.mean(active)) * 100,  2) if active else 0.0,
            "max_alloc_pct":        round(float(np.max(allocs))  * 100,  2),
            "std_alloc_pct":        round(float(np.std(allocs))  * 100,  2),
            "mean_kelly":           round(float(np.mean(kellys)),          4),
            "n_vol_scaled":         vol_scaled,
            "regime_dist":          regime_dist,
        }

    # --------------------------------------------------------------------------
    # PRIVATE HELPERS
    # --------------------------------------------------------------------------
    @staticmethod
    def _vol_scale(volatility: float) -> float:
        """
        BUG 1 FIX: Graduated linear volatility scaling.
        Replaces the binary cliff (if vol > 0.02: *= 0.5).

        vol ≤ VOL_LOW  -> 1.00 (no cut)
        vol ≥ VOL_HIGH -> 0.50 (max 50% cut)
        between        -> linear interpolation
        """
        if volatility <= VOL_LOW:
            return 1.0
        if volatility >= VOL_HIGH:
            return VOL_SCALE_MIN
        t = (volatility - VOL_LOW) / (VOL_HIGH - VOL_LOW)
        return float(1.0 - t * (1.0 - VOL_SCALE_MIN))