"""
ml_engine/correlation_agent.py  HOLD  CorrelationDivergenceDetector v2.2
=======================================================================
CHANGES vs v2.1:

  Fix 4 HOLD Two-tier beta scaling (MCD/KO fix)
    v2.1 used a single LOW_BETA_SCALE_FACTOR=0.50 for all tickers
    with |SPY corr| < 0.20. MCD (corr=-0.09) was scaled enough to
    drop from 0.79 -> 0.63 but still above the 0.60 threshold, so
    it kept getting flagged as HIGH divergence incorrectly.

    v2.2 adds a second tier for extremely decorrelated tickers
    (|SPY corr| < 0.10): scale factor = 0.35 (stronger suppression).
    This drops MCD from 0.63 -> ~0.55, below the threshold.

    Tier 1: 0.10 ≤ |corr| < 0.20  -> scale x 0.50  (moderate low-beta)
    Tier 2:       |corr| < 0.10   -> scale x 0.35  (extremely decorrelated)

    Tickers affected: MCD (corr≈-0.09), KO (corr≈0.02), WMT (corr≈-0.13
    is in tier 1 at 0.13, unchanged).

All other fixes from v2.1 retained:
  Fix 1 HOLD XOM + energy equities skip (neutral 0.35)
  Fix 2 HOLD Beta-adjusted divergence for |SPY corr| < 0.20
  Fix 3 HOLD Predictive penalty gated by |SPY corr| >= 0.25
"""

import pickle
import os
import yfinance as yf
import pandas as pd
import numpy as np
import logging
from collections import deque

logger = logging.getLogger("CorrelationAgent")


class CorrelationDivergenceDetector:
    """
    Detects Systemic Risk by analyzing the 'Graph' of market assets.

    Core hypothesis: assets generally move in sync with their underlying
    market factors (SPY, QQQ, Rates, Volatility). When an asset breaks
    this correlation significantly, it signals an idiosyncratic anomaly
    or a potential trend reversal (Systemic Divergence).

    Divergence history is persisted to disk so the warm-up period survives
    server restarts. Ready after the first 10 unique analysis calls.
    """

    def __init__(self, lookback_window=60, cache_path=None):
        self.assets          = ["SPY", "QQQ", "TLT", "VIXY"]
        self.lookback_window = lookback_window

        if cache_path is None:
            base_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cache_path = os.path.join(base_dir, "data", "meta", "divergence_cache.pkl")
        self.cache_path = cache_path

        self.divergence_history = self._load_history()

        print("   [OK] Correlation Graph Engine Initialized.")
        if len(self.divergence_history) > 0:
            print(
                f"      [OK] Divergence history restored "
                f"({len(self.divergence_history)}/{lookback_window} samples)"
            )

    # ------------------------------------------------------------------
    # PERSISTENCE
    # ------------------------------------------------------------------
    def _load_history(self):
        try:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, "rb") as f:
                    loaded = pickle.load(f)
                return deque(loaded, maxlen=self.lookback_window)
        except Exception as e:
            logger.warning(f"Could not load divergence cache: {e}")
        return deque(maxlen=self.lookback_window)

    def _save_history(self):
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump(list(self.divergence_history), f)
        except Exception as e:
            logger.warning(f"Could not save divergence cache: {e}")

    def __repr__(self):
        return (
            f"<CorrelationDivergenceDetector "
            f"history={len(self.divergence_history)}/{self.lookback_window}>"
        )

    # ------------------------------------------------------------------
    # TICKER CLASSIFICATION
    # ------------------------------------------------------------------

    # Broad market / bond / precious metal / commodity ETFs
    MACRO_TICKERS = {
        "SPY", "QQQ", "TLT", "GLD", "SLV", "USO", "UNG",
        "DIA", "IWM", "EEM",
    }

    # Fix 1: Energy/commodity equities HOLD follow oil, not SPY
    COMMODITY_EQUITY_SKIP = {
        "XOM", "CVX", "COP", "OXY", "PSX", "VLO", "MPC",
    }

    # Fix 2 + Fix 4: Two-tier beta scaling thresholds
    VERY_LOW_BETA_CORR   = 0.10   # tier 2: extremely decorrelated (MCD, KO)
    VERY_LOW_BETA_SCALE  = 0.35   # stronger suppression for tier 2
    LOW_BETA_CORR        = 0.20   # tier 1: moderate low-beta (WMT, BA, JNJ, NFLX)
    LOW_BETA_SCALE       = 0.50   # standard suppression for tier 1

    # Fix 3: Predictive penalty only fires for correlated tickers
    PREDICTIVE_PENALTY_MIN_CORR = 0.25

    # ------------------------------------------------------------------
    # MAIN
    # ------------------------------------------------------------------
    def get_market_context(self, target_ticker="AAPL"):
        """
        Returns:
            risk_score  (float 0->1) : systemic divergence risk
            corr_matrix (DataFrame) : adjacency matrix (None for skipped tickers)
        """
        clean_target = target_ticker.replace("^", "").upper()
        tickers      = [target_ticker] + self.assets

        print(
            f"   🕸️  [Correlation Agent] Building Market Graph: "
            f"{target_ticker} vs {self.assets}..."
        )

        # Skip broad macro/commodity ETFs
        if clean_target in self.MACRO_TICKERS:
            print(
                f"      ℹ️ {clean_target} is a macro/commodity ETF "
                f"HOLD skipping equity divergence check."
            )
            return 0.3, None

        # Fix 1: Skip energy/commodity equities
        if clean_target in self.COMMODITY_EQUITY_SKIP:
            print(
                f"      ℹ️ {clean_target} is a commodity-correlated equity "
                f"HOLD equity graph not applicable. Returning neutral score."
            )
            return 0.35, None

        try:
            # 1. Fetch 6 months of data
            data = yf.download(tickers, period="6mo", progress=False)["Close"]
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data.columns = [c.replace("^", "").upper() for c in data.columns]

            missing = (
                set(t.replace("^", "").upper() for t in tickers) - set(data.columns)
            )
            if missing:
                print(f"      [WARN] Missing Graph Nodes: {missing}. Using partial graph.")
                if clean_target not in data.columns:
                    print(f"      [BAD] Target {clean_target} missing. Aborting.")
                    return 0.5, None

            # 2. Daily returns
            returns = data.pct_change().dropna()
            if len(returns) < 30:
                print("      [WARN] Insufficient data (need >30 days).")
                return 0.5, None

            # 3. Adjacency matrix HOLD last 30 days
            recent_returns = returns.tail(30)
            corr_matrix    = recent_returns.corr()

            # SPY correlation for this ticker
            spy_corr     = float(corr_matrix[clean_target].get("SPY", 0.0)) \
                           if clean_target in corr_matrix.columns else 0.0
            abs_spy_corr = abs(spy_corr)
            print(f"      - Correlation with SPY: {spy_corr:.3f}")
            if "TLT" in corr_matrix.columns and clean_target in corr_matrix.columns:
                print(
                    f"      - Correlation with TLT: "
                    f"{corr_matrix[clean_target].get('TLT', 0.0):.3f}"
                )

            # Fix 3: Predictive divergence penalty HOLD only for correlated tickers
            predictive_penalty = 0.0
            prev_returns       = returns.iloc[-60:-30] if len(returns) >= 60 else None
            if (prev_returns is not None
                    and len(prev_returns) >= 30
                    and abs_spy_corr >= self.PREDICTIVE_PENALTY_MIN_CORR):
                prev_matrix = prev_returns.corr()
                if clean_target in prev_matrix.columns:
                    prev_corr  = prev_matrix[clean_target].drop(clean_target)
                    curr_corr  = corr_matrix[clean_target].drop(clean_target)
                    corr_drops = (curr_corr.abs() - prev_corr.abs()) < -0.15
                    if corr_drops.sum() >= 2:
                        print(
                            f"      🚨 [Predictive] Correlation shrinking rapidly "
                            f"with {corr_drops.sum()} peers."
                        )
                        predictive_penalty = 0.15

            # 4. Graph convolution HOLD expected move
            target_corr_vector = corr_matrix[clean_target].drop(clean_target)
            latest_moves       = returns.iloc[-1]
            market_moves       = latest_moves.drop(clean_target)

            weights    = target_corr_vector.abs()
            weight_sum = weights.sum()

            if weight_sum < 1e-6:
                print("      [WARN] Weak correlations HOLD using market mean.")
                expected_move = market_moves.mean()
            else:
                expected_move = (
                    target_corr_vector * market_moves
                ).sum() / weight_sum

            actual_move    = float(latest_moves.get(clean_target, 0.0))
            raw_divergence = abs(actual_move - expected_move)

            # Fix 2 + Fix 4: Two-tier beta-adjusted divergence
            if abs_spy_corr < self.VERY_LOW_BETA_CORR:
                # Tier 2 HOLD extremely decorrelated (MCD corr≈-0.09, KO corr≈0.02)
                adjusted_divergence = raw_divergence * self.VERY_LOW_BETA_SCALE
                print(
                    f"      ℹ️ [Beta tier-2] |SPY corr|={abs_spy_corr:.2f} "
                    f"< {self.VERY_LOW_BETA_CORR} -> "
                    f"divergence {raw_divergence:.5f} x {self.VERY_LOW_BETA_SCALE} "
                    f"= {adjusted_divergence:.5f}"
                )
            elif abs_spy_corr < self.LOW_BETA_CORR:
                # Tier 1 HOLD moderate low-beta (WMT corr≈-0.13, BA corr≈0.14, JNJ≈-0.11)
                adjusted_divergence = raw_divergence * self.LOW_BETA_SCALE
                print(
                    f"      ℹ️ [Beta tier-1] |SPY corr|={abs_spy_corr:.2f} "
                    f"< {self.LOW_BETA_CORR} -> "
                    f"divergence {raw_divergence:.5f} x {self.LOW_BETA_SCALE} "
                    f"= {adjusted_divergence:.5f}"
                )
            else:
                adjusted_divergence = raw_divergence

            # 5. Z-score normalisation (persisted history)
            self.divergence_history.append(adjusted_divergence)
            self._save_history()

            if len(self.divergence_history) >= 10:
                mean_div = np.mean(self.divergence_history)
                std_div  = np.std(self.divergence_history)
                if std_div > 1e-6:
                    z_score    = (adjusted_divergence - mean_div) / std_div
                    risk_score = 1.0 / (1.0 + np.exp(-z_score))
                else:
                    risk_score = 0.5
            else:
                warm_up = len(self.divergence_history)
                print(f"      ℹ️ Warming up ({warm_up}/10 samples)...")
                risk_score = 0.5

            risk_score = float(max(0.0, min(1.0, risk_score + predictive_penalty)))
            return risk_score, corr_matrix

        except Exception as e:
            print(f"      [WARN] Graph Calculation Error: {e}")
            import traceback
            traceback.print_exc()
            return 0.5, None