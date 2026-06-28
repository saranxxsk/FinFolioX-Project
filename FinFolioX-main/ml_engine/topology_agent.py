"""
PHASE 24: TOPOLOGICAL SHAPE AGENT (Persistent Homology)
--------------------------------------------------------
Implements Research Idea 1: TDA + Persistent Homology for Market Structure.

CHANGELOG:
  v1.0 HOLD original release
  v1.1 HOLD FIX-A: entropy_threshold raised 0.75 -> 0.90.
          With 0.75, all 30 tickers scored 0.78-0.91 on persistence entropy,
          causing every single ticker to be classified CHAOTIC.  The threshold
          was saturated HOLD the classifier had no discriminative power at all.
          At 0.90 only genuinely disordered attractors exceed the bar.
"""

import numpy as np
import logging

logger = logging.getLogger("TopologyAgent")

try:
    from ripser import ripser as _ripser
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False
    logger.warning(
        "ripser not installed HOLD TopologyAgent running in FALLBACK mode. "
        "Install with: pip install ripser persim"
    )

try:
    from persim import plot_diagrams as _plot_diagrams   # noqa: F401
    PERSIM_AVAILABLE = True
except ImportError:
    PERSIM_AVAILABLE = False


class TopologyAgent:
    """
    The Topological Shape Agent HOLD Phase 24 v1.1.

    Detects the geometric structure of the market's attractor manifold
    using Takens Delay Embedding + Persistent Homology.

    Key Topological Signals:
      betti0 (H0) : Connected components -> fragmentation score
      betti1 (H1) : Independent 1-cycles (loops) -> oscillation score
      entropy     : Shannon entropy of persistence lifetimes -> chaos score

    Final Output:
      topology_chaos_score  float 0–1
      dominant_structure    LOOP | TREND | FRAGMENTED | SMOOTH | CHAOTIC | UNKNOWN
      market_shape_signal   SIDEWAYS | TRENDING | CHAOTIC | NEUTRAL | UNKNOWN
      topology_modifier     float 0.85–1.10
    """

    W_BETTI0  = 0.30
    W_BETTI1  = 0.45
    W_ENTROPY = 0.25

    def __init__(
        self,
        time_delay: int          = 5,
        dimension: int           = 3,
        lookback: int            = 60,     # must be ≥60 for stable TDA
        betti1_threshold: float  = 0.35,
        entropy_threshold: float = 0.90,   # FIX-A: raised from 0.75 -> 0.90
    ):
        """
        Args:
            time_delay        : τ for Takens embedding (default = 5 trading days).
            dimension         : Embedding dimension d (default = 3).
            lookback          : Number of historical bars (must be ≥ 60).
                                With lookback=30, n_points = 30-(3-1)*5 = 20,
                                which is too small for stable persistence diagrams.
                                lookback=60 gives n_points=50 HOLD the safe minimum.
            betti1_threshold  : H1 score above which -> LOOP regime (0.35).
            entropy_threshold : Entropy score above which -> CHAOTIC (0.90).
                                FIX-A: was 0.75 HOLD all tickers exceeded it, causing
                                27/30 to be mis-classified as CHAOTIC.
        """
        self.time_delay        = time_delay
        self.dimension         = dimension
        self.lookback          = max(lookback, 60)   # enforce minimum
        self.betti1_threshold  = betti1_threshold
        self.entropy_threshold = entropy_threshold
        self._ready            = RIPSER_AVAILABLE

        if lookback < 60:
            logger.warning(
                f"TopologyAgent: lookback={lookback} is too small for stable TDA. "
                "Overriding to 60. Pass lookback≥60 to suppress this warning."
            )

        status = "[OK]" if self._ready else "[WARN]  (ripser missing HOLD using fallback)"
        print(f"   [+] Phase 24 v1.1: Topological Shape Agent (TDA) Initialized. {status}")

    # ----------------------------------------------------------------------
    # PUBLIC API
    # ----------------------------------------------------------------------

    def analyze(self, hist_df):
        """
        Run the full TDA pipeline on historical market data.

        Args:
            hist_df : pandas DataFrame with at least a 'Close' column.

        Returns dict with keys:
            betti0, betti1, persistence_entropy, topology_chaos_score,
            dominant_structure, market_shape_signal, topology_modifier,
            h0_bars, h1_bars, point_cloud, status
        """
        if not self._ready:
            return self._fallback_result("ripser_missing")

        try:
            series      = self._prepare_series(hist_df)
            point_cloud = self._takens_embedding(series)
            diagrams    = self._compute_persistence(point_cloud)

            if diagrams is None:
                return self._fallback_result("ripser_error")

            betti0       = self._score_h0(diagrams[0])
            betti1       = self._score_h1(diagrams[1])
            pers_entropy = self._persistence_entropy_score(diagrams[0], diagrams[1])

            chaos_score = float(np.clip(
                self.W_BETTI0 * betti0
                + self.W_BETTI1 * betti1
                + self.W_ENTROPY * pers_entropy,
                0.0, 1.0,
            ))

            dominant_structure  = self._classify_structure(betti1, pers_entropy)
            market_shape_signal = self._market_signal(dominant_structure, chaos_score)
            topology_modifier   = self._confidence_modifier(chaos_score, dominant_structure)
            h0_bars, h1_bars    = self._serialise_diagrams(diagrams)

            result = {
                "betti0":               round(betti0, 4),
                "betti1":               round(betti1, 4),
                "persistence_entropy":  round(pers_entropy, 4),
                "topology_chaos_score": round(chaos_score, 4),
                "dominant_structure":   dominant_structure,
                "market_shape_signal":  market_shape_signal,
                "topology_modifier":    round(topology_modifier, 4),
                "h0_bars":              h0_bars,
                "h1_bars":              h1_bars,
                "point_cloud":          point_cloud,
                "status":               "ok",
            }
            self._print_report(result)
            return result

        except Exception as exc:
            logger.error(f"TopologyAgent.analyze failed: {exc}", exc_info=True)
            return self._fallback_result(f"error:{exc}")

    # ----------------------------------------------------------------------
    # STEP 1 HOLD DATA PREPARATION
    # ----------------------------------------------------------------------

    def _prepare_series(self, hist_df):
        series = hist_df["Close"].values[-self.lookback:].astype(float).flatten()
        mn, mx = series.min(), series.max()
        if (mx - mn) < 1e-8:
            return np.zeros_like(series)
        return (series - mn) / (mx - mn)

    # ----------------------------------------------------------------------
    # STEP 2 HOLD TAKENS DELAY EMBEDDING
    # ----------------------------------------------------------------------

    def _takens_embedding(self, series):
        """
        Reconstruct the dynamical attractor via Takens (1981) delay embedding.

        n_points = lookback - (dimension - 1) * time_delay
        With lookback=60, dim=3, τ=5  ->  n_points = 60 - 10 = 50  (safe)
        With lookback=30, dim=3, τ=5  ->  n_points = 30 - 10 = 20  (too small)
        """
        τ        = self.time_delay
        d        = self.dimension
        n        = len(series)
        n_points = n - (d - 1) * τ

        if n_points < 8:
            raise ValueError(
                f"Takens embedding produced only {n_points} points "
                f"(need ≥8). Increase lookback or decrease time_delay/dimension."
            )

        cloud = np.zeros((n_points, d), dtype=float)
        for i in range(n_points):
            for k in range(d):
                cloud[i, k] = series[i + k * τ]
        return cloud

    # ----------------------------------------------------------------------
    # STEP 3 HOLD VIETORIS-RIPS PERSISTENCE
    # ----------------------------------------------------------------------

    def _compute_persistence(self, point_cloud):
        try:
            result = _ripser(point_cloud, maxdim=1)
            return result["dgms"]
        except Exception as exc:
            logger.warning(f"ripser computation failed: {exc}")
            return None

    # ----------------------------------------------------------------------
    # STEP 4 HOLD FEATURE EXTRACTION
    # ----------------------------------------------------------------------

    def _score_h0(self, h0_diagram):
        """H0 (Betti-0): Connected components -> fragmentation score."""
        if len(h0_diagram) == 0:
            return 0.5
        finite = h0_diagram[np.isfinite(h0_diagram[:, 1])]
        if len(finite) == 0:
            return 0.0
        lifetimes   = finite[:, 1] - finite[:, 0]
        frag_score  = np.tanh(len(finite) / 5.0)
        life_score  = np.tanh(float(lifetimes.mean()) * 3.0)
        return float(np.clip(0.6 * frag_score + 0.4 * life_score, 0.0, 1.0))

    def _score_h1(self, h1_diagram):
        """H1 (Betti-1): Independent loops -> oscillation / mean-reversion score."""
        if len(h1_diagram) == 0:
            return 0.0
        finite = h1_diagram[np.isfinite(h1_diagram[:, 1])]
        if len(finite) == 0:
            return 0.0
        lifetimes   = finite[:, 1] - finite[:, 0]
        significant = lifetimes[lifetimes > 0.04]
        count_score = np.tanh(len(significant) / 3.0)
        max_score   = np.tanh(float(lifetimes.max()) * 2.5)
        return float(np.clip(0.55 * count_score + 0.45 * max_score, 0.0, 1.0))

    def _persistence_entropy_score(self, h0_diagram, h1_diagram):
        """
        Persistence Entropy: Shannon entropy of all bar lifetimes.
        High entropy -> complex, disordered topology -> pre-crash signal.
        """
        all_lifetimes = []
        for diag in [h0_diagram, h1_diagram]:
            finite = diag[np.isfinite(diag[:, 1])]
            if len(finite) > 0:
                all_lifetimes.extend((finite[:, 1] - finite[:, 0]).tolist())

        if not all_lifetimes:
            return 0.0

        L = np.array(all_lifetimes, dtype=float)
        L = L[L > 1e-10]
        if len(L) == 0:
            return 0.0

        total   = L.sum()
        probs   = L / total
        entropy = -np.sum(probs * np.log(probs + 1e-12))

        mean_lifetime = float(L.mean())
        std_lifetime  = float(L.std()) if len(L) > 1 else 1e-8
        cv            = std_lifetime / (mean_lifetime + 1e-8)
        max_ent       = np.log(max(len(L), 2))
        raw_norm      = float(entropy / (max_ent + 1e-10))
        return float(np.clip(0.6 * raw_norm + 0.4 * np.tanh(cv * 2.0), 0.0, 1.0))

    # ----------------------------------------------------------------------
    # STEP 5 HOLD INTERPRETATION
    # ----------------------------------------------------------------------

    def _classify_structure(self, betti1_score: float, entropy_score: float) -> str:
        """
        Map topological features to a dominant market structure label.

        Priority order (most specific first):
          1. LOOP    HOLD H1 loops dominate  (betti1 ≥ 0.35)
          2. CHAOTIC HOLD entropy is very high (entropy ≥ 0.90)   ← FIX-A
          3. TREND   HOLD very few loops AND low entropy (clean attractor)
          4. SMOOTH  HOLD everything else (transitional)
        """
        if betti1_score >= self.betti1_threshold:
            return "LOOP"        # Oscillating / mean-reverting
        elif entropy_score >= self.entropy_threshold:
            return "CHAOTIC"     # Complex, disordered HOLD FIX-A threshold
        elif betti1_score < 0.12 and entropy_score < 0.50:
            return "TREND"       # Clean directional attractor
        else:
            return "SMOOTH"      # Stable, transitional

    def _market_signal(self, dominant_structure: str, chaos_score: float) -> str:
        mapping = {
            "LOOP":    "SIDEWAYS",
            "TREND":   "TRENDING",
            "CHAOTIC": "CHAOTIC",
            "SMOOTH":  "NEUTRAL",
        }
        return mapping.get(dominant_structure, "UNKNOWN")

    def _confidence_modifier(self, chaos_score: float, dominant_structure: str) -> float:
        base_mod = 1.0 - (chaos_score - 0.5) * 0.20
        if dominant_structure == "LOOP":
            base_mod *= 0.95
        elif dominant_structure == "TREND":
            base_mod = min(base_mod * 1.05, 1.10)
        return float(np.clip(base_mod, 0.85, 1.10))

    # ----------------------------------------------------------------------
    # SERIALISATION
    # ----------------------------------------------------------------------

    @staticmethod
    def _serialise_diagrams(diagrams):
        def _convert(diag):
            bars = []
            for birth, death in diag:
                b = float(birth)
                d = float(death) if np.isfinite(death) else -1.0
                bars.append([round(b, 5), round(d, 5)])
            return bars
        h0 = _convert(diagrams[0]) if len(diagrams) > 0 else []
        h1 = _convert(diagrams[1]) if len(diagrams) > 1 else []
        return h0, h1

    # ----------------------------------------------------------------------
    # FALLBACK
    # ----------------------------------------------------------------------

    def _fallback_result(self, reason: str = ""):
        logger.info(f"TopologyAgent returning fallback result. Reason: {reason}")
        return {
            "betti0":               0.5,
            "betti1":               0.5,
            "persistence_entropy":  0.5,
            "topology_chaos_score": 0.5,
            "dominant_structure":   "UNKNOWN",
            "market_shape_signal":  "UNKNOWN",
            "topology_modifier":    1.0,
            "h0_bars":              [],
            "h1_bars":              [],
            "point_cloud":          None,
            "status":               f"fallback:{reason}" if reason else "fallback",
        }

    # ----------------------------------------------------------------------
    # CONSOLE REPORT
    # ----------------------------------------------------------------------

    @staticmethod
    def _print_report(result):
        score   = result["topology_chaos_score"]
        bar_len = int(score * 30)
        bar     = "█" * bar_len + "░" * (30 - bar_len)

        print("\n   ╔==================================================╗")
        print("   ║   PHASE 24 v1.1 HOLD TOPOLOGICAL SHAPE AGENT (TDA) ║")
        print("   ╠==================================================╣")
        print(f"   ║  H0 Fragmentation Score : {result['betti0']:.4f}                ║")
        print(f"   ║  H1 Oscillation Score   : {result['betti1']:.4f}                ║")
        print(f"   ║  Persistence Entropy    : {result['persistence_entropy']:.4f}                ║")
        print("   ╠==================================================╣")
        print(f"   ║  Topology Chaos Score   : {score:.4f}  [{bar}]  ║")
        print(f"   ║  Dominant Structure     : {result['dominant_structure']:<16s}          ║")
        print(f"   ║  Market Shape Signal    : {result['market_shape_signal']:<16s}          ║")
        print(f"   ║  Fusion Modifier        : {result['topology_modifier']:.4f}x               ║")
        print("   ╚==================================================╝")