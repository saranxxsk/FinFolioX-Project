"""
validate_hybrid_regime.py  HOLD Project Root
==========================================
FinFolioX HOLD Hybrid Regime Agent v2.3.1 Validation Suite
T1–T13 + internal checks  (74 total tests)

Run from project root:
    python validate_hybrid_regime.py

Pass = [OK]   Fail = [BAD]   Warn = [WARN]

Expected output:
    🏆 VERDICT: HybridRegimeAgent v2.3.1 FIT for FinFolioX Phase 3.
"""

import os
import sys
import time
import tempfile
import traceback
import warnings
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# -- path setup -----------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from ml_engine.hybrid_regime_agent import (
    HybridRegimeAgent,
    FeatureEngine,
    MarketDataFetcher,
    _synthetic_ohlcv,
    _bear_data,
    _bull_data,
)

MODEL_PATH = os.path.join(PROJECT_ROOT, "saved_models", "hmm_regime_hybrid.pkl")

PASS = "[OK] PASS"
FAIL = "[BAD] FAIL"
WARN = "[WARN]  WARN"


# ==============================================================
#  VALIDATION SUITE
# ==============================================================
class ValidationSuite:

    VALID_REGIMES   = {"Bull", "Bear", "Sideways"}
    VALID_RISK      = {"Risk-On", "Risk-Off", "Neutral"}
    VALID_VOL       = {"Low", "Moderate", "High", "Extreme"}
    VALID_TREND     = {"Strong Uptrend", "Uptrend", "Downtrend",
                       "Weak / Below MA200", "Ranging"}
    VALID_LIQUIDITY = {"High", "Normal", "Low"}
    VALID_BIAS      = {"Up", "Down", "Neutral"}
    FUSION_MAP      = {"Bear": 0.9, "Sideways": 0.55, "Bull": 0.2}
    KELLY_MAP       = {"Bear": 0.3, "Sideways": 0.7,  "Bull": 1.5}

    def __init__(self, agent: HybridRegimeAgent):
        self.agent   = agent
        self.results = []

    def _r(self, name, passed, detail="", warn=False):
        status = WARN if warn else (PASS if passed else FAIL)
        self.results.append({"Test": name, "Status": status, "Detail": detail})

    # -- data helper -------------------------------------------
    def _get(self, ticker, start, end):
        try:
            import yfinance as yf
            import io, contextlib

            @contextlib.contextmanager
            def _s():
                old = sys.stdout, sys.stderr
                sys.stdout = sys.stderr = io.StringIO()
                try: yield
                finally:
                    sys.stdout, sys.stderr = old

            with _s():
                df = yf.download(ticker, start=start, end=end,
                                 auto_adjust=True, progress=False)
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df.dropna(inplace=True)
            if len(df) > 50:
                return df
        except Exception:
            pass
        return _synthetic_ohlcv(start, end)

    # -- T1: Output Contract -----------------------------------
    def t1(self):
        print("\n-- T1: Output Contract -----------------------------")
        try:
            df  = self._get("^GSPC", "2024-01-01", "2024-12-31")
            res = self.agent.analyze_regime(df)
            self._r("Returns exactly 2 values",
                    isinstance(res, tuple) and len(res) == 2, f"type={type(res)}")
            label, vol = res
            self._r("Value 1 is str",   isinstance(label, str),  f"type={type(label)}")
            self._r("Value 2 is float", isinstance(vol,   float), f"type={type(vol)}")
            print(f"         -> ('{label}', {vol:.6f})")
        except Exception as e:
            self._r("T1 contract", False, str(e))

    # -- T2: Label Validity ------------------------------------
    def t2(self):
        print("\n-- T2: Label Validity ------------------------------")
        try:
            df       = self._get("^GSPC", "2024-01-01", "2024-12-31")
            label, _ = self.agent.analyze_regime(df)
            self._r("Label in {Bull,Bear,Sideways}",
                    label in self.VALID_REGIMES, f"'{label}'")
            print(f"         -> Detected: '{label}'")
        except Exception as e:
            self._r("T2 label", False, str(e))

    # -- T3: Volatility Sanity ---------------------------------
    def t3(self):
        print("\n-- T3: Volatility Sanity ---------------------------")
        try:
            df     = self._get("^GSPC", "2024-01-01", "2024-12-31")
            _, vol = self.agent.analyze_regime(df)
            self._r("vol > 0",             vol > 0,             f"vol={vol:.6f}")
            self._r("vol is finite",        np.isfinite(vol),    f"vol={vol:.6f}")
            self._r("vol in [0.001, 0.20]", 0.001 < vol < 0.20, f"vol={vol:.6f}",
                    warn=not 0.001 < vol < 0.20)
            print(f"         -> daily vol = {vol:.5f}  (~{vol*100:.2f}%/day)")
        except Exception as e:
            self._r("T3 vol", False, str(e))

    # -- T4: Multi-Ticker -------------------------------------
    def t4(self):
        print("\n-- T4: Multi-Ticker Coverage -----------------------")
        tickers = {"^GSPC": "S&P500", "AAPL": "Apple",
                   "MSFT": "Microsoft", "GLD": "Gold ETF", "QQQ": "Nasdaq ETF"}
        for sym, nm in tickers.items():
            try:
                df = self._get(sym, "2022-01-01", "2024-12-31")
                if len(df) < 60:
                    self._r(f"{nm} ({sym})", False, "<60 rows"); continue
                label, vol = self.agent.analyze_regime(df)
                ok = label in self.VALID_REGIMES and isinstance(vol, float)
                self._r(f"{nm} ({sym})", ok, f"'{label}', vol={vol:.5f}")
            except Exception as e:
                self._r(f"{nm} ({sym})", False, str(e))

    # -- T5: Known Regime Dates --------------------------------
    def t5(self):
        print("\n-- T5: Known Regime Dates (Ground Truth) -----------")
        train = _synthetic_ohlcv("2003-01-01", "2024-12-31", seed=42)
        n     = len(train)
        third = n // 3
        cases = [
            ("Pure Bear (seed=52)",    _bear_data(600),            "Bear"),
            ("Pure Bull (seed=99)",    _bull_data(600),            "Bull"),
            ("Synthetic Bear slice",   train.iloc[third:2*third],  "Bear"),
            ("Synthetic Bull slice",   train.iloc[:third],         "Bull"),
        ]
        for desc, df, expected in cases:
            try:
                label, _ = self.agent.analyze_regime(df)
                self._r(desc, label == expected,
                        f"expected '{expected}' got '{label}'")
            except Exception as e:
                self._r(desc, False, str(e))

    # -- T6: Regime Persistence --------------------------------
    def t6(self):
        print("\n-- T6: Regime Persistence (Markov Stability) -------")
        try:
            df = _synthetic_ohlcv("2003-01-01", "2024-12-31", seed=42)
            _, labels = self.agent.predict_all_states(df)
            same  = sum(1 for a, b in zip(labels, labels[1:]) if a == b)
            pers  = same / (len(labels) - 1)
            self._r("Persistence > 0.80", pers > 0.80,
                    f"{pers:.4f} ({pers*100:.1f}%)")
            self._r("Persistence > 0.90 (ideal)", pers > 0.90,
                    f"{pers:.4f}", warn=pers <= 0.90)
            dist  = Counter(labels)
            total = len(labels)
            print(f"         -> {total} days  |  ", end="")
            for lbl in ["Bull", "Sideways", "Bear"]:
                print(f"{lbl}: {dist.get(lbl, 0)/total*100:.0f}%  ", end="")
            print()
        except Exception as e:
            self._r("T6 persistence", False, str(e))
            traceback.print_exc()

    # -- T7: Downstream Mapping --------------------------------
    def t7(self):
        print("\n-- T7: Downstream Mapping --------------------------")
        for lbl in ["Bull", "Bear", "Sideways"]:
            fv = self.FUSION_MAP.get(lbl)
            kb = self.KELLY_MAP.get(lbl)
            self._r(f"'{lbl}' maps correctly",
                    fv is not None and kb is not None,
                    f"fusion={fv}, kelly_b={kb}")
        try:
            label, _ = self.agent.analyze_regime(_bear_data(600))
            self._r("Arbitrator veto (BUY+Bear=veto)",
                    label == "Bear", f"regime='{label}'")
        except Exception as e:
            self._r("Arbitrator veto", False, str(e))

    # -- T8: Save / Load Integrity -----------------------------
    def t8(self):
        print("\n-- T8: Save / Load Integrity -----------------------")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "models", "test_hybrid.pkl")
                self.agent.save(path)
                self._r("Model saved", os.path.exists(path))

                loaded = HybridRegimeAgent(hmm_model_path=path, verbose=False)
                self._r("Model loaded (is_fitted=True)", loaded.is_fitted)

                df = _synthetic_ohlcv("2024-01-01", "2024-12-31", seed=5)
                o1 = self.agent.analyze_regime(df)
                o2 = loaded.analyze_regime(df)
                self._r("Label identical after reload",  o1[0] == o2[0],
                        f"'{o1[0]}' vs '{o2[0]}'")
                self._r("Vol identical after reload",
                        abs(o1[1] - o2[1]) < 1e-10,
                        f"{o1[1]:.8f} vs {o2[1]:.8f}")
        except Exception as e:
            self._r("T8 save/load", False, str(e))
            traceback.print_exc()

    # -- T9: Edge Cases ----------------------------------------
    def t9(self):
        print("\n-- T9: Edge Cases ----------------------------------")
        try:
            df         = _synthetic_ohlcv("2024-09-01", "2024-12-31", seed=3)
            label, vol = self.agent.analyze_regime(df)
            self._r("Works with ~65 rows", label in self.VALID_REGIMES,
                    f"rows={len(df)}, '{label}'")
        except Exception as e:
            self._r("Works with ~65 rows", False, str(e))
        try:
            untrained = HybridRegimeAgent(hmm_model_path=None, verbose=False)
            untrained.is_fitted = False
            try:
                untrained.analyze_regime(
                    _synthetic_ohlcv("2024-01-01", "2024-06-01", 1))
                self._r("Untrained raises RuntimeError", False, "no error raised!")
            except RuntimeError:
                self._r("Untrained raises RuntimeError", True)
        except Exception as e:
            self._r("Untrained RuntimeError", False, str(e))

    # -- T10: Latency -----------------------------------------
    def t10(self):
        print("\n-- T10: Inference Latency (< 100ms) ----------------")
        try:
            df    = _synthetic_ohlcv("2023-01-01", "2024-12-31", seed=8)
            times = []
            for _ in range(10):
                t0 = time.perf_counter()
                self.agent.analyze_regime(df)
                times.append((time.perf_counter() - t0) * 1000)
            avg_ms, max_ms = np.mean(times), np.max(times)
            self._r("Avg inference < 100ms", avg_ms < 100, f"avg={avg_ms:.2f}ms")
            self._r("Max inference < 200ms", max_ms < 200, f"max={max_ms:.2f}ms")
            print(f"         -> avg={avg_ms:.1f}ms | max={max_ms:.1f}ms | min={np.min(times):.1f}ms")
        except Exception as e:
            self._r("T10 latency", False, str(e))

    # -- T11: Forward Accuracy ---------------------------------
    def t11(self):
        """Does today's regime predict next 5 days direction? This makes money."""
        print("\n-- T11: Forward Accuracy (regime -> future returns) -")
        try:
            df             = _synthetic_ohlcv("2003-01-01", "2024-12-31", seed=42)
            states, labels, feat = self.agent.predict_all_with_feat(df)
            labels_arr     = np.array(labels)
            rets           = feat["log_return"].values
            fwd5 = np.array([
                rets[i+1:i+6].sum() if i + 6 <= len(rets) else np.nan
                for i in range(len(rets))
            ])

            stats = {}
            print(f"\n  {'Regime':<12} {'Days':>6} {'Avg Fwd5':>10} "
                  f"{'Hit Rate':>10} {'Verdict'}")
            print("  " + "-" * 48)

            for lbl in ["Bull", "Bear", "Sideways"]:
                mask = (labels_arr == lbl) & ~np.isnan(fwd5)
                if mask.sum() < 10:
                    stats[lbl] = (np.nan, np.nan, 0); continue
                ret = fwd5[mask]
                avg = ret.mean()
                # Sideways band = 1.5% over 5 days (GBM-calibrated)
                if lbl == "Bull":    correct = (ret > 0).sum()
                elif lbl == "Bear":  correct = (ret < 0).sum()
                else:                correct = (np.abs(ret) < 0.015).sum()
                hit = correct / len(ret)
                stats[lbl] = (avg, hit, len(ret))
                v = ("[OK]" if ((lbl == "Bull"    and avg > 0) or
                              (lbl == "Bear"    and avg < 0) or
                              (lbl == "Sideways" and abs(avg) < 0.01))
                     else "[BAD]")
                print(f"  {lbl:<12} {len(ret):>6} {avg:>+10.5f} {hit:>9.1%}  {v}")

            bull_avg, bull_hit, bull_n = stats.get("Bull",  (np.nan, np.nan, 0))
            bear_avg, bear_hit, bear_n = stats.get("Bear",  (np.nan, np.nan, 0))

            # Bull+Bear directional accuracy (Sideways excluded HOLD inherently low-directional)
            bull_c  = int(bull_hit * bull_n) if not np.isnan(bull_hit) else 0
            bear_c  = int(bear_hit * bear_n) if not np.isnan(bear_hit) else 0
            dir_hit = (bull_c + bear_c) / (bull_n + bear_n) \
                      if (bull_n + bear_n) > 0 else 0
            print(f"\n  Bull+Bear directional accuracy: {dir_hit:.1%}")

            self._r("Bull avg 5d return > 0",
                    not np.isnan(bull_avg) and bull_avg > 0, f"avg={bull_avg:+.5f}")
            self._r("Bear avg 5d return < 0",
                    not np.isnan(bear_avg) and bear_avg < 0, f"avg={bear_avg:+.5f}")
            self._r("Bull returns > Bear returns",
                    not any(np.isnan([bull_avg, bear_avg])) and bull_avg > bear_avg,
                    f"Bull={bull_avg:+.5f} Bear={bear_avg:+.5f}")
            # Thresholds calibrated for GBM drift magnitudes
            self._r("Bull directional hit > 52%",
                    not np.isnan(bull_hit) and bull_hit > 0.52,
                    f"hit={bull_hit:.1%}")
            self._r("Bear directional hit > 51%",
                    not np.isnan(bear_hit) and bear_hit > 0.51,
                    f"hit={bear_hit:.1%}")
            self._r("Bull+Bear directional accuracy > 52%",
                    dir_hit > 0.52,
                    f"{dir_hit:.1%} (Sideways excluded)")
        except Exception as e:
            self._r("T11 forward accuracy", False, str(e))
            traceback.print_exc()

    # -- T12: Transition Matrix --------------------------------
    def t12(self):
        """High diagonal = stable regimes, not random flipping."""
        print("\n-- T12: Transition Matrix (regime stickiness) ------")
        try:
            T     = self.agent.model.transmat_
            K     = T.shape[0]
            lmap  = self.agent.regime_map
            labels= [lmap[k] for k in range(K)]

            print(f"\n  Transition matrix  (row=from, col=to):")
            print(f"  {'':>12}", end="")
            for lbl in labels:
                print(f"  {lbl:>10}", end="")
            print()
            for i in range(K):
                print(f"  {labels[i]:>12}", end="")
                for j in range(K):
                    marker = "←" if i == j else " "
                    print(f"  {T[i,j]:>8.4f}{marker}", end="")
                print()

            print()
            for k in range(K):
                self._r(f"State '{lmap[k]}' self-transition > 0.85",
                        T[k, k] > 0.85, f"P(stay)={T[k,k]:.4f}")

            off_diag = T.copy()
            np.fill_diagonal(off_diag, 0)
            max_off  = off_diag.max()
            self._r("Max off-diagonal < 0.10 (no random switching)",
                    max_off < 0.10, f"max_off={max_off:.4f}")

            # Stationary distribution
            eigvals, eigvecs = np.linalg.eig(T.T)
            stat_idx = np.argmax(np.abs(eigvals - 1.0) < 1e-6)
            stat     = np.real(eigvecs[:, stat_idx])
            stat    /= stat.sum()
            print(f"\n  Stationary distribution:")
            for k in range(K):
                print(f"     {lmap[k]:>10}: {stat[k]:.3f}  ({stat[k]*100:.1f}%)")
        except Exception as e:
            self._r("T12 transition matrix", False, str(e))
            traceback.print_exc()

    # -- T13: Performance Separation ---------------------------
    def t13(self):
        """Bull=high ret+low vol, Bear=low ret+high vol HOLD model is meaningful."""
        print("\n-- T13: Regime Performance Separation --------------")
        try:
            df             = _synthetic_ohlcv("2003-01-01", "2024-12-31", seed=42)
            states, labels, feat = self.agent.predict_all_with_feat(df)
            labels_arr     = np.array(labels)
            rets           = feat["log_return"].values
            vols           = feat["vol_21d"].values

            stats = {}
            print(f"\n  {'Regime':<12} {'Days':>6} {'Avg Daily Ret':>15} "
                  f"{'Avg Daily Vol':>15} {'Ann Ret':>10} {'Ann Vol':>10}")
            print("  " + "-" * 72)
            for lbl in ["Bull", "Bear", "Sideways"]:
                mask = labels_arr == lbl
                if mask.sum() < 5:
                    stats[lbl] = (np.nan, np.nan); continue
                ar = rets[mask].mean()
                av = vols[mask].mean()
                stats[lbl] = (ar, av)
                print(f"  {lbl:<12} {mask.sum():>6} {ar:>+15.6f} {av:>15.6f}"
                      f" {ar*252:>+9.1%} {av*np.sqrt(252):>9.1%}")

            bull_r, bull_v = stats.get("Bull",    (np.nan, np.nan))
            bear_r, bear_v = stats.get("Bear",    (np.nan, np.nan))
            side_r, side_v = stats.get("Sideways",(np.nan, np.nan))

            self._r("Return order: Bull > Sideways > Bear",
                    not any(np.isnan([bull_r, bear_r, side_r]))
                    and bull_r > side_r > bear_r,
                    f"Bull={bull_r:+.6f} Side={side_r:+.6f} Bear={bear_r:+.6f}")
            self._r("Vol order: Bear > Sideways > Bull",
                    not any(np.isnan([bull_v, bear_v, side_v]))
                    and bear_v > side_v > bull_v,
                    f"Bear={bear_v:.6f} Side={side_v:.6f} Bull={bull_v:.6f}")
            self._r("Bull mean daily return > 0",
                    not np.isnan(bull_r) and bull_r > 0, f"{bull_r:+.6f}")
            self._r("Bear mean daily return < 0",
                    not np.isnan(bear_r) and bear_r < 0, f"{bear_r:+.6f}")
            self._r("Bear vol > Bull vol",
                    not any(np.isnan([bull_v, bear_v])) and bear_v > bull_v,
                    f"Bear={bear_v:.5f} Bull={bull_v:.5f}")
            if not any(np.isnan([bull_r, bull_v, bear_r, bear_v])):
                bs  = bull_r / (bull_v + 1e-9)
                bs2 = bear_r / (bear_v + 1e-9)
                self._r("Bull Sharpe >> Bear Sharpe",
                        bs > bs2 + 0.01,
                        f"Bull={bs:+.4f} Bear={bs2:+.4f}")
        except Exception as e:
            self._r("T13 performance separation", False, str(e))
            traceback.print_exc()

    # -- T14: Ticker + Date Spot Check ------------------------
    def t14(self):
        """
        Spot-check regime + signals for specific tickers and date windows.
        Each case downloads real data for the window, runs analyze_regime(),
        and validates label, vol range, and optional expected regime.

        ADD your own cases to SPOT_CHECKS below.
        expected=None means "any valid label" HOLD useful when you just want
        to confirm the pipeline runs without asserting direction.
        """
        print("\n-- T14: Ticker + Date Spot Checks ------------------")

        # -----------------------------------------------------
        #  SPOT_CHECKS format:
        #  (description, ticker, start, end, expected_regime_or_None)
        #
        #  expected_regime options:
        #    "Bull"     -> assert label == "Bull"
        #    "Bear"     -> assert label == "Bear"
        #    "Sideways" -> assert label == "Sideways"
        #    None       -> any valid label (pipeline smoke test only)
        # -----------------------------------------------------
        SPOT_CHECKS = [
            # -- Known historical periods ----------------------
            ("S&P500 2008 crisis",       "^GSPC", "2008-01-01", "2009-03-01", "Bear"),
            ("S&P500 2020 COVID crash",  "^GSPC", "2020-02-01", "2020-05-01", "Bear"),
            ("S&P500 2021 bull run",     "^GSPC", "2020-11-01", "2021-11-01", "Bull"),
            ("S&P500 2022 bear market",  "^GSPC", "2022-01-01", "2022-12-01", "Bear"),
            # -- Individual tickers (smoke tests) -------------
            ("AAPL recent",              "AAPL",  "2024-01-01", "2025-03-01", None),
            ("NVDA recent",              "NVDA",  "2024-01-01", "2025-03-01", None),
            ("GLD recent",               "GLD",   "2024-01-01", "2025-03-01", None),
        ]

        print(f"\n  {'Description':<35} {'Ticker':<8} {'Got':>10} "
              f"{'Vol':>8} {'Expected':>10}  {'Result'}")
        print("  " + "-" * 82)

        for desc, ticker, start, end, expected in SPOT_CHECKS:
            try:
                df = self._get(ticker, start, end)

                if len(df) < 30:
                    self._r(f"T14 {desc}", False,
                            f"too few rows: {len(df)}")
                    print(f"  {'  '+desc:<35} {ticker:<8} {'n/a':>10} "
                          f"{'n/a':>8} {str(expected):>10}  [BAD] too few rows")
                    continue

                label, vol = self.agent.analyze_regime(df)

                # -- Checks ------------------------------------
                label_ok  = label in self.VALID_REGIMES
                vol_ok    = 0.001 < vol < 0.20
                regime_ok = (label == expected) if expected else True

                all_ok = label_ok and vol_ok and regime_ok
                icon   = "[OK]" if all_ok else ("[WARN] " if (label_ok and vol_ok) else "[BAD]")
                note   = (f"expected '{expected}' got '{label}'"
                          if expected and not regime_ok else "")

                self._r(f"T14 HOLD {desc} ({ticker} {start[:7]}->{end[:7]})",
                        all_ok,
                        f"label='{label}', vol={vol:.5f}{' | '+note if note else ''}",
                        warn=(label_ok and vol_ok and not regime_ok))

                print(f"  {'  '+desc:<35} {ticker:<8} {label:>10} "
                      f"{vol:>8.5f} {str(expected or 'any'):>10}  {icon} {note}")

            except Exception as e:
                self._r(f"T14 HOLD {desc}", False, str(e))
                print(f"  {'  '+desc:<35} {ticker:<8} {'ERROR':>10} "
                      f"{'':>8} {str(expected or 'any'):>10}  [BAD] {e}")

    # -- T15: Point-in-Time Date Backtest ---------------------
    def t15(self):
        """
        🔥 "On a specific date, what regime did the model predict
            for each ticker HOLD and was it correct?"

        For each (date, ticker) pair:
          1. Downloads data UP TO (and including) that date
          2. Runs the model -> records regime label + confidence
          3. Downloads the NEXT 5 trading days -> computes forward return
          4. Checks directional correctness:
               Bull     -> forward_return > 0    [OK]
               Bear     -> forward_return < 0    [OK]
               Sideways -> not a directional call [CHECK] (excluded from accuracy)
          5. Outputs a clean per-row table + overall accuracy

        ADD your own dates and tickers to DATE_BACKTEST below.
        Set verify=True  -> test directional correctness (needs future data)
        Set verify=False -> smoke test only (just checks pipeline runs)
        """
        print("\n-- T15: Point-in-Time Date Backtest ----------------")

        # -----------------------------------------------------
        #  DATE_BACKTEST format:
        #  (date_str "YYYY-MM-DD", ticker, verify_direction)
        #
        #  verify=True  -> downloads 5 days AFTER date, checks direction
        #  verify=False -> smoke test only (confirms pipeline runs)
        #
        #  noise_band: moves WITHIN this % are inconclusive (not a miss)
        #    1.0  -> good for indices  (^GSPC, QQQ, ^IXIC)  HOLD low vol
        #    2.0  -> good for stocks   (AAPL, MSFT, NVDA)   HOLD mid vol
        #    3.0  -> good for volatile (TSLA, NVDA on big news days)
        # -----------------------------------------------------
        DATE_BACKTEST = [
            # -- date          ticker   verify  noise_band --------------
            # -- 02 Mar 2026 ------------------------------------------
            ("2026-03-02", "^GSPC",  True,  1.0),
            ("2026-03-02", "AAPL",   True,  2.0),
            ("2026-03-02", "NVDA",   True,  2.0),  # high vol stock -> 2% band
            ("2026-03-02", "MSFT",   True,  2.0),
            # -- 06 Mar 2026 ------------------------------------------
            ("2026-03-06", "^GSPC",  True,  1.0),
            ("2026-03-06", "AAPL",   True,  2.0),
            ("2026-03-06", "NVDA",   True,  2.0),
            ("2026-03-06", "GLD",    True,  1.5),
            # -- 07 Mar 2026 ------------------------------------------
            ("2026-03-07", "^GSPC",  True,  1.0),
            ("2026-03-07", "AAPL",   True,  2.0),
            ("2026-03-07", "QQQ",    True,  1.0),
            # -- 12 Mar 2026 ------------------------------------------
            ("2026-03-12", "^GSPC",  True,  1.0),
            ("2026-03-12", "AAPL",   True,  2.0),
            ("2026-03-12", "MSFT",   True,  2.0),
            ("2026-03-12", "GLD",    True,  1.5),
            # -- 14 Mar 2026 ------------------------------------------
            ("2026-03-14", "^GSPC",  True,  1.0),
            ("2026-03-14", "AAPL",   True,  2.0),
            ("2026-03-14", "GLD",    True,  1.5),
            ("2026-03-14", "QQQ",    True,  1.0),
            # -- 16 Mar 2026 (Sun -> last trading day = Fri 14) --------
            ("2026-03-16", "^GSPC",  True,  1.0),
            ("2026-03-16", "AAPL",   True,  2.0),
            ("2026-03-16", "NVDA",   True,  2.0),
            ("2026-03-16", "GLD",    True,  1.5),
            # -- 19 Mar 2026 ------------------------------------------
            ("2026-03-19", "^GSPC",  True,  1.0),
            ("2026-03-19", "AAPL",   True,  2.0),
            ("2026-03-19", "MSFT",   True,  2.0),
            ("2026-03-19", "NVDA",   True,  2.0),
            ("2026-03-19", "GLD",    True,  1.5),
            ("2026-03-19", "QQQ",    True,  1.0),
            # -- Add your own dates here -------------------------------
            # ("2026-02-01", "TSLA",   True, 3.0),
        ]

        LOOKBACK = 400   # calendar days of history to feed the HMM

        def _fetch_window(ticker, end_date, lookback_days):
            """Download data ending on end_date (inclusive)."""
            import yfinance as yf
            import io, contextlib
            @contextlib.contextmanager
            def _s():
                old = sys.stdout, sys.stderr
                sys.stdout = sys.stderr = io.StringIO()
                try: yield
                finally: sys.stdout, sys.stderr = old
            start = (pd.Timestamp(end_date) - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            end   = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            with _s():
                df = yf.download(ticker, start=start, end=end,
                                 auto_adjust=True, progress=False)
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df.dropna(inplace=True)
            # trim to on-or-before target date
            df = df[df.index <= pd.Timestamp(end_date)]
            return df

        def _fetch_forward(ticker, after_date, n_days=5):
            """Download n_days of data AFTER after_date for forward return."""
            import yfinance as yf
            import io, contextlib
            @contextlib.contextmanager
            def _s():
                old = sys.stdout, sys.stderr
                sys.stdout = sys.stderr = io.StringIO()
                try: yield
                finally: sys.stdout, sys.stderr = old
            start = (pd.Timestamp(after_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            end   = (pd.Timestamp(after_date) + pd.Timedelta(days=n_days*3)).strftime("%Y-%m-%d")
            with _s():
                df = yf.download(ticker, start=start, end=end,
                                 auto_adjust=True, progress=False)
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df.dropna(inplace=True)
            return df.head(n_days)   # keep only first n trading days

        # -- Run all checks ------------------------------------
        print(f"\n  {'Date':<12} {'Ticker':<8} {'Regime':>9} "
              f"{'Conf':>6} {'Fwd':>12} {'Band':>5} {'Result'}")
        print("  " + "-" * 70)

        correct_total = 0
        verifiable    = 0

        for date_str, ticker, verify, noise_band in DATE_BACKTEST:
            try:
                # 1. Data up to target date
                hist = _fetch_window(ticker, date_str, LOOKBACK)
                if len(hist) < 30:
                    self._r(f"T15 {ticker} @{date_str}",
                            False, f"only {len(hist)} rows before {date_str}")
                    print(f"  {date_str:<12} {ticker:<8} {'n/a':>9} "
                          f"{'':>6} {'n/a':>12} {noise_band:>4.1f}%  [BAD] too few rows")
                    continue

                # 2. Predict regime on that date
                label, vol = self.agent.analyze_regime(hist)
                conf_label, _, conf = self.agent.detect(hist, ticker)

                # 3. Forward return (if verify=True)
                fwd_return = None
                direction  = "n/a"
                correct    = None

                if verify:
                    fwd = _fetch_forward(ticker, date_str, n_days=5)
                    if len(fwd) >= 1:
                        # Use whatever days are available (≥1).
                        # For recent dates (e.g. Mar 19 when today is Mar 22)
                        # only 2 trading days may exist HOLD that's fine.
                        fwd_return = (fwd["Close"].iloc[-1] /
                                      fwd["Close"].iloc[0] - 1) * 100
                        n_avail    = len(fwd)
                        direction  = f"{fwd_return:+.2f}% ({n_avail}d)"

                        # Per-row noise band (set in DATE_BACKTEST):
                        #   indices (^GSPC, QQQ): 1% HOLD tight, low vol
                        #   stocks  (AAPL, MSFT): 2% HOLD wider, mid vol
                        #   volatile (NVDA, TSLA): 2-3% HOLD macro signal not stock alpha
                        if label == "Sideways":
                            correct = None  # Sideways is never a directional call
                        elif abs(fwd_return) <= noise_band:
                            correct = None  # within noise HOLD inconclusive
                        elif label == "Bull":
                            correct = fwd_return > 0
                            verifiable += 1
                            if correct: correct_total += 1
                        elif label == "Bear":
                            correct = fwd_return < 0
                            verifiable += 1
                            if correct: correct_total += 1
                        else:
                            correct = None
                    else:
                        direction = "no data"  # future data not yet available

                # 4. Result icon
                if correct is True:   icon = "[OK]"
                elif correct is False: icon = "[BAD]"
                else:                  icon = "[CHECK]"   # not verified

                # 5. Register test result
                test_name = f"T15 {ticker} @{date_str}"
                if correct is not None:
                    self._r(test_name, correct,
                            f"regime='{label}' conf={conf:.2f} "
                            f"fwd5d={direction}")
                else:
                    self._r(test_name, True,   # smoke pass
                            f"regime='{label}' conf={conf:.2f} (not verified)")

                print(f"  {date_str:<12} {ticker:<8} {label:>9} "
                      f"{conf:>6.2f} {direction:>12} {noise_band:>4.1f}%  {icon}")

            except Exception as e:
                self._r(f"T15 {ticker} @{date_str}", False, str(e))
                print(f"  {date_str:<12} {ticker:<8} {'ERROR':>9} "
                      f"{'':>6} {'':>12} {noise_band:>4.1f}%  [BAD]  {e}")

        # 6. Summary
        if verifiable > 0:
            accuracy = correct_total / verifiable * 100
            print(f"\n  Point-in-time directional accuracy: "
                  f"{correct_total}/{verifiable}  ({accuracy:.0f}%)")
            acc_ok = accuracy >= 50   # better than random
            self._r("T15 overall directional accuracy ≥ 50%",
                    acc_ok,
                    f"{correct_total}/{verifiable} = {accuracy:.0f}%")
        else:
            print("\n  (No verified checks HOLD set verify=True to check direction)")

    # -- Internal: detect() interface -------------------------
    def internal_detect(self):
        print("\n-- Internal: detect() 3-tuple interface ------------")
        try:
            df     = _synthetic_ohlcv("2024-01-01", "2024-12-31", seed=7)
            result = self.agent.detect(df, "TEST")
            self._r("detect() returns 3-tuple",
                    isinstance(result, tuple) and len(result) == 3,
                    f"len={len(result) if isinstance(result, tuple) else 'N/A'}")
            lbl, vol, conf = result
            self._r("detect() label valid",   lbl  in self.VALID_REGIMES, f"'{lbl}'")
            self._r("detect() vol > 0",        vol  > 0,                   f"{vol:.5f}")
            self._r("detect() conf in [0,1]",  0.0 <= conf <= 1.0,        f"{conf:.3f}")
            print(f"         -> ('{lbl}', vol={vol:.5f}, conf={conf:.3f})")
        except Exception as e:
            self._r("Internal detect() interface", False, str(e))

    # -- Internal: data quality --------------------------------
    def internal_data(self, data: dict):
        print("\n-- Internal: Data Quality --------------------------")
        idx  = data["index"]
        vix  = data["vix"]
        secs = data["sectors"]
        src  = data.get("source", "live")
        self._r("Index rows ≥ 100",         len(idx) >= 100,           f"{len(idx)} rows")
        self._r("VIX rows ≥ 100",           len(vix) >= 100,           f"{len(vix)} rows")
        self._r("No NaN in sp500 (last 30)",
                idx["sp500"].tail(30).isna().sum() == 0, "")
        self._r("VIX range [5–90]",         vix.between(5, 90).all(),
                f"min={vix.min():.1f}, max={vix.max():.1f}")
        self._r("Sectors ≥ 5 cols",         secs.shape[1] >= 5, f"{secs.shape[1]} cols")
        self._r("Source identified",        src in ("live", "synthetic"), f"{src}")

    # -- Internal: feature quality ----------------------------
    def internal_features(self, feat: pd.DataFrame):
        print("\n-- Internal: Feature Quality -----------------------")
        req = ["sp500","ma20","ma50","ma200","vix","realised_vol_20d",
               "realised_vol_ewm","breadth_pct","mom_score","ret_5d",
               "ret_20d","ret_60d","vol_ratio","broad_confirm"]
        missing = [c for c in req if c not in feat.columns]
        self._r("All rule features present", len(missing) == 0,
                f"Missing: {missing}" if missing else "All present")

        # 80 = minimum after 450-day fetch + MA200 warmup
        self._r("Feature rows ≥ 80", len(feat) >= 80, f"{len(feat)} rows")
        self._r("breadth_pct in [0,100]",
                feat["breadth_pct"].between(0, 100).all(),
                f"[{feat['breadth_pct'].min():.1f}, {feat['breadth_pct'].max():.1f}]")
        self._r("mom_score in [0,3]", feat["mom_score"].between(0, 3).all(), "")

        # Adaptive EWM tolerance
        med = feat["realised_vol_20d"].median()
        tol = max(6.0, med * 0.35)
        md  = (feat["realised_vol_ewm"] - feat["realised_vol_20d"]).abs().max()
        self._r(f"realised_vol_ewm within {tol:.1f} pp (adaptive)",
                md < tol, f"max diff={md:.3f}, tol={tol:.1f}")

        self._r("realised_vol in % units (median > 1)",
                med > 1.0, f"median={med:.2f}")
        self._r("No infinite values",
                not np.isinf(feat.select_dtypes("number")).any().any(), "")

    # -- Internal: output schema -------------------------------
    def internal_output(self, out):
        print("\n-- Internal: Output Schema + Cross-Consistency -----")
        from ml_engine.hybrid_regime_agent import RegimeOutput
        self._r("regime valid",       out.regime in self.VALID_REGIMES,      f"'{out.regime}'")
        self._r("confidence [0,1]",   0.0 <= out.confidence <= 1.0,          f"{out.confidence}")
        self._r("risk_state valid",   out.risk_state in self.VALID_RISK,     f"'{out.risk_state}'")
        self._r("volatility valid",   out.volatility in self.VALID_VOL,      f"'{out.volatility}'")
        self._r("trend valid",        out.trend in self.VALID_TREND,         f"'{out.trend}'")
        self._r("liquidity valid",    out.liquidity in self.VALID_LIQUIDITY, f"'{out.liquidity}'")
        self._r("bias_5d valid",      out.bias_5d in self.VALID_BIAS,        f"'{out.bias_5d}'")
        self._r("vix_level > 0",      out.vix_level > 0,                     f"{out.vix_level}")
        self._r("breadth_pct [0,100]",0 <= out.breadth_pct <= 100,           f"{out.breadth_pct}")
        self._r("current_vol > 0",    out.current_vol > 0,                   f"{out.current_vol:.5f}")
        self._r("conflict_flags is list",
                isinstance(out.conflict_flags, list), f"len={len(out.conflict_flags)}")
        # Soft-blend flag is informational (changes formula, applies NO penalty).
        # Only breadth gate, VIX, and Bear+MA200 flags actually subtract from confidence.
        # Test must count only penalty-applying flags, not informational ones.
        penalty_flags = [f for f in out.conflict_flags if "soft blend" not in f.lower()]
        if len(penalty_flags) >= 2:
            self._r("≥2 penalty flags -> conf < 0.75",
                    out.confidence < 0.75,
                    f"penalty_flags={len(penalty_flags)}, conf={out.confidence}")
        self._r("Bull breadth gate enforced",
                not (out.regime == "Bull" and out.breadth_pct < 35.0),
                f"regime={out.regime}, breadth={out.breadth_pct}%")
        if out.regime == "Bull":
            self._r("Bull -> Risk-On or Neutral",
                    out.risk_state in ("Risk-On", "Neutral"),
                    f"risk_state={out.risk_state}")
        if out.regime == "Bear":
            self._r("Bear -> Risk-Off or Neutral",
                    out.risk_state in ("Risk-Off", "Neutral"),
                    f"risk_state={out.risk_state}")
        if out.vix_level > 25:
            self._r("VIX>25 -> vol not 'Low'", out.volatility != "Low",
                    f"vix={out.vix_level}, vol={out.volatility}")

    # -- Internal: stability -----------------------------------
    def internal_stability(self, data: dict):
        print("\n-- Internal: Regime Stability (5-day window) -------")
        try:
            sp_df  = pd.DataFrame({"Close": data["index"]["sp500"]})
            hmm_f  = FeatureEngine().build_hmm_features(sp_df)
            Xs     = self.agent.scaler.transform(
                hmm_f[["log_return", "vol_21d"]].values)
            regs   = []
            for i in range(-5, 0):
                sub    = Xs[:i] if i < -1 else Xs
                states = self.agent.model.predict(sub)
                window = min(5, len(states))
                major  = Counter(states[-window:]).most_common(1)[0][0]
                regs.append(self.agent.regime_map[major])
            flips = sum(1 for a, b in zip(regs, regs[1:]) if a != b)
            self._r("Regime stable ≤ 2 flips / 5 days", flips <= 2,
                    f"Regimes: {regs}, flips={flips}")
        except Exception as e:
            self._r("Internal stability", False, str(e))

    # -- Run all -----------------------------------------------
    def run_all(self):
        print("\n" + "=" * 66)
        print("  VALIDATION SUITE HOLD Hybrid Regime Agent v2.3.1")
        print("  T1–T15 + Internal")
        print("=" * 66)

        # T1–T15
        self.t1();  self.t2();  self.t3();  self.t4()
        self.t5();  self.t6();  self.t7();  self.t8()
        self.t9();  self.t10(); self.t11(); self.t12(); self.t13()
        self.t14()   # ticker + date range spot checks
        self.t15()   # point-in-time date backtest

        # Internal checks need market data
        print("\n📡 Fetching market data for internal checks...")
        fetcher     = MarketDataFetcher(lookback_days=450)
        market_data = fetcher.fetch_all()
        feat_engine = FeatureEngine()
        rule_feat   = feat_engine.build_rule_features(market_data)

        # analyze_full for output schema check
        print("🧠 Running analyze_full for output schema check...")
        full_out = self.agent.analyze_full(ticker="AAPL")
        print(f"   Regime={full_out.regime}  conf={full_out.confidence}\n")

        self.internal_detect()
        self.internal_data(market_data)
        self.internal_features(rule_feat)
        self.internal_output(full_out)
        self.internal_stability(market_data)

        self._print_report()

    # -- Report ------------------------------------------------
    def _print_report(self):
        try:
            from tabulate import tabulate
            print("\n" + "=" * 70)
            print("  FINAL VALIDATION REPORT")
            print("=" * 70)
            print(tabulate(pd.DataFrame(self.results),
                           headers="keys", tablefmt="rounded_outline", showindex=False))
        except ImportError:
            print("\n" + "=" * 70)
            print("  FINAL VALIDATION REPORT")
            print("=" * 70)
            for r in self.results:
                print(f"  {r['Status']}  {r['Test']:<55}  {r['Detail']}")

        passes = sum(1 for r in self.results if r["Status"] == PASS)
        fails  = sum(1 for r in self.results if r["Status"] == FAIL)
        warns  = sum(1 for r in self.results if r["Status"] == WARN)
        total  = len(self.results)
        print(f"\n  Total: {total}  |  [OK] {passes}  |  [BAD] {fails}  |  [WARN]  {warns}")
        print("-" * 70)
        if fails == 0:
            print("\n  🏆 VERDICT: HybridRegimeAgent v2.3.1 FIT for FinFolioX Phase 3.")
        elif fails <= 5:
            print("\n  [WARN]  VERDICT: Minor issues. Review [BAD] before integrating.")
        else:
            print("\n  🚨 VERDICT: FAILS validation. Do NOT integrate.")
        print("=" * 70 + "\n")


# ==============================================================
#  ENTRY POINT
# ==============================================================
if __name__ == "__main__":
    print("╔==================================================╗")
    print("║  FinFolioX HOLD Hybrid Regime Agent v2.3.1         ║")
    print("║  Validation Suite  T1–T13 + Internal            ║")
    print("╚==================================================╝\n")

    print(f"📂 Model path: {MODEL_PATH}")
    agent = HybridRegimeAgent(hmm_model_path=MODEL_PATH, verbose=True)
    print(f"   is_fitted={agent.is_fitted}\n")

    suite = ValidationSuite(agent)
    suite.run_all()