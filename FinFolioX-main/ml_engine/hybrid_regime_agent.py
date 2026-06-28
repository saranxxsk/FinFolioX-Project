"""
ml_engine/hybrid_regime_agent.py
==================================
FinFolioX HOLD Hybrid Regime Agent v2.3.1
=======================================
Backbone : Pure-NumPy GaussianHMM  (Baum-Welch EM + Viterbi)
Amplifier: Rule-based scoring (MA, VIX, Breadth, Momentum)
Fusion   : Soft blend 0.65/0.35 when HMM/Rules disagree
Stability: 5-day Viterbi majority vote

Public interface consumed by orchestrator and master system:
    from ml_engine.hybrid_regime_agent import HybridRegimeAgent

    agent = HybridRegimeAgent(hmm_model_path="saved_models/hmm_regime_hybrid.pkl")
    label, vol, conf = agent.detect(hist_df, ticker)

    Returns
    -------
    label : str   HOLD "Bull" | "Bear" | "Sideways"
    vol   : float HOLD 21-day decimal daily vol (e.g. 0.012)
    conf  : float HOLD fusion confidence 0.25 – 0.98

No hmmlearn required. Only: numpy, pandas, scikit-learn, joblib.
"""

from __future__ import annotations
import io, os, sys, time, warnings, contextlib
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# -------------------------------------------------------------
#  SILENCE HELPER
# -------------------------------------------------------------
@contextlib.contextmanager
def _silent():
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err


# -------------------------------------------------------------
#  SYNTHETIC DATA  (used for auto-training + validation)
# -------------------------------------------------------------
def _synthetic_ohlcv(start="2003-01-01", end="2024-12-31", seed=42):
    rng   = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    n     = len(dates)
    if n < 60:
        dates = pd.bdate_range(start=start, periods=300); n = 300
    third = n // 3
    rets  = np.concatenate([
        rng.normal( 0.0006, 0.007, third),
        rng.normal(-0.0008, 0.018, third),
        rng.normal( 0.0001, 0.011, n - 2 * third),
    ])
    price = 4000 * np.exp(np.cumsum(rets))
    c     = pd.Series(price, index=dates[:len(rets)])
    df    = pd.DataFrame({
        "Open"  : c.shift(1).fillna(c.iloc[0]),
        "High"  : c * (1 + np.abs(rng.normal(0, 0.005, len(c)))),
        "Low"   : c * (1 - np.abs(rng.normal(0, 0.005, len(c)))),
        "Close" : c,
        "Volume": rng.integers(1_000_000, 5_000_000, len(c)),
    }, index=c.index)
    df.index.name = "Date"
    return df


def _bear_data(n=600):
    rng = np.random.default_rng(52)
    r   = rng.normal(-0.0009, 0.019, n)
    p   = 4000 * np.exp(np.cumsum(r))
    idx = pd.bdate_range("2007-01-01", periods=n)
    c   = pd.Series(p, index=idx)
    return pd.DataFrame({"Open": c, "High": c*1.01, "Low": c*0.99,
                          "Close": c, "Volume": 1_000_000})


def _bull_data(n=600):
    rng = np.random.default_rng(99)
    r   = rng.normal(0.0007, 0.007, n)
    p   = 4000 * np.exp(np.cumsum(r))
    idx = pd.bdate_range("2019-01-01", periods=n)
    c   = pd.Series(p, index=idx)
    return pd.DataFrame({"Open": c, "High": c*1.01, "Low": c*0.99,
                          "Close": c, "Volume": 1_000_000})


# ==============================================================
#  PURE-NUMPY GAUSSIAN HMM
#  Baum-Welch EM + Viterbi. No hmmlearn dependency.
# ==============================================================
class GaussianHMM:

    def __init__(self, n_components=3, covariance_type="full",
                 n_iter=100, tol=1e-4, random_state=42):
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol    = tol
        self.rng    = np.random.default_rng(random_state)
        self.startprob_: Optional[np.ndarray] = None
        self.transmat_:  Optional[np.ndarray] = None
        self.means_:     Optional[np.ndarray] = None
        self.covars_:    Optional[np.ndarray] = None

    def _log_gauss(self, X):
        T, D = X.shape; K = self.n_components
        log_p = np.zeros((T, K))
        for k in range(K):
            diff = X - self.means_[k]
            cov  = self.covars_[k]
            try:
                L       = np.linalg.cholesky(cov + 1e-6 * np.eye(D))
                log_det = 2 * np.sum(np.log(np.diag(L)))
                sol     = np.linalg.solve(L, diff.T)
                maha    = np.sum(sol ** 2, axis=0)
            except np.linalg.LinAlgError:
                cov_r   = cov + 1e-4 * np.eye(D)
                _, log_det = np.linalg.slogdet(cov_r)
                inv     = np.linalg.inv(cov_r)
                maha    = np.einsum("ti,ij,tj->t", diff, inv, diff)
            log_p[:, k] = -0.5 * (D * np.log(2 * np.pi) + log_det + maha)
        return log_p

    def _forward(self, log_p):
        T, K  = log_p.shape
        log_A = np.log(self.transmat_ + 1e-300)
        alpha = np.full((T, K), -np.inf)
        alpha[0] = np.log(self.startprob_ + 1e-300) + log_p[0]
        for t in range(1, T):
            for k in range(K):
                alpha[t, k] = (np.logaddexp.reduce(alpha[t-1] + log_A[:, k])
                               + log_p[t, k])
        return alpha

    def _backward(self, log_p):
        T, K  = log_p.shape
        log_A = np.log(self.transmat_ + 1e-300)
        beta  = np.full((T, K), -np.inf)
        beta[-1] = 0.0
        for t in range(T - 2, -1, -1):
            for k in range(K):
                beta[t, k] = np.logaddexp.reduce(
                    log_A[k] + log_p[t + 1] + beta[t + 1])
        return beta

    def _e_step(self, X):
        log_p = self._log_gauss(X)
        alpha = self._forward(log_p)
        beta  = self._backward(log_p)
        log_A = np.log(self.transmat_ + 1e-300)
        T, K  = log_p.shape
        log_ll = np.logaddexp.reduce(alpha[-1])
        log_gamma = alpha + beta
        log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)
        log_xi = np.full((T - 1, K, K), -np.inf)
        for t in range(T - 1):
            for i in range(K):
                for j in range(K):
                    log_xi[t, i, j] = (alpha[t, i] + log_A[i, j]
                                       + log_p[t+1, j] + beta[t+1, j])
            log_xi[t] -= np.logaddexp.reduce(log_xi[t].reshape(-1))
        xi = np.exp(log_xi)
        return gamma, xi, log_ll

    def _m_step(self, X, gamma, xi):
        T, D = X.shape; K = self.n_components
        self.startprob_ = gamma[0] / (gamma[0].sum() + 1e-300)
        self.transmat_  = (xi.sum(axis=0) /
                           (xi.sum(axis=0).sum(axis=1, keepdims=True) + 1e-300))
        for k in range(K):
            w = gamma[:, k]; W = w.sum() + 1e-300
            self.means_[k] = (w @ X) / W
            diff = X - self.means_[k]
            cov  = (w[:, None, None] * (diff[:, :, None] @ diff[:, None, :])).sum(0)
            self.covars_[k] = cov / W + 1e-4 * np.eye(D)

    def _init_params(self, X):
        T, D = X.shape; K = self.n_components
        self.startprob_ = np.full(K, 1.0 / K)
        self.transmat_  = np.full((K, K), 0.1 / (K - 1))
        np.fill_diagonal(self.transmat_, 0.8)
        idx   = np.argsort(X[:, 0])
        chunk = T // K
        self.means_  = np.array([X[idx[k*chunk:(k+1)*chunk]].mean(0) for k in range(K)])
        self.covars_ = np.array([np.cov(X.T) + 1e-4 * np.eye(D)] * K)

    def fit(self, X):
        self._init_params(X)
        prev_ll = -np.inf
        for _ in range(self.n_iter):
            gamma, xi, log_ll = self._e_step(X)
            self._m_step(X, gamma, xi)
            if abs(log_ll - prev_ll) < self.tol:
                break
            prev_ll = log_ll
        return self

    def predict(self, X):
        log_p = self._log_gauss(X)
        T, K  = log_p.shape
        log_A = np.log(self.transmat_ + 1e-300)
        delta = np.full((T, K), -np.inf)
        psi   = np.zeros((T, K), dtype=int)
        delta[0] = np.log(self.startprob_ + 1e-300) + log_p[0]
        for t in range(1, T):
            for k in range(K):
                scores    = delta[t-1] + log_A[:, k]
                psi[t, k] = scores.argmax()
                delta[t, k]= scores.max() + log_p[t, k]
        states = np.zeros(T, dtype=int)
        states[-1] = delta[-1].argmax()
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states

    def predict_proba(self, X):
        log_p     = self._log_gauss(X)
        alpha     = self._forward(log_p)
        beta      = self._backward(log_p)
        log_gamma = alpha + beta
        log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
        return np.exp(log_gamma)

    def score(self, X):
        return float(np.logaddexp.reduce(
            self._forward(self._log_gauss(X))[-1]))


# ==============================================================
#  FEATURE ENGINE
# ==============================================================
class FeatureEngine:
    VOL_WINDOW = 21

    def build_hmm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"].squeeze().astype(float)
        out   = pd.DataFrame(index=df.index)
        out["log_return"] = np.log(close / close.shift(1))
        out["vol_21d"]    = out["log_return"].rolling(self.VOL_WINDOW).std()
        return out.dropna()

    def build_rule_features(self, market_data: dict) -> pd.DataFrame:
        idx  = market_data["index"].copy()
        vix  = market_data["vix"]
        nq   = market_data["nasdaq"]
        rut  = market_data["russell"]
        secs = market_data["sectors"]
        sp   = idx["sp500"].astype(float)

        for w in [20, 50, 200]:
            idx[f"ma{w}"] = sp.rolling(w).mean()
        idx["above_ma20"]  = (sp > idx["ma20"]).astype(int)
        idx["above_ma50"]  = (sp > idx["ma50"]).astype(int)
        idx["above_ma200"] = (sp > idx["ma200"]).astype(int)
        idx["ma50_vs_ma200"]  = (idx["ma50"] > idx["ma200"]).astype(int)
        idx["ma20_vs_ma50"]   = (idx["ma20"] > idx["ma50"]).astype(int)
        idx["price_vs_ma50_pct"]  = (sp - idx["ma50"])  / idx["ma50"]  * 100
        idx["price_vs_ma200_pct"] = (sp - idx["ma200"]) / idx["ma200"] * 100

        for d in [1, 5, 20, 60]:
            idx[f"ret_{d}d"] = sp.pct_change(d) * 100
        idx["mom_score"] = (
            (idx["ret_5d"]  > 0).astype(int) +
            (idx["ret_20d"] > 0).astype(int) +
            (idx["ret_60d"] > 0).astype(int)
        )

        idx["vix"]             = vix.reindex(idx.index, method="ffill")
        idx["realised_vol_20d"]= idx["ret_1d"].rolling(20).std() * np.sqrt(252)
        idx["realised_vol_ewm"]= idx["realised_vol_20d"].ewm(span=5, adjust=False).mean()
        idx["vix_change_5d"]   = idx["vix"].pct_change(5) * 100

        sec_above = {}
        for col in secs.columns:
            sec_above[col] = (secs[col] > secs[col].rolling(50).mean()).astype(int)
        idx["breadth_pct"] = (
            pd.DataFrame(sec_above).mean(axis=1)
            .reindex(idx.index, method="ffill") * 100
        )
        idx["nq_above_ma50"]  = (
            (nq > nq.rolling(50).mean()).astype(int)
            .reindex(idx.index, method="ffill")
        )
        idx["rut_above_ma50"] = (
            (rut > rut.rolling(50).mean()).astype(int)
            .reindex(idx.index, method="ffill")
        )
        idx["broad_confirm"] = idx["nq_above_ma50"] + idx["rut_above_ma50"]
        idx["vol_ratio"]     = idx["volume"] / idx["volume"].rolling(20).mean()
        return idx.dropna()


# ==============================================================
#  MARKET DATA FETCHER
# ==============================================================
class MarketDataFetcher:
    INDEX = "^GSPC"; VIX = "^VIX"; NASDAQ = "^IXIC"; RUSSELL = "^RUT"
    SECTORS = {
        "XLK": "Technology",  "XLF": "Financials", "XLE": "Energy",
        "XLV": "Healthcare",  "XLI": "Industrials", "XLP": "ConsumerStaples",
        "XLY": "ConsumerDisc","XLU": "Utilities",  "XLRE": "RealEstate",
        "XLB": "Materials",   "XLC": "Communication",
    }

    def __init__(self, lookback_days=450):
        # 450 days -> ~320 trading days.
        # MA200 warmup consumes 200 rows -> ~120 clean rows after dropna.
        # 252 only gave ~215 trading days -> 16 rows after dropna -> validation FAIL.
        self.end   = datetime.today()
        self.start = self.end - timedelta(days=lookback_days + 60)

    @staticmethod
    def _flatten(df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        return df

    def _yf(self, ticker, retries=2):
        import yfinance as yf
        for attempt in range(retries):
            try:
                with _silent():
                    df = yf.download(ticker, start=self.start, end=self.end,
                                     progress=False, auto_adjust=True)
                df = self._flatten(df)
                if not df.empty and len(df) > 30:
                    return df
            except Exception as e:
                if attempt == retries - 1:
                    raise
                time.sleep(1)
        raise ValueError(f"empty for {ticker}")

    def _stooq(self, ticker):
        import pandas_datareader.data as pdr
        sym = {"^GSPC": "^SPX", "^VIX": "^VIX",
               "^IXIC": "^NDQ", "^RUT": "^RUT"}.get(ticker, ticker)
        with _silent():
            df = pdr.DataReader(sym, "stooq", start=self.start, end=self.end)
        df = self._flatten(df).sort_index()
        if df.empty or len(df) < 30:
            raise ValueError(f"empty for {sym}")
        return df

    def _download(self, ticker):
        for fn, nm in [(self._yf, "yfinance"), (self._stooq, "stooq")]:
            try:
                return fn(ticker)
            except Exception as e:
                print(f"   [WARN]  {nm} failed for {ticker}: {type(e).__name__}")
        raise ConnectionError(f"All live sources failed for {ticker}")

    def fetch_all(self):
        print("📡 Fetching market data...")
        try:
            idx  = self._download(self.INDEX)
            idx  = idx[["Close", "Volume"]].rename(
                columns={"Close": "sp500", "Volume": "volume"})
            vix  = self._download(self.VIX)["Close"].rename("vix")
            nq   = self._download(self.NASDAQ)["Close"].rename("nasdaq")
            rut  = self._download(self.RUSSELL)["Close"].rename("russell")
            secs = {}
            for sym, nm in self.SECTORS.items():
                try:
                    secs[nm] = self._download(sym)["Close"]
                except Exception:
                    pass
            if len(secs) < 5:
                raise ValueError("Too few sectors.")
            data = dict(index=idx, vix=vix, nasdaq=nq, russell=rut,
                        sectors=pd.DataFrame(secs), source="live")
            print(f"   [OK] Live data  |  {len(idx)} trading days\n")
            return data
        except Exception as e:
            print(f"\n   [BAD] Live fetch failed: {e}")
            print("   🔄 Falling back to synthetic market data...\n")
            return self._synthetic_fallback()

    def _synthetic_fallback(self):
        print("[WARN]  OFFLINE MODE HOLD Synthetic market data.\n")
        gen = _synthetic_ohlcv("2022-01-01", "2024-12-31", seed=7)
        sp  = gen["Close"].values
        rng = np.random.default_rng(7)
        idx = pd.DataFrame({"sp500": sp, "volume": gen["Volume"].values},
                           index=gen.index)
        vix = pd.Series(np.clip(15 + 5*rng.standard_normal(len(sp)), 8, 80),
                        index=gen.index, name="vix")
        nasdaq  = pd.Series(sp * 3.5,  index=gen.index, name="nasdaq")
        russell = pd.Series(sp * 0.42, index=gen.index, name="russell")
        secs = pd.DataFrame(
            {nm: sp * (0.9 + 0.2*i/len(self.SECTORS))
             for i, nm in enumerate(self.SECTORS.values())},
            index=gen.index
        )
        return dict(index=idx, vix=vix, nasdaq=nasdaq,
                    russell=russell, sectors=secs, source="synthetic")


# ==============================================================
#  REGIME OUTPUT DATACLASS
# ==============================================================
@dataclass
class RegimeOutput:
    timestamp:      str
    regime:         str
    confidence:     float
    trend:          str
    volatility:     str
    risk_state:     str
    liquidity:      str
    vix_level:      float
    breadth_pct:    float
    momentum_score: int
    bias_5d:        str
    current_vol:    float
    data_source:    str = "live"
    hmm_state:      int = -1
    hmm_agreement:  bool = True
    conflict_flags: list = field(default_factory=list)
    policy_hint:    dict = field(default_factory=dict)
    ticker_context: Optional[dict] = None
    raw_scores:     dict = field(default_factory=dict)


# ==============================================================
#  TICKER CONTEXT ENRICHER
# ==============================================================
class TickerContextEnricher:
    DEFENSIVE = {"JNJ","PG","KO","PEP","WMT","XLP","XLU","XLRE","GLD","TLT"}
    CYCLICAL  = {"NVDA","AMD","TSLA","META","AMZN","GOOGL","XLK","XLY","XLE","XLF"}

    def enrich(self, ticker: str, output: RegimeOutput) -> dict:
        t   = ticker.upper()
        reg = output.regime
        cat = ("Defensive" if t in self.DEFENSIVE else
               "Cyclical"  if t in self.CYCLICAL  else "General")
        impact = {
            ("Bull",    "Defensive"): "May underperform HOLD rotate to growth",
            ("Bull",    "Cyclical"):  "Favourable HOLD momentum strategies apply",
            ("Bull",    "General"):   "Positive HOLD normal conviction entries",
            ("Bear",    "Defensive"): "Outperform HOLD safe-haven candidate",
            ("Bear",    "Cyclical"):  "High risk HOLD avoid or hedge",
            ("Bear",    "General"):   "Caution HOLD verify thesis before entry",
            ("Sideways","Defensive"): "Neutral HOLD range-bound",
            ("Sideways","Cyclical"):  "Low conviction HOLD avoid breakouts",
            ("Sideways","General"):   "Low conviction HOLD wait for clarity",
        }.get((reg, cat), "No specific guidance")
        action = {
            "Bull":     "Normal entries HOLD momentum / dip-buy",
            "Bear":     "Avoid new longs HOLD strong stock signal required",
            "Sideways": "Reduce size HOLD fade extremes",
        }.get(reg, "Neutral")
        return {
            "ticker": t, "sector_category": cat,
            "regime_impact": impact, "suggested_action": action,
            "note": f"[WARN]  Market-level context only HOLD not a prediction for {t}.",
        }


# ==============================================================
#  HYBRID REGIME AGENT  ← main class used by project
#
#  Used as: self.master.hybrid_regime in orchestrator
#  Key methods:
#    detect(hist_df, ticker)  -> (label, vol, confidence)   ← orchestrator
#    analyze_regime(df)       -> (label, vol)               ← test suite T1-T10
#    analyze_full(ticker)     -> RegimeOutput               ← detailed output
# ==============================================================
class HybridRegimeAgent:
    """
    FinFolioX Hybrid Regime Agent v2.3.1

    Primary method called by langgraph_orchestrator.py Node 2:
        label, vol, conf = agent.detect(hist_df, ticker)

    Also used by finfolio_system.py analyze_stock():
        label, vol, conf = self.hybrid_regime.detect(hist, ticker)
    """

    VIX_LOW          = 15
    VIX_MED          = 20
    VIX_HIGH         = 30
    BULL_MIN_BREADTH = 35.0
    CONFLICT_PENALTY = 0.12

    def __init__(self, hmm_model_path: str = None,
                 n_components: int = 3,
                 n_iter: int = 100,
                 verbose: bool = True):
        self._verbose      = verbose
        self._feat         = FeatureEngine()
        self._enrich       = TickerContextEnricher()
        self.n_components  = n_components
        self.n_iter        = n_iter

        # HMM core
        self.model:      Optional[GaussianHMM] = None
        self.scaler:     StandardScaler        = StandardScaler()
        self.regime_map: dict                  = {}
        self.is_fitted:  bool                  = False

        # Auto-load or auto-train
        if hmm_model_path and os.path.exists(hmm_model_path):
            self._load(hmm_model_path)
        else:
            if verbose:
                if hmm_model_path:
                    print(f"   [HybridRegime] {hmm_model_path} not found HOLD "
                          "auto-training on synthetic data...")
                else:
                    print("   [HybridRegime] No model path given HOLD "
                          "auto-training on synthetic data...")
            self._auto_train()
            if hmm_model_path:
                try:
                    self.save(hmm_model_path)
                except Exception as e:
                    if verbose:
                        print(f"   [HybridRegime] Could not save: {e}")

    # -- Primary interface (orchestrator + master system) ------

    def detect(self, df: pd.DataFrame,
               ticker: str = "") -> Tuple[str, float, float]:
        """
        Called by langgraph_orchestrator node_market_context and
        finfolio_system analyze_stock.

        Parameters
        ----------
        df     : hist_data (OHLCV, optionally with SMA_50/SMA_200/RSI/MACD)
        ticker : optional HOLD passed through for context, doesn't affect HMM

        Returns
        -------
        regime_label : "Bull" | "Bear" | "Sideways"
        current_vol  : decimal 21-day daily vol  (e.g. 0.012)
        confidence   : float 0.25 – 0.98
        """
        if not self.is_fitted:
            raise RuntimeError("HybridRegimeAgent not fitted.")
        try:
            hmm_label, current_vol, hmm_conf = self._hmm_predict(df)
            rule_label, rule_norm            = self._rule_score(df)
            regime, confidence, _            = self._fuse(
                hmm_label, hmm_conf, rule_label, rule_norm, df)
            return regime, current_vol, confidence
        except Exception as e:
            if self._verbose:
                print(f"   [HybridRegime] detect() error: {e} HOLD safe fallback")
            vol = float(df["Close"].pct_change().rolling(21).std().iloc[-1]) \
                  if len(df) > 21 else 0.015
            return "Sideways", vol, 0.50

    # -- T1-T10 validation interface ---------------------------

    def analyze_regime(self, df: pd.DataFrame = None,
                       market_ticker: str = "^GSPC",
                       lookback_days: int = 300) -> Tuple[str, float]:
        """
        Strict 2-tuple interface for validation test suite T1-T10.
        current_vol is decimal daily vol satisfying T3: 0.001 < vol < 0.20.
        """
        if not self.is_fitted:
            raise RuntimeError("HybridRegimeAgent not fitted.")

        if df is not None:
            df_raw = df.copy()
        else:
            try:
                import yfinance as yf
                end   = datetime.today().strftime("%Y-%m-%d")
                start = (datetime.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
                with _silent():
                    df_raw = yf.download(market_ticker, start=start, end=end,
                                         auto_adjust=True, progress=False)
                df_raw.columns = [c[0] if isinstance(c, tuple) else c
                                  for c in df_raw.columns]
            except Exception:
                df_raw = _synthetic_ohlcv()

        feat   = self._feat.build_hmm_features(df_raw)
        Xs     = self.scaler.transform(feat[["log_return", "vol_21d"]].values)
        states = self.model.predict(Xs)
        window = min(5, len(states))
        major  = Counter(states[-window:]).most_common(1)[0][0]
        label  = self.regime_map[major]
        vol    = float(feat["vol_21d"].iloc[-1])
        return label, vol

    # -- Full RegimeOutput (detailed production use) ------------

    def analyze_full(self, ticker: str = None) -> RegimeOutput:
        """
        Returns the full RegimeOutput dataclass with all signals.
        Used directly when you need VIX, breadth, trend, policy hints.
        """
        fetcher   = MarketDataFetcher(lookback_days=450)
        data      = fetcher.fetch_all()
        source    = data.get("source", "live")
        rule_feat = self._feat.build_rule_features(data)

        sp_df    = pd.DataFrame({"Close": data["index"]["sp500"]})
        hmm_feat = self._feat.build_hmm_features(sp_df)
        Xs       = self.scaler.transform(
            hmm_feat[["log_return", "vol_21d"]].values)

        states_all = self.model.predict(Xs)
        posteriors = self.model.predict_proba(Xs)
        window     = min(5, len(states_all))
        major      = Counter(states_all[-window:]).most_common(1)[0][0]
        hmm_label  = self.regime_map[major]
        hmm_conf   = float(posteriors[-window:, major].mean())
        current_vol= float(hmm_feat["vol_21d"].iloc[-1])

        rule_label, rule_norm = self._rule_score_full(rule_feat)
        rule_scores           = self._raw_scores(rule_feat)
        # Must pass iloc[-1] (Series) HOLD _fuse() expects a Series, not a DataFrame.
        # Passing the full DataFrame causes KeyError:'Close' because _fuse()
        # tries row["vix"] which returns a sub-Series on a DataFrame.
        regime, confidence, conflict_flags = self._fuse(
            hmm_label, hmm_conf, rule_label, rule_norm, rule_feat.iloc[-1])

        row = rule_feat.iloc[-1]
        b   = float(row["breadth_pct"])
        vix = float(row["vix"])

        trend = ("Strong Uptrend"    if row["above_ma200"] and row["above_ma50"] and row["ma50_vs_ma200"]
                 else "Uptrend"      if row["above_ma200"] and row["ma50_vs_ma200"]
                 else "Downtrend"    if not row["above_ma200"] and not row["ma50_vs_ma200"]
                 else "Weak / Below MA200" if not row["above_ma200"]
                 else "Ranging")

        vol_label = ("Low"      if vix < self.VIX_LOW  else
                     "Moderate" if vix < self.VIX_MED  else
                     "High"     if vix < self.VIX_HIGH else "Extreme")

        risk_state = ("Risk-On"  if regime == "Bull"  else
                      "Risk-Off" if regime == "Bear"  else "Neutral")

        liq = ("High" if float(row["vol_ratio"]) > 1.1 else
               "Low"  if float(row["vol_ratio"]) < 0.85 else "Normal")

        bias_5d = ("Up"   if row["ret_5d"] > 0.5  and rule_norm > 0 else
                   "Down" if row["ret_5d"] < -0.5 and rule_norm < 0 else "Neutral")

        out = RegimeOutput(
            timestamp      = rule_feat.index[-1].strftime("%Y-%m-%d"),
            regime         = regime,
            confidence     = confidence,
            trend          = trend,
            volatility     = vol_label,
            risk_state     = risk_state,
            liquidity      = liq,
            vix_level      = round(vix, 2),
            breadth_pct    = round(b, 1),
            momentum_score = int(row["mom_score"]),
            bias_5d        = bias_5d,
            current_vol    = current_vol,
            data_source    = source,
            hmm_state      = int(major),
            hmm_agreement  = (hmm_label == rule_label),
            conflict_flags = conflict_flags,
            policy_hint    = self._policy(regime),
            raw_scores     = rule_scores,
        )
        if ticker:
            out.ticker_context = self._enrich.enrich(ticker, out)
        return out

    # -- Training ----------------------------------------------

    def train_on_df(self, df: pd.DataFrame,
                    run_bic: bool = True) -> "HybridRegimeAgent":
        self._fit(df, run_bic=run_bic)
        return self

    def train(self, ticker: str = "^GSPC",
              start: str = "2003-01-01",
              end:   str = "2024-12-31",
              run_bic: bool = True) -> "HybridRegimeAgent":
        try:
            import yfinance as yf
            if self._verbose:
                print(f"   [HybridRegime] Downloading {ticker} {start}->{end}")
            with _silent():
                df = yf.download(ticker, start=start, end=end,
                                 auto_adjust=True, progress=False)
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df.dropna(inplace=True)
            if len(df) < 200:
                raise ValueError("Too few rows")
        except Exception as e:
            if self._verbose:
                print(f"   [HybridRegime] Download failed ({e}) HOLD using synthetic data")
            df = _synthetic_ohlcv(start, end)
        self._fit(df, run_bic=run_bic)
        return self

    def _auto_train(self):
        df = _synthetic_ohlcv("2003-01-01", "2024-12-31", seed=42)
        self._fit(df, run_bic=False)

    def _fit(self, df: pd.DataFrame, run_bic: bool = True):
        feat = self._feat.build_hmm_features(df)
        X    = feat[["log_return", "vol_21d"]].values
        Xs   = self.scaler.fit_transform(X)

        if run_bic and self._verbose:
            self._bic_validation(Xs)

        if self._verbose:
            print(f"   [HybridRegime] Training HMM | n={self.n_components} | iter={self.n_iter}")

        self.model = GaussianHMM(n_components=self.n_components,
                                  n_iter=self.n_iter, random_state=42)
        self.model.fit(Xs)
        states         = self.model.predict(Xs)
        self.regime_map= self._label_by_return(feat, states)
        self.is_fitted  = True

        if self._verbose:
            print("   [HybridRegime] Training complete.")

    def _bic_validation(self, Xs: np.ndarray, max_states: int = 5):
        print("   [HybridRegime] BIC validation:")
        bic = {}
        for n in range(2, max_states + 1):
            try:
                m = GaussianHMM(n_components=n, n_iter=50, random_state=42)
                m.fit(Xs)
                ll    = m.score(Xs)
                n_p   = n ** 2 + 2 * n * Xs.shape[1]
                bic[n]= -2 * ll + n_p * np.log(len(Xs))
                print(f"      n={n}: BIC={bic[n]:.1f}")
            except Exception as e:
                print(f"      n={n}: skipped ({e})")
        if bic:
            best = min(bic, key=bic.get)
            print(f"      BIC-optimal={best}  (using n=3 per FinFolioX contract)")

    def _label_by_return(self, feat: pd.DataFrame, states: np.ndarray) -> dict:
        avg_ret = {s: feat.loc[states == s, "log_return"].mean()
                   for s in range(self.n_components)}
        ss = sorted(avg_ret, key=avg_ret.get)
        rmap = {ss[0]: "Bear", ss[1]: "Sideways", ss[2]: "Bull"}
        if self._verbose:
            for s, lbl in rmap.items():
                print(f"      State {s} -> {lbl:8s}  avg_ret={avg_ret[s]:.5f}")
        return rmap

    # -- Save / Load -------------------------------------------

    def save(self, path: str):
        if not self.is_fitted:
            raise RuntimeError("Cannot save HOLD model not fitted.")
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".",
                    exist_ok=True)
        joblib.dump({
            "model":        self.model,
            "scaler":       self.scaler,
            "regime_map":   self.regime_map,
            "n_components": self.n_components,
            "version":      "v2.3.1",
        }, path)
        if self._verbose:
            print(f"   [HybridRegime] Saved -> {path}")

    def _load(self, path: str):
        p = joblib.load(path)
        self.model       = p["model"]
        self.scaler      = p["scaler"]
        self.regime_map  = p["regime_map"]
        self.n_components= p.get("n_components", 3)
        self.is_fitted   = True
        if self._verbose:
            ver = p.get("version", "?")
            print(f"   [HybridRegime] Loaded {ver} ← {path}")

    # -- Internal helpers --------------------------------------

    def _hmm_predict(self, df: pd.DataFrame) -> Tuple[str, float, float]:
        feat   = self._feat.build_hmm_features(df)
        Xs     = self.scaler.transform(feat[["log_return", "vol_21d"]].values)
        states = self.model.predict(Xs)
        proba  = self.model.predict_proba(Xs)
        window = min(5, len(states))
        major  = Counter(states[-window:]).most_common(1)[0][0]
        label  = self.regime_map[major]
        conf   = float(proba[-window:, major].mean())
        vol    = float(feat["vol_21d"].iloc[-1])
        return label, vol, conf

    def _rule_score(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Lightweight rule score from a plain OHLCV DataFrame."""
        close = df["Close"].squeeze().astype(float)
        ma50  = (df["SMA_50"].iloc[-1]  if "SMA_50"  in df.columns
                 else close.rolling(50).mean().iloc[-1])
        ma200 = (df["SMA_200"].iloc[-1] if "SMA_200" in df.columns
                 else close.rolling(200).mean().iloc[-1])
        ma20  = close.rolling(20).mean().iloc[-1]
        cur   = float(close.iloc[-1])

        S = {}
        S["above_ma200"] = +2 if cur > ma200 else -2
        S["above_ma50"]  = +1 if cur > ma50  else -1
        S["above_ma20"]  = +1 if cur > ma20  else -1
        S["golden"]      = +2 if ma50 > ma200 else -2

        def ret(d):
            return (close.iloc[-1] / close.iloc[-d] - 1) * 100 if len(close) >= d else 0.0

        r5, r20, r60 = ret(5), ret(20), ret(60)
        S["ret_5d"]  = +1 if r5  > 0 else -1
        S["ret_20d"] = +2 if r20 > 1 else (-2 if r20 < -1 else 0)
        S["ret_60d"] = +2 if r60 > 3 else (-2 if r60 < -3 else 0)

        vol_20 = close.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100
        S["vol"] = +1 if vol_20 < 15 else (-2 if vol_20 > 30 else -1)

        total  = sum(S.values())
        max_sc = sum(abs(v) for v in S.values())
        norm   = total / max_sc if max_sc else 0.0
        label  = "Bull" if norm >= 0.25 else "Bear" if norm <= -0.25 else "Sideways"
        return label, norm

    def _rule_score_full(self, feat: pd.DataFrame) -> Tuple[str, float]:
        """Rule score from a fully engineered rule_feat DataFrame."""
        row   = feat.iloc[-1]
        S     = {}
        S["above_ma200"]    = +2 if row["above_ma200"]    else -2
        S["above_ma50"]     = +1 if row["above_ma50"]     else -1
        S["above_ma20"]     = +1 if row["above_ma20"]     else -1
        S["golden_cross"]   = +2 if row["ma50_vs_ma200"]  else -2
        S["ma20_vs_ma50"]   = +1 if row["ma20_vs_ma50"]   else -1
        S["price_vs_ma200"] = +1 if row["price_vs_ma200_pct"] > 0 else -1
        S["ret_5d"]  = +1 if row["ret_5d"]  > 0 else -1
        S["ret_20d"] = +2 if row["ret_20d"] > 1 else (-2 if row["ret_20d"] < -1 else 0)
        S["ret_60d"] = +2 if row["ret_60d"] > 3 else (-2 if row["ret_60d"] < -3 else 0)
        S["mom_score"] = int(row["mom_score"]) - 1
        b = float(row["breadth_pct"])
        S["breadth"]       = +2 if b > 65 else +1 if b > 50 else -1 if b < 40 else -2
        S["broad_confirm"] = int(row["broad_confirm"]) - 1
        vix = float(row["vix"])
        S["vix_level"]  = +2 if vix < self.VIX_LOW else +1 if vix < self.VIX_MED else -2 if vix > self.VIX_HIGH else -1
        S["vix_change"] = -1 if row["vix_change_5d"] > 10 else +1 if row["vix_change_5d"] < -10 else 0
        total  = sum(S.values())
        max_sc = sum(abs(v) for v in S.values())
        norm   = total / max_sc if max_sc else 0.0
        label  = "Bull" if norm >= 0.25 else "Bear" if norm <= -0.25 else "Sideways"
        return label, norm

    def _raw_scores(self, feat: pd.DataFrame) -> dict:
        """Return raw integer scores for debug output."""
        row = feat.iloc[-1]
        S   = {}
        S["above_ma200"]    = +2 if row["above_ma200"]    else -2
        S["above_ma50"]     = +1 if row["above_ma50"]     else -1
        S["above_ma20"]     = +1 if row["above_ma20"]     else -1
        S["golden_cross"]   = +2 if row["ma50_vs_ma200"]  else -2
        S["ma20_vs_ma50"]   = +1 if row["ma20_vs_ma50"]   else -1
        S["price_vs_ma200"] = +1 if row["price_vs_ma200_pct"] > 0 else -1
        S["ret_5d"]   = +1 if row["ret_5d"]  > 0 else -1
        S["ret_20d"]  = +2 if row["ret_20d"] > 1 else (-2 if row["ret_20d"] < -1 else 0)
        S["ret_60d"]  = +2 if row["ret_60d"] > 3 else (-2 if row["ret_60d"] < -3 else 0)
        S["mom_score"]= int(row["mom_score"]) - 1
        b = float(row["breadth_pct"])
        S["breadth"]       = +2 if b > 65 else +1 if b > 50 else -1 if b < 40 else -2
        S["broad_confirm"] = int(row["broad_confirm"]) - 1
        vix = float(row["vix"])
        S["vix_level"]  = +2 if vix < self.VIX_LOW else +1 if vix < self.VIX_MED else -2 if vix > self.VIX_HIGH else -1
        S["vix_change"] = -1 if row["vix_change_5d"] > 10 else +1 if row["vix_change_5d"] < -10 else 0
        return {k: int(v) for k, v in S.items()}

    def _fuse(self, hmm_label: str, hmm_conf: float,
              rule_label: str, rule_norm: float,
              df_or_row) -> Tuple[str, float, list]:
        """Soft blend fusion. HMM is the stable backbone."""
        flags = []

        # Get vix and breadth proxy from df or row
        if isinstance(df_or_row, pd.DataFrame):
            close = df_or_row["Close"].squeeze().astype(float)
            vix   = float(df_or_row["vix"].iloc[-1]) if "vix" in df_or_row.columns else 18.0
            ret5  = (close.iloc[-1]/close.iloc[-5] - 1)*100 if len(close) >= 5 else 0.0
            vol20 = close.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100
            b     = 50.0  # neutral proxy when breadth not available
        else:
            # It's a Series (rule_feat row)
            row   = df_or_row
            vix   = float(row["vix"])
            ret5  = float(row["ret_5d"])
            vol20 = float(row.get("realised_vol_20d", 20.0))
            b     = float(row["breadth_pct"])

        if hmm_label == rule_label:
            regime     = hmm_label
            confidence = min(0.98, hmm_conf + abs(rule_norm) * 0.15)
        else:
            regime     = hmm_label  # HMM wins stability
            confidence = min(0.98, 0.65 * hmm_conf + 0.35 * abs(rule_norm))
            flags.append(f"HMM={hmm_label} vs Rules={rule_label} HOLD soft blend (0.65/0.35)")

        if regime == "Bull" and b < self.BULL_MIN_BREADTH:
            regime = "Sideways"
            confidence -= self.CONFLICT_PENALTY
            flags.append(f"Breadth gate: {b:.1f}% < {self.BULL_MIN_BREADTH}% HOLD Bull->Sideways")

        if regime in ("Bull", "Sideways") and vix > 22:
            flags.append(f"Elevated VIX ({vix:.1f}) for {regime}")
            confidence -= self.CONFLICT_PENALTY

        if regime == "Bear" and isinstance(df_or_row, pd.Series):
            if df_or_row.get("above_ma200", 0):
                flags.append("Price above MA200 in Bear regime")
                confidence -= self.CONFLICT_PENALTY

        confidence = round(max(min(confidence, 0.98), 0.25), 2)
        return regime, confidence, flags

    def _policy(self, regime: str) -> dict:
        return {
            "Bull":     {"position_size": "Increase",
                         "strategy":      "Momentum / Buy dips",
                         "risk":          "Normal",
                         "hedge":         "Optional"},
            "Bear":     {"position_size": "Reduce significantly",
                         "strategy":      "Short / Hedge",
                         "risk":          "Defensive",
                         "hedge":         "Required"},
            "Sideways": {"position_size": "Moderate",
                         "strategy":      "Range trading",
                         "risk":          "Selective",
                         "hedge":         "Light"},
        }.get(regime, {})

    # -- Helpers for validation suite --------------------------

    def predict_all_states(self, df: pd.DataFrame) -> Tuple[np.ndarray, list]:
        """Used by T6 (persistence) and T11 (forward accuracy)."""
        feat   = self._feat.build_hmm_features(df)
        Xs     = self.scaler.transform(feat[["log_return", "vol_21d"]].values)
        states = self.model.predict(Xs)
        labels = [self.regime_map[s] for s in states]
        return states, labels

    def predict_all_with_feat(self, df: pd.DataFrame):
        """Returns (states, labels, feat_df) for T11/T13."""
        feat   = self._feat.build_hmm_features(df)
        Xs     = self.scaler.transform(feat[["log_return", "vol_21d"]].values)
        states = self.model.predict(Xs)
        labels = [self.regime_map[s] for s in states]
        return states, labels, feat


# ==============================================================
#  PRETTY PRINTER  (standalone use / debugging)
# ==============================================================
def print_regime_output(out: RegimeOutput):
    ICONS  = {"Bull": "🟢", "Bear": "🔴", "Sideways": "🟡"}
    icon   = ICONS.get(out.regime, "❓")
    src    = "🌐 LIVE" if out.data_source == "live" else "🧪 SYNTHETIC"
    agree  = ("[OK] HMM+Rules agree" if out.hmm_agreement
               else "[WARN]  HMM/Rules disagree (soft blend)")
    print("\n" + "="*62)
    print(f"  {icon}  REGIME OUTPUT  HOLD  {out.timestamp}  [{src}]")
    print("="*62)
    print(f"  Regime         : {icon} {out.regime}")
    print(f"  Confidence     : {out.confidence:.0%}   ({agree})")
    print(f"  HMM State      : {out.hmm_state}  ->  {out.regime}")
    print(f"  Trend          : {out.trend}")
    print(f"  Volatility     : {out.volatility}  (VIX={out.vix_level})")
    print(f"  Daily Vol (dec): {out.current_vol:.4f}  (~{out.current_vol*100:.2f}%/day)")
    print(f"  Risk State     : {out.risk_state}")
    print(f"  Breadth        : {out.breadth_pct:.1f}% sectors above MA50")
    print(f"  Momentum       : {out.momentum_score}/3  |  5d-Bias: {out.bias_5d}")
    if out.conflict_flags:
        print("-"*62)
        print("  [WARN]  CONFLICT FLAGS:")
        for cf in out.conflict_flags:
            print(f"     • {cf}")
    print("-"*62)
    print("  📋 POLICY")
    for k, v in out.policy_hint.items():
        print(f"     {k:<18}: {v}")
    if out.ticker_context:
        ctx = out.ticker_context
        print("-"*62)
        print(f"  🔗 TICKER: {ctx['ticker']}  ({ctx['sector_category']})")
        print(f"     Impact  : {ctx['regime_impact']}")
        print(f"     Action  : {ctx['suggested_action']}")
    print("="*62)