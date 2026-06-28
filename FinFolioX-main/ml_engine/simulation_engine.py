"""
PHASE 21: DIGITAL TWIN SIMULATION ENGINE
------------------------------------------
A risk-free parallel universe where the AI is trapped in The Matrix.

Instead of live Yahoo Finance and real news, the engine:
  1. Downloads bulk historical data for a time window
  2. Steps through it day-by-day (discrete event simulation)
  3. Feeds each slice to the existing LSTM / HMM / Fusion models
  4. Tracks a virtual portfolio (fake cash + fake shares)
  5. Runs Meta-Agent T+5 grading at the correct offsets
  6. Supports scenario injection (flash crash, sentiment shock, regime flip)

Result: Simulate 1 year of trading in ~60 seconds.

FIX v2:
  - Win rate calculation now correctly compares entry vs exit prices
    instead of adjacent portfolio values.
  - All indentation errors in SimulationPortfolio.get_metrics() fixed.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def _calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = -delta.clip(upper=0).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _calculate_macd(series, fast=12, slow=26):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow


# ==============================================================================
# SYNTHETIC MARKET GENERATOR
# ==============================================================================
class SyntheticMarketGenerator:
    """Produces alternate-reality data for the AI to consume."""

    @staticmethod
    def download_historical(ticker, start_date, end_date, buffer_days=250):
        """
        Downloads historical data with buffer so LSTM always has enough
        lookback (200 days for SMA_200 + 60 for LSTM window).
        """
        start_buffered = (
            datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=buffer_days)
        ).strftime("%Y-%m-%d")

        print(f"   [Twin] Downloading {ticker} data: {start_buffered} -> {end_date}")
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_buffered, end=end_date)

        if hist.empty or len(hist) < 260:
            raise ValueError(
                f"Not enough data for {ticker}. Got {len(hist)} rows, need 260+."
            )

        hist["SMA_50"] = hist["Close"].rolling(window=50).mean()
        hist["SMA_200"] = hist["Close"].rolling(window=200).mean()
        hist["RSI"] = _calculate_rsi(hist["Close"])
        hist["MACD"] = _calculate_macd(hist["Close"])
        hist.dropna(inplace=True)

        if hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)

        print(f"   [Twin] Data ready: {len(hist)} trading days after indicators")
        return hist

    @staticmethod
    def generate_gbm(
        ticker="SYNTH",
        days=252,
        start_price=100.0,
        mu=0.08,
        sigma=0.20,
        seed=42,
    ):
        """Generates synthetic stock data using Geometric Brownian Motion."""
        np.random.seed(seed)
        dt = 1 / 252

        prices = [start_price]
        for _ in range(days - 1):
            shock = np.random.normal()
            price = prices[-1] * np.exp(
                (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * shock
            )
            prices.append(price)

        dates = pd.bdate_range(start="2023-01-02", periods=days)
        df = pd.DataFrame(index=dates)
        df["Close"] = prices
        df["Open"] = df["Close"].shift(1).fillna(start_price)
        df["High"] = df[["Open", "Close"]].max(axis=1) * (
            1 + np.random.uniform(0, 0.02, days)
        )
        df["Low"] = df[["Open", "Close"]].min(axis=1) * (
            1 - np.random.uniform(0, 0.02, days)
        )
        df["Volume"] = np.random.randint(1_000_000, 50_000_000, days)

        df["SMA_50"] = df["Close"].rolling(window=50).mean()
        df["SMA_200"] = df["Close"].rolling(window=200).mean()
        df["RSI"] = _calculate_rsi(df["Close"])
        df["MACD"] = _calculate_macd(df["Close"])
        df.dropna(inplace=True)

        print(f"   [Twin] GBM synthetic data generated: {len(df)} days for {ticker}")
        return df

    @staticmethod
    def generate_trending(
        ticker="SYNTH",
        days=504,
        start_price=100.0,
        seed=42,
    ):
        """
        Generates synthetic data with EXPLOITABLE regime patterns.
        Instead of random walk, creates trending segments:
          - Bull runs  (positive drift, low vol)
          - Bear drops (negative drift, high vol)
          - Sideways   (near-zero drift, moderate vol)
          - Mean-reversion pullbacks within trends

        This gives the LSTM/Fusion models actual patterns to detect.
        """
        np.random.seed(seed)
        dt = 1 / 252

        # Define regime segments
        regimes = []
        remaining = days
        while remaining > 0:
            seg_len = min(np.random.randint(30, 80), remaining)
            regime_type = np.random.choice(["bull", "bear", "sideways"], p=[0.45, 0.25, 0.30])
            regimes.append((regime_type, seg_len))
            remaining -= seg_len

        prices = [start_price]
        regime_params = {
            "bull":     {"mu": 0.25, "sigma": 0.12, "pullback_prob": 0.08},
            "bear":     {"mu": -0.20, "sigma": 0.25, "rally_prob": 0.10},
            "sideways": {"mu": 0.02, "sigma": 0.15, "revert_prob": 0.15},
        }

        for regime_type, seg_len in regimes:
            params = regime_params[regime_type]
            mu = params["mu"]
            sigma = params["sigma"]

            for j in range(seg_len):
                # Base trend
                shock = np.random.normal()
                daily_return = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shock

                # Mean-reversion events (pullbacks in bull, rallies in bear)
                if regime_type == "bull" and np.random.random() < params["pullback_prob"]:
                    daily_return -= np.random.uniform(0.01, 0.03)  # pullback
                elif regime_type == "bear" and np.random.random() < params["rally_prob"]:
                    daily_return += np.random.uniform(0.01, 0.025)  # bear rally
                elif regime_type == "sideways" and np.random.random() < params["revert_prob"]:
                    # Mean revert toward segment start
                    seg_start = prices[-1]
                    if len(prices) > 5:
                        dev = (prices[-1] - prices[-5]) / prices[-5]
                        daily_return -= dev * 0.3  # pull back toward mean

                new_price = prices[-1] * np.exp(daily_return)
                prices.append(max(new_price, 1.0))  # floor at $1

        prices = prices[:days]  # trim to exact length

        dates = pd.bdate_range(start="2023-01-02", periods=len(prices))
        df = pd.DataFrame(index=dates)
        df["Close"] = prices
        df["Open"] = df["Close"].shift(1).fillna(start_price)
        df["High"] = df[["Open", "Close"]].max(axis=1) * (
            1 + np.random.uniform(0, 0.02, len(prices))
        )
        df["Low"] = df[["Open", "Close"]].min(axis=1) * (
            1 - np.random.uniform(0, 0.02, len(prices))
        )
        df["Volume"] = np.random.randint(1_000_000, 50_000_000, len(prices))

        df["SMA_50"] = df["Close"].rolling(window=50).mean()
        df["SMA_200"] = df["Close"].rolling(window=200).mean()
        df["RSI"] = _calculate_rsi(df["Close"])
        df["MACD"] = _calculate_macd(df["Close"])
        df.dropna(inplace=True)

        print(f"   [Twin] Trending synthetic data generated: {len(df)} days for {ticker}")
        print(f"   [Twin] Regime segments: {[(r, l) for r, l in regimes]}")
        return df


# ==============================================================================
# SCENARIO INJECTOR
# ==============================================================================
class ScenarioInjector:
    """Injects Black Swan events into the simulation timeline."""

    @staticmethod
    def flash_crash(df, day_index, drop_pct=0.15):
        idx = df.index[day_index]
        df.loc[idx, "Close"] *= (1.0 - drop_pct)
        df.loc[idx, "Low"] = df.loc[idx, "Close"]
        df["RSI"] = _calculate_rsi(df["Close"])
        df["MACD"] = _calculate_macd(df["Close"])
        return df

    @staticmethod
    def regime_shift(df, day_index, direction="bear"):
        idx = df.index[day_index]
        if direction == "bear":
            df.loc[idx, "Close"] *= 0.95
            df.loc[idx, "Volume"] *= 3
        else:
            df.loc[idx, "Close"] *= 1.05
            df.loc[idx, "Volume"] *= 2
        df["RSI"] = _calculate_rsi(df["Close"])
        df["MACD"] = _calculate_macd(df["Close"])
        return df


# ==============================================================================
# SIMULATION PORTFOLIO
# ==============================================================================
class SimulationPortfolio:
    """Virtual wallet HOLD tracks fake cash and fake shares."""

    def __init__(self, starting_capital=10_000.0):
        self.starting_capital = starting_capital
        self.cash = starting_capital
        self.shares = 0
        self.avg_cost = 0.0
        self.trades = []
        self.daily_values = []

    def execute_decision(self, decision, price, alloc_pct, day_date):
        action_taken = "HOLD"
        pnl = 0.0
        is_profitable = False

        if "BUY" in decision.upper() and alloc_pct > 0 and price > 0:
            invest_amount = self.cash * alloc_pct
            new_shares = int(invest_amount / price)
            if new_shares > 0:
                cost = new_shares * price
                total_cost = (self.shares * self.avg_cost) + cost
                self.shares += new_shares
                self.avg_cost = total_cost / self.shares
                self.cash -= cost
                action_taken = "BUY"

        elif "SELL" in decision.upper() and self.shares > 0:
            revenue = self.shares * price
            pnl = (price - self.avg_cost) * self.shares
            is_profitable = price > self.avg_cost
            self.cash += revenue
            self.shares = 0
            self.avg_cost = 0.0
            action_taken = "SELL"

        portfolio_value = self.cash + (self.shares * price)
        self.daily_values.append({
            "date": str(day_date)[:10],
            "portfolio_value": round(portfolio_value, 2),
            "cash": round(self.cash, 2),
            "shares": self.shares,
            "price": round(price, 2),
            "action": action_taken,
        })
        self.trades.append({
            "date": str(day_date)[:10],
            "action": action_taken,
            "price": round(price, 2),
            "shares": self.shares,
            "portfolio_value": round(portfolio_value, 2),
            "pnl": round(pnl, 2),
            "is_profitable": is_profitable,
        })

    def record_idle_day(self, price, day_date):
        portfolio_value = self.cash + (self.shares * price)
        self.daily_values.append({
            "date": str(day_date)[:10],
            "portfolio_value": round(portfolio_value, 2),
            "cash": round(self.cash, 2),
            "shares": self.shares,
            "price": round(price, 2),
            "action": "IDLE",
        })

    def get_metrics(self):
        """
        Calculates institutional performance metrics.

        FIX v2: Win rate now correctly compares BUY entry price vs SELL exit
        price for each completed round-trip, instead of comparing adjacent
        portfolio values (which was inflating win rate on idle days).
        """
        if not self.daily_values:
            return {}

        values = [d["portfolio_value"] for d in self.daily_values]
        returns = pd.Series(values).pct_change().dropna()

        total_return = (values[-1] / self.starting_capital - 1) * 100

        # Max drawdown
        max_val = values[0]
        max_drawdown = 0.0
        for v in values:
            if v > max_val:
                max_val = v
            dd = (max_val - v) / max_val
            if dd > max_drawdown:
                max_drawdown = dd

        # Sharpe ratio
        sharpe = 0.0
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)

        # FIX v3: Track average cost basis to properly calculate win_rate on SELL even with pyramiding
        buy_trades = [t for t in self.trades if t["action"] == "BUY"]
        sell_trades = [t for t in self.trades if t["action"] == "SELL" and "is_profitable" in t]

        completed_pairs = len(sell_trades)
        profitable = sum(1 for t in sell_trades if t["is_profitable"])

        win_rate = (profitable / completed_pairs * 100) if completed_pairs > 0 else 0.0

        return {
            "total_return_pct": round(total_return, 2),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "sharpe_ratio": round(sharpe, 4),
            "total_trades": len(buy_trades),
            "win_rate_pct": round(win_rate, 1),
            "final_value": round(values[-1], 2),
            "starting_capital": self.starting_capital,
        }


# ==============================================================================
# DIGITAL TWIN SIMULATOR
# ==============================================================================
class DigitalTwinSimulator:
    """
    Master controller. Steps through time day-by-day, feeds data slices
    to the existing AI models, and records everything.

    Decision logic:
      - 5-layer confirmation (LSTM + Momentum + RSI + MACD + Trend)
      - Trailing stop-loss (10% base, tightens to 6% after >15% gain)
      - Regime-aware gating (no BUY in Bear)
      - GDI gate (no BUY when boardroom tension > 45%)
      - Pyramiding (scale in to winners with +5% PnL)
    """

    def __init__(self, system=None):
        if system is None:
            from ml_engine.master_system import FinFolioSystem
            self.system = FinFolioSystem()
        else:
            self.system = system

        self.decisions_log = []
        self.trust_evolution = []
        self._buy_price = 0.0
        self._peak_price = 0.0
        self._hold_intervals = 0  # track how long position held

    def run_simulation(
        self,
        ticker,
        start_date,
        end_date,
        starting_capital=10_000.0,
        decision_interval=5,
        scenarios=None,
        data_mode="historical",
        gbm_params=None,
    ):
        """Main entry point. Runs the full simulation."""
        print("\n" + "=" * 70)
        print("  DIGITAL TWIN SIMULATION ENGINE (v2 HOLD Optimized)")
        print("=" * 70)
        print(f"  Ticker: {ticker}")
        print(f"  Period: {start_date} -> {end_date}")
        print(f"  Capital: ${starting_capital:,.2f}")
        print(f"  Mode: {data_mode.upper()}")
        print("=" * 70)

        # Step 1: Load data
        if data_mode == "gbm":
            params = gbm_params or {}
            full_data = SyntheticMarketGenerator.generate_gbm(
                ticker=ticker,
                days=params.get("days", 504),
                start_price=params.get("start_price", 150.0),
                mu=params.get("mu", 0.08),
                sigma=params.get("sigma", 0.20),
                seed=params.get("seed", 42),
            )
            sim_start_idx = 200
        elif data_mode == "trending":
            params = gbm_params or {}
            full_data = SyntheticMarketGenerator.generate_trending(
                ticker=ticker,
                days=params.get("days", 504),
                start_price=params.get("start_price", 150.0),
                seed=params.get("seed", 42),
            )
            sim_start_idx = 200
        else:
            full_data = SyntheticMarketGenerator.download_historical(
                ticker, start_date, end_date
            )
            start_dt = pd.Timestamp(start_date)
            sim_start_idx = full_data.index.searchsorted(start_dt)
            if sim_start_idx < 60:
                sim_start_idx = 60

        # Step 2: Scenario injections
        if scenarios:
            for sc in scenarios:
                day = sc.get("day", 0) + sim_start_idx
                sc_type = sc.get("type", "")
                params = sc.get("params", {})
                if day < len(full_data):
                    if sc_type == "flash_crash":
                        full_data = ScenarioInjector.flash_crash(
                            full_data, day, params.get("drop_pct", 0.15)
                        )
                        print(f"   [Scenario] Flash Crash injected on day {sc.get('day')}")
                    elif sc_type == "regime_shift":
                        full_data = ScenarioInjector.regime_shift(
                            full_data, day, params.get("direction", "bear")
                        )
                        print(f"   [Scenario] Regime Shift injected on day {sc.get('day')}")

        # Step 3: Initialize
        portfolio = SimulationPortfolio(starting_capital)
        self.decisions_log = []
        self.trust_evolution = []
        self._buy_price = 0.0
        self._peak_price = 0.0
        self._hold_intervals = 0

        total_sim_days = len(full_data) - sim_start_idx
        print(f"\n   [Twin] Simulating {total_sim_days} trading days...")

        # Step 4: Time loop
        for i in range(sim_start_idx, len(full_data)):
            day_index = i - sim_start_idx
            current_date = full_data.index[i]
            current_price = float(full_data.iloc[i]["Close"])

            if portfolio.shares > 0 and current_price > self._peak_price:
                self._peak_price = current_price

            data_slice = full_data.iloc[: i + 1].copy()

            if day_index % decision_interval == 0 and day_index > 0:
                try:
                    result = self._run_ai_on_slice(
                        data_slice, ticker, current_date, portfolio, current_price
                    )
                    decision = result.get("decision", "HOLD")
                    alloc_pct = result.get("alloc_pct", 0.0)
                    confidence = result.get("confidence", 0.0)
                    gdi = result.get("gdi", 0.0)
                    regime = result.get("regime", "Unknown")

                    portfolio.execute_decision(decision, current_price, alloc_pct, current_date)

                    if decision == "BUY":
                        self._buy_price = current_price
                        self._peak_price = current_price
                        self._hold_intervals = 0
                    elif decision == "SELL":
                        self._buy_price = 0.0
                        self._peak_price = 0.0
                        self._hold_intervals = 0
                    elif portfolio.shares > 0:
                        self._hold_intervals += 1

                    self.decisions_log.append({
                        "day": day_index,
                        "date": str(current_date)[:10],
                        "decision": decision,
                        "confidence": round(confidence, 4),
                        "price": round(current_price, 2),
                        "alloc_pct": round(alloc_pct * 100, 2),
                        "regime": regime,
                        "gdi": round(gdi, 1),
                        "portfolio_value": portfolio.daily_values[-1]["portfolio_value"],
                    })

                    trust = self._get_trust_scores()
                    trust["day"] = day_index
                    trust["date"] = str(current_date)[:10]
                    self.trust_evolution.append(trust)

                    if len(self.decisions_log) % 20 == 0:
                        pv = portfolio.daily_values[-1]["portfolio_value"]
                        print(
                            f"   [Day {day_index:>4}] {decision:>10} | "
                            f"Price: ${current_price:.2f} | "
                            f"Portfolio: ${pv:,.2f}"
                        )

                except Exception as e:
                    print(f"   [Day {day_index}] Error: {e}")
                    portfolio.record_idle_day(current_price, current_date)
            else:
                portfolio.record_idle_day(current_price, current_date)

        # Step 5: Results
        metrics = portfolio.get_metrics()

        print(f"\n   {'=' * 50}")
        print("   SIMULATION COMPLETE (v2 HOLD Optimized)")
        print(f"   {'=' * 50}")
        print(f"   Starting Capital : ${starting_capital:>12,.2f}")
        print(f"   Final Value      : ${metrics['final_value']:>12,.2f}")
        print(f"   Total Return     : {metrics['total_return_pct']:>11.2f}%")
        print(f"   Max Drawdown     : {metrics['max_drawdown_pct']:>11.2f}%")
        print(f"   Sharpe Ratio     : {metrics['sharpe_ratio']:>11.4f}")
        print(f"   Win Rate         : {metrics['win_rate_pct']:>11.1f}%")
        print(f"   Total Trades     : {metrics['total_trades']:>11}")
        print(f"   {'=' * 50}")

        return {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "data_mode": data_mode,
            "metrics": metrics,
            "equity_curve": portfolio.daily_values,
            "decisions": self.decisions_log,
            "trust_evolution": self.trust_evolution,
            "trades": [t for t in portfolio.trades if t["action"] != "HOLD"],
        }

    def _run_ai_on_slice(
        self, data_slice, ticker, current_date, portfolio, current_price
    ):
        """
        Runs AI models with institutional trend-following logic.
        Optimized for high Win Rate (>60%) and high Sharpe Ratio (>1.0).
        """
        system = self.system

        # 1. Technical Analysis
        last_60 = data_slice[
            ["Close", "Volume", "SMA_50", "SMA_200", "RSI", "MACD"]
        ].tail(60)
        if len(last_60) < 60:
            return {
                "decision": "HOLD", "alloc_pct": 0, "confidence": 0,
                "gdi": 0, "regime": "Unknown",
            }

        lstm_signal = system.tech_agent.predict(last_60)

        # 2. Regime Detection
        current_vol = data_slice["Close"].pct_change().rolling(10).std().iloc[-1]
        if pd.isna(current_vol):
            current_vol = 0.015

        sma_50 = float(data_slice["SMA_50"].iloc[-1])
        sma_200 = float(data_slice["SMA_200"].iloc[-1])

        if sma_50 > sma_200 and current_vol < 0.02:
            regime_label = "Bull"
        elif sma_50 < sma_200 and current_vol > 0.015:
            regime_label = "Bear"
        else:
            regime_label = "Sideways"

        # H1 FIX: Short-term override to prevent buying the top of a cliff
        if len(data_slice) >= 5:
            ret_5d = float(data_slice["Close"].iloc[-1] / data_slice["Close"].iloc[-5] - 1.0)
        else:
            ret_5d = 0.0
            
        rsi_now = float(data_slice["RSI"].iloc[-1]) if "RSI" in data_slice.columns else 50.0
        
        if regime_label == "Bull" and (ret_5d < -0.015 or rsi_now < 45):
            regime_label = "Sideways"

        # 3. Momentum
        recent_prices = data_slice["Close"].tail(5)
        momentum_5d = float((recent_prices.iloc[-1] / recent_prices.iloc[0]) - 1)

        # 4. Synthetic Sentiment (independent RSI + mean-reversion based)
        # Decoupled from momentum to avoid circular dependency
        rsi_sent = (rsi_now - 50.0) / 50.0  # RSI>50 → bullish, RSI<50 → bearish
        # Mean-reversion signal: price vs SMA_50
        price_vs_sma = (current_price - sma_50) / sma_50 if sma_50 > 0 else 0.0
        sent_score = float(np.clip(rsi_sent * 0.6 + price_vs_sma * 0.4, -1.0, 1.0))

        # 5. Fusion
        vol_input = (
            0.9 if regime_label == "Bear"
            else 0.2 if regime_label == "Bull"
            else 0.5
        )
        trust_scores = self._get_trust_scores()
        confidence, _ = system.fusion_agent.predict(
            lstm_p=lstm_signal,
            sent_s=sent_score,
            vol_v=vol_input,
            trust_scores=trust_scores,
        )

        # 6. Heatmap GDI
        gdi, gdi_penalty = 0.0, 1.0
        if hasattr(system, "heatmap_agent") and system.heatmap_agent:
            heatmap_result = system.heatmap_agent.analyze(
                lstm_score=lstm_signal,
                sent_score=sent_score,
                regime_label=regime_label,
                regime_vol=current_vol,
            )
            gdi = heatmap_result["gdi"] * 100
            gdi_penalty = heatmap_result["penalty"]

        # 7. Kelly Sizing (regime-aware)
        alloc_pct, _ = system.risk_engine.calculate_position_size(
            confidence, current_vol,
            disagreement_penalty=gdi_penalty,
            regime=regime_label,
        )

        # ================================================================
        # 8. DECISION LOGIC v3 — Fixed over-filtering and missing sells
        # ================================================================
        decision = "HOLD"
        is_invested = portfolio.shares > 0

        if is_invested:
            # --- SELL LOGIC (when holding shares) ---
            pnl_pct = (
                (current_price - self._buy_price) / self._buy_price
                if self._buy_price > 0 else 0
            )
            drawdown_from_peak = (
                (self._peak_price - current_price) / self._peak_price
                if self._peak_price > 0 else 0
            )

            # Dynamic stop-loss (widens with volatility)
            dynamic_stop_loss = max(0.06, current_vol * 2.0)
            if pnl_pct > 0.08:
                dynamic_stop_loss = 0.04  # tighten after decent gain

            # Sell trigger 1: Stop-loss hit
            if drawdown_from_peak >= dynamic_stop_loss:
                decision = "SELL"
            # Sell trigger 2: Profit-taking (>2% unrealized gain — lock in profits)
            elif pnl_pct > 0.02:
                decision = "SELL"
            # Sell trigger 3: Regime flips to Bear with low confidence
            elif regime_label == "Bear" and confidence < 0.50:
                decision = "SELL"
            # Sell trigger 4: LSTM flips bearish while holding
            elif lstm_signal < 0.45 and confidence < 0.50:
                decision = "SELL"
            # Sell trigger 5: Held too long with no gain (stale position)
            elif self._hold_intervals >= 4 and pnl_pct < 0.005:
                decision = "SELL"
            # Pyramid: scale into winners (only in strong trends)
            elif pnl_pct > 0.03 and confidence >= 0.60 and regime_label == "Bull":
                decision = "BUY"
                alloc_pct = min(alloc_pct, 0.15)

            if decision == "SELL":
                alloc_pct = 0.0

        else:
            # --- BUY LOGIC (when not holding) ---
            # Aggressive entry: exploit every signal worth following
            base_conf = 0.45 if regime_label == "Bull" else 0.52

            if regime_label == "Bear":
                # Enter Bear only with strong bullish LSTM divergence
                if confidence >= 0.62 and lstm_signal > 0.55 and gdi < 50:
                    decision = "BUY"
                    alloc_pct = min(alloc_pct * 0.5, 0.10)  # half size
            elif confidence >= base_conf and gdi < 60:
                decision = "BUY"
                alloc_pct = min(alloc_pct * 1.5, 0.30)
            # Fallback: LSTM very bullish → override moderate confidence
            elif lstm_signal > 0.75 and confidence >= 0.40:
                decision = "BUY"
                alloc_pct = min(alloc_pct, 0.15)  # smaller cautious position

        return {
            "decision": decision,
            "alloc_pct": alloc_pct,
            "confidence": confidence,
            "gdi": gdi,
            "regime": regime_label,
            "lstm_signal": lstm_signal,
            "sent_score": sent_score,
        }

    def _get_trust_scores(self):
        """Reads current trust scores from the Meta-Agent."""
        if hasattr(self.system, "meta_agent") and self.system.meta_agent:
            scores = self.system.meta_agent.get_trust_scores()
            return {
                "technical": scores.get("technical", 1.0),
                "sentiment": scores.get("sentiment", 1.0),
                "regime": scores.get("regime", 1.0),
            }
        return {"technical": 1.0, "sentiment": 1.0, "regime": 1.0}