import unittest
import pandas as pd
from datetime import datetime, timedelta

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml_engine.simulation_engine import SimulationPortfolio, DigitalTwinSimulator, SyntheticMarketGenerator

class TestSimulationEngine(unittest.TestCase):

    def test_simulation_portfolio_win_rate_with_pyramiding(self):
        """Test if the portfolio accurately calculates win rate when scaling in (pyramiding) to a position."""
        portfolio = SimulationPortfolio()
        
        # Action 1: Buy 100 shares at $10.0 (uses all cash? Let's use exact amounts by passing alloc_pct)
        # portfolio starts with 10_000. 1000 spent.
        portfolio.execute_decision("BUY", 10.0, 0.10, "2026-03-01")
        self.assertEqual(portfolio.shares, 100)
        self.assertEqual(portfolio.avg_cost, 10.0)
        
        # Action 2: Buy 200 shares at $13.0 (pyramid success)
        # 100 * 10 = 1000. 200 * 13 = 2600. Total cost = 3600. Shares = 300. Avg cost = 12.0
        # Add slight float precision padding to avoid int() rounding 199.999 down to 199.
        portfolio.execute_decision("BUY", 13.0, 2600.01 / portfolio.cash, "2026-03-05")
        self.assertEqual(portfolio.shares, 300)
        self.assertEqual(portfolio.avg_cost, 12.0)
        
        # Action 3: Sell all 300 shares at $15.0. 
        # This is 1 round trip (completion). Average cost is 12.0. Margin = +3.0. Win.
        portfolio.execute_decision("SELL", 15.0, 0.0, "2026-03-10")
        self.assertEqual(portfolio.shares, 0)
        self.assertEqual(portfolio.avg_cost, 0.0)
        
        metrics = portfolio.get_metrics()
        
        self.assertEqual(metrics["win_rate_pct"], 100.0) # 1 profitable sell / 1 total sells
        self.assertEqual(metrics["total_trades"], 2) # 2 buys
        
        # Setup loss round-trip
        portfolio.execute_decision("BUY", 20.0, 0.10, "2026-04-01")
        portfolio.execute_decision("SELL", 15.0, 0.0, "2026-04-05")
        
        metrics = portfolio.get_metrics()
        self.assertEqual(metrics["win_rate_pct"], 50.0) # 1 profitable, 1 loss = 50%
        
    def test_synthetic_market_generator(self):
        """Test if the GBM data generator works properly for simulations."""
        df = SyntheticMarketGenerator.generate_gbm(ticker="TEST", days=270)
        self.assertIn("SMA_50", df.columns)
        self.assertIn("RSI", df.columns)
        
        # After dropping NAs (needs 200 days for SMA_200), we should have > 0 rows
        self.assertGreater(len(df), 0)

    def test_run_simulation_short(self):
        """Test the master process on a small batch of synthetic data."""
        # Mocking the AI model responses by using a dummy class
        class DummySystem:
            def __init__(self):
                class DummyAgent:
                    def predict(self, *args, **kwargs):
                        return 0.9  # bullish
                
                class DummyFusion:
                    def predict(self, *args, **kwargs):
                        return 0.8, None  # confidence, (ignored)

                class DummyRisk:
                    def calculate_position_size(self, *args, **kwargs):
                        return 0.1, None  # 10% allocation
                
                self.tech_agent = DummyAgent()
                self.fusion_agent = DummyFusion()
                self.risk_engine = DummyRisk()
                self.meta_agent = None  # Bypass meta agent testing since we just tested it elsewhere

        # Run
        simulator = DigitalTwinSimulator(system=DummySystem())
        # Total days needs to be large enough because generate_gbm drops the first 200 days for SMA_200.
        params = {"days": 500, "start_price": 100.0, "seed": 42} # trending with patterns
        
        results = simulator.run_simulation(
            ticker="SYNTH",
            start_date="2025-03-01",
            end_date="2026-03-31",
            decision_interval=2,
            data_mode="trending",
            gbm_params=params
        )
        self.assertEqual(results["ticker"], "SYNTH")
        self.assertIn("metrics", results)
        self.assertGreaterEqual(results["metrics"]["total_trades"], 1)

    def test_run_20_different_dates(self):
        """Test the master process on an array of 20 different timelines."""
        class DummySystem:
            def __init__(self):
                class DummyAgent:
                    def predict(self, *args, **kwargs): return 0.9
                class DummyFusion:
                    def predict(self, *args, **kwargs): return 0.8, None
                class DummyRisk:
                    def calculate_position_size(self, *args, **kwargs): return 0.1, None
                self.tech_agent = DummyAgent()
                self.fusion_agent = DummyFusion()
                self.risk_engine = DummyRisk()
                self.meta_agent = None

        simulator = DigitalTwinSimulator(system=DummySystem())
        base_date = datetime(2023, 1, 1)
        
        # Loop over 20 iterations pushing the timeline by 30 days
        for i in range(20):
            start = (base_date + timedelta(days=i*30)).strftime("%Y-%m-%d")
            end = (base_date + timedelta(days=i*30 + 180)).strftime("%Y-%m-%d")
            
            # Using trending data with exploitable regime patterns
            params = {"days": 500, "start_price": 50.0 + i*5, "seed": 42+i}
            results = simulator.run_simulation(
                ticker=f"TICK{i}",
                start_date=start,
                end_date=end,
                decision_interval=2,
                data_mode="trending",
                gbm_params=params
            )
            self.assertEqual(results["start_date"], start)
            self.assertEqual(results["end_date"], end)
            self.assertIn("metrics", results)

if __name__ == "__main__":
    unittest.main()
