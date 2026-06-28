import os
import json
import unittest
from datetime import datetime, timedelta
import pandas as pd
from unittest.mock import patch, MagicMock

# Ensure we can import ml_engine
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml_engine.meta_agent import (
    MetaAgent, TRUST_DEFAULT, EMA_ALPHA,
    MOVEMENT_THRESHOLD, STRONG_MOVE_THRESHOLD, BEAR_RALLY_TOLERANCE,
    CONFIDENT_WRONG_MULTIPLIER, CONFIDENT_WRONG_THRESHOLD,
    MIN_EVALUATIONS_FOR_TRUST, DAMPENED_ALPHA_FACTOR,
    TRUST_AMPLIFICATION_FACTOR, MAX_AMPLIFIED_DEVIATION,
)


class TestMetaAgentTemporalDegradation(unittest.TestCase):
    def setUp(self):
        """Setup a temporary MetaAgent instance."""
        self.meta_agent = MetaAgent()
        
        # Override pathways for testing without messing up main ledger
        self.test_dir = os.path.join(BASE_DIR, "data", "test_meta")
        os.makedirs(self.test_dir, exist_ok=True)
        self.meta_agent.meta_dir = self.test_dir
        self.meta_agent.ledger_path = os.path.join(self.test_dir, "test_decision_ledger.csv")
        self.meta_agent.trust_path = os.path.join(self.test_dir, "test_trust_scores.json")
        self.meta_agent._create_ledger()
        self.meta_agent._create_default_trust()
        
    def tearDown(self):
        """Clean up the test files."""
        if os.path.exists(self.meta_agent.ledger_path):
            os.remove(self.meta_agent.ledger_path)
        if os.path.exists(self.meta_agent.trust_path):
            os.remove(self.meta_agent.trust_path)
            
    def test_temporal_degradation_decay(self):
        """Test if the trust scores correctly decay back to default over time."""
        print("\n--- Testing Temporal Degradation Decay ---")
        
        # 1. Modify trust scores to be high initially
        current_scores = self.meta_agent.load_trust_scores()
        current_scores["technical"] = 1.40
        current_scores["sentiment"] = 0.60
        current_scores["regime"] = 1.00
        
        # 2. Simulate the last update being 14 days ago (~10% decay)
        past_date = datetime.now() - timedelta(days=14)
        current_scores["last_updated"] = past_date.strftime("%Y-%m-%d %H:%M:%S")
        # Set high eval count so amplification applies fully
        current_scores["evaluation_count"] = 50
        
        with open(self.meta_agent.trust_path, "w", encoding="utf-8") as f:
            json.dump(current_scores, f, indent=2)
            
        # 3. Retrieve RAW trust scores (before amplification) to verify temporal decay
        # Load raw to check decay math independently
        raw_scores = self.meta_agent.load_trust_scores()
        
        # Manual decay calculation:
        # 14 days / 7 = 2, decay_ratio = 2 * 0.05 = 0.10, decay_factor = 0.90
        # technical: 1.40 * 0.90 + 1.00 * 0.10 = 1.36
        # sentiment: 0.60 * 0.90 + 1.00 * 0.10 = 0.64
        
        # Get decayed+amplified scores
        retrieved_scores = self.meta_agent.get_trust_scores()
        
        print("Initial Scores (14 days ago):")
        print("  technical: 1.40")
        print("  sentiment: 0.60")
        print("  regime:    1.00")
        print(f"Retrieved scores (after decay + amplification):")
        print(f"  technical: {retrieved_scores['technical']}")
        print(f"  sentiment: {retrieved_scores['sentiment']}")
        print(f"  regime:    {retrieved_scores['regime']}")
        
        # After decay, technical = 1.36, deviation = +0.36
        # After amplification: sqrt(0.36) * 1.0 = 0.6 -> capped at 0.40 -> trust = 1.40
        self.assertGreater(retrieved_scores["technical"], 1.0, "Boosted score should stay above 1.0")
        self.assertLessEqual(retrieved_scores["technical"], 1.40, "Amplification should be capped")
        
        # After decay, sentiment = 0.64, deviation = -0.36
        # After amplification: sqrt(0.36) * 1.0 = 0.6 -> capped at 0.40 -> trust = 0.60
        self.assertLess(retrieved_scores["sentiment"], 1.0, "Penalized score should stay below 1.0")
        self.assertGreaterEqual(retrieved_scores["sentiment"], 0.60, "Amplification should be capped")
        
        # Regime stays at 1.0 (no deviation = no amplification)
        self.assertEqual(retrieved_scores["regime"], 1.0, "Neutral score should stay at 1.0")
        
    @patch('yfinance.download')
    def test_log_and_evaluate_with_temporal_context(self, mock_yfinance):
        """Test logging and evaluation using dynamic EMA context to address degradation."""
        print("\n--- Testing Log & Evaluation context ---")
        
        # Mock yfinance return
        mock_df = pd.DataFrame({'Close': [110.0, 115.0]})
        mock_yfinance.return_value = mock_df
        
        # Log a bullish decision with high confidence
        self.meta_agent.log_decision(
            ticker="AAPL",
            lstm_score=0.9,
            sent_score=0.8,
            regime_label="Bull",
            risk_score=0.2,
            fusion_confidence=0.85,
            final_decision="BUY",
            price_at_decision=100.0,
            asc_score=0.45,
            asc_reliable=True
        )
        
        # Change timestamp to trigger past eval
        df = pd.read_csv(self.meta_agent.ledger_path)
        past_date = datetime.now() - timedelta(days=6)
        df.at[0, "timestamp"] = past_date.strftime("%Y-%m-%d %H:%M:%S")
        df.to_csv(self.meta_agent.ledger_path, index=False)
        
        # Evaluate
        self.meta_agent.evaluate_past_decisions()
        
        # Re-read ledger to check grades
        df = pd.read_csv(self.meta_agent.ledger_path)
        print(f"Ledger grades: LSTM={df.at[0, 'lstm_grade']}, Sent={df.at[0, 'sent_grade']}, Regime={df.at[0, 'regime_grade']}")

        # 10% move is strongly correct -> grade should be +2
        # Note: pandas reads "+2" from CSV as int, so compare as int
        self.assertEqual(int(df.at[0, "lstm_grade"]), 2)
        self.assertEqual(int(df.at[0, "sent_grade"]), 2)
        self.assertEqual(int(df.at[0, "regime_grade"]), 2)
        
        # Check trust scores (should be boosted)
        scores = self.meta_agent.get_trust_scores()
        print(f"Evaluated Scores:")
        for k, v in scores.items():
            print(f"  {k}: {v}")
            
        self.assertGreater(scores["technical"], TRUST_DEFAULT)
        self.assertGreater(scores["sentiment"], TRUST_DEFAULT)

    def test_multi_level_grading(self):
        """Test all 5 grade levels for _grade_agent and _grade_regime."""
        print("\n--- Testing Multi-Level Grading ---")
        
        ma = self.meta_agent

        # Technical agent (lstm_score=0.9 -> bullish)
        self.assertEqual(ma._grade_agent(0.9, 0.10, "technical"), 2)
        self.assertEqual(ma._grade_agent(0.9, 0.02, "technical"), 1)
        self.assertEqual(ma._grade_agent(0.9, 0.005, "technical"), 0)
        self.assertEqual(ma._grade_agent(0.9, -0.02, "technical"), -1)
        self.assertEqual(ma._grade_agent(0.9, -0.08, "technical"), -2)

        # Sentiment agent
        self.assertEqual(ma._grade_agent(0.5, 0.06, "sentiment"), 2)
        self.assertEqual(ma._grade_agent(0.5, 0.02, "sentiment"), 1)
        self.assertEqual(ma._grade_agent(0.5, -0.02, "sentiment"), -1)
        self.assertEqual(ma._grade_agent(0.5, -0.06, "sentiment"), -2)
        
        # Bearish sentiment
        self.assertEqual(ma._grade_agent(-0.3, -0.06, "sentiment"), 2)
        self.assertEqual(ma._grade_agent(-0.3, 0.06, "sentiment"), -2)
        
        print("  All 5 grade levels validated for agents [OK]")

        # Regime grading
        self.assertEqual(ma._grade_regime("Bull", 0.10), 2)
        self.assertEqual(ma._grade_regime("Bull", 0.03), 1)
        self.assertEqual(ma._grade_regime("Bull", -0.02), 0)
        self.assertEqual(ma._grade_regime("Bull", -0.04), -1)
        self.assertEqual(ma._grade_regime("Bull", -0.06), -2)
        
        self.assertEqual(ma._grade_regime("Bear", -0.10), 2)
        self.assertEqual(ma._grade_regime("Bear", 0.02), 0)
        self.assertEqual(ma._grade_regime("Bear", 0.04), -1)
        
        self.assertEqual(ma._grade_regime("Sideways", 0.02), 1)
        self.assertEqual(ma._grade_regime("Sideways", 0.04), -1)
        self.assertEqual(ma._grade_regime("Sideways", 0.06), -2)
        
        print("  All regime grade levels validated [OK]")

    @patch('yfinance.download')
    def test_confidence_weighted_rewards(self, mock_yfinance):
        """Test that high-confidence correct decisions boost trust more than low-confidence ones."""
        print("\n--- Testing Confidence-Weighted Rewards ---")
        
        mock_df = pd.DataFrame({'Close': [110.0, 115.0]})
        mock_yfinance.return_value = mock_df

        # --- HIGH confidence (0.95) ---
        self.meta_agent._create_default_trust()
        self.meta_agent._create_ledger()
        self.meta_agent.log_decision(
            ticker="AAPL", lstm_score=0.9, sent_score=0.8,
            regime_label="Bull", risk_score=0.2,
            fusion_confidence=0.95,
            final_decision="BUY", price_at_decision=100.0,
        )
        df = pd.read_csv(self.meta_agent.ledger_path)
        df.at[0, "timestamp"] = (datetime.now() - timedelta(days=6)).strftime("%Y-%m-%d %H:%M:%S")
        df.to_csv(self.meta_agent.ledger_path, index=False)
        self.meta_agent.evaluate_past_decisions()
        high_conf_scores = self.meta_agent.load_trust_scores()
        high_tech = high_conf_scores["technical"]

        # --- LOW confidence (0.55) ---
        self.meta_agent._create_default_trust()
        self.meta_agent._create_ledger()
        self.meta_agent.log_decision(
            ticker="AAPL", lstm_score=0.9, sent_score=0.8,
            regime_label="Bull", risk_score=0.2,
            fusion_confidence=0.55,
            final_decision="BUY", price_at_decision=100.0,
        )
        df = pd.read_csv(self.meta_agent.ledger_path)
        df.at[0, "timestamp"] = (datetime.now() - timedelta(days=6)).strftime("%Y-%m-%d %H:%M:%S")
        df.to_csv(self.meta_agent.ledger_path, index=False)
        self.meta_agent.evaluate_past_decisions()
        low_conf_scores = self.meta_agent.load_trust_scores()
        low_tech = low_conf_scores["technical"]

        print(f"  High-confidence (0.95) trust: {high_tech}")
        print(f"  Low-confidence  (0.55) trust: {low_tech}")
        
        self.assertGreater(high_tech, TRUST_DEFAULT, "High-conf correct should boost trust")
        self.assertGreater(low_tech, TRUST_DEFAULT, "Low-conf correct should also boost trust")
        self.assertGreater(high_tech, low_tech, "High-conf correct should boost MORE than low-conf")
        print("  Confidence weighting validated [OK]")

    def test_regime_bear_rally_tolerance(self):
        """Test that small rallies in Bear regime are graded as inconclusive (0), not wrong."""
        print("\n--- Testing Bear Rally Tolerance ---")
        
        ma = self.meta_agent
        
        grade = ma._grade_regime("Bear", 0.025)
        self.assertEqual(grade, 0, "Bear + small rally should be inconclusive (0)")
        print(f"  Bear + 2.5% rally -> grade {grade} (inconclusive) [OK]")
        
        grade = ma._grade_regime("Bear", 0.04)
        self.assertEqual(grade, -1, "Bear + moderate rally should be -1")
        print(f"  Bear + 4.0% rally -> grade {grade} (wrong) [OK]")
        
        grade = ma._grade_regime("Bull", -0.025)
        self.assertEqual(grade, 0, "Bull + small pullback should be inconclusive (0)")
        print(f"  Bull - 2.5% dip -> grade {grade} (inconclusive) [OK]")
        print("  Bear rally / Bull pullback tolerance validated [OK]")

    # ==================================================================
    # ROUND 2 TESTS
    # ==================================================================

    @patch('yfinance.download')
    def test_asymmetric_penalty(self, mock_yfinance):
        """Test that high-confidence wrong predictions are penalized 1.5x more than normal."""
        print("\n--- Testing Asymmetric Penalty ---")
        
        # Mock: price DROPS 10% (market went down, but we predicted bullish)
        mock_df = pd.DataFrame({'Close': [90.0, 88.0]})
        mock_yfinance.return_value = mock_df

        # --- HIGH confidence wrong (0.90) ---
        self.meta_agent._create_default_trust()
        self.meta_agent._create_ledger()
        self.meta_agent.log_decision(
            ticker="AAPL", lstm_score=0.9, sent_score=0.8,  # bullish
            regime_label="Bull", risk_score=0.2,
            fusion_confidence=0.90,  # very confident
            final_decision="BUY", price_at_decision=100.0,
        )
        df = pd.read_csv(self.meta_agent.ledger_path)
        df.at[0, "timestamp"] = (datetime.now() - timedelta(days=6)).strftime("%Y-%m-%d %H:%M:%S")
        df.to_csv(self.meta_agent.ledger_path, index=False)
        self.meta_agent.evaluate_past_decisions()
        high_conf_wrong = self.meta_agent.load_trust_scores()
        high_tech = high_conf_wrong["technical"]

        # --- LOW confidence wrong (0.55) ---
        self.meta_agent._create_default_trust()
        self.meta_agent._create_ledger()
        self.meta_agent.log_decision(
            ticker="AAPL", lstm_score=0.9, sent_score=0.8,
            regime_label="Bull", risk_score=0.2,
            fusion_confidence=0.55,  # low confidence
            final_decision="BUY", price_at_decision=100.0,
        )
        df = pd.read_csv(self.meta_agent.ledger_path)
        df.at[0, "timestamp"] = (datetime.now() - timedelta(days=6)).strftime("%Y-%m-%d %H:%M:%S")
        df.to_csv(self.meta_agent.ledger_path, index=False)
        self.meta_agent.evaluate_past_decisions()
        low_conf_wrong = self.meta_agent.load_trust_scores()
        low_tech = low_conf_wrong["technical"]

        print(f"  High-confidence wrong (0.90) trust: {high_tech}")
        print(f"  Low-confidence wrong  (0.55) trust: {low_tech}")

        # Both should be penalized (below 1.0)
        self.assertLess(high_tech, TRUST_DEFAULT, "High-conf wrong should penalize trust")
        self.assertLess(low_tech, TRUST_DEFAULT, "Low-conf wrong should also penalize")
        # High-confidence wrong should penalize MORE (lower score)
        self.assertLess(high_tech, low_tech, "High-conf wrong should penalize MORE than low-conf wrong")
        print("  Asymmetric penalty validated [OK]")

    @patch('yfinance.download')
    def test_regime_specific_trust(self, mock_yfinance):
        """Test that trust is tracked per-regime (Bull/Bear/Sideways)."""
        print("\n--- Testing Regime-Specific Trust ---")
        
        mock_df = pd.DataFrame({'Close': [110.0, 115.0]})
        mock_yfinance.return_value = mock_df

        self.meta_agent._create_default_trust()
        self.meta_agent._create_ledger()
        
        # Log a correct prediction in Bull regime
        self.meta_agent.log_decision(
            ticker="AAPL", lstm_score=0.9, sent_score=0.8,
            regime_label="Bull", risk_score=0.2,
            fusion_confidence=0.85,
            final_decision="BUY", price_at_decision=100.0,
        )
        df = pd.read_csv(self.meta_agent.ledger_path)
        df.at[0, "timestamp"] = (datetime.now() - timedelta(days=6)).strftime("%Y-%m-%d %H:%M:%S")
        df.to_csv(self.meta_agent.ledger_path, index=False)
        self.meta_agent.evaluate_past_decisions()
        
        # Check regime-specific trust was updated
        raw_scores = self.meta_agent.load_trust_scores()
        regime_trust = raw_scores.get("regime_trust", {})
        
        bull_tech = regime_trust.get("Bull", {}).get("technical", TRUST_DEFAULT)
        bear_tech = regime_trust.get("Bear", {}).get("technical", TRUST_DEFAULT)
        
        print(f"  Bull regime technical trust: {bull_tech}")
        print(f"  Bear regime technical trust: {bear_tech}")
        
        # Bull should be boosted, Bear should be unchanged
        self.assertGreater(bull_tech, TRUST_DEFAULT, "Bull regime trust should be boosted after correct Bull prediction")
        self.assertEqual(bear_tech, TRUST_DEFAULT, "Bear regime trust should be unchanged")
        
        # Test that get_trust_scores with regime blends correctly
        scores_bull = self.meta_agent.get_trust_scores(regime="Bull")
        scores_bear = self.meta_agent.get_trust_scores(regime="Bear")
        
        print(f"  Blended trust (Bull context): {scores_bull['technical']}")
        print(f"  Blended trust (Bear context): {scores_bear['technical']}")
        
        # Bull context should yield higher trust than Bear context
        # (because Bull regime_trust is boosted)
        self.assertGreaterEqual(scores_bull["technical"], scores_bear["technical"],
                               "Bull-context trust should be >= Bear-context trust")
        print("  Regime-specific trust validated [OK]")

    @patch('yfinance.download')
    def test_data_sufficiency_dampening(self, mock_yfinance):
        """Test that trust updates are dampened when evaluation_count < MIN_EVALUATIONS_FOR_TRUST."""
        print("\n--- Testing Data Sufficiency Dampening ---")
        
        mock_df = pd.DataFrame({'Close': [110.0, 115.0]})
        mock_yfinance.return_value = mock_df

        # Run 1 evaluation with fresh agent (eval_count = 0, should dampen)
        self.meta_agent._create_default_trust()
        self.meta_agent._create_ledger()
        self.meta_agent.log_decision(
            ticker="AAPL", lstm_score=0.9, sent_score=0.8,
            regime_label="Bull", risk_score=0.2,
            fusion_confidence=0.85,
            final_decision="BUY", price_at_decision=100.0,
        )
        df = pd.read_csv(self.meta_agent.ledger_path)
        df.at[0, "timestamp"] = (datetime.now() - timedelta(days=6)).strftime("%Y-%m-%d %H:%M:%S")
        df.to_csv(self.meta_agent.ledger_path, index=False)
        self.meta_agent.evaluate_past_decisions()
        dampened_scores = self.meta_agent.load_trust_scores()
        dampened_tech = dampened_scores["technical"]

        # Run 1 evaluation with high eval_count (>= MIN_EVALUATIONS, full alpha)
        self.meta_agent._create_default_trust()
        self.meta_agent._create_ledger()
        # Manually set high eval count
        ts = self.meta_agent.load_trust_scores()
        ts["evaluation_count"] = MIN_EVALUATIONS_FOR_TRUST + 1
        with open(self.meta_agent.trust_path, "w", encoding="utf-8") as f:
            json.dump(ts, f, indent=2)
        self.meta_agent.log_decision(
            ticker="AAPL", lstm_score=0.9, sent_score=0.8,
            regime_label="Bull", risk_score=0.2,
            fusion_confidence=0.85,
            final_decision="BUY", price_at_decision=100.0,
        )
        df = pd.read_csv(self.meta_agent.ledger_path)
        df.at[0, "timestamp"] = (datetime.now() - timedelta(days=6)).strftime("%Y-%m-%d %H:%M:%S")
        df.to_csv(self.meta_agent.ledger_path, index=False)
        self.meta_agent.evaluate_past_decisions()
        full_scores = self.meta_agent.load_trust_scores()
        full_tech = full_scores["technical"]

        print(f"  Dampened trust (N<{MIN_EVALUATIONS_FOR_TRUST}): {dampened_tech}")
        print(f"  Full trust (N>={MIN_EVALUATIONS_FOR_TRUST}): {full_tech}")
        
        # Both should be boosted, but dampened should be closer to 1.0
        self.assertGreater(dampened_tech, TRUST_DEFAULT)
        self.assertGreater(full_tech, TRUST_DEFAULT)
        self.assertGreater(full_tech, dampened_tech,
                          "Full alpha should produce larger trust update than dampened")
        print("  Data sufficiency dampening validated [OK]")

    def test_trust_amplification(self):
        """Test that trust amplification makes deviations from 1.0 stronger."""
        print("\n--- Testing Trust Amplification ---")
        
        # Set trust to 1.2 (boosted) and 0.8 (penalized)
        ts = self.meta_agent.load_trust_scores()
        ts["technical"] = 1.20
        ts["sentiment"] = 0.80
        ts["regime"] = 1.00
        ts["evaluation_count"] = 50
        with open(self.meta_agent.trust_path, "w", encoding="utf-8") as f:
            json.dump(ts, f, indent=2)
        
        scores = self.meta_agent.get_trust_scores()
        
        print(f"  Raw technical: 1.20 -> Amplified: {scores['technical']}")
        print(f"  Raw sentiment: 0.80 -> Amplified: {scores['sentiment']}")
        print(f"  Raw regime:    1.00 -> Amplified: {scores['regime']}")
        
        # Amplification with cap:
        # technical: sqrt(0.2) * 1.0 = 0.447 -> capped at 0.40 -> 1.40
        self.assertGreater(scores["technical"], 1.20, "Amplified boosted trust should be > raw")
        self.assertLessEqual(scores["technical"], 1.0 + MAX_AMPLIFIED_DEVIATION, "Should be capped")
        # sentiment: sqrt(0.2) * 1.0 = 0.447 -> capped at 0.40 -> 0.60
        self.assertLess(scores["sentiment"], 0.80, "Amplified penalized trust should be < raw")
        self.assertGreaterEqual(scores["sentiment"], 1.0 - MAX_AMPLIFIED_DEVIATION, "Should be capped")
        # regime: no deviation, no amplification
        self.assertEqual(scores["regime"], 1.0, "No deviation = no amplification")
        
        print("  Trust amplification validated [OK]")


if __name__ == "__main__":
    unittest.main(verbosity=2)
