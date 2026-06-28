"""
PHASE 14: Daily Meta-Evaluation Script
---------------------------------------
Run this script once per day (or before every analysis session)
to evaluate past decisions and update agent trust multipliers.

Usage:
    python scripts/run_meta_evaluation.py
"""

import os
import sys

# Ensure project root is on the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_engine.meta_agent import MetaAgent


def main():
    print("\n" + "=" * 60)
    print("  FINFOLIO-X: META-AGENT DAILY EVALUATION")
    print("=" * 60)

    # 1. Initialize Meta-Agent
    meta = MetaAgent()

    # 2. Show current trust scores BEFORE evaluation
    print("\n  --- BEFORE Evaluation ---")
    current = meta.get_trust_scores()
    meta.print_trust_report(current)

    # 3. Run hindsight evaluation
    meta.evaluate_past_decisions()

    # 4. Show updated trust scores AFTER evaluation
    print("\n  --- AFTER Evaluation ---")
    updated = meta.get_trust_scores()
    meta.print_trust_report(updated)

    print("\n" + "=" * 60)
    print("  Evaluation Complete. Trust scores are ready for next run.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
