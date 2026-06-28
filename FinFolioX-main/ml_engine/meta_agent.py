"""
ml_engine/meta_agent.py  HOLD  Self-Correcting Meta-Agent (Phase 14 + Phase 26)
=============================================================================
Maintains a CSV decision ledger, evaluates past predictions T+5 days later,
and dynamically adjusts per-agent trust multipliers via EMA. Phase 26 extends
the ledger with two new columns: asc_score (the Agent Sycophancy Coefficient
at decision time) and asc_reliable (whether the buffer had enough data). The
evaluate_past_decisions() method now produces an ASC accuracy correlation
table showing decision accuracy bucketed by ASC range (low/medium/high) HOLD
the core empirical validation for the IEEE paper hypothesis that high-ASC
decisions have systematically lower outcome accuracy than low-ASC decisions.
"""

import os
import json
import csv
import numpy as np
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger("MetaAgent")

try:
    from counterfactual_engine import CounterfactualEngine
except ImportError:
    try:
        from ml_engine.counterfactual_engine import CounterfactualEngine
    except ImportError:
        CounterfactualEngine = None

# ==============================================================================
# CONFIGURATION
# ==============================================================================

TRUST_MIN         = 0.30
TRUST_MAX         = 1.80
TRUST_DEFAULT     = 1.0
EMA_ALPHA         = 0.07
EVAL_LOOKBACK_DAYS = 5
MOVEMENT_THRESHOLD = 0.01
STRONG_MOVE_THRESHOLD = 0.05   # >=5% move = strong correct/wrong
BEAR_RALLY_TOLERANCE  = 0.03   # bear regime + up <=3% = inconclusive
SIDEWAYS_TOLERANCE    = 0.03   # sideways regime tolerance band

# Asymmetric penalty: high-confidence mistakes penalized extra
CONFIDENT_WRONG_MULTIPLIER = 1.5   # 50% extra penalty if conf > 0.70
CONFIDENT_WRONG_THRESHOLD  = 0.70  # confidence threshold for asymmetric penalty

# Data sufficiency: dampened learning until enough evaluations
MIN_EVALUATIONS_FOR_TRUST  = 10    # minimum decisions before full trust updates
DAMPENED_ALPHA_FACTOR      = 0.5   # EMA_ALPHA *= this when N < MIN_EVALUATIONS

# Trust amplification: amplify deviation for stronger decision influence
TRUST_AMPLIFICATION_FACTOR = 1.0   # amplify how far trust deviates from 1.0
MAX_AMPLIFIED_DEVIATION    = 0.40  # cap amplified deviation to prevent explosion

# ASC accuracy bucket thresholds (for paper validation table)
ASC_LOW_THRESHOLD  = 0.30
ASC_HIGH_THRESHOLD = 0.70


class MetaAgent:
    """
    The Self-Correcting Meta-Agent.
    Extends Phase 14 with Phase 26 ASC tracking for empirical validation.
    """

    def __init__(self):
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.meta_dir    = os.path.join(BASE_DIR, "data", "meta")
        self.ledger_path = os.path.join(self.meta_dir, "decision_ledger.csv")
        self.trust_path  = os.path.join(self.meta_dir, "trust_scores.json")

        os.makedirs(self.meta_dir, exist_ok=True)

        if not os.path.exists(self.ledger_path):
            self._create_ledger()

        if not os.path.exists(self.trust_path):
            self._create_default_trust()

        print("   [+] Phase 14+26: Meta-Agent (Self-Correcting + ASC Tracking) Initialized.")

    # ----------------------------------------------------------------------
    # LEDGER
    # ----------------------------------------------------------------------

    def _create_ledger(self):
        """Create CSV with Phase 14 + Phase 26 columns."""
        headers = [
            # Phase 14 original columns
            "timestamp", "ticker", "lstm_score", "sent_score",
            "regime_label", "risk_score", "fusion_confidence",
            "final_decision", "price_at_decision", "evaluated",
            "actual_price_t5", "price_change_pct",
            "lstm_grade", "sent_grade", "regime_grade",
            "hypothetical_buy_pnl", "hypothetical_sell_pnl",
            "optimal_decision", "regret_score", "llm_retrospective",
            # Phase 26 new columns
            "asc_score", "asc_reliable",
        ]
        with open(self.ledger_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        print("      [+] Created decision ledger with Phase 26 ASC columns.")

    def log_decision(
        self,
        ticker,
        lstm_score,
        sent_score,
        regime_label,
        risk_score,
        fusion_confidence,
        final_decision,
        price_at_decision,
        # Phase 26 new parameters (optional with defaults for backward compat)
        asc_score: float = 0.5,
        asc_reliable: bool = False,
    ):
        """Record a decision to the ledger, including Phase 26 ASC data."""
        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ticker,
            round(lstm_score, 4),
            round(sent_score, 4),
            regime_label,
            round(risk_score, 4),
            round(fusion_confidence, 4),
            final_decision,
            round(price_at_decision, 2),
            "NO",         # evaluated
            "", "", "", "", "",    # t5 columns
            "", "", "", "", "",    # counterfactual columns
            round(asc_score, 4),  # Phase 26
            str(asc_reliable),    # Phase 26
        ]
        with open(self.ledger_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
        print(f"   [Meta-Agent] Decision logged: {ticker} @ ${price_at_decision:.2f} | ASC={asc_score:.3f}")

    # ----------------------------------------------------------------------
    # HINDSIGHT EVALUATOR
    # ----------------------------------------------------------------------

    def evaluate_past_decisions(self):
        """
        Evaluate un-graded decisions, update trust scores, and run the
        Phase 26 ASC accuracy correlation analysis for paper validation.
        """
        print("\n" + "=" * 60)
        print("[Meta-Agent] HINDSIGHT EVALUATION SESSION (Phase 14+26)")
        print("=" * 60)

        try:
            df = pd.read_csv(self.ledger_path, encoding="utf-8", on_bad_lines="skip")
            str_cols = [
                "evaluated", "actual_price_t5", "price_change_pct",
                "lstm_grade", "sent_grade", "regime_grade",
                "hypothetical_buy_pnl", "hypothetical_sell_pnl",
                "optimal_decision", "regret_score", "llm_retrospective",
            ]
            for col in str_cols:
                if col in df.columns:
                    df[col] = df[col].astype(object)

            # Ensure Phase 26 columns exist (backward compat with old ledgers)
            if "asc_score" not in df.columns:
                df["asc_score"] = 0.5
            if "asc_reliable" not in df.columns:
                df["asc_reliable"] = False

        except Exception as e:
            print(f"   [!] Cannot read ledger: {e}")
            return

        if df.empty:
            print("   [i] Ledger is empty. Nothing to evaluate.")
            return

        unevaluated = df[df["evaluated"] == "NO"].copy()
        if unevaluated.empty:
            print("   [i] All decisions already evaluated.")
            # Still run ASC accuracy table on already-evaluated decisions
            self._print_asc_accuracy_table(df)
            return

        unevaluated["timestamp_dt"] = pd.to_datetime(unevaluated["timestamp"])
        grades_log = []

        for idx, row in unevaluated.iterrows():
            ticker         = row["ticker"]
            decision_date  = row["timestamp_dt"]
            decision_price = float(row["price_at_decision"])
            lstm_score     = float(row["lstm_score"])
            sent_score     = float(row["sent_score"])
            regime_label   = row["regime_label"]
            final_decision = row["final_decision"]

            print(f"\n   --- Evaluating {ticker} from {decision_date.strftime('%Y-%m-%d')} ---")

            actual_price = self._get_price_t5(ticker, decision_date)
            if actual_price is None:
                print("      [!] Could not fetch T+5 price. Skipping.")
                continue

            price_change_pct = (actual_price - decision_price) / decision_price
            confidence = float(row.get("fusion_confidence", 0.5))
            print(f"      Decision Price : ${decision_price:.2f}")
            print(f"      Actual Price   : ${actual_price:.2f}")
            print(f"      Change         : {price_change_pct * 100:.2f}%")

            lstm_grade   = self._grade_agent(lstm_score, price_change_pct, "technical")
            sent_grade   = self._grade_agent(sent_score, price_change_pct, "sentiment")
            regime_grade = self._grade_regime(regime_label, price_change_pct)

            print(f"      Grades -> LSTM: {lstm_grade:+d}, Sent: {sent_grade:+d}, Regime: {regime_grade:+d}")

            df.at[idx, "evaluated"]        = "YES"
            df.at[idx, "actual_price_t5"]  = round(actual_price, 2)
            df.at[idx, "price_change_pct"] = round(price_change_pct, 4)
            df.at[idx, "lstm_grade"]       = f"{lstm_grade:+d}"
            df.at[idx, "sent_grade"]       = f"{sent_grade:+d}"
            df.at[idx, "regime_grade"]     = f"{regime_grade:+d}"

            grades_log.append({
                "ticker": ticker,
                "lstm": lstm_grade,
                "sentiment": sent_grade,
                "regime": regime_grade,
                "regime_label": regime_label,
                "confidence": confidence,
                "regret_penalty": 0.0,
            })

            # Phase 15 counterfactual
            if CounterfactualEngine:
                if not hasattr(self, "_cf_engine"):
                    self._cf_engine = CounterfactualEngine()
                cf_result = self._cf_engine.analyze(
                    actual_decision=final_decision,
                    decision_price=decision_price,
                    actual_price_t5=actual_price,
                    confidence=float(row.get("fusion_confidence", 0.5)),
                )
                retrospective = self._cf_engine.generate_retrospective(
                    ticker=ticker,
                    decision_date=decision_date.strftime("%Y-%m-%d"),
                    cf_result=cf_result,
                    regime_label=regime_label,
                    confidence=float(row.get("fusion_confidence", 0.5)),
                )
                df.at[idx, "hypothetical_buy_pnl"]  = round(cf_result["hypothetical_buy_pnl"], 6)
                df.at[idx, "hypothetical_sell_pnl"] = round(cf_result["hypothetical_sell_pnl"], 6)
                df.at[idx, "optimal_decision"]      = cf_result["optimal_decision"]
                df.at[idx, "regret_score"]          = round(cf_result["regret_score"], 6)
                df.at[idx, "llm_retrospective"]     = retrospective[:500]
                regret_penalty = self._cf_engine.get_regret_penalty(cf_result["regret_score"])
                grades_log[-1]["regret_penalty"] = regret_penalty

        if "timestamp_dt" in df.columns:
            df = df.drop(columns=["timestamp_dt"])
        df.to_csv(self.ledger_path, index=False, encoding="utf-8")
        print(f"\n   [+] Ledger updated ({len(grades_log)} decisions graded).")

        if grades_log:
            self._update_trust_scores(grades_log)

        # Phase 26: Print ASC accuracy correlation table
        self._print_asc_accuracy_table(df)

    def _print_asc_accuracy_table(self, df: pd.DataFrame):
        """
        Phase 26 paper validation: print accuracy bucketed by ASC range.
        Tests hypothesis: low ASC -> higher decision accuracy.
        """
        evaluated = df[df["evaluated"] == "YES"].copy()
        if evaluated.empty or "asc_score" not in evaluated.columns:
            return

        evaluated["asc_score_num"] = pd.to_numeric(evaluated["asc_score"], errors="coerce").fillna(0.5)
        # Multi-level grade mapping: grades >= +1 count as correct, <= -1 as wrong
        def _grade_to_correct(g):
            try:
                val = int(g)
            except (ValueError, TypeError):
                # Backward compat: handle legacy RIGHT/WRONG strings
                if str(g).upper() == "RIGHT":  return 1
                if str(g).upper() == "WRONG":  return 0
                return None
            if val >= 1:  return 1
            if val <= -1: return 0
            return None  # grade 0 = inconclusive
        evaluated["correct"] = evaluated["lstm_grade"].apply(_grade_to_correct)
        evaluated_valid = evaluated.dropna(subset=["correct"])

        if evaluated_valid.empty:
            return

        buckets = {
            f"Low ASC (< {ASC_LOW_THRESHOLD})":
                evaluated_valid[evaluated_valid["asc_score_num"] < ASC_LOW_THRESHOLD],
            f"Medium ASC ({ASC_LOW_THRESHOLD}–{ASC_HIGH_THRESHOLD})":
                evaluated_valid[
                    (evaluated_valid["asc_score_num"] >= ASC_LOW_THRESHOLD) &
                    (evaluated_valid["asc_score_num"] < ASC_HIGH_THRESHOLD)
                ],
            f"High ASC (>= {ASC_HIGH_THRESHOLD})":
                evaluated_valid[evaluated_valid["asc_score_num"] >= ASC_HIGH_THRESHOLD],
        }

        try:
            print("\n   ╔==========================================================╗")
            print("   ║   PHASE 26 HOLD ASC ACCURACY CORRELATION TABLE              ║")
            print("   ║   Hypothesis: Low ASC -> Higher Decision Accuracy          ║")
            print("   ╠==========================================================╣")
            print(f"   ║  {'ASC Bucket':<35s} {'N':>4s}  {'Accuracy':>8s}          ║")
            print("   ╠==========================================================╣")

            for label, subset in buckets.items():
                n = len(subset)
                if n == 0:
                    print(f"   ║  {label:<35s} {'0':>4s}  {'N/A':>8s}          ║")
                else:
                    accuracy = subset["correct"].mean() * 100
                    bar_len  = int(accuracy / 5)
                    bar      = "█" * bar_len + "░" * (20 - bar_len)
                    print(f"   ║  {label:<35s} {n:>4d}  {accuracy:>6.1f}%  {bar}  ║")

            print("   ╠==========================================================╣")

            # Overall Pearson correlation between asc_score and correctness
            if len(evaluated_valid) >= 5:
                try:
                    corr = float(np.corrcoef(
                        evaluated_valid["asc_score_num"].values,
                        evaluated_valid["correct"].values,
                    )[0, 1])
                    direction = "inverse (supports hypothesis)" if corr < 0 else "positive (rejects hypothesis)"
                    print(f"   ║  Pearson r(ASC, accuracy) = {corr:+.4f}  {direction}  ║")
                except Exception:
                    pass

            print("   ╚==========================================================╝\n")
        except UnicodeEncodeError:
            # Fallback for terminals that cannot render box-drawing characters
            print("\n   [ASC ACCURACY TABLE]")
            for label, subset in buckets.items():
                n = len(subset)
                if n == 0:
                    print(f"     {label:<35s}  N=0  N/A")
                else:
                    accuracy = subset["correct"].mean() * 100
                    print(f"     {label:<35s}  N={n}  {accuracy:.1f}%")
            print()

    # ----------------------------------------------------------------------
    # T+5 PRICE FETCH
    # ----------------------------------------------------------------------

    def _get_price_t5(self, ticker, decision_date):
        try:
            start = decision_date + timedelta(days=5)
            end   = decision_date + timedelta(days=12)
            hist  = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                                end=end.strftime("%Y-%m-%d"), progress=False)
            if not hist.empty:
                close_col = hist["Close"]
                if hasattr(close_col, "columns"):
                    close_col = close_col.iloc[:, 0]
                return float(close_col.iloc[0])
            # Proxy with latest price
            latest = yf.download(ticker, period="5d", progress=False)
            if not latest.empty:
                close_col = latest["Close"]
                if hasattr(close_col, "columns"):
                    close_col = close_col.iloc[:, 0]
                return float(close_col.iloc[-1])
            return None
        except Exception as e:
            logger.warning(f"T+5 price fetch failed for {ticker}: {e}")
            return None

    # ----------------------------------------------------------------------
    # GRADING
    # ----------------------------------------------------------------------

    def _grade_agent(self, agent_score, price_change_pct, agent_type):
        """
        Multi-level grading: returns integer in {-2, -1, 0, +1, +2}.
        Replaces binary RIGHT/WRONG to capture signal strength.
        """
        if agent_type == "technical":
            predicted_bullish = agent_score > 0.5
        elif agent_type == "sentiment":
            predicted_bullish = agent_score > 0.0
        else:
            predicted_bullish = agent_score > 0.5

        abs_change = abs(price_change_pct)
        if abs_change < MOVEMENT_THRESHOLD:
            return 0   # inside noise band → inconclusive

        market_went_up = price_change_pct > 0
        direction_correct = (predicted_bullish == market_went_up)

        if direction_correct:
            return 2 if abs_change >= STRONG_MOVE_THRESHOLD else 1
        else:
            return -2 if abs_change >= STRONG_MOVE_THRESHOLD else -1

    def _grade_regime(self, regime_label, price_change_pct):
        """
        Multi-level regime grading with bear-rally and bull-pullback tolerance.
        Returns integer in {-2, -1, 0, +1, +2}.
        """
        abs_change = abs(price_change_pct)

        if abs_change < MOVEMENT_THRESHOLD:
            return 0   # noise band

        if regime_label == "Sideways":
            # Sideways is correct if within tolerance, neutral if slightly outside
            if abs_change < SIDEWAYS_TOLERANCE:
                return 1
            elif abs_change < STRONG_MOVE_THRESHOLD:
                return -1  # sideways missed a moderate move
            else:
                return -2  # sideways missed a strong move

        if regime_label == "Bull":
            if price_change_pct > 0:
                return 2 if abs_change >= STRONG_MOVE_THRESHOLD else 1
            else:
                # Bull pullback: small dip is inconclusive, not flat wrong
                if abs_change <= BEAR_RALLY_TOLERANCE:
                    return 0  # bull pullback tolerance
                return -2 if abs_change >= STRONG_MOVE_THRESHOLD else -1

        if regime_label == "Bear":
            if price_change_pct < 0:
                return 2 if abs_change >= STRONG_MOVE_THRESHOLD else 1
            else:
                # Bear rally: small rise is inconclusive, not flat wrong
                if abs_change <= BEAR_RALLY_TOLERANCE:
                    return 0  # bear rally tolerance
                return -2 if abs_change >= STRONG_MOVE_THRESHOLD else -1

        return 0  # unknown regime label

    # ----------------------------------------------------------------------
    # TRUST SCORE MANAGER
    # ----------------------------------------------------------------------

    def _create_default_trust(self):
        regime_default = {"technical": TRUST_DEFAULT, "sentiment": TRUST_DEFAULT, "regime": TRUST_DEFAULT}
        default_scores = {
            "technical": TRUST_DEFAULT,
            "sentiment": TRUST_DEFAULT,
            "regime": TRUST_DEFAULT,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "evaluation_count": 0,
            # Regime-specific trust: tracks agent performance per market regime
            "regime_trust": {
                "Bull":     dict(regime_default),
                "Bear":     dict(regime_default),
                "Sideways": dict(regime_default),
            },
        }
        with open(self.trust_path, "w", encoding="utf-8") as f:
            json.dump(default_scores, f, indent=2)

    def _update_trust_scores(self, grades_log):
        current = self.load_trust_scores()
        agent_rewards = {"technical": [], "sentiment": [], "regime": []}
        ticker_rewards: dict = {}
        regime_rewards: dict = {}  # per-regime trust tracking

        # Data sufficiency: dampen learning when evaluation_count is low
        eval_count = current.get("evaluation_count", 0)
        effective_alpha = EMA_ALPHA
        if eval_count < MIN_EVALUATIONS_FOR_TRUST:
            effective_alpha = EMA_ALPHA * DAMPENED_ALPHA_FACTOR
            print(f"\n   [Meta-Agent] Data sufficiency warning: only {eval_count} evaluations "
                  f"(need {MIN_EVALUATIONS_FOR_TRUST}). Using dampened alpha={effective_alpha:.4f}")

        for entry in grades_log:
            tk = entry.get("ticker", "GLOBAL")
            regret_penalty = entry.get("regret_penalty", 0.0)
            confidence = entry.get("confidence", 0.5)
            regime_label = entry.get("regime_label", "Sideways")
            if tk not in ticker_rewards:
                ticker_rewards[tk] = {"technical": [], "sentiment": [], "regime": []}
            if regime_label not in regime_rewards:
                regime_rewards[regime_label] = {"technical": [], "sentiment": [], "regime": []}

            for agent_key in agent_rewards:
                grade_key = "lstm" if agent_key == "technical" else agent_key
                grade     = entry.get(grade_key, 0)
                # Multi-level: grade is already numeric (-2, -1, 0, +1, +2)
                # Normalize to [-1, +1] range for reward calculation
                reward    = float(grade) / 2.0
                # Confidence weighting: high-confidence mistakes hurt more
                reward   *= confidence
                # Asymmetric penalty: high-confidence wrong predictions hurt 1.5x extra
                if reward < 0 and confidence > CONFIDENT_WRONG_THRESHOLD:
                    reward *= CONFIDENT_WRONG_MULTIPLIER
                reward   += regret_penalty
                agent_rewards[agent_key].append(reward)
                ticker_rewards[tk][agent_key].append(reward)
                regime_rewards[regime_label][agent_key].append(reward)

        print("\n   [Meta-Agent] Updating Trust Scores (EMA):")
        for agent_key in ["technical", "sentiment", "regime"]:
            old_trust = current.get(agent_key, TRUST_DEFAULT)
            rewards   = agent_rewards[agent_key]
            if not rewards:
                continue
            avg_reward = np.mean(rewards)
            target     = TRUST_DEFAULT + (avg_reward * 0.5)
            new_trust  = old_trust + effective_alpha * (target - old_trust)
            new_trust  = max(TRUST_MIN, min(TRUST_MAX, new_trust))
            direction  = "+" if new_trust > old_trust else "-" if new_trust < old_trust else "="
            print(f"      {agent_key:12s}: {old_trust:.3f} -> {new_trust:.3f} "
                  f"({direction}) [avg_reward={avg_reward:+.2f}]")
            current[agent_key] = round(new_trust, 4)

        # Per-ticker trust
        for tk, tk_rewards in ticker_rewards.items():
            ticker_key = f"ticker_{tk.upper()}"
            existing   = current.get(ticker_key, {})
            for agent_key in ["technical", "sentiment", "regime"]:
                rewards = tk_rewards[agent_key]
                if not rewards:
                    continue
                avg_reward = np.mean(rewards)
                old_t      = existing.get(agent_key, TRUST_DEFAULT)
                target     = TRUST_DEFAULT + (avg_reward * 0.5)
                new_t      = old_t + effective_alpha * (target - old_t)
                new_t      = max(TRUST_MIN, min(TRUST_MAX, new_t))
                existing[agent_key] = round(new_t, 4)
            current[ticker_key] = existing

        # Per-regime trust (new)
        if "regime_trust" not in current:
            current["regime_trust"] = {
                "Bull": {"technical": TRUST_DEFAULT, "sentiment": TRUST_DEFAULT, "regime": TRUST_DEFAULT},
                "Bear": {"technical": TRUST_DEFAULT, "sentiment": TRUST_DEFAULT, "regime": TRUST_DEFAULT},
                "Sideways": {"technical": TRUST_DEFAULT, "sentiment": TRUST_DEFAULT, "regime": TRUST_DEFAULT},
            }
        for rl, rl_rewards in regime_rewards.items():
            if rl not in current["regime_trust"]:
                current["regime_trust"][rl] = {"technical": TRUST_DEFAULT, "sentiment": TRUST_DEFAULT, "regime": TRUST_DEFAULT}
            for agent_key in ["technical", "sentiment", "regime"]:
                rewards = rl_rewards[agent_key]
                if not rewards:
                    continue
                avg_reward = np.mean(rewards)
                old_rt     = current["regime_trust"][rl].get(agent_key, TRUST_DEFAULT)
                target     = TRUST_DEFAULT + (avg_reward * 0.5)
                new_rt     = old_rt + effective_alpha * (target - old_rt)
                new_rt     = max(TRUST_MIN, min(TRUST_MAX, new_rt))
                current["regime_trust"][rl][agent_key] = round(new_rt, 4)

        current["last_updated"]     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current["evaluation_count"] = eval_count + len(grades_log)

        with open(self.trust_path, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)
        print("      [+] Trust scores saved.")

    def load_trust_scores(self):
        try:
            with open(self.trust_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"technical": TRUST_DEFAULT, "sentiment": TRUST_DEFAULT, "regime": TRUST_DEFAULT}

    def get_trust_scores(self, ticker=None, regime=None):
        scores = self.load_trust_scores()
        
        # Phase Temporal: Decay trust scores toward default if stale to fix temporal degradation problem
        last_updated_str = scores.get("last_updated")
        decay_factor = 1.0
        if last_updated_str:
            try:
                # Assuming format %Y-%m-%d %H:%M:%S
                last_updated = datetime.strptime(last_updated_str, "%Y-%m-%d %H:%M:%S")
                # For testing or simulation, if date is in the past, decay kicks in
                days_since_update = (datetime.now() - last_updated).days
                if days_since_update > 0:
                    # Decay 5% towards TRUST_DEFAULT for every 7 days without update, max decay is 100%
                    decay_ratio = min(1.0, (days_since_update / 7.0) * 0.05)
                    decay_factor = 1.0 - decay_ratio
            except Exception as e:
                logger.warning(f"Error parsing last_updated in meta agent: {e}")

        global_scores = {
            "technical": scores.get("technical", TRUST_DEFAULT),
            "sentiment": scores.get("sentiment", TRUST_DEFAULT),
            "regime":    scores.get("regime", TRUST_DEFAULT),
        }
        
        # Apply temporal decay to global scores
        for agent in global_scores:
            current_score = global_scores[agent]
            global_scores[agent] = round(current_score * decay_factor + TRUST_DEFAULT * (1.0 - decay_factor), 4)

        if ticker:
            ticker_key    = f"ticker_{ticker.upper()}"
            ticker_scores = scores.get(ticker_key, {})
            if ticker_scores:
                for agent in list(global_scores.keys()):
                    if agent in ticker_scores:
                        ticker_score = ticker_scores[agent]
                        # Apply temporal decay to ticker specific score as well
                        ticker_score_decayed = ticker_score * decay_factor + TRUST_DEFAULT * (1.0 - decay_factor)
                        global_scores[agent] = round(
                            0.70 * global_scores[agent] + 0.30 * ticker_score_decayed, 4
                        )

        # Regime-specific trust blending: global * 0.6 + regime_specific * 0.4
        if regime:
            regime_trust = scores.get("regime_trust", {})
            rt = regime_trust.get(regime, {})
            if rt:
                for agent in list(global_scores.keys()):
                    if agent in rt:
                        regime_val = rt[agent]
                        # Apply temporal decay to regime trust
                        regime_val_decayed = regime_val * decay_factor + TRUST_DEFAULT * (1.0 - decay_factor)
                        global_scores[agent] = round(
                            0.60 * global_scores[agent] + 0.40 * regime_val_decayed, 4
                        )

        # Trust amplification: sqrt the deviation from 1.0 to make trust
        # actually change behavior in the fusion layer, capped to prevent explosion
        for agent in global_scores:
            raw = global_scores[agent]
            deviation = raw - TRUST_DEFAULT  # positive = boosted, negative = penalized
            sign = 1.0 if deviation >= 0 else -1.0
            amp_dev = (abs(deviation) ** 0.5) * TRUST_AMPLIFICATION_FACTOR
            amp_dev = min(amp_dev, MAX_AMPLIFIED_DEVIATION)  # prevent explosion
            amplified = TRUST_DEFAULT + sign * amp_dev
            amplified = max(TRUST_MIN, min(TRUST_MAX, amplified))
            global_scores[agent] = round(amplified, 4)

        return global_scores

    @staticmethod
    def print_trust_report(trust_scores):
        print("\n   [Meta-Agent] Current Agent Trust Multipliers:")
        print("   " + "-" * 50)
        for agent, score in trust_scores.items():
            if agent in ("technical", "sentiment", "regime"):
                bar_len = int(score * 20)
                bar     = "#" * bar_len + "." * (30 - bar_len)
                status  = "BOOSTED" if score > 1.05 else "PENALIZED" if score < 0.95 else "NORMAL"
                print(f"      {agent:12s}: {score:.3f}  [{bar}]  {status}")
        print("   " + "-" * 50)