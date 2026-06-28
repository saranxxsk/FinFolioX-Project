import torch
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_engine.fusion_agent import FusionAgent

def run_sanity_check():
    print("🕵️ STARTING PHASE 5 VERIFICATION (THE LIE DETECTOR)...")
    print("-" * 60)

    # 1. Load the Trained Agent
    model_path = os.path.join("saved_models", "attention_fusion.pth")
    if not os.path.exists(model_path):
        print(f"[BAD] ERROR: Model not found at {model_path}. Did you run Phase 5 training?")
        return

    agent = FusionAgent(model_path)
    print("[OK] Fusion Agent Loaded Successfully.\n")

    # ==========================================
    # TEST CASE 1: THE BULL MARKET (Calm & Up)
    # ==========================================
    # Scenario: 
    # - LSTM says BUY (0.9)
    # - Sentiment is NEUTRAL (0.1)
    # - Volatility is LOW (0.1) -> "Calm waters"
    # EXPECTATION: Trust the LSTM.
    
    print("🧪 TEST 1: Calm Bull Market (Should Trust LSTM)")
    conf, weights = agent.predict(lstm_p=0.9, sent_s=0.1, vol_v=0.1)
    
    print(f"   Input: LSTM=0.9 | Sentiment=0.1 | Volatility=0.1")
    print(f"   🤖 Output Confidence: {conf:.4f} (Expected > 0.8)")
    print(f"   🧠 Attention Weights: {weights}")
    
    if conf > 0.8: 
        print("   [OK] PASS: AI correctly bought the dip based on technicals.")
    else: 
        print("   [BAD] FAIL: AI was too scared.")
    print("-" * 30)

    # ==========================================
    # TEST CASE 2: THE MARKET CRASH (Panic & Trap)
    # ==========================================
    # Scenario: 
    # - LSTM says BUY (0.8) -> "Buying the dip" (This is a TRAP)
    # - Sentiment is BAD (-0.9) -> "Breaking News: War/Crisis"
    # - Volatility is HIGH (0.9) -> "Panic selling"
    # EXPECTATION: Ignore LSTM, Trust Sentiment -> SELL.
    
    print("🧪 TEST 2: Market Crash (Should Trust Sentiment/Panic)")
    conf, weights = agent.predict(lstm_p=0.8, sent_s=-0.9, vol_v=0.9)
    
    print(f"   Input: LSTM=0.8 (Trap) | Sentiment=-0.9 (Panic) | Volatility=0.9")
    print(f"   🤖 Output Confidence: {conf:.4f} (Expected < 0.3)")
    print(f"   🧠 Attention Weights: {weights}")

    if conf < 0.3:
        print("   [OK] PASS: AI correctly ignored the trap and sold!")
    else:
        print("   [BAD] FAIL: AI got tricked by the chart and bought into a crash.")
    print("-" * 30)

    # ==========================================
    # TEST CASE 3: INTERPRETABILITY CHECK
    # ==========================================
    # We check if the AI actually "looked" at the Sentiment during the crash.
    # In Test 2 (Crash), 'Sentiment_Focus' should be high.
    
    print("🧪 TEST 3: Attention Mechanism Check")
    sent_focus = weights['Sentiment_Focus']
    vol_focus = weights['Volatility_Focus']
    
    print(f"   In Crash Scenario -> Sentiment Focus: {sent_focus:.4f} | Volatility Focus: {vol_focus:.4f}")
    
    # We expect Sentiment or Volatility focus to be significant (e.g., > 0.3 or 30%)
    if sent_focus > 0.25 or vol_focus > 0.25:
        print("   [OK] PASS: The AI is paying attention to the risk factors.")
    else:
        print("   [WARN] WARNING: Attention is low. It might be guessing.")

    print("\n" + "="*60)
    print("🏆 VERIFICATION COMPLETE")

if __name__ == "__main__":
    run_sanity_check()