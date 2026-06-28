import sys
import os

# Add the project root to python path so we can import from ml_engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_engine.sentiment_agent import SentimentAgent

def test_sentiment_logic():
    # 1. Initialize Agent
    agent = SentimentAgent()
    
    # 2. Define Dummy Data (Mixed bag of news)
    news_headlines = [
        "Apple reports record-breaking revenue for Q4, beating expectations.",  # Should be POSITIVE
        "Supply chain issues continue to plague iPhone production lines.",      # Should be NEGATIVE
        "Analysts are uncertain about the new tech regulation policies.",       # Should be NEUTRAL or slightly NEGATIVE
        "Apple announces new buyback program, shareholders rejoice.",           # Should be POSITIVE
    ]
    
    # 3. Run Analysis
    print("\n🚀 STARTING SENTIMENT ANALYSIS TEST")
    print("="*60)
    
    label, score = agent.analyze_daily_headlines(news_headlines)
    
    print("="*60)
    print(f"📊 DAILY SUMMARY:")
    print(f"   Final Label: {label.upper()}")
    print(f"   Aggregate Score: {score:.4f} (-1.0 to +1.0)")
    print("="*60)

    # 4. Interpretation
    if score > 0:
        print("[OK] SUCCESS: The AI correctly identified a generally POSITIVE day.")
    else:
        print("[BAD] CHECK: The score seems low. Verify the headlines.")

if __name__ == "__main__":
    test_sentiment_logic()