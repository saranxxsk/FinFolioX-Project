import sys
import os

# 1. Setup Path to find 'ml_engine'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_engine.master_system import FinFolioSystem

def main():
    try:
        # 2. Initialize the AI System (Once)
        # This loads the heavy models so you don't wait every time
        app = FinFolioSystem()
        
        # 3. Continuous Loop
        while True:
            print("\n" + "="*60)
            ticker = input("🔎 ENTER TICKER (e.g., AAPL, NVDA, TSLA) or 'q' to quit: ").strip().upper()
            
            if ticker == 'Q':
                print("   👋 Shutting down FinFolio-X...")
                break
            
            if not ticker: continue
            
            # 4. Run Analysis
            app.analyze_stock(ticker)
            
    except KeyboardInterrupt:
        print("\n   [WARN] System halted by user.")
    except Exception as e:
        print(f"   [BAD] Critical Error: {e}")
        print("   Tip: Check if your models are trained and saved in 'saved_models/'")

if __name__ == "__main__":
    main()