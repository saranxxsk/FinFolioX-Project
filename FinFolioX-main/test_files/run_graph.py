import os
import sys

# Append project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_engine.master_system import FinFolioSystem
from ml_engine.langgraph_orchestrator import FinFolioGraphOrchestrator

if __name__ == "__main__":
    # 1. Initialize the heavy math agents (Loads models into memory)
    master = FinFolioSystem()
    
    # 2. Hand them over to the LangGraph Supervisor
    orchestrator = FinFolioGraphOrchestrator(master)
    
    # 3. Ask for a ticker
    ticker = input("\n🔎 Enter Ticker for LangGraph Analysis (e.g., AAPL): ").upper()
    
    # 4. Watch the Magic Happen
    orchestrator.run_analysis(ticker)