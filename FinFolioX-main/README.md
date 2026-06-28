# 🚀 FinFolio-X

**An Advanced Agentic AI Framework for Intelligent Financial Decision-Making**

FinFolio-X is a state-of-the-art multi-agent, multi-modal AI system designed to support intelligent trading decisions. By combining technical analysis, global financial news sentiment, market regime detection, causal discovery, topological data analysis, and risk-aware position sizing, the system emulates the rigor of quantitative trading desks.

At the core of FinFolio-X is a **Mixture of Experts (MoE)** architecture, orchestrated via **LangGraph**, where specialized AI agents independently analyze different facets of the market. Their outputs are synthesized using a hierarchical multi-head attention mechanism and conflict resolution logic, ensuring dynamic signal prioritization based on market context, volatility, and model confidence.

⚠️ **Status:** Actively under development (Current Version: `v27.0 - Hybrid Regime + ASC Sycophancy Detection`). 

---

## 🧠 System Architecture & Agent Specializations

FinFolio-X operates through an extensive pipeline of AI agents, each contributing a unique perspective to the final decision. 

### Core Analytical Agents
1. **Technical Agent (LSTM):** Deep learning model trained to analyze price trends and historical patterns.
2. **Sentiment Agent (FinBERT + MCP):** Analyzes global news sentiment using an MCP (Model Context Protocol) server to gather real-time macroeconomic context.
3. **Hybrid Regime Agent (HMM + Rule-Based):** Detects hidden market states (Bull, Bear, Sideways) by fusing rule-based heuristics with Hidden Markov Models (v2.3).
4. **Correlation Agent:** Employs statistical graph theory to detect cross-asset divergence and systemic risks.

### Advanced Mathematical & Causal Agents
5. **Topological Shape Agent (TDA):** Phase 24 module that applies Persistent Homology (Ripser) to compute the underlying geometric "shape" and chaos of market data.
6. **Causal Discovery Agent:** Phase 25 module running the PC Algorithm to distinguish true causal drivers from spurious correlations across a macro universe.
7. **Uncertainty Agent (Bayesian):** Quantifies confidence distance and model uncertainty using Monte Carlo dropout/Bayesian approximations.
8. **Explainability Agent (SHAP):** Runs perturbation analysis to explain the *why* behind predictions, extracting top market drivers.

### Orchestration, Fusion & Risk Agents
9. **LangGraph Orchestrator & LLM Supervisor:** Manages the multi-agent asynchronous graph flow, utilizing Groq LLMs for executive summarization and process oversight.
10. **Fusion Agent (Multi-Head Attention):** The synthesis layer. Learns to weigh inputs dynamically (e.g., favoring volatility metrics in a Bear regime).
11. **ASC Memory Engine (Phase 26):** Tracks historical agent decisions to compute the **Agent Sycophancy Coefficient**. Detects and penalizes sycophantic/herd behavior among agents to maintain contrarian robustness.
12. **Conflict Resolver (Phase 13):** Arbitrates disagreements between the Technical, Sentiment, and Regime agents using structured Group Disagreement Index (GDI) metrics (Heatmap Agent - Phase 16).
13. **Adversarial Tester (Red Team - Phase 11):** Simulates live flash crashes to ensure the pipeline's robustness before issuing a final BUY signal.
14. **Risk Engine (Kelly Criterion):** Calculates optimal position sizing and portfolio allocation strictly capping downside risk.
15. **Meta-Agent (Phase 14):** Maintains dynamic trust scores for each agent based on their historical accuracy within specific market regimes.

---

## 🛠️ Tech Stack

### AI / Machine Learning
- **Orchestration:** LangGraph, LangChain, Groq LLM
- **Deep Learning / NLP:** TensorFlow (Keras), PyTorch, HuggingFace Transformers (FinBERT)
- **Math & Stats:** `causal-learn` (Causal Discovery), `ripser` & `persim` (Topological Data Analysis), `hmmlearn` (Hidden Markov Models), `scikit-learn`
- **Data & APIs:** `yfinance`, `pandas_datareader`, `networkx`

### Backend
- **Framework:** Django 5.0, Django REST Framework
- **Architecture:** API-driven REST endpoints mapping to the LangGraph/Master System orchestrator.

### Frontend
- **Framework:** React 19, Vite
- **UI/UX:** Framer Motion (for dynamic animations), Lucide React (icons)
- **Routing:** React Router DOM

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+
- API Keys: Groq API Key (for LLM Supervisor)

### Backend Setup
```bash
# 1. Clone and navigate to the project directory
cd d:\FinFolioX

# 2. Create and activate a Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
# Ensure your .env file has required keys like GROQ_API_KEY
cp .env.example .env 

# 5. Run Django Migrations
python manage.py migrate

# 6. Start the Backend Server
python manage.py runserver
```

### Frontend Setup
```bash
# 1. Navigate to the frontend directory
cd frontend

# 2. Install NPM packages
npm install

# 3. Start the Vite development server
npm run dev
```

---

## 🔮 Future Roadmap

- **Phase 28:** Advanced multi-asset portfolio rebalancing integration.
- **Phase 29:** Reinforcement Learning from Human Feedback (RLHF) for tuning the Conflict Arbitrator.
- **Continuous:** Expanding the MCP protocol plugins for deeper on-chain data and alternative data sources.

---

> **Disclaimer:** FinFolio-X is currently a research project and educational tool. It does not constitute financial advice. Trading in financial markets involves significant risk.
