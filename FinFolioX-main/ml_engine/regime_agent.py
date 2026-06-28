import numpy as np
from hmmlearn import hmm
import joblib
import os

class RegimeAgent:
    def __init__(self, model_path=None):
        """
        Market Regime Detection using Gaussian HMM.
        Automatically maps hidden states to: Bull, Bear, Sideways.
        """
        self.model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
        self.model_path = model_path
        self.regime_map = {}  # Stores mapping: {0: 'Bull', 1: 'Sideways', ...}
        
        # Load if exists
        if model_path and os.path.exists(model_path):
            try:
                saved_data = joblib.load(model_path)
                # Check if it's the old format or new dictionary format
                if isinstance(saved_data, dict):
                    self.model = saved_data['model']
                    self.regime_map = saved_data.get('regime_map', {})
                else:
                    self.model = saved_data # Fallback for old models
                print(f"[OK] Regime Agent Loaded from {model_path}")
            except Exception as e:
                print(f"[BAD] Error loading model: {e}")
        else:
            print("[WARN] Regime Agent initialized (Untrained).")

    def train(self, data):
        """
        Train the HMM and automatically figure out which state is which.
        """
        print("⏳ Training HMM on Market Data...")
        self.model.fit(data)
        
        # --- AUTO-LABELING LOGIC ---
        # We need to find out: Which state is 'Bull'? Which is 'Bear'?
        self._label_regimes(data)
        print("[OK] HMM Training Complete & Regimes Labeled.")

    def _label_regimes(self, data):
        """
        Internal function to map random State numbers (0,1,2) to Human Labels.
        Logic:
        - Highest Average Return -> Bull
        - Lowest Average Return -> Bear
        - The Middle one -> Sideways
        """
        states = self.model.predict(data)
        
        # Calculate stats for each state
        state_stats = {}
        for state in range(3):
            mask = (states == state)
            if np.sum(mask) > 0:
                state_data = data[mask]
                mean_return = state_data[:, 0].mean() # Column 0 is Returns
                mean_vol = state_data[:, 1].mean()    # Column 1 is Volatility
                state_stats[state] = {'ret': mean_return, 'vol': mean_vol}
            else:
                state_stats[state] = {'ret': -999, 'vol': 999} # Empty state handling

        # Sort states by Return (High to Low)
        sorted_states = sorted(state_stats.items(), key=lambda x: x[1]['ret'], reverse=True)
        
        bull_state = sorted_states[0][0]     # Highest Return
        sideways_state = sorted_states[1][0] # Middle
        bear_state = sorted_states[2][0]     # Lowest Return
        
        self.regime_map = {
            bull_state: 'Bull',
            sideways_state: 'Sideways',
            bear_state: 'Bear'
        }
        
        print("\n📊 Discovered Market Regimes:")
        for s, label in self.regime_map.items():
            print(f"   State {s} -> {label} (Avg Ret: {state_stats[s]['ret']:.4f}, Vol: {state_stats[s]['vol']:.4f})")

    def save(self, save_path):
        # Save both the model AND the label mapping
        payload = {
            'model': self.model,
            'regime_map': self.regime_map
        }
        joblib.dump(payload, save_path)
        print(f"💾 Regime Agent saved to {save_path}")

    def predict_regime(self, recent_data):
        """Returns the State ID (int)"""
        try:
            if recent_data.ndim == 1:
                recent_data = recent_data.reshape(1, -1)
            state = self.model.predict(recent_data)[-1]  # [-1] = today, not 30 days ago
            return state
        except:
            return 0
            
    def get_regime_label(self, recent_data):
        """Returns the Human Label (str)"""
        state = self.predict_regime(recent_data)
        return self.regime_map.get(state, "Unknown")