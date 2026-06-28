import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_engine.fusion_agent import MultiHeadFusion

# --- CONFIGURATION ---
SAVE_PATH = os.path.join("saved_models", "attention_fusion.pth")
EPOCHS = 300
LEARNING_RATE = 0.001

def generate_market_scenarios(samples=5000):
    """
    Generates synthetic data with RANDOMIZED thresholds to prevent overfitting.
    """
    data = []
    targets = []
    
    for _ in range(samples):
        lstm_val = np.random.uniform(0, 1)      # LSTM Prediction
        sent_val = np.random.uniform(-1, 1)     # Sentiment Score
        volatility = np.random.uniform(0, 1)    # Volatility Index
        
        sent_norm = (sent_val + 1) / 2 
        
        # --- RANDOMIZED THRESHOLDS ---
        # Real markets don't have exact cutoffs. We add noise to the rules.
        calm_threshold = np.random.uniform(0.25, 0.35)  # Around 0.3
        panic_threshold = np.random.uniform(0.65, 0.75) # Around 0.7
        
        if volatility < calm_threshold:
            # RULE 1: CALM MARKET (Trust Technicals)
            target = (lstm_val * 0.9) + (sent_norm * 0.1)
            
        elif volatility > panic_threshold:
            # RULE 2: PANIC MARKET (Trust News)
            target = (lstm_val * 0.1) + (sent_norm * 0.9)
            
        else:
            # RULE 3: NORMAL MARKET (Balanced)
            if abs(sent_val) > 0.8: 
                target = (lstm_val * 0.3) + (sent_norm * 0.7)
            else:
                target = (lstm_val * 0.6) + (sent_norm * 0.4)
        
        data.append([lstm_val, sent_val, volatility])
        targets.append([target])
        
    return torch.tensor(data, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

def train_advanced():
    print("🧠 Generating Synthetic Scenarios with Randomized Thresholds...")
    X, y = generate_market_scenarios()
    
    # Initialize the Model
    model = MultiHeadFusion(d_model=16, nhead=4) 
    
    # --- FIX IS HERE: Corrected the double 'optim' typo ---
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    criterion = nn.MSELoss()
    
    print("🏋️ Training Multi-Head Attention Engine...")
    print("-" * 40)
    
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        # Prepare inputs
        lstm_in = X[:, 0].unsqueeze(1)
        sent_in = X[:, 1].unsqueeze(1)
        vol_in  = X[:, 2].unsqueeze(1)
        
        # Forward pass
        preds, _ = model(lstm_in, sent_in, vol_in)
        loss = criterion(preds, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"   Epoch {epoch}: Loss = {loss.item():.6f}")

    # Save
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH)
    print("-" * 40)
    print(f"[OK] Advanced Fusion Model Saved to: {SAVE_PATH}")

    # --- FINAL EXAM ---
    print("\n🧐 FINAL EXAM: Testing Logic & Interpretability")
    model.eval()
    
    # Scenario: LSTM says Buy (0.9), Sentiment says Sell (-0.9), Volatility is High (0.9)
    t_lstm = torch.tensor([[0.9]])
    t_sent = torch.tensor([[-0.9]]) 
    t_vol = torch.tensor([[0.9]])   
    
    with torch.no_grad():
        conf, weights = model(t_lstm, t_sent, t_vol)
    
    print(f"   Scenario: LSTM=Buy | Sentiment=Sell | Volatility=High")
    print(f"   🤖 Fused Confidence Output: {conf.item():.4f}")
    
    # In a high volatility market with negative sentiment, the confidence should be LOW (Sell)
    if conf.item() < 0.4:
        print("   [OK] PASSED: The AI correctly ignored the Technicals!")
    else:
        print("   [BAD] FAILED: The AI is still confused.")

if __name__ == "__main__":
    train_advanced()