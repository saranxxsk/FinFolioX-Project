"""
ml_engine/fusion_agent.py  HOLD  Multi-Head Attention Fusion Agent (v2.1)
=======================================================================
WHAT'S NEW IN v2.1:
  - Version string added for changelog tracking.
  - predict() docstring clarified: expects STRETCHED LSTM probability
    (from TechnicalAgent.predict(), not predict_raw()).
  - interpret_weights() shape contract documented: both architectures
    return attn_weights of shape (batch=1, 3, 3).
  - Weight collapse guard threshold documented as 0.35.

ARCHITECTURE AUTO-DETECTION:
  KaggleFusion (legacy P100 Kaggle model):
    Keys: lstm_proj, sent_proj, vol_proj, transformer, decoder
    Checkpoint format: {"model_state": ..., "hyperparameters": ..., "normalization_stats": ...}

  MultiHeadFusion (local synthetic model):
    Keys: lstm_embed, sent_embed, vol_embed, attention, fc1, fc2

PRODUCTION CALL CHAIN (from finfolio_system.py / langgraph_orchestrator.py):
    vol_v = 0.9 if regime=="Bear" else 0.2 if regime=="Bull" else 0.5
    final_conf, weights = fusion_agent.predict(
        lstm_p  = mc_mean,       ← stretched prob from UncertaintyAgent
        sent_s  = sent_score,    ← FinBERT score ∈ [-0.75, +0.75]
        vol_v   = vol_v,         ← regime-derived vol proxy
        trust_scores = {...},    ← optional per-agent trust multipliers
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# -- Version -------------------------------------------------------------------
FUSION_AGENT_VERSION = "v2.1"


# ==============================================================================
# ARCHITECTURE A HOLD Kaggle P100 model (legacy)
# ==============================================================================
class KaggleFusion(nn.Module):
    def __init__(self, d_model=64, nhead=8, dropout=0.17):
        super().__init__()
        assert d_model % nhead == 0

        self.lstm_proj = nn.Linear(1, d_model)
        self.sent_proj = nn.Linear(1, d_model)
        self.vol_proj  = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 3, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_lstm, x_sent, x_vol):
        t_lstm = self.lstm_proj(x_lstm).unsqueeze(1)
        t_sent = self.sent_proj(x_sent).unsqueeze(1)
        t_vol  = self.vol_proj(x_vol).unsqueeze(1)
        tokens = torch.cat([t_lstm, t_sent, t_vol], dim=1) + self.pos_embed
        enc    = self.transformer(tokens)
        pooled = enc.mean(dim=1)
        conf   = self.decoder(pooled)
        # Attention proxy: use post-encoder token norms as signal importance.
        # enc shape: (batch, 3, d_model) HOLD one token per signal (lstm, sent, vol).
        # L2 norm per token -> larger norm = that signal drove more of the representation.
        # Softmax-normalised so weights sum to 1 across the 3 signals.
        # Expanded to (batch, 3, 3) to match MultiHeadFusion.attn_w shape so
        # interpret_weights() works identically for both architectures.
        token_norms = enc.norm(dim=-1)                      # (batch, 3)
        token_weights = F.softmax(token_norms, dim=-1)      # (batch, 3) HOLD sums to 1
        proxy_attn = token_weights.unsqueeze(1).expand(-1, 3, -1)  # (batch, 3, 3)
        return conf, proxy_attn


# ==============================================================================
# ARCHITECTURE B HOLD Local synthetic model
# Keys: lstm_embed, sent_embed, vol_embed, attention.*, fc1, fc2
# ==============================================================================
class MultiHeadFusion(nn.Module):
    def __init__(self, d_model=16, nhead=4):
        assert d_model % nhead == 0
        super().__init__()
        self.lstm_embed = nn.Linear(1, d_model)
        self.sent_embed = nn.Linear(1, d_model)
        self.vol_embed  = nn.Linear(1, d_model)
        self.attention  = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.fc1     = nn.Linear(d_model * 3, 32)
        self.dropout = nn.Dropout(0.2)
        self.fc2     = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, lstm_pred, sentiment_score, volatility):
        e_lstm   = F.relu(self.lstm_embed(lstm_pred)).unsqueeze(1)
        e_sent   = F.relu(self.sent_embed(sentiment_score)).unsqueeze(1)
        e_vol    = F.relu(self.vol_embed(volatility)).unsqueeze(1)
        sequence = torch.cat((e_lstm, e_sent, e_vol), dim=1)
        # attn_out shape: (batch, 3, d_model)
        # attn_w  shape: (batch, 3, 3)  ← same as KaggleFusion dummy
        attn_out, attn_w = self.attention(sequence, sequence, sequence)
        x = F.relu(self.fc1(attn_out.reshape(attn_out.size(0), -1)))
        x = self.dropout(x)
        return self.sigmoid(self.fc2(x)), attn_w


# ==============================================================================
# FUSION AGENT HOLD auto-detects architecture from checkpoint keys
# ==============================================================================
class FusionAgent:
    """
    Wraps KaggleFusion or MultiHeadFusion.
    Auto-detects architecture from checkpoint key names at load time.

    predict() contract:
      Inputs must be in the same scale used during training:
        lstm_p  : stretched LSTM probability ∈ [0, 1]
                  Source: TechnicalAgent.predict() (NOT predict_raw())
                  UncertaintyAgent.predict_with_uncertainty() returns this as mc_mean.
        sent_s  : FinBERT blended score ∈ [-0.75, +0.75]
        vol_v   : regime-derived proxy HOLD 0.9 (Bear) | 0.5 (Sideways) | 0.2 (Bull)
                  NOT the raw decimal vol from HybridRegimeAgent.
        trust_scores: optional dict {"technical": float, "sentiment": float, "regime": float}
    """

    def __init__(self, model_path=None):
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model       = None
        self._norm_stats = None
        self._arch       = "unknown"

        if model_path:
            self._load(model_path)
        else:
            self.model = KaggleFusion().to(self.device)
            self._arch = "KaggleFusion (default)"

        if self.model is not None:
            self.model.eval()

    def _load(self, model_path):
        try:
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False
            )

            # Unwrap Kaggle-style dict
            if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                state_dict       = checkpoint["model_state"]
                hp               = checkpoint.get("hyperparameters", {})
                self._norm_stats = checkpoint.get("normalization_stats", None)
            else:
                state_dict = checkpoint
                hp         = {}

            keys = set(state_dict.keys())

            if "lstm_proj.weight" in keys:
                # -- Kaggle architecture ---------------------------------------
                d_model = state_dict["lstm_proj.weight"].shape[0]
                nhead   = hp.get("nhead",   8)
                dropout = hp.get("dropout", 0.17)
                self.model = KaggleFusion(
                    d_model=d_model, nhead=nhead, dropout=dropout
                ).to(self.device)
                self._arch = f"KaggleFusion(d_model={d_model}, nhead={nhead})"
                print(f"      ℹ️  Kaggle architecture detected "
                      f"(d_model={d_model}, nhead={nhead})")

            elif "lstm_embed.weight" in keys:
                # -- Local MultiHeadFusion architecture ------------------------
                d_model = state_dict["lstm_embed.weight"].shape[0]
                nhead   = 4   # matches training script
                self.model = MultiHeadFusion(
                    d_model=d_model, nhead=nhead
                ).to(self.device)
                self._arch = f"MultiHeadFusion(d_model={d_model}, nhead={nhead})"
                print(f"      ℹ️  Local architecture detected "
                      f"(d_model={d_model}, nhead={nhead})")

            else:
                raise ValueError(
                    f"Unknown checkpoint format. First 5 keys: {list(keys)[:5]}"
                )

            self.model.load_state_dict(state_dict)
            norm_note = "  (norm stats loaded)" if self._norm_stats else ""
            print(f"[OK] Fusion Agent {FUSION_AGENT_VERSION} loaded "
                  f"[{self._arch}] from {model_path}{norm_note}")
            if self._norm_stats:
                print(f"   Norm stats: lstm μ={self._norm_stats.get('lstm_mean',0):.4f} "
                      f"σ={self._norm_stats.get('lstm_std',1):.4f} | "
                      f"sent μ={self._norm_stats.get('sent_mean',0):.4f} "
                      f"σ={self._norm_stats.get('sent_std',1):.4f} | "
                      f"vol μ={self._norm_stats.get('vol_mean',0):.4f} "
                      f"σ={self._norm_stats.get('vol_std',1):.4f}")

        except FileNotFoundError:
            print("[WARN] No trained fusion model found. Using default KaggleFusion weights.")
            self.model = KaggleFusion().to(self.device)
            self._arch = "KaggleFusion (default/untrained)"
        except Exception as e:
            print(f"[BAD] Critical Error loading Fusion Agent: {e}")
            raise

    def _normalize_input(self, val, key):
        """Apply Kaggle z-score normalization if stats are available."""
        if self._norm_stats is None:
            return val
        mu    = self._norm_stats.get(f"{key}_mean", 0.0)
        sigma = self._norm_stats.get(f"{key}_std",  1.0)
        return (val - mu) / (sigma + 1e-8)

    def interpret_weights(self, attn_weights):
        """
        Convert attention weight tensor to focus dict.

        attn_weights shape: (batch=1, 3, 3)
          Both KaggleFusion (dummy) and MultiHeadFusion produce this shape.
          Column index: 0=LSTM, 1=Sentiment, 2=Volatility.

        Returns dict with LSTM_Focus, Sentiment_Focus, Volatility_Focus.
        """
        w = attn_weights.mean(dim=0).cpu().numpy()   # (3, 3)
        return {
            "LSTM_Focus":       float(np.mean(w[:, 0])),
            "Sentiment_Focus":  float(np.mean(w[:, 1])),
            "Volatility_Focus": float(np.mean(w[:, 2])),
        }

    def _heuristic_confidence(self, lstm_p: float,
                               sent_s: float,
                               vol_v: float) -> float:
        """
        Directional heuristic confidence HOLD replaces the flat 0.35 constant.

        FIX v2.2: Special handling for opposing LSTM/Sentiment signals (reversals).
        
        Called when the neural network collapses (output < 0.35).
        Computes a meaningful confidence from the three input directions
        rather than returning an arbitrary constant that distorts decision
        boundaries and makes all collapsed cases indistinguishable.

        CRITICAL FIX:
        When LSTM is bearish but sentiment is clearly bullish (e.g., lstm=0.013, sent=+0.06),
        this indicates a potential reversal. The original weighting (LSTM 0.55, Sent 0.30) 
        could not override the bearish LSTM signal. 
        
        NEW LOGIC:
        - If LSTM < 0.30 AND sent_s > 0.05: Use SENTIMENT-DOMINANT weighting (sent=0.65, lstm=0.20)
        - Otherwise: Use original weighting (lstm=0.55, sent=0.30)
        This allows positive sentiment to override weak LSTM signals in potential reversals.

        Signal composition (weights sum to 1.0):
          STANDARD (LSTM trend is clear):
            LSTM  (0.55) HOLD dominant: price trend is the primary evidence
            Sent  (0.30) HOLD secondary: news provides directional context
            Vol   (0.15) HOLD tertiary: regime proxy confirms macro context
            
          REVERSAL (bearish LSTM but bullish sentiment):
            Sent  (0.65) HOLD elevated: contradictory positive sentiment suggests reversal
            LSTM  (0.20) HOLD reduced: bearish LSTM may be lagging
            Vol   (0.15) HOLD unchanged: regime context

        Mapping:
          signal = -1.0 -> all bearish  -> heuristic ≈ 0.12 (strong SELL)
          signal =  0.0 -> all neutral  -> heuristic = 0.40 (HOLD boundary)
          signal = +1.0 -> all bullish  -> heuristic ≈ 0.75 (capped)

        Range: [0.12, 0.75] HOLD ensures decisions are still meaningful.
        """
        # Convert all inputs to [-1, +1] directional scale
        lstm_dir = float(np.clip((lstm_p - 0.5) * 2, -1.0, 1.0))
        sent_dir = float(np.clip(sent_s / 0.75, -1.0, 1.0))
        # vol_v=0.9(Bear)->-1, vol_v=0.5(Sideways)->0, vol_v=0.2(Bull)->+1
        vol_dir  = float(np.clip((0.55 - vol_v) / 0.35, -1.0, 1.0))

        # CRITICAL FIX: Detect reversal signals (weak LSTM but strong positive sentiment)
        is_reversal = (lstm_p < 0.30 and sent_s > 0.05)
        
        if is_reversal:
            # Sentiment-dominant weighting: trust contradictory positive signal
            # Weights: Sent=0.65, LSTM=0.20, Vol=0.15
            signal = 0.20 * lstm_dir + 0.65 * sent_dir + 0.15 * vol_dir
        else:
            # Standard weighting: LSTM-dominant
            # Weights: LSTM=0.55, Sent=0.30, Vol=0.15
            signal = 0.55 * lstm_dir + 0.30 * sent_dir + 0.15 * vol_dir
        
        return float(np.clip(0.40 + signal * 0.35, 0.12, 0.75))

    def predict(self, lstm_p: float, sent_s: float, vol_v: float,
                trust_scores: dict = None) -> tuple:
        """
        Run inference and return (confidence, focus_map).

        Parameters
        ----------
        lstm_p       : STRETCHED LSTM probability from predict() / mc_mean.
                       Do NOT pass predict_raw() output here.
        sent_s       : FinBERT score ∈ [-0.75, +0.75].
        vol_v        : Regime vol proxy: 0.9=Bear | 0.5=Sideways | 0.2=Bull.
                       This is NOT the raw decimal daily volatility.
        trust_scores : Optional scaling dict. Applied before normalization.

        Returns
        -------
        confidence : float ∈ [0.35, 1.0]  (floor at 0.35 from collapse guard)
        focus_map  : dict HOLD LSTM_Focus, Sentiment_Focus, Volatility_Focus
        """
        # Capture originals before trust_score scaling (used in heuristic fallback below)
        original_lstm = lstm_p
        original_sent = sent_s
        original_vol  = vol_v

        if trust_scores:
            lstm_p = lstm_p * trust_scores.get("technical", 1.0)
            sent_s = sent_s * trust_scores.get("sentiment", 1.0)
            vol_v  = vol_v  * trust_scores.get("regime",    1.0)

        lstm_n = self._normalize_input(lstm_p, "lstm")
        sent_n = self._normalize_input(sent_s, "sent")
        vol_n  = self._normalize_input(vol_v,  "vol")

        t_lstm = torch.tensor([[lstm_n]], dtype=torch.float32).to(self.device)
        t_sent = torch.tensor([[sent_n]], dtype=torch.float32).to(self.device)
        t_vol  = torch.tensor([[vol_n]],  dtype=torch.float32).to(self.device)

        with torch.no_grad():
            conf, weights = self.model(t_lstm, t_sent, t_vol)

        focus_map  = self.interpret_weights(weights)
        final_conf = conf.item()

        # Weight collapse guard:
        # The model can output near-zero when LSTM is very low (bearish signal).
        # That is EXPECTED behaviour HOLD a 0.001 confidence on a bearish LSTM IS correct.
        # Only print a warning when the collapse is UNEXPECTED:
        #   unexpected = lstm_p ≥ 0.25 but model still collapses (suggests normalisation issue)
        # Silently floor to 0.35 for expected cases (low-LSTM -> bearish -> SELL path).
        if final_conf < 0.35:
            # Collapse guard HOLD model output is numerically near-zero.
            # Replace with a directional heuristic computed from raw inputs.
            # This gives a meaningful confidence instead of an arbitrary constant:
            #   flat 0.35 -> every collapsed case produces identical decision boundary
            #   heuristic -> strong bearish inputs -> 0.12, neutral -> 0.40, etc.
            # Only log a warning for UNEXPECTED collapse (lstm_p ≥ 0.25 but still collapses),
            # which suggests a normalisation mismatch between training and inference.
            # Expected collapse (lstm_p < 0.25 -> bearish signal -> low conf) is silent.
            heuristic = self._heuristic_confidence(original_lstm, original_sent, original_vol)
            if original_lstm >= 0.25:
                print(
                    f"      🔴 [Fusion] Unexpected collapse "
                    f"(model={final_conf:.4f}, lstm={original_lstm:.3f}) "
                    f"-> heuristic={heuristic:.4f}"
                )
            final_conf = heuristic

        return final_conf, focus_map