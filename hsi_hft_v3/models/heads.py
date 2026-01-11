import torch
import torch.nn as nn
from hsi_hft_v3.core.config import K_BARS, BLACKBOX_DIM

class HitHead(nn.Module):
    """Binary Classification: P(Hit)"""
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.fc(x) # Logits

class HazardHead(nn.Module):
    """Discrete Time Survival: P(T=k | T>=k)"""
    def __init__(self, input_dim, k_bins=K_BARS):
        super().__init__()
        self.fc = nn.Linear(input_dim, k_bins)
        
    def forward(self, x):
        return self.fc(x) # Logits for each k

class RiskHead(nn.Module):
    """Competing Risk: P(Adverse | Adverse or Hit)"""
    def __init__(self, input_dim):
        super().__init__()
        # Simplified Risk Head: P(Adverse Before Hit)
        self.fc = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.fc(x) # Logits

class ResidualCombine(nn.Module):
    """
    Final Output = WhiteBox Logic + BlackBox Correction
    """
    def __init__(self, white_dim, black_dim=BLACKBOX_DIM):
        super().__init__()
        
        # WhiteBox Proxy: Simple Linear Map from White Factors to Logits
        # This represents the "Explainable Baseline"
        self.white_hit = nn.Linear(white_dim, 1)
        self.white_hazard = nn.Linear(white_dim, K_BARS)
        self.white_risk = nn.Linear(white_dim, 1)
        
        # BlackBox Correction: Map Latent to Delta Logits
        self.delta_hit = nn.Linear(black_dim, 1)
        self.delta_hazard = nn.Linear(black_dim, K_BARS)
        self.delta_risk = nn.Linear(black_dim, 1)
        
        # Initialize White proxies (e.g., identity-like or small random)
        # Initialize Deltas to zero to start from baseline
        nn.init.zeros_(self.delta_hit.weight)
        nn.init.zeros_(self.delta_hazard.weight)
        nn.init.zeros_(self.delta_risk.weight)

    def forward(self, white_feats, deep_factors):
        # Base Logits
        base_hit = self.white_hit(white_feats)
        base_hazard = self.white_hazard(white_feats)
        base_risk = self.white_risk(white_feats)
        
        # Delta Logits
        d_hit = self.delta_hit(deep_factors)
        d_hazard = self.delta_hazard(deep_factors)
        d_risk = self.delta_risk(deep_factors)
        
        # Combine
        return {
            "logit_hit": base_hit + d_hit,
            "logit_hazard": base_hazard + d_hazard,
            "logit_risk": base_risk + d_risk,
            # For audit/debug
            "base_hit": base_hit,
            "delta_hit": d_hit
        }
