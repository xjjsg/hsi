import torch
import torch.nn as nn
import torch.nn.functional as F
from hsi_hft_v3.core.config import BLACKBOX_DIM, LOOKBACK_BARS

# Pseudo-Mamba Block (Simplified for prototype, ideally import real mamba_ssm)
class SimpleSSM(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.act = nn.SiLU()
    
    def forward(self, x):
        # x: (B, T, D)
        x = x.transpose(1, 2) # (B, D, T)
        x = self.act(self.conv(x))
        x = x.transpose(1, 2)
        return self.proj(x)

class LocalLOBEncoder(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(d_model)
    
    def forward(self, x):
        # x: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x.transpose(1, 2) # (B, T, D)

class FiLMFusion(nn.Module):
    """Feature-wise Linear Modulation: Target conditioned on Aux"""
    def __init__(self, d_model):
        super().__init__()
        self.scale = nn.Linear(d_model, d_model)
        self.shift = nn.Linear(d_model, d_model)
        
    def forward(self, tgt, aux, aux_mask):
        # aux: (B, T, D), aux_mask: (B, T, 1)
        # Apply mask to aux
        aux_masked = aux * aux_mask
        gamma = self.scale(aux_masked)
        beta = self.shift(aux_masked)
        return tgt * (1 + gamma) + beta

class DeepFactorMinerV5(nn.Module):
    """
    Dual-Stream Mamba + FiLM + VICReg Projector
    Output: 32-dim Latent Factors
    """
    def __init__(self, input_dim_raw, d_model=64, out_dim=BLACKBOX_DIM):
        super().__init__()
        self.d_model = d_model
        
        # Encoders
        self.encoder_tgt = LocalLOBEncoder(input_dim_raw, d_model)
        self.encoder_aux = LocalLOBEncoder(input_dim_raw, d_model)
        
        # SSM Backbone (Mamba-like)
        self.ssm_tgt = SimpleSSM(d_model)
        self.ssm_aux = SimpleSSM(d_model)
        
        # Fusion
        self.film = FiLMFusion(d_model)
        
        # Mixing
        self.mixer = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        
        # Projector (Information Bottleneck)
        self.projector = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, out_dim) # 32-dim
        )
        
        # Output Heads (Delta Logits)
        # These will be initialized in the main model (Heads.py), 
        # Miner only outputs latent factors.

    def forward(self, x_tgt, x_aux, aux_mask, has_fut_mask=None):
        """
        x_tgt: (B, T, F)
        x_aux: (B, T, F)
        aux_mask: (B, T, 1)
        """
        # 1. Local Encode
        e_tgt = self.encoder_tgt(x_tgt)
        e_aux = self.encoder_aux(x_aux)
        
        # 2. SSM Scan (Sequence Modeling)
        h_tgt = self.ssm_tgt(e_tgt)
        h_aux = self.ssm_aux(e_aux)
        
        # 3. FiLM Fusion
        # Modulate Target state with Aux state
        h_fused = self.film(h_tgt, h_aux, aux_mask)
        
        # 4. Mixing
        h_final = self.mixer(h_fused)
        
        # 5. Pooling (Last state)
        # (B, T, D) -> (B, D) taking the last time step
        z_pool = h_final[:, -1, :]
        
        # 6. Projection
        deep_factors = self.projector(z_pool) # (B, 32)
        
        return deep_factors

# VICReg Loss Implementation (Reused Concept)
def vicreg_loss(z, batch_size):
    # Variance
    std_z = torch.sqrt(z.var(dim=0) + 0.0001)
    std_loss = torch.mean(torch.relu(1 - std_z))
    
    # Covariance
    z = z - z.mean(dim=0)
    cov_z = (z.T @ z) / (batch_size - 1)
    # Off-diagonal
    off_diag = cov_z.flatten()[:-1].view(31, 33)[:, 1:].flatten()
    cov_loss = off_diag.pow(2).sum() / 32
    
    return std_loss, cov_loss
