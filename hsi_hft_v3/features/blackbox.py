import torch
import torch.nn as nn
import torch.nn.functional as F
from hsi_hft_v3.core.config import BLACKBOX_DIM, LOOKBACK_BARS

# Pseudo-Mamba Block (Simplified for prototype, ideally import real mamba_ssm)
# Selective Scan (Pure PyTorch Implementation for Windows/CPU compatibility)
# Strictly follows Mamba: h_t = A_bar * h_{t-1} + B_bar * x_t
class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state
        
        # In-projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2) 

        # x_proj takes x to (dt, B, C)
        self.dt_rank = d_model // 16 if d_model // 16 > 0 else 1
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2) 
        
        # Parameters
        # A: (d_inner, d_state)
        self.A_log = nn.Parameter(torch.log(torch.randn(self.d_inner, d_state).abs()))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # dt proj
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.act = nn.SiLU()

    def forward(self, u):
        # u: (B, T, D)
        B, T, D = u.shape
        
        # 1. Expand
        u_inner = self.in_proj(u)
        x, z = u_inner.chunk(2, dim=-1) # (B, T, d_inner)
        
        # 2. Conv1d (Short local conv) - standard Mamba block has this before SSM
        # For strict compliance, we should use a conv here or assume pre-conv.
        # User spec mentions "MambaEncoder" -> "SimpleSSM". 
        # In Mamba paper: Block = Conv -> SSM -> Gated MLP.
        # We will assume 'x' input here needs the SSM part.
        
        # 3. Dynamic Projections (Selection)
        # Scan Input x: (B, T, d_inner)
        # Params: (B, T, dt_rank + 2*d_state)
        delta_bc = self.x_proj(x)
        
        # Split
        delta_raw, B_ssm, C_ssm = torch.split(delta_bc, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        # Softplus delta
        delta = F.softplus(self.dt_proj(delta_raw)) # (B, T, d_inner)
        
        # 4. Discretization
        # A_bar = exp(delta * A)
        # B_bar = (delta * A)^-1 (exp(delta * A) - I) * delta * B  ~ approx delta * B
        
        A = -torch.exp(self.A_log) # A must be negative
        
        # Sequential Scan (Slow but Correct Pure Torch)
        # h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
        # y_t = C_t * h_t
        
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        y_list = []
        
        for t in range(T):
            # Slice time t
            dt_t = delta[:, t, :] # (B, d_inner)
            dA = torch.exp(torch.einsum('bd,dn->bdn', dt_t, A)) # (B, d_inner, d_state)
            dB = torch.einsum('bd,bn->bdn', dt_t, B_ssm[:, t, :]) # (B, d_inner, d_state)
            
            xt = x[:, t, :] # (B, d_inner)
            
            # Recurrence
            # h: (B, d_inner, d_state)
            # x_t * dB -> (B, d_inner, 1) * (B, d_inner, d_state) ??
            # Wait, standard SISO: x_t is scalar per channel. 
            # In Mamba D_inner channels are independent.
            # So x_t (B, d_inner) broadcasts to (B, d_inner, 1)
            
            h = h * dA + xt.unsqueeze(-1) * dB
            
            # Output y_t = h_t * C_t
            # C_ssm: (B, d_state) -> Shared across channels? 
            # In Mamba, C is (B, T, d_state). 
            # Projection: y = sum(h * C, dim=-1) -> (B, d_inner)
            
            Ct = C_ssm[:, t, :] # (B, d_state)
            yt = torch.einsum('bdn,bn->bd', h, Ct)
            y_list.append(yt)
            
        y = torch.stack(y_list, dim=1) # (B, T, d_inner)
        
        # 5. Residual + Gate
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        y = y * self.act(z)
        
        return self.out_proj(y)

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
        self.ssm_tgt = SelectiveSSM(d_model)
        self.ssm_aux = SelectiveSSM(d_model)
        
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
