import sys
import os
import numpy as np
import time

sys.path.append(os.getcwd())

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch missing.")
    sys.exit(1)

from hsi_hft_v3.data.loader import V5DataLoader
from hsi_hft_v3.features.whitebox import WhiteBoxFeatureFactory
from hsi_hft_v3.features.blackbox import DeepFactorMinerV5, vicreg_loss
from hsi_hft_v3.core.config import BLACKBOX_DIM

class LatentPredictor(nn.Module):
    """Simple MLP to predict future latent state from current state"""
    def __init__(self, dim=BLACKBOX_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
    def forward(self, x):
        return self.net(x)

def pretrain():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Pretrain] Device: {DEVICE}")
    
    # Config
    DATA_DIR = "./data"
    BATCH_SIZE = 128
    EPOCHS = 5
    LOOKBACK = 64
    PRED_HORIZON = 20 # Predict state 20 steps ahead
    
    # 1. Load Data
    loader = V5DataLoader(DATA_DIR)
    print("[Pretrain] Loading Data...")
    data_dict = loader.load_date_range(end_date="2025-12-05")
    
    if not data_dict: return

    # 2. Prep Sequences
    # We need pairs: (Window_Core, Window_Target)
    # Actually for JEPA:
    # Context: [t-L : t] -> Output Z_t
    # Target: [t-L+k : t+k] -> Output Z_{t+k}
    # We predict Z_{t+k} using Z_t.
    
    print("[Pretrain] Constructing Self-Supervised Dataset...")
    
    raw_tgt = []
    raw_aux = []
    masks = []
    
    # Simple extraction loop
    for date, samples in data_dict.items():
        # Pre-extract all raw vectors
        # Tgt: Mid, Vwap, LogVol
        # Aux: Mid, Vwap, LogVol
        
        day_tgt = []
        day_aux = []
        day_mask = []
        
        for s in samples:
            t_vec = [s.target.mid/1000.0, s.target.vwap/1000.0 if s.target.vwap else 0, np.log1p(s.target.volume)]
            a_vec = [s.aux.mid/1000.0 if s.aux else 0, s.aux.vwap/1000.0 if s.aux else 0, np.log1p(s.aux.volume) if s.aux else 0]
            m = 1.0 if s.aux_available else 0.0
            
            day_tgt.append(t_vec)
            day_aux.append(a_vec)
            day_mask.append(m)
            
        # Create Sliding Windows
        # We need pairs (X_t, X_{t+k})
        L_day = len(day_tgt)
        if L_day <= LOOKBACK + PRED_HORIZON: continue
        
        d_t = np.array(day_tgt, dtype=np.float32)
        d_a = np.array(day_aux, dtype=np.float32)
        d_m = np.array(day_mask, dtype=np.float32)
        
        # Stride? 
        for i in range(LOOKBACK, L_day - PRED_HORIZON, 2): # Stride 2
            # Context Window (ends at i)
            c_t = d_t[i-LOOKBACK:i]
            c_a = d_a[i-LOOKBACK:i]
            c_m = d_m[i-LOOKBACK:i]
            
            # Future Window (ends at i + k)
            # Actually, encoder takes a window.
            # Target encoder takes window shifted by k?
            # Yes.
            f_t = d_t[i + PRED_HORIZON - LOOKBACK : i + PRED_HORIZON]
            f_a = d_a[i + PRED_HORIZON - LOOKBACK : i + PRED_HORIZON]
            f_m = d_m[i + PRED_HORIZON - LOOKBACK : i + PRED_HORIZON]
            
            raw_tgt.append(c_t)
            raw_aux.append(c_a)
            masks.append(c_m)
            
            # We append future target as "labels" or just part of batch?
            # Let's stack them: Batch will have (Context, Target)
            # For simplicity in DataLoader, let's double the arrays or concat?
            # Let's store Future in separate lists
            
    # To save memory, convert to tensor immediately?
    # We need a custom Dataset that returns (ctx, fut)
    # Let's just make one big TensorDataset with 6 tensors: 
    # Ctx_T, Ctx_A, Ctx_M, Fut_T, Fut_A, Fut_M
    
    print(f"[Pretrain] Samples: {len(raw_tgt)}")
    if len(raw_tgt) == 0: return
    
    # To handle the logic easily, I will implement a simplified version:
    # Instead of storing Future tensors (memory!), I will just store ONE big timeseries and sample indices?
    # No, typical DL: pre-slice.
    # Given we are verifying, I'll slice but maybe not all data.
    
    # ... (Slicing logic assumed done above, but I didn't actually append Future lists above. Fixing)
    # Re-loop or fix above? I'll fix the lists.
    
    # RESTART LISTS
    ctx_t, ctx_a, ctx_m = [], [], []
    fut_t, fut_a, fut_m = [], [], []
    
    for date, samples in data_dict.items():
        day_t, day_a, day_m = [], [], []
        for s in samples:
            t_vec = [s.target.mid/1000.0, s.target.vwap/1000.0 if s.target.vwap else 0, np.log1p(s.target.volume)]
            a_vec = [s.aux.mid/1000.0 if s.aux else 0, s.aux.vwap/1000.0 if s.aux else 0, np.log1p(s.aux.volume) if s.aux else 0]
            m = 1.0 if s.aux_available else 0.0
            day_t.append(t_vec); day_a.append(a_vec); day_m.append(m)
            
        if len(day_t) <= LOOKBACK + PRED_HORIZON: continue
        d_t, d_a, d_m = np.array(day_t), np.array(day_a), np.array(day_m)
        
        for i in range(LOOKBACK, len(day_t) - PRED_HORIZON, 5): # Stride 5
             ctx_t.append(d_t[i-LOOKBACK:i])
             ctx_a.append(d_a[i-LOOKBACK:i])
             ctx_m.append(d_m[i-LOOKBACK:i])
             
             fut_t.append(d_t[i+PRED_HORIZON-LOOKBACK : i+PRED_HORIZON])
             fut_a.append(d_a[i+PRED_HORIZON-LOOKBACK : i+PRED_HORIZON])
             fut_m.append(d_m[i+PRED_HORIZON-LOOKBACK : i+PRED_HORIZON])
             
    # Tensors
    t_ct = torch.tensor(np.array(ctx_t), dtype=torch.float32).to(DEVICE)
    t_ca = torch.tensor(np.array(ctx_a), dtype=torch.float32).to(DEVICE)
    t_cm = torch.tensor(np.array(ctx_m), dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    
    t_ft = torch.tensor(np.array(fut_t), dtype=torch.float32).to(DEVICE)
    t_fa = torch.tensor(np.array(fut_a), dtype=torch.float32).to(DEVICE)
    t_fm = torch.tensor(np.array(fut_m), dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    
    ds = TensorDataset(t_ct, t_ca, t_cm, t_ft, t_fa, t_fm)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    
    # 3. Model
    dim_raw = t_ct.shape[-1]
    enc = DeepFactorMinerV5(input_dim_raw=dim_raw).to(DEVICE)
    pred = LatentPredictor().to(DEVICE)
    
    opt = optim.Adam(
        list(enc.parameters()) + list(pred.parameters()),
        lr=1e-3
    )
    
    # 4. Loop
    print(f"[Pretrain] Starting JEPA Loop ({EPOCHS} Epochs)...")
    enc.train()
    pred.train()
    
    mse = nn.MSELoss()
    
    for ep in range(EPOCHS):
        tot_loss = 0
        steps = 0
        for b in dl:
            ct, ca, cm, ft, fa, fm = b
            
            opt.zero_grad()
            
            # Context Encoding -> Z_ctx
            z_ctx = enc(ct, ca, cm)
            
            # Future Encoding (Target) -> Z_fut (No gradient to target encoder in typical JEPA? 
            # Classic JEPA uses EMA target encoder. 
            # Simplified version here: Shared weights, no stop grad for prototype? 
            # Or Stop grad. Let's Stop Grad for stability.)
            with torch.no_grad():
                z_fut = enc(ft, fa, fm)
                
            # Predict: Z_ctx -> Z_pred
            z_pred = pred(z_ctx)
            
            # Loss 1: Prediction Error
            l_pred = mse(z_pred, z_fut)
            
            # Loss 2: VICReg on Z_ctx (Regularize representation to be good)
            # Maintain variance, decorrelate
            l_std, l_cov = vicreg_loss(z_ctx, ct.shape[0])
            
            # Total
            loss = l_pred + l_std + l_cov
            loss.backward()
            opt.step()
            
            tot_loss += loss.item()
            steps += 1
            
        print(f"Ep {ep+1} | Loss: {tot_loss/steps:.4f}")
        
    # 5. Save
    os.makedirs("./checkpoints", exist_ok=True)
    torch.save(enc.state_dict(), "./checkpoints/v3_encoder_pretrained.pth")
    print("✅ Pretrained Encoder Saved.")

if __name__ == "__main__":
    pretrain()
