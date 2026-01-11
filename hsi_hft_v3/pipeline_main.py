import sys
import os
import torch
import numpy as np

# Hack to make local imports work if run as script
sys.path.append(os.getcwd())

from hsi_hft_v3.core.config import TARGET_SYMBOL, AUX_SYMBOL, PolicyConfig, BLACKBOX_DIM
from hsi_hft_v3.data.loader import V5DataLoader
from hsi_hft_v3.features.whitebox import WhiteBoxFeatureFactory
from hsi_hft_v3.features.blackbox import DeepFactorMinerV5
from hsi_hft_v3.models.heads import ResidualCombine
from hsi_hft_v3.policy.state_machine import StateMachine
from hsi_hft_v3.backtest.engine import BacktestEngine
from hsi_hft_v3.backtest.metrics import calculate_metrics

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[V3 Production] Starting Pipeline on {DEVICE}...")
    
    # 1. Config & Paths
    DATA_DIR = "./data"
    CHECKPOINT_PATH = "./checkpoints/v3_model_latest.pth"
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Checkpoint not found at {CHECKPOINT_PATH}. Run training first.")
        return

    # 2. Data Loading (Real)
    print(f"[Prod] Loading Data from {DATA_DIR}...")
    loader = V5DataLoader(DATA_DIR)
    # Load all available data or specific range
    data_dict = loader.load_date_range(end_date="2025-12-05")
    
    if not data_dict:
        print("❌ No data found.")
        return
        
    samples = []
    for date, date_samples in data_dict.items():
        samples.extend(date_samples)
    print(f"[Prod] Total Data Samples: {len(samples)}")

    # 3. Model Loading
    print("[Prod] Loading Models...")
    try:
        # Load weights
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        
        # Init Architectures
        # Need dim_raw from data
        # Let's peek at one sample to get dims if not hardcoded?
        # But for Mamba, input_dim is fixed by feature size.
        # Tgt: Mid, Vwap, LogVol (3)
        # Aux: Mid, Vwap, LogVol (3)
        # Masks: 1
        # Hardcoding dims for now based on training logic
        DIM_RAW = 3 
        # White dim? 
        # WhiteBox returns features. We need to know the count.
        # Run one feature calc to check dim
        wb_tmp = WhiteBoxFeatureFactory().compute(samples[0])
        DIM_WHITE = len([k for k in wb_tmp["white_derived"] if "_z_" in k])
        print(f"  -> Features: Raw={DIM_RAW}, White={DIM_WHITE}")
        
        miner = DeepFactorMinerV5(input_dim_raw=DIM_RAW, out_dim=BLACKBOX_DIM).to(DEVICE)
        combine = ResidualCombine(white_dim=DIM_WHITE, black_dim=BLACKBOX_DIM).to(DEVICE)
        
        miner.load_state_dict(checkpoint["miner"])
        combine.load_state_dict(checkpoint["combine"])
        
        miner.eval()
        combine.eval()
        print("✅ Models Loaded.")
        
    except Exception as e:
        print(f"❌ Model Loading Failed: {e}")
        return

    # 4. Inference & Backtest Loop
    print("[Prod] Running Inference & Backtest...")
    
    wb_factory = WhiteBoxFeatureFactory()
    model_outputs = []
    
    # Pre-processing buffers for sequence model (Stateful or Windowed?)
    # Training used sliding window of 64.
    # Inference needs to do the same.
    # We must maintain a buffer of last 64 raw vectors.
    
    LOOKBACK = 64
    raw_tgt_buf = []
    raw_aux_buf = []
    mask_buf = []
    
    # We iterate samples sequentially as in real-time
    with torch.no_grad():
        for i, s in enumerate(samples):
            # A. WhiteBox
            wb_out = wb_factory.compute(s)
            z_feats = [v for k,v in wb_out["white_derived"].items() if "_z_" in k]
            if not z_feats: z_feats = [0.0]*DIM_WHITE
            
            # B. Raw Buffers
            t_vec = [s.target.mid/1000.0, s.target.vwap/1000.0 if s.target.vwap else 0, np.log1p(s.target.volume)]
            a_vec = [s.aux.mid/1000.0 if s.aux else 0, s.aux.vwap/1000.0 if s.aux else 0, np.log1p(s.aux.volume) if s.aux else 0]
            m = 1.0 if s.aux_available else 0.0
            
            raw_tgt_buf.append(t_vec)
            raw_aux_buf.append(a_vec)
            mask_buf.append(m)
            
            # Standardize buffer length
            if len(raw_tgt_buf) > LOOKBACK:
                raw_tgt_buf.pop(0)
                raw_aux_buf.pop(0)
                mask_buf.pop(0)
                
            # C. Inference
            if len(raw_tgt_buf) == LOOKBACK:
                # Prepare Tensor (Batch=1)
                b_t = torch.tensor([raw_tgt_buf], dtype=torch.float32).to(DEVICE)
                b_a = torch.tensor([raw_aux_buf], dtype=torch.float32).to(DEVICE)
                b_m = torch.tensor([mask_buf], dtype=torch.float32).unsqueeze(-1).to(DEVICE) # (1, T, 1)
                b_w = torch.tensor([z_feats], dtype=torch.float32).to(DEVICE)
                
                deep_factors = miner(b_t, b_a, b_m)
                out = combine(b_w, deep_factors)
                
                # Sigmoid for binary probabilities
                p_hit = torch.sigmoid(out["logit_hit"]).item()
                risk = torch.sigmoid(out["logit_risk"]).item()
                # Hazard: P(T=k | T>=k) = sigmoid(logit)
                h_k = torch.sigmoid(out["logit_hazard"]).squeeze(0).cpu().numpy() # (K,)
                
                # S(k) = Prod(1 - h_i)
                # P(T <= k) = 1 - S(k)
                # 60s = 20 bars
                k_60 = 20
                if len(h_k) >= k_60:
                    surv_60 = np.prod(1 - h_k[:k_60])
                    p_tau_60 = 1.0 - surv_60
                else:
                    p_tau_60 = 0.0
                
                # Depth (L1)
                depth_q = 0
                if s.target.bids and s.target.asks:
                     depth_q = s.target.bids[0][1] + s.target.asks[0][1]
                
                white_risk_dict = {
                    "spread_bps": wd.get("spread_bps", 0.0),
                    "vpin_z": wd.get("VPIN_20_z_100", 0.0),
                    "kyle_z": wd.get("KyleLambda_20_z_100", 0.0), # Added Kyle
                    "depth_qty": depth_q
                }
                
                model_outputs.append({
                    "p_hit": p_hit,
                    "risk": risk,
                    "P_tau_le_60s": float(p_tau_60),
                    "white_risk": white_risk_dict # Pass this alongside
                })
            else:
                # Warming up
                model_outputs.append({"p_hit": 0.0, "risk": 0.0, "P_tau_le_60s": 0.0})

    # 5. Policy Execution
    print(f"[Prod] Generated {len(model_outputs)} signals. Executing Strategy...")
    policy = StateMachine(PolicyConfig())
    engine = BacktestEngine(policy)
    
    # Run loop needs modification to unpack white_risk from model_outputs
    # Engine.run expects list of dicts. 
    # But Engine.run logic separates model_out and white_risk?
    # Engine.run() implementation:
    #   model_out = model_outputs[i]
    #   white_risk = {"spread_bps": 5.0} ... HARDCODED there.
    # We must fix Engine loop too?
    # Actually Engine is external. We should modify Engine to accept white_risk in input or 
    # extract it from model_outputs if we bundled it there.
    # Let's modify BacktestEngine.run to look for 'white_risk' in model_outputs or 
    # pass a separate list.
    # Easier: Engine.run takes models_outputs. We bundled 'white_risk' into it above.
    # So we just update Engine.run to read it.
    
    engine.run(samples, model_outputs)
    
    # 6. Metrics
    metrics = calculate_metrics(engine.trades, engine.equity_curve)
    print("\n=== V3 System Report ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
        
    # Debug: Model Distribution
    df_out = pd.DataFrame(model_outputs)
    if not df_out.empty:
        print("\n=== Model Prediction Stats ===")
        print(df_out.describe(percentiles=[0.5, 0.9, 0.95, 0.99]))
        
    print("[Prod] Finished.")

if __name__ == "__main__":
    main()
