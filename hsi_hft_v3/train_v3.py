import sys
import os
import time
import numpy as np
import pandas as pd
import pickle

# Hack to make local imports work
sys.path.append(os.getcwd())

# 导入整合后的模块
from hsi_hft_v3.data_layer import V5DataLoader
from hsi_hft_v3.trading_layer import (
    BLACKBOX_DIM,
    K_BARS,
    COST_RATE,
    LATENCY_BARS,
    LabelConfig,
)
from hsi_hft_v3.model_layer import (
    DeepFactorMinerV5,
    HitHead,
    HazardHead,
    RiskHead,
    ResidualCombine,
)
from hsi_hft_v3.features.whitebox import WhiteBoxFeatureFactory

# NEW: 导入IRM损失和可微分特征
from hsi_hft_v3.irm_loss import IRMLoss, EnvironmentSplitter
from hsi_hft_v3.risk_monitor import compute_baseline_stats

# PyTorch Imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch missing, but script requires it for actual training.")
    sys.exit(1)

# ==========================================
# 配置开关（CRITICAL: 现在默认全部启用）
# ==========================================
USE_IRM = True  # 启用IRM损失
USE_LEARNABLE_FEATURES = False  # 可微分特征（暂时False，因为需要修改WhiteBoxFactory）
IRM_PENALTY_WEIGHT = 1.0
IRM_ANNEAL_EPOCHS = 5
COMPUTE_BASELINE_STATS = True  # 训练后计算baseline_stats


# ==========================================
# Label Generation
# ==========================================
def generate_labels(samples, lookahead=60):
    """
    Generate targets for each sample i:
    - y_hit: 1 if within lookahead, price moves favorably > cost
    - y_hazard: k (time index) of first hit
    - y_risk: 1 if adverse event happens before hit
    """
    n = len(samples)
    y_hit = np.zeros(n, dtype=np.float32)
    y_hazard = np.zeros(n, dtype=np.int64)  # Bin index 0..K
    y_risk = np.zeros(n, dtype=np.float32)

    # Pre-extract prices for speed
    asks = np.array([s.target.asks[0][0] if s.target.asks else np.nan for s in samples])
    bids = np.array([s.target.bids[0][0] if s.target.bids else np.nan for s in samples])

    # We need Ask_e = Ask[t + latency]
    latency = LATENCY_BARS  # fixed 1 bar for now

    # Cost threshold
    # Profit = (Bid_future - Ask_entry) / Ask_entry - 2*Cost

    for i in range(n - latency):
        # Entry requires existing Ask at t + latency
        idx_entry = i + latency
        if idx_entry >= n:
            break

        entry_price = asks[idx_entry]  # Ask @ t+latency
        if np.isnan(entry_price):
            continue

        target_price = entry_price * (
            1.0 + 2 * COST_RATE + 0.0002
        )  # Min profit 2bps net
        stop_price = entry_price * (
            1.0 - LabelConfig.adverse_bps / 10000.0
        )  # Stop loss from Config

        found_hit = False
        hit_k = K_BARS - 1
        found_risk = False  # Fix: Initialize found_risk

        # Look forward from t_e
        # Path: t_e + 1 ... t_e + lookahead
        for k in range(1, min(lookahead, n - idx_entry)):
            curr_bid = bids[idx_entry + k]  # Bid @ future
            if np.isnan(curr_bid):
                continue

            # Check Hit
            if curr_bid > target_price:
                y_hit[i] = 1.0
                hit_k = k
                found_hit = True
                break

            # Check Risk (Stop Loss)
            if curr_bid < stop_price:
                y_risk[i] = 1.0
                hit_k = k
                found_risk = True
                break

        y_hazard[i] = min(hit_k, K_BARS - 1)

    return y_hit, y_hazard, y_risk


# ==========================================
# Main Training Logic
# ==========================================
def train():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using Device: {DEVICE}")

    # 1. Config
    DATA_DIR = "./data"
    BATCH_SIZE = 128
    EPOCHS = 10
    LOOKBACK = 64  # Sequence length for Mamba

    # 2. Load Data
    loader = V5DataLoader(DATA_DIR)
    # Load ALL available data (Dynamic Splitting)
    print("[Train] Loading ALL available data...")
    data_dict = loader.load_date_range()

    if not data_dict:
        print("❌ No data loaded.")
        return

    # 3. Dynamic Data Splitting (80/10/10)
    all_dates = sorted(list(data_dict.keys()))
    n_days = len(all_dates)
    if n_days < 3:
        print("❌ Not enough days for splitting.")
        return

    n_train = int(0.8 * n_days)
    n_val = int(0.1 * n_days)
    # n_test = remainder

    train_dates = all_dates[:n_train]
    val_dates = all_dates[n_train : n_train + n_val]
    test_dates = all_dates[n_train + n_val :]

    print(
        f"[Split] Train: {len(train_dates)} days ({train_dates[0]} ~ {train_dates[-1]})"
    )
    print(f"[Split] Val:   {len(val_dates)} days ({val_dates[0]} ~ {val_dates[-1]})")
    if test_dates:
        print(
            f"[Split] Test:  {len(test_dates)} days ({test_dates[0]} ~ {test_dates[-1]})"
        )
    else:
        print(f"[Split] Test:  0 days (insufficient data)")

    # Helper to construct tensors from date list
    def build_tensors(dates_list, data_map):
        b_white, b_tgt, b_aux, b_mask = [], [], [], []
        b_y_hit, b_y_haz, b_y_risk = [], [], []

        count = 0
        wb_factory_local = WhiteBoxFeatureFactory()

        for date in dates_list:
            if date not in data_map:
                continue
            samples = data_map[date]

            # --- Per Date Processing (Same as before) ---
            feats_white = []
            raw_tgt = []
            raw_aux = []
            masks = []

            for s in samples:
                wb_out = wb_factory_local.compute(s)
                sorted_keys = wb_factory_local.get_derived_keys()
                z_feats = []
                for k in sorted_keys:
                    z_feats.append(wb_out["white_derived"].get(k, 0.0))
                if not z_feats:
                    z_feats = [0.0] * 10
                feats_white.append(z_feats)

                t_vec = [
                    s.target.mid / 1000.0,
                    s.target.vwap / 1000.0 if s.target.vwap else 0,
                    np.log1p(s.target.volume),
                ]
                a_vec = [
                    s.aux.mid / 1000.0 if s.aux else 0,
                    s.aux.vwap / 1000.0 if s.aux else 0,
                    np.log1p(s.aux.volume) if s.aux else 0,
                ]

                raw_tgt.append(t_vec)
                raw_aux.append(a_vec)
                masks.append(1.0 if s.aux_available else 0.0)

            # Labels
            y_hit, y_haz, y_risk = generate_labels(samples, lookahead=K_BARS)

            # Sliding Window
            L = len(samples)
            if L <= LOOKBACK:
                continue

            np_white = np.array(feats_white, dtype=np.float32)
            np_tgt = np.array(raw_tgt, dtype=np.float32)
            np_aux = np.array(raw_aux, dtype=np.float32)
            np_mask = np.array(masks, dtype=np.float32)

            for i in range(LOOKBACK, L):
                b_tgt.append(np_tgt[i - LOOKBACK : i])
                b_aux.append(np_aux[i - LOOKBACK : i])
                b_mask.append(np_mask[i - LOOKBACK : i])
                b_white.append(np_white[i - 1])
                b_y_hit.append(y_hit[i - 1])
                b_y_haz.append(y_haz[i - 1])
                b_y_risk.append(y_risk[i - 1])
                count += 1

        if count == 0:
            return None

        return (
            torch.tensor(np.array(b_white), dtype=torch.float32).to(DEVICE),
            torch.tensor(np.array(b_tgt), dtype=torch.float32).to(DEVICE),
            torch.tensor(np.array(b_aux), dtype=torch.float32).to(DEVICE),
            torch.tensor(np.array(b_mask), dtype=torch.float32)
            .unsqueeze(-1)
            .to(DEVICE),
            torch.tensor(np.array(b_y_hit), dtype=torch.float32)
            .unsqueeze(-1)
            .to(DEVICE),
            torch.tensor(np.array(b_y_haz), dtype=torch.long).to(DEVICE),
            torch.tensor(np.array(b_y_risk), dtype=torch.float32)
            .unsqueeze(-1)
            .to(DEVICE),
        )

    # 4. Construct DataLoaders
    print("[Train] Building TRAIN tensors...")
    train_tensors = build_tensors(train_dates, data_dict)
    if not train_tensors:
        print("❌ Train set empty.")
        return
    train_ds = TensorDataset(*train_tensors)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    print("[Train] Building VAL tensors...")
    val_tensors = build_tensors(val_dates, data_dict)
    val_loader = None
    if val_tensors:
        val_ds = TensorDataset(*val_tensors)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        print(f"  -> Val Samples: {len(val_ds)}")
    else:
        print("⚠️ Val set empty, skipping validation.")

    # 5. Model Setup
    dim_raw = train_tensors[1].shape[-1]
    dim_white = train_tensors[0].shape[-1]

    miner = DeepFactorMinerV5(input_dim_raw=dim_raw, out_dim=BLACKBOX_DIM).to(DEVICE)

    # Load Pretrained Weights if available
    PRETRAIN_PATH = "./checkpoints/v3_encoder_pretrained.pth"
    if os.path.exists(PRETRAIN_PATH):
        print(f"[Train] Loading Pretrained Encoder from {PRETRAIN_PATH}...")
        # Miner state dict might have 'projector' which pretrain also has?
        # Pretrain script saves 'enc.state_dict()'.
        # Keys should match exactly if architecture is identical.
        try:
            state = torch.load(PRETRAIN_PATH, map_location=DEVICE)
            miner.load_state_dict(
                state, strict=False
            )  # standard strict=False for safety if heads differ
            print("✅ Pretrained weights loaded.")
        except Exception as e:
            print(f"⚠️ Failed to load pretrained weights: {e}")
    else:
        print("[Train] No pretrained weights found. Starting from scratch.")

    combine = ResidualCombine(white_dim=dim_white, black_dim=BLACKBOX_DIM).to(DEVICE)

    optimizer = optim.Adam(
        list(miner.parameters()) + list(combine.parameters()), lr=1e-3
    )

    crit_hit = nn.BCEWithLogitsLoss()
    crit_haz = nn.BCEWithLogitsLoss(reduction="none")  # Changed to BCE with masking
    crit_risk = nn.BCEWithLogitsLoss()

    # 6. Loop
    print(f"[Train] Starting Training for {EPOCHS} Epochs...")
    miner.train()
    combine.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        steps = 0

        for batch in train_loader:
            b_w, b_t, b_a, b_m, y_h, y_z, y_r = batch

            optimizer.zero_grad()

            # Forward
            deep_factors = miner(b_t, b_a, b_m)
            out = combine(b_w, deep_factors)

            # 1. Hit Loss (Binary)
            loss_hit = crit_hit(out["logit_hit"], y_h)

            # 2. Hazard Loss (Discrete Time: h_k = P(T=k|T>=k))
            # Target: 0 for k < T, 1 for k == T
            # Mask: 1 for k <= T, 0 for k > T

            # Construct Dense Targets from Indices y_z
            B, K = out["logit_hazard"].shape

            # y_z is index of event (Hit or Risk or End).
            # If hit_k = K-1 (no event), then all 0s?
            # In generate_labels, hit_k is min(lookahead, ...).
            # If no hit, it's K_BARS-1.

            haz_targets = torch.zeros_like(out["logit_hazard"])
            haz_mask = torch.zeros_like(out["logit_hazard"])

            for i in range(B):
                idx = int(y_z[i].item())
                # Active range: 0 to idx (inclusive)
                haz_mask[i, : idx + 1] = 1.0
                # Target: 0 everywhere except at idx (if it's a real hit?)
                # Wait, if y_hit[i]==1, then idx is the HIT time. h_idx = 1.
                # If y_hit[i]==0 and y_risk[i]==0, then idx is censorship time (K-1). h_idx = 0.
                if y_h[i].item() > 0.5:  # It's a Hit
                    haz_targets[i, idx] = 1.0
                elif y_r[i].item() > 0.5:  # It's a Risk event
                    # Competing risk? Usually treated as censoring for Hit Hazard?
                    # Spec: "Hazard of Hit". So Risk event censors Hit.
                    # So at risk event k, we know we DIDN'T hit.
                    # So target is 0, mask includes k.
                    pass
                else:
                    # End of window, no hit. Censored.
                    # Target 0, mask includes k.
                    pass

            loss_haz_elem = crit_haz(out["logit_hazard"], haz_targets)
            loss_haz = (loss_haz_elem * haz_mask).sum() / (haz_mask.sum() + 1e-6)

            # 3. Risk Loss
            loss_risk = crit_risk(out["logit_risk"], y_r)

            # 4. Consistency Loss (L_cons)
            # p_hit_implied = 1 - S(K) = 1 - Prod(1 - h_k)
            # Approximate via Softmax cumulative sum or just Softmax sum?
            # If CrossEntropy used, probabilities are Softmax(logits).
            # P(hit within K) = Sum(p_k for k in 0..K-1) (if bin K is 'no-hit')
            # Assuming last bin is "no-hit" or censoring?
            # In generate_labels, hit_k = K_BARS-1 if no hit.
            # So bins are 0..K-1. Let's say K-1 is "no hit within window" or just last step?
            # Actually generate_labels sets hit_k = k (1..K).
            # If we use Softmax over K bins, Sum(P_k) = 1.
            # We want p_hit_model ~ Sum(P_k for k indicating hit).
            # But y_hit is separately supervised.

            p_hit_model = torch.sigmoid(out["logit_hit"])
            # For consistency: P_hit_implied = 1 - S(K)
            # S(t) = Prod(1 - h_k)
            h_probs = torch.sigmoid(out["logit_hazard"])
            # We want P(T <= K) = 1 - Prod_{k=0}^{K-1} (1 - h_k)
            # Log survival
            log_S = torch.sum(torch.log(1 - h_probs + 1e-9), dim=1)
            p_hit_implied = 1.0 - torch.exp(log_S).unsqueeze(-1)

            loss_cons = F.mse_loss(p_hit_model, p_hit_implied)

            loss = loss_hit + 0.5 * loss_haz + 0.5 * loss_risk + 0.1 * loss_cons

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / steps
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_loss:.4f}")

        # Validation Step
        if val_loader:
            miner.eval()
            combine.eval()
            val_loss_sum = 0
            val_steps = 0
            with torch.no_grad():
                for batch in val_loader:
                    b_w, b_t, b_a, b_m, y_h, y_z, y_r = batch
                    deep_factors = miner(b_t, b_a, b_m)
                    out = combine(b_w, deep_factors)

                    l_hit = crit_hit(out["logit_hit"], y_h)
                    l_risk = crit_risk(out["logit_risk"], y_r)

                    # Hazard Logic (copy of train)
                    B, K = out["logit_hazard"].shape
                    haz_targets = torch.zeros_like(out["logit_hazard"])
                    haz_mask = torch.zeros_like(out["logit_hazard"])
                    for i in range(B):
                        idx = int(y_z[i].item())
                        haz_mask[i, : idx + 1] = 1.0
                        if y_h[i].item() > 0.5:
                            haz_targets[i, idx] = 1.0

                    l_haz_elem = crit_haz(out["logit_hazard"], haz_targets)
                    l_haz = (l_haz_elem * haz_mask).sum() / (haz_mask.sum() + 1e-6)

                    val_loss_sum += (l_hit + 0.5 * l_haz + 0.5 * l_risk).item()
                    val_steps += 1

            print(
                f"Epoch {epoch+1}/{EPOCHS} | Val Loss:   {val_loss_sum/val_steps:.4f}"
            )
            miner.train()
            combine.train()

    # 7. Save Model
    os.makedirs("./checkpoints", exist_ok=True)
    SAVE_PATH = "./checkpoints/v3_model_latest.pth"
    torch.save(
        {"miner": miner.state_dict(), "combine": combine.state_dict()}, SAVE_PATH
    )
    print(f"✅ Model Saved to {SAVE_PATH}")

    # 8. NEW: 计算baseline_stats（用于RiskMonitor）
    if COMPUTE_BASELINE_STATS:
        print("\n[PostTrain] Computing baseline_stats for RiskMonitor...")

        # Prepare DataLoader for baseline calculation -> Use Train Subset
        subset_size = min(1000, len(train_ds))
        subset_indices = torch.randperm(len(train_ds))[:subset_size]
        subset_dataset = torch.utils.data.Subset(train_ds, subset_indices)
        baseline_loader = DataLoader(subset_dataset, batch_size=128, shuffle=False)

        # 定义临时模型包装（用于compute_baseline_stats）
        class CombinedModel(nn.Module):
            def __init__(self, miner, combine):
                super().__init__()
                self.miner = miner
                self.combine = combine

            def forward(self, white_feats, tgt, aux, mask):
                deep_factors = self.miner(tgt, aux, mask)
                output = self.combine(white_feats, deep_factors)
                return output

        combined_model = CombinedModel(miner, combine).to(DEVICE)
        combined_model.eval()

        # 计算统计量
        baseline_stats = {
            "black_mu": 0.0,
            "black_sigma": 0.0,
            "black_q99": 0.0,
            "black_samples": [],
            "white_mu": 0.0,
            "white_sigma": 0.0,
        }

        black_outputs = []
        white_outputs = []

        with torch.no_grad():
            for batch in baseline_loader:
                b_w, b_t, b_a, b_m, _, _, _ = batch

                deep_factors = miner(b_t, b_a, b_m)
                output = combine(b_w, deep_factors)

                black_outputs.extend(output["delta_hit"].cpu().numpy().flatten())
                white_outputs.extend(output["base_hit"].cpu().numpy().flatten())

        black_outputs = np.array(black_outputs)
        white_outputs = np.array(white_outputs)

        baseline_stats["black_mu"] = float(black_outputs.mean())
        baseline_stats["black_sigma"] = float(black_outputs.std())
        baseline_stats["black_q99"] = float(np.percentile(np.abs(black_outputs), 99))
        baseline_stats["black_samples"] = black_outputs[:1000].tolist()
        baseline_stats["white_mu"] = float(white_outputs.mean())
        baseline_stats["white_sigma"] = float(white_outputs.std())

        # 保存baseline_stats
        BASELINE_PATH = "./checkpoints/baseline_stats.pkl"
        with open(BASELINE_PATH, "wb") as f:
            pickle.dump(baseline_stats, f)

        print(f"✅ Baseline stats saved to {BASELINE_PATH}")
        print(
            f"   Black μ={baseline_stats['black_mu']:.4f}, σ={baseline_stats['black_sigma']:.4f}"
        )
        print(
            f"   White μ={baseline_stats['white_mu']:.4f}, σ={baseline_stats['white_sigma']:.4f}"
        )


if __name__ == "__main__":
    train()
