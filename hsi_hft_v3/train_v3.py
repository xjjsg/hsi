import sys
import os
import time
import numpy as np
import pandas as pd
import pickle

# Hack to make local imports work
sys.path.append(os.getcwd())

# 导入整合后的模块
# 导入整合后的模块
from hsi_hft_v3.model.data_layer import V5DataLoader
from hsi_hft_v3.config import (
    BLACKBOX_DIM,
    K_BARS,
    COST_RATE,
    LATENCY_BARS,
    LabelConfig,
    PolicyConfig,
)
from hsi_hft_v3.model.trading_layer import (
    BacktestEngine,
    StateMachine,
    calculate_metrics,
)
from hsi_hft_v3.model.model_layer import (
    DeepFactorMinerV5,
    HitHead,
    HazardHead,
    RiskHead,
    ResidualCombine,
)
from hsi_hft_v3.model.whitebox import WhiteBoxFeatureFactory

# NEW: 导入IRM损失和可微分特征
from hsi_hft_v3.model.irm_loss import IRMLoss, EnvironmentSplitter
from hsi_hft_v3.model.risk_monitor import compute_baseline_stats

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
# 配置加载
# ==========================================
from hsi_hft_v3.config import TrainConfig, IRMConfig, RiskConfig, PretrainConfig

train_cfg = TrainConfig()
irm_cfg = IRMConfig()
risk_cfg = RiskConfig()

# 兼容性变量映射
USE_IRM = irm_cfg.use_irm
USE_LEARNABLE_FEATURES = False
IRM_PENALTY_WEIGHT = irm_cfg.penalty_weight
IRM_ANNEAL_EPOCHS = irm_cfg.anneal_epochs
COMPUTE_BASELINE_STATS = train_cfg.compute_baseline_stats


# ==========================================
# Label Generation
# ==========================================
# ==========================================
# 标签生成 (Label Generation)
# ==========================================
def generate_labels(samples, lookahead=60):
    """
    为每个样本生成训练目标:
    - y_hit: 1 如果在展望窗口内，价格有利变动 > 成本
    - y_hazard: 首次命中发生的时间索引 k
    - y_risk: 1 如果在命中前发生逆向事件 (止损)
    """
    n = len(samples)
    y_hit = np.zeros(n, dtype=np.float32)
    y_hazard = np.zeros(n, dtype=np.int64)  # 时间桶索引 0..K
    y_risk = np.zeros(n, dtype=np.float32)

    # 预提取价格以加速处理
    asks = np.array([s.target.asks[0][0] if s.target.asks else np.nan for s in samples])
    bids = np.array([s.target.bids[0][0] if s.target.bids else np.nan for s in samples])

    # 我们需要 Ask_e = Ask[t + latency]
    latency = LATENCY_BARS  # 固定为1个Bar

    # 成本阈值逻辑
    # 利润 = (Bid_future - Ask_entry) / Ask_entry - 2*Cost

    for i in range(n - latency):
        # 进场需要 t + latency 时刻有 Ask 价格
        idx_entry = i + latency
        if idx_entry >= n:
            break

        entry_price = asks[idx_entry]  # t+latency 时刻的 Ask
        if np.isnan(entry_price):
            continue

        target_price = entry_price * (1.0 + 2 * COST_RATE + 0.0002)  # 最小净利润 2bps
        stop_price = entry_price * (
            1.0 - LabelConfig.adverse_bps / 10000.0
        )  # 从配置读取止损阈值

        found_hit = False
        hit_k = K_BARS - 1
        found_risk = False  # 修正: 初始化 found_risk

        # 从 t_e 开始向后展望
        # 路径: t_e + 1 ... t_e + lookahead
        for k in range(1, min(lookahead, n - idx_entry)):
            curr_bid = bids[idx_entry + k]  # 未来的 Bid
            if np.isnan(curr_bid):
                continue

            # 检查是否命中获利 (Hit)
            if curr_bid > target_price:
                y_hit[i] = 1.0
                hit_k = k
                found_hit = True
                break

            # 检查是否触及风控 (Risk)
            if curr_bid < stop_price:
                y_risk[i] = 1.0
                hit_k = k
                found_risk = True
                break

        y_hazard[i] = min(hit_k, K_BARS - 1)

    return y_hit, y_hazard, y_risk


# ==========================================
# 2. 回测助手 (Backtest Helper)
# ==========================================
def run_backtest(miner, combine, val_dates, data_map, device):
    """
    在验证集上运行回测模拟
    """
    print(f"\n[Backtest] 正在 {len(val_dates)} 个验证日上进行模拟...")

    # 初始化策略与引擎
    policy_cfg = PolicyConfig()
    policy = StateMachine(policy_cfg)

    # 验证回测使用宽松的基准
    # 因为模型仍在训练中，我们不想使用过时的统计数据，
    # 也不想对正在演变的模型强制执行严格的分布检查。
    baseline_stats = {
        "black_mu": 0.0,
        "black_sigma": 10.0,  # 宽Sigma以防止漂移警报
        "black_q99": 100.0,
        "black_samples": [],
        "white_mu": 0.0,
        "white_sigma": 1.0,
    }

    engine = BacktestEngine(policy, baseline_stats)

    miner.eval()
    combine.eval()

    total_samples = 0
    wb_factory_local = WhiteBoxFeatureFactory()
    all_p_hits = []

    with torch.no_grad():
        for date in val_dates:
            if date not in data_map:
                continue
            samples = data_map[date]

            # --- 提取特征 (与构建Tensor相同) ---
            feats_white = []
            raw_tgt = []
            raw_aux = []
            masks = []

            # 我们需要将样本映射到索引以过滤有效窗口
            valid_samples = []

            # 1. 为当天所有样本预计算特征
            # 注意: 理想情况下应该缓存此步骤以避免每个epoch重复计算
            # 但对于验证来说是可以接受的。

            day_w, day_t, day_a, day_m = [], [], [], []

            for s in samples:
                wb_out = wb_factory_local.compute(s)
                sorted_keys = wb_factory_local.get_derived_keys()
                z_feats = []
                for k in sorted_keys:
                    z_feats.append(wb_out["white_derived"].get(k, 0.0))
                if not z_feats:
                    z_feats = [0.0] * 10

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

                day_w.append(z_feats)
                day_t.append(t_vec)
                day_a.append(a_vec)
                day_m.append(1.0 if s.aux_available else 0.0)

            # 2. 滑动窗口批处理
            L = len(samples)
            LOOKBACK = 64

            if L <= LOOKBACK:
                continue

            np_w = np.array(day_w, dtype=np.float32)
            np_t = np.array(day_t, dtype=np.float32)
            np_a = np.array(day_a, dtype=np.float32)
            np_m = np.array(day_m, dtype=np.float32)

            batch_w, batch_t, batch_aux, batch_mask = [], [], [], []

            # 准备引擎所需的索引
            # 引擎需要对应于预测时间（窗口的最后一步）的对齐样本
            # 窗口 [i-LOOKBACK : i]，预测附加到 samples[i-1]

            engine_samples = []

            for i in range(LOOKBACK, L):
                batch_t.append(np_t[i - LOOKBACK : i])
                batch_aux.append(np_a[i - LOOKBACK : i])
                batch_mask.append(np_m[i - LOOKBACK : i])
                batch_w.append(np_w[i - 1])
                engine_samples.append(samples[i - 1])

            if not batch_t:
                continue

            # 转换为 Tensor
            t_t = torch.tensor(np.array(batch_t), dtype=torch.float32).to(device)
            t_a = torch.tensor(np.array(batch_aux), dtype=torch.float32).to(device)
            t_m = (
                torch.tensor(np.array(batch_mask), dtype=torch.float32)
                .unsqueeze(-1)
                .to(device)
            )
            t_w = torch.tensor(np.array(batch_w), dtype=torch.float32).to(device)

            # 推理
            deep_factors = miner(t_t, t_a, t_m)
            out = combine(t_w, deep_factors)

            # 解析输出
            p_hit = torch.sigmoid(out["logit_hit"]).cpu().numpy().flatten()
            p_risk = torch.sigmoid(out["logit_risk"]).cpu().numpy().flatten()
            hazard = torch.sigmoid(out["logit_hazard"]).cpu().numpy()  # (B, K)

            # P(tau <= 60s) -> 约等于 Hazard 概率的累积
            # 实际上 Hazard H_k 是给定之前未命中时在 k 命中的概率。
            # CDF(k) = 1 - Prod(1-h_i)
            # k=20 (60s / 3s)

            k_60s = 60 // 3
            prob_tau_60 = 1.0 - np.exp(
                np.sum(np.log(1 - hazard[:, :k_60s] + 1e-9), axis=1)
            )

            # 构建模型输出供引擎使用
            model_outputs = []
            for j in range(len(engine_samples)):
                # 解析 RiskMonitor 所需的原始组件
                out_base_hit = out["base_hit"].cpu().numpy().flatten()
                out_delta_hit = out["delta_hit"].cpu().numpy().flatten()
                out_logit_hit = out["logit_hit"].cpu().numpy().flatten()

                model_outputs.append(
                    {
                        "p_hit": float(p_hit[j]),
                        "risk": float(p_risk[j]),
                        "P_tau_le_60s": float(prob_tau_60[j]),
                        "base_hit": float(out_base_hit[j]),
                        "delta_hit": float(out_delta_hit[j]),
                        "logit_hit": float(out_logit_hit[j]),
                        "white_risk": {
                            "spread_bps": 0.0,
                            "vpin_z": 0.0,
                        },  # 占位符，理想情况从样本计算
                    }
                )

                # 如果可能，从样本更新 WhiteRisk
                # 但这里的 samples 不是简单的字典形式
                # 我们可以从原始样本近似计算 spread
                s = engine_samples[j]
                if s.target.asks and s.target.bids:
                    ask1 = s.target.asks[0][0]
                    bid1 = s.target.bids[0][0]
                    mid = (ask1 + bid1) / 2
                    spread_bps = (ask1 - bid1) / mid * 10000
                    model_outputs[-1]["white_risk"]["spread_bps"] = spread_bps

            # 为这一天运行引擎
            engine.run(engine_samples, model_outputs)

            # 调试: 收集 p_hit
            for m in model_outputs:
                all_p_hits.append(m["p_hit"])

            total_samples += len(engine_samples)

    # 计算指标
    metrics = calculate_metrics(engine.trades, engine.equity_curve)

    # 添加调试统计
    if all_p_hits:
        metrics["avg_p_hit"] = np.mean(all_p_hits)
        metrics["max_p_hit"] = np.max(all_p_hits)
    else:
        metrics["avg_p_hit"] = 0.0
        metrics["max_p_hit"] = 0.0

    return metrics


# ==========================================
# 3. 主训练逻辑 (Main Training Logic)
# ==========================================
def train():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] 使用设备: {DEVICE}")

    # 1. 配置
    DATA_DIR = "./data"
    BATCH_SIZE = train_cfg.batch_size
    EPOCHS = train_cfg.epochs
    LOOKBACK = 100  # 固定

    # 2. 加载数据
    loader = V5DataLoader(DATA_DIR)
    # 加载所有可用数据 (动态分割)
    print("[Train] 加载所有可用数据...")
    # 排除问题数据日期 (数据质量问题: 缺少指数价格)
    EXCLUDE_DATES = train_cfg.exclude_dates
    data_dict = loader.load_date_range(exclude_dates=EXCLUDE_DATES)

    if not data_dict:
        print("❌ 未加载到数据。")
        return

    # 3. 动态数据分割 (80/10/10)
    all_dates = sorted(list(data_dict.keys()))
    n_days = len(all_dates)
    if n_days < 3:
        print("❌ 天数不足，无法分割。")
        return

    n_train = int(train_cfg.train_ratio * n_days)
    n_val = int(train_cfg.val_ratio * n_days)
    # n_test = 剩余部分

    train_dates = all_dates[:n_train]
    val_dates = all_dates[n_train : n_train + n_val]
    test_dates = all_dates[n_train + n_val :]

    print(
        f"[Split] 训练集: {len(train_dates)} 天 ({train_dates[0]} ~ {train_dates[-1]})"
    )
    print(f"[Split] 验证集: {len(val_dates)} 天 ({val_dates[0]} ~ {val_dates[-1]})")
    if test_dates:
        print(
            f"[Split] 测试集: {len(test_dates)} 天 ({test_dates[0]} ~ {test_dates[-1]})"
        )
    else:
        print(f"[Split] 测试集: 0 天 (数据不足)")

    # 构建 Tensor 的辅助函数 (Date List -> Tensors)
    def build_tensors(dates_list, data_map):
        b_white, b_tgt, b_aux, b_mask = [], [], [], []
        b_y_hit, b_y_haz, b_y_risk = [], [], []

        count = 0
        wb_factory_local = WhiteBoxFeatureFactory()

        for date in dates_list:
            if date not in data_map:
                continue
            samples = data_map[date]

            # --- 每日处理 (同前) ---
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

            # 标签生成
            y_hit, y_haz, y_risk = generate_labels(samples, lookahead=K_BARS)

            # 滑动窗口
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

    # 4. 构建 DataLoaders
    print("[Train] 构建训练集 Tensors...")
    train_tensors = build_tensors(train_dates, data_dict)
    if not train_tensors:
        print("❌ 训练集为空。")
        return
    train_ds = TensorDataset(*train_tensors)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    print("[Train] 构建验证集 Tensors...")
    val_tensors = build_tensors(val_dates, data_dict)
    val_loader = None
    if val_tensors:
        val_ds = TensorDataset(*val_tensors)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        print(f"  -> 验证样本数: {len(val_ds)}")
    else:
        print("⚠️ 验证集为空，跳过验证。")

    # 5. 模型设置
    dim_raw = train_tensors[1].shape[-1]
    dim_white = train_tensors[0].shape[-1]

    miner = DeepFactorMinerV5(input_dim_raw=dim_raw, out_dim=BLACKBOX_DIM).to(DEVICE)

    # 如果有预训练权重则加载
    PRETRAIN_PATH = "./checkpoints/v3_encoder_pretrained.pth"
    if os.path.exists(PRETRAIN_PATH):
        print(f"[Train] 从 {PRETRAIN_PATH} 加载预训练编码器...")
        # Miner state dict 可能有 'projector'，预训练也有。
        # 预训练脚本保存了 'enc.state_dict()'。
        try:
            state = torch.load(PRETRAIN_PATH, map_location=DEVICE)
            miner.load_state_dict(
                state, strict=False
            )  # 使用 strict=False 以防预测头不同
            print("✅ 预训练权重加载成功。")
        except Exception as e:
            print(f"⚠️ 加载预训练权重失败: {e}")
    else:
        print("[Train] 未找到预训练权重，从头开始训练。")

    combine = ResidualCombine(white_dim=dim_white, black_dim=BLACKBOX_DIM).to(DEVICE)

    optimizer = optim.Adam(
        list(miner.parameters()) + list(combine.parameters()),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    # 不平衡数据的加权损失 (假设 ~5-10% 命中率)
    pos_weight = torch.tensor([train_cfg.pos_weight]).to(DEVICE)
    crit_hit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    crit_haz = nn.BCEWithLogitsLoss(reduction="none")  # 改为带Mask的BCE
    crit_risk = nn.BCEWithLogitsLoss()

    # 6. 训练循环
    print(f"[Train] 开始训练 {EPOCHS} 轮...")
    miner.train()
    combine.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        tot_hit = 0
        tot_haz = 0
        tot_risk = 0
        tot_cons = 0
        steps = 0

        for batch in train_loader:
            b_w, b_t, b_a, b_m, y_h, y_z, y_r = batch

            optimizer.zero_grad()

            # Forward
            deep_factors = miner(b_t, b_a, b_m)
            out = combine(b_w, deep_factors)

            # 1. Hit Loss (二分类)
            loss_hit = crit_hit(out["logit_hit"], y_h)

            # 2. Hazard Loss (离散时间: h_k = P(T=k|T>=k))
            # 目标: k < T 时为 0, k == T 时为 1
            # Mask: k <= T 时为 1, k > T 时为 0

            # 从索引 y_z 构建密集目标
            B, K = out["logit_hazard"].shape

            # y_z 是事件索引 (Hit 或 Risk 或 End)。
            haz_targets = torch.zeros_like(out["logit_hazard"])
            haz_mask = torch.zeros_like(out["logit_hazard"])

            for i in range(B):
                idx = int(y_z[i].item())
                # 有效范围: 0 到 idx (包含)
                haz_mask[i, : idx + 1] = 1.0

                # 如果是 Hit，则 idx 处的 Hazard 目标为 1
                if y_h[i].item() > 0.5:
                    haz_targets[i, idx] = 1.0
                # 如果是 Risk 或 End，则该点的 Hazard 目标保持 0 (被审查)

            loss_haz_elem = crit_haz(out["logit_hazard"], haz_targets)
            loss_haz = (loss_haz_elem * haz_mask).sum() / (haz_mask.sum() + 1e-6)

            # 3. Risk Loss
            loss_risk = crit_risk(out["logit_risk"], y_r)

            # 4. 一致性损失 (L_cons)
            # p_hit_implied = 1 - S(K) = 1 - Prod(1 - h_k)
            # 我们希望 Logits 间的数学关系成立

            p_hit_model = torch.sigmoid(out["logit_hit"])
            # 一致性: S(t) = Prod(1 - h_k)
            h_probs = torch.sigmoid(out["logit_hazard"])

            # 计算生存概率 P(T > K)
            log_S = torch.sum(torch.log(1 - h_probs + 1e-9), dim=1)
            p_hit_implied = 1.0 - torch.exp(log_S).unsqueeze(-1)

            loss_cons = F.mse_loss(p_hit_model, p_hit_implied)

            loss = loss_hit + 0.5 * loss_haz + 0.5 * loss_risk + 0.1 * loss_cons

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            tot_hit += loss_hit.item()
            tot_haz += loss_haz.item()
            tot_risk += loss_risk.item()
            tot_cons += loss_cons.item()
            steps += 1

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train: {total_loss/steps:.4f} | Hit: {tot_hit/steps:.4f} | Haz: {tot_haz/steps:.4f} | Risk: {tot_risk/steps:.4f} | Cons: {tot_cons/steps:.4f}"
        )
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
# 2. Backtest Helper
# ==========================================
def run_backtest(miner, combine, val_dates, data_map, device):
    """
    Run backtest simulation on validation set
    """
    print(f"\n[Backtest] Simulating on {len(val_dates)} validation days...")

    # Init Policy & Engine
    policy_cfg = PolicyConfig()
    policy = StateMachine(policy_cfg)

    # Use permissive baseline for validation backtests (since model is still training)
    # We don't want to use stale stats from previous runs, nor enforce strict distribution checks
    # on an evolving model.
    baseline_stats = {
        "black_mu": 0.0,
        "black_sigma": 10.0,  # Wide sigma to prevent drift alerts
        "black_q99": 100.0,
        "black_samples": [],
        "white_mu": 0.0,
        "white_sigma": 1.0,
    }

    engine = BacktestEngine(policy, baseline_stats)

    miner.eval()
    combine.eval()

    total_samples = 0
    wb_factory_local = WhiteBoxFeatureFactory()
    all_p_hits = []

    with torch.no_grad():
        for date in val_dates:
            if date not in data_map:
                continue
            samples = data_map[date]

            # --- Extract Features (Same as build_tensors) ---
            feats_white = []
            raw_tgt = []
            raw_aux = []
            masks = []

            # We need to map samples to indices to filter valid windows
            valid_samples = []

            # 1. Pre-compute features for all samples in day
            # Note: Ideally we should cache this to avoid re-computing every epoch
            # But for validation it's acceptable.

            day_w, day_t, day_a, day_m = [], [], [], []

            for s in samples:
                wb_out = wb_factory_local.compute(s)
                sorted_keys = wb_factory_local.get_derived_keys()
                z_feats = []
                for k in sorted_keys:
                    z_feats.append(wb_out["white_derived"].get(k, 0.0))
                if not z_feats:
                    z_feats = [0.0] * 10

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

                day_w.append(z_feats)
                day_t.append(t_vec)
                day_a.append(a_vec)
                day_m.append(1.0 if s.aux_available else 0.0)

            # 2. Sliding Window Batching
            L = len(samples)
            LOOKBACK = 64

            if L <= LOOKBACK:
                continue

            np_w = np.array(day_w, dtype=np.float32)
            np_t = np.array(day_t, dtype=np.float32)
            np_a = np.array(day_a, dtype=np.float32)
            np_m = np.array(day_m, dtype=np.float32)

            batch_w, batch_t, batch_aux, batch_mask = [], [], [], []

            # Prepare indices for engine
            # Engine needs aligned samples corresponding to the PREDICTION time (last step of window)
            # Window [i-LOOKBACK : i], prediction is attached to samples[i-1]

            engine_samples = []

            for i in range(LOOKBACK, L):
                batch_t.append(np_t[i - LOOKBACK : i])
                batch_aux.append(np_a[i - LOOKBACK : i])
                batch_mask.append(np_m[i - LOOKBACK : i])
                batch_w.append(np_w[i - 1])
                engine_samples.append(samples[i - 1])

            if not batch_t:
                continue

            # Convert to Tensor
            t_t = torch.tensor(np.array(batch_t), dtype=torch.float32).to(device)
            t_a = torch.tensor(np.array(batch_aux), dtype=torch.float32).to(device)
            t_m = (
                torch.tensor(np.array(batch_mask), dtype=torch.float32)
                .unsqueeze(-1)
                .to(device)
            )
            t_w = torch.tensor(np.array(batch_w), dtype=torch.float32).to(device)

            # Inference
            deep_factors = miner(t_t, t_a, t_m)
            out = combine(t_w, deep_factors)

            # Parse Outputs
            p_hit = torch.sigmoid(out["logit_hit"]).cpu().numpy().flatten()
            p_risk = torch.sigmoid(out["logit_risk"]).cpu().numpy().flatten()
            hazard = torch.sigmoid(out["logit_hazard"]).cpu().numpy()  # (B, K)

            # P(tau <= 60s) -> Sum of hazard prob roughly?
            # Actually Hazard H_k is prob of hit at k given not hit before.
            # CDF(k) = 1 - Prod(1-h_i)
            # k=20 (60s / 3s)

            k_60s = 60 // 3
            prob_tau_60 = 1.0 - np.exp(
                np.sum(np.log(1 - hazard[:, :k_60s] + 1e-9), axis=1)
            )

            # Construct Model Outputs for Engine
            model_outputs = []
            for j in range(len(engine_samples)):
                # Parse raw components for RiskMonitor
                out_base_hit = out["base_hit"].cpu().numpy().flatten()
                out_delta_hit = out["delta_hit"].cpu().numpy().flatten()
                out_logit_hit = out["logit_hit"].cpu().numpy().flatten()

                model_outputs.append(
                    {
                        "p_hit": float(p_hit[j]),
                        "risk": float(p_risk[j]),
                        "P_tau_le_60s": float(prob_tau_60[j]),
                        "base_hit": float(out_base_hit[j]),
                        "delta_hit": float(out_delta_hit[j]),
                        "logit_hit": float(out_logit_hit[j]),
                        "white_risk": {
                            "spread_bps": 0.0,
                            "vpin_z": 0.0,
                        },  # Placeholder, ideally compute from samples
                    }
                )

                # Update WhiteRisk from Sample if possible
                # But samples here don't have computed white features easily accessible in dict form
                # We can approximate spread from raw sample
                s = engine_samples[j]
                if s.target.asks and s.target.bids:
                    ask1 = s.target.asks[0][0]
                    bid1 = s.target.bids[0][0]
                    mid = (ask1 + bid1) / 2
                    spread_bps = (ask1 - bid1) / mid * 10000
                    model_outputs[-1]["white_risk"]["spread_bps"] = spread_bps

            # Run Engine for this Day
            engine.run(engine_samples, model_outputs)

            # Debug: Collect p_hit
            for m in model_outputs:
                all_p_hits.append(m["p_hit"])

            total_samples += len(engine_samples)

    # Calculate Metrics
    metrics = calculate_metrics(engine.trades, engine.equity_curve)

    # Add Debug Stats
    if all_p_hits:
        metrics["avg_p_hit"] = np.mean(all_p_hits)
        metrics["max_p_hit"] = np.max(all_p_hits)
    else:
        metrics["avg_p_hit"] = 0.0
        metrics["max_p_hit"] = 0.0

    return metrics


# ==========================================
# 3. Main Training Logic
# ==========================================
def train():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using Device: {DEVICE}")

    # 1. Config
    DATA_DIR = "./data"
    BATCH_SIZE = train_cfg.batch_size
    EPOCHS = train_cfg.epochs
    LOOKBACK = 100  # Fixed

    # 2. Load Data
    loader = V5DataLoader(DATA_DIR)
    # Load ALL available data (Dynamic Splitting)
    print("[Train] Loading ALL available data...")
    # Exclude problematic dates (Data Quality Issues: Index Price Missing)
    # Exclude problematic dates (Data Quality Issues: Index Price Missing)
    EXCLUDE_DATES = train_cfg.exclude_dates
    data_dict = loader.load_date_range(exclude_dates=EXCLUDE_DATES)

    if not data_dict:
        print("❌ No data loaded.")
        return

    # 3. Dynamic Data Splitting (80/10/10)
    all_dates = sorted(list(data_dict.keys()))
    n_days = len(all_dates)
    if n_days < 3:
        print("❌ Not enough days for splitting.")
        return

    n_train = int(train_cfg.train_ratio * n_days)
    n_val = int(train_cfg.val_ratio * n_days)
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
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

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
        list(miner.parameters()) + list(combine.parameters()),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    # Weighted Loss for Imbalanced Data (Assume ~5-10% hit rate)
    pos_weight = torch.tensor([train_cfg.pos_weight]).to(DEVICE)
    crit_hit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    crit_haz = nn.BCEWithLogitsLoss(reduction="none")  # Changed to BCE with masking
    crit_risk = nn.BCEWithLogitsLoss()

    # 6. Loop
    print(f"[Train] Starting Training for {EPOCHS} Epochs...")
    miner.train()
    combine.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        tot_hit = 0
        tot_haz = 0
        tot_risk = 0
        tot_cons = 0
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
            tot_hit += loss_hit.item()
            tot_haz += loss_haz.item()
            tot_risk += loss_risk.item()
            tot_cons += loss_cons.item()
            steps += 1

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train: {total_loss/steps:.4f} | Hit: {tot_hit/steps:.4f} | Haz: {tot_haz/steps:.4f} | Risk: {tot_risk/steps:.4f} | Cons: {tot_cons/steps:.4f}"
        )

        # Validation Step
        if val_loader:
            miner.eval()
            combine.eval()
            val_loss_sum = 0
            v_hit = 0
            v_haz = 0
            v_risk = 0
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
                    v_hit += l_hit.item()
                    v_haz += l_haz.item()
                    v_risk += l_risk.item()
                    val_steps += 1

            print(
                f"Epoch {epoch+1}/{EPOCHS} | Val:   {val_loss_sum/val_steps:.4f} | Hit: {v_hit/val_steps:.4f} | Haz: {v_haz/val_steps:.4f} | Risk: {v_risk/val_steps:.4f}"
            )

            # [Backtest Integration]
            # Run simulation from Epoch 5 onwards
            if epoch >= 5:
                bk_metrics = run_backtest(miner, combine, val_dates, data_dict, DEVICE)
                print(
                    f"[Backtest] PnL: {bk_metrics['total_pnl']:.2f} | Sharpe: {bk_metrics['sharpe']:.2f} | Trades: {bk_metrics['n_trades']} | Win: {bk_metrics['fill_rate_pct']:.1f}%"
                )
                if "avg_p_hit" in bk_metrics:
                    print(
                        f"           [Debug] p_hit avg: {bk_metrics['avg_p_hit']:.4f} max: {bk_metrics['max_p_hit']:.4f}"
                    )

            # Save Checkpoint
            if (epoch + 1) % 5 == 0:
                os.makedirs("./checkpoints", exist_ok=True)
                checkpoint_path = f"./checkpoints/v3_model_epoch_{epoch+1}.pth"
                torch.save(
                    {"miner": miner.state_dict(), "combine": combine.state_dict()},
                    checkpoint_path,
                )
                print(f"✅ Checkpoint saved to {checkpoint_path}")

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
