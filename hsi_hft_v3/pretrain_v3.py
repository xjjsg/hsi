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

# 导入整合后的模块
from hsi_hft_v3.model.data_layer import V5DataLoader
from hsi_hft_v3.config import BLACKBOX_DIM
from hsi_hft_v3.model.model_layer import DeepFactorMinerV5, vicreg_loss
from hsi_hft_v3.model.whitebox import WhiteBoxFeatureFactory


class LatentPredictor(nn.Module):
    """Simple MLP to predict future latent state from current state"""

    def __init__(self, dim=BLACKBOX_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Linear(dim, dim)
        )

    def forward(self, x):
        return self.net(x)


def pretrain():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Pretrain] Device: {DEVICE}")

    # Config
    from hsi_hft_v3.config import PretrainConfig

    cfg = PretrainConfig()

    DATA_DIR = "./data"
    BATCH_SIZE = cfg.batch_size
    EPOCHS = cfg.epochs
    LOOKBACK = 64  # Fixed by model arch
    PRED_HORIZON = cfg.pred_horizon

    # 1. Load Data
    loader = V5DataLoader(DATA_DIR)
    print("[Pretrain] Loading Data...")
    # Use full dataset but exclude known bad dates
    EXCLUDE_DATES = cfg.exclude_dates
    data_dict = loader.load_date_range(exclude_dates=EXCLUDE_DATES)

    if not data_dict:
        return

    # 2. 准备序列
    # 我们需要成对数据: (Window_Core, Window_Target)
    # 对于 JEPA (联合嵌入预测架构):
    # 上下文 (Context): [t-L : t] -> 输出 Z_t
    # 目标 (Target):   [t-L+k : t+k] -> 输出 Z_{t+k}
    # 我们的目标是利用 Z_t 预测 Z_{t+k}

    # 2. 数据分割 (80/10/10)
    all_dates = sorted(list(data_dict.keys()))
    n_days = len(all_dates)
    n_train = int(0.8 * n_days)
    n_val = int(0.1 * n_days)

    train_dates = all_dates[:n_train]
    val_dates = all_dates[n_train : n_train + n_val]
    # 测试集日期在预训练中被隐式排除，以防止泄漏

    print(f"[Pretrain] 数据分割:")
    print(f"  训练集: {len(train_dates)} 天")
    print(f"  验证集: {len(val_dates)} 天")
    print(f"  测试集: {len(all_dates) - n_train - n_val} 天 (已排除)")

    def build_dataset(dates_list):
        ctx_t, ctx_a, ctx_m = [], [], []
        fut_t, fut_a, fut_m = [], [], []

        for date in dates_list:
            if date not in data_dict:
                continue
            samples = data_dict[date]

            day_t, day_a, day_m = [], [], []
            for s in samples:
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
                m = 1.0 if s.aux_available else 0.0
                day_t.append(t_vec)
                day_a.append(a_vec)
                day_m.append(m)

            if len(day_t) <= LOOKBACK + PRED_HORIZON:
                continue
            d_t, d_a, d_m = np.array(day_t), np.array(day_a), np.array(day_m)

            for i in range(LOOKBACK, len(day_t) - PRED_HORIZON, 5):  # 步长 5
                ctx_t.append(d_t[i - LOOKBACK : i])
                ctx_a.append(d_a[i - LOOKBACK : i])
                ctx_m.append(d_m[i - LOOKBACK : i])

                fut_t.append(d_t[i + PRED_HORIZON - LOOKBACK : i + PRED_HORIZON])
                fut_a.append(d_a[i + PRED_HORIZON - LOOKBACK : i + PRED_HORIZON])
                fut_m.append(d_m[i + PRED_HORIZON - LOOKBACK : i + PRED_HORIZON])

        if not ctx_t:
            return None

        # Tensors
        t_ct = torch.tensor(np.array(ctx_t), dtype=torch.float32).to(DEVICE)
        t_ca = torch.tensor(np.array(ctx_a), dtype=torch.float32).to(DEVICE)
        t_cm = (
            torch.tensor(np.array(ctx_m), dtype=torch.float32).unsqueeze(-1).to(DEVICE)
        )

        t_ft = torch.tensor(np.array(fut_t), dtype=torch.float32).to(DEVICE)
        t_fa = torch.tensor(np.array(fut_a), dtype=torch.float32).to(DEVICE)
        t_fm = (
            torch.tensor(np.array(fut_m), dtype=torch.float32).unsqueeze(-1).to(DEVICE)
        )

        return TensorDataset(t_ct, t_ca, t_cm, t_ft, t_fa, t_fm)

    print("[Pretrain] 构建训练数据集...")
    train_ds = build_dataset(train_dates)
    if train_ds is None:
        print("错误: 未生成训练数据")
        return
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    print("[Pretrain] 构建验证数据集...")
    val_ds = build_dataset(val_dates)
    val_dl = (
        DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False) if val_ds else None
    )

    print(f"[Pretrain] 训练样本数: {len(train_ds)}")
    if val_ds:
        print(f"[Pretrain] 验证样本数: {len(val_ds)}")

    # 从训练集获取输入维度
    dim_raw = train_ds[0][0].shape[-1]
    enc = DeepFactorMinerV5(input_dim_raw=dim_raw).to(DEVICE)
    pred = LatentPredictor().to(DEVICE)

    opt = optim.Adam(
        list(enc.parameters()) + list(pred.parameters()), lr=cfg.learning_rate
    )

    # 4. 训练循环
    print(f"[Pretrain] 开始 JEPA 预训练循环 ({EPOCHS} 轮)...")
    enc.train()
    pred.train()

    mse = nn.MSELoss()

    for ep in range(EPOCHS):
        tot_loss = 0
        tot_pred = 0
        tot_std = 0
        tot_cov = 0
        steps = 0
        for b in train_dl:
            ct, ca, cm, ft, fa, fm = b

            opt.zero_grad()

            # 上下文编码 -> Z_ctx
            z_ctx = enc(ct, ca, cm)

            # 未来编码 (目标) -> Z_fut
            # 经典 JEPA 使用 EMA 目标编码器。
            # 此处采用简化版：共享权重但停止目标分支梯度 (Stop Grad)，
            # 以保持稳定性并防止坍塌 (类似 SimSiam)。
            with torch.no_grad():
                z_fut = enc(ft, fa, fm)

            # 预测: Z_ctx -> Z_pred
            z_pred = pred(z_ctx)

            # 损失 1: 预测误差
            l_pred = mse(z_pred, z_fut)

            # 损失 2: Z_ctx 的 VICReg (正则化表示以保持信息量)
            # 维持方差，去相关
            l_std, l_cov = vicreg_loss(z_ctx, ct.shape[0])

            # Total
            loss = l_pred + l_std + l_cov
            loss.backward()
            opt.step()

            tot_loss += loss.item()
            tot_pred += l_pred.item()
            tot_std += l_std.item()
            tot_cov += l_cov.item()
            steps += 1

        print(
            f"Ep {ep+1} | Total: {tot_loss/steps:.4f} | Pred: {tot_pred/steps:.4f} | Std: {tot_std/steps:.4f} | Cov: {tot_cov/steps:.4f}"
        )

    # 5. Save
    os.makedirs("./checkpoints", exist_ok=True)
    torch.save(enc.state_dict(), "./checkpoints/v3_encoder_pretrained.pth")
    print("✅ Pretrained Encoder Saved.")


if __name__ == "__main__":
    pretrain()
