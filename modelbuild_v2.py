import os
import glob
import warnings
import math
from datetime import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# ==========================================
# 0. 全局配置 (CIO Directives)
# ==========================================
CONFIG = {
    # --- [关键修改] 路径配置 ---
    # 使用 "." 表示从当前目录开始递归搜索
    "DATA_DIR": ".",  
    "SYMBOL": "sz159920",
    
    # --- 核心时序参数 ---
    "RESAMPLE_FREQ": "3S",      # 3秒快照 (去噪)
    "PREDICT_HORIZON": 600,     # 30分钟 = 600 bars (趋势策略)
    "LOOKBACK": 100,            # 输入窗口
    
    # --- 交易费率 ---
    "TRADE_COST": 0.0001,       # 单边万1
    
    # --- 训练参数 ---
    "BATCH_SIZE": 256,
    "EPOCHS": 50,
    "LR": 1e-4,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "PATIENCE": 10,
    "TIMEZONE": "Asia/Shanghai" # 明确时区
}

warnings.filterwarnings("ignore")
DEVICE = CONFIG["DEVICE"]

# ==========================================
# 1. 核心模型: DeepLOB-Transformer
# ==========================================
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding='same')
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding='same')
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), padding='same')
        self.pool = nn.MaxPool2d(kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.project = nn.Conv2d(4 * out_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        c1 = self.conv1(x)
        c3 = self.conv3(x)
        c5 = self.conv5(x)
        p = self.pool(x)
        out = torch.cat([c1, c3, c5, p], dim=1)
        out = self.project(out)
        out = self.bn(out)
        return self.relu(out)

class DeepLOBTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2, num_classes=3):
        super().__init__()
        # Input: (Batch, 1, 20, Lookback) -> 20特征 (5档价+5档量 * 2边? 这里简化取前5档价+量)
        # 你的数据有 5档BP/SP/BV/SV -> 20个特征
        self.cnn = nn.Sequential(
            InceptionBlock(1, 16),
            InceptionBlock(16, 32),
            InceptionBlock(32, 64),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, CONFIG["LOOKBACK"], d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                 dim_feedforward=128, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.LeakyReLU(),
            nn.Linear(32, num_classes)
        )
        self._init_bias()

    def _init_bias(self):
        prior_prob = np.array([0.8, 0.1, 0.1])
        bias = -np.log((1 - prior_prob) / prior_prob)
        self.fc[-1].bias.data = torch.from_numpy(bias).float()

    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(1)
        feat = self.cnn(x) 
        feat = feat.mean(dim=2) 
        feat = feat.permute(0, 2, 1) 
        feat = feat + self.pos_embedding
        trans_out = self.transformer(feat) 
        context = trans_out[:, -1, :]
        return self.fc(context)

# ==========================================
# 2. 抗躺平损失函数
# ==========================================
class AlphaFocalLoss(nn.Module):
    def __init__(self, alpha=[1.0, 3.0, 3.0], gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha).to(DEVICE)
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        logpt = -self.ce(logits, targets)
        pt = torch.exp(logpt)
        alpha_t = self.alpha[targets]
        focal_loss = -alpha_t * (1 - pt)**self.gamma * logpt
        return focal_loss.mean()

# ==========================================
# 3. 数据管道: AlphaForge V2 (集成标准Loader)
# ==========================================
class AlphaForgeV2:
    def __init__(self, cfg):
        self.cfg = cfg
        self.scaler = StandardScaler()

    def _load_raw_data(self) -> pd.DataFrame:
        """
        [Standardized Loader Integration]
        完全复刻 load_data_like_model 的逻辑
        """
        root_dir = self.cfg["DATA_DIR"]
        symbol = self.cfg["SYMBOL"]
        
        # 1. 递归查找文件
        pattern = os.path.join(root_dir, "**", f"{symbol}*.csv")
        print(f"🕵️ [Loader] 正在递归搜索: {os.path.abspath(pattern)}")
        files = sorted(glob.glob(pattern, recursive=True))
        
        print(f"🔎 [Loader] 扫描到 {len(files)} 个文件")
        
        df_list = []
        for f in files:
            try:
                # 2. 读取与时间清洗
                # 必须包含 5档 量/价 (共20列) + 时间
                usecols = ["tx_local_time", 
                           "bp1", "bv1", "sp1", "sv1", 
                           "bp2", "bv2", "sp2", "sv2", 
                           "bp3", "bv3", "sp3", "sv3", 
                           "bp4", "bv4", "sp4", "sv4", 
                           "bp5", "bv5", "sp5", "sv5"]
                
                # 预读检查
                preview = pd.read_csv(f, nrows=1)
                valid_cols = [c for c in usecols if c in preview.columns]
                
                if "tx_local_time" not in preview.columns:
                    print(f"⚠️ 跳过 {f}: 缺少 tx_local_time")
                    continue

                df = pd.read_csv(f, usecols=valid_cols)
                
                # 核心时间处理 (Standard Logic)
                dt_utc = pd.to_datetime(df["tx_local_time"], unit="ms", utc=True, errors="coerce")
                df["datetime"] = dt_utc.dt.tz_convert(self.cfg["TIMEZONE"]).dt.tz_localize(None)
                
                # 数值强制转换
                num_cols = [c for c in df.columns if c not in ["datetime", "tx_local_time"]]
                for c in num_cols:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                
                # 排序与去重
                df = df.sort_values("datetime")
                df = df.drop_duplicates(subset="datetime", keep="last")
                df = df.set_index("datetime").sort_index()
                
                # 3. 重采样 (Standard Logic)
                df_res = df.resample(self.cfg["RESAMPLE_FREQ"]).last().dropna()
                
                # 补充计算 Mid Price
                if "bp1" in df_res.columns and "sp1" in df_res.columns:
                    df_res["mid"] = (df_res["bp1"] + df_res["sp1"]) / 2
                else:
                    continue

                # 过滤无效盘口
                df_res = df_res[(df_res["bp1"] > 0) & (df_res["sp1"] > 0)]
                
                df_list.append(df_res)
                
            except Exception as e:
                print(f"❌ 读取错误 {f}: {e}")
                continue

        if not df_list:
            raise ValueError(f"❌ 未找到有效数据! 请检查路径 {root_dir}")

        full_df = pd.concat(df_list).sort_index()
        print(f"✅ [Loader] 加载完成: {len(full_df)} 行 (频率: {self.cfg['RESAMPLE_FREQ']})")
        return full_df

    def _filter_trading_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """[CIO 风控] 剔除垃圾时间，只保留黄金波动窗口"""
        t = df.index.time
        # 早盘 09:30 - 10:15 | 尾盘 14:00 - 14:45
        mask_am = (t >= time(9, 30)) & (t <= time(10, 15))
        mask_pm = (t >= time(14, 0)) & (t <= time(14, 45))
        return df[mask_am | mask_pm]

    def _make_dynamic_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """[CIO 核心] 动态三屏障打标"""
        horizon = self.cfg["PREDICT_HORIZON"]
        comm = self.cfg["TRADE_COST"]
        
        future_mid = df["mid"].shift(-horizon)
        ret = (future_mid - df["mid"]) / df["mid"]
        
        spread_bps = (df["sp1"] - df["bp1"]) / df["mid"]
        spread_bps = spread_bps.clip(lower=0.0002) 
        
        vol = df["mid"].pct_change().rolling(200).std()
        threshold = spread_bps + (2 * comm) + (0.5 * vol)
        
        labels = np.zeros(len(df), dtype=int)
        labels[ret > threshold] = 1   # Buy
        labels[ret < -threshold] = 2  # Sell
        
        df["label"] = labels
        df["future_ret"] = ret 
        return df.dropna()

    def _normalize_features(self, df: pd.DataFrame, is_train=False) -> np.ndarray:
        # 选取前20个特征 (5档 价+量)
        feature_cols = ["bp1", "bv1", "sp1", "sv1", 
                        "bp2", "bv2", "sp2", "sv2", 
                        "bp3", "bv3", "sp3", "sv3", 
                        "bp4", "bv4", "sp4", "sv4", 
                        "bp5", "bv5", "sp5", "sv5"]
        
        # 确保列存在
        valid_cols = [c for c in feature_cols if c in df.columns]
        X = df[valid_cols].values
        
        if is_train:
            self.scaler.fit(X)
        return self.scaler.transform(X)

    def prepare_data(self):
        df = self._load_raw_data()
        df = self._filter_trading_hours(df)
        df = self._make_dynamic_labels(df)
        
        n = len(df)
        train_idx = int(n * 0.8)
        val_idx = int(n * 0.9)
        
        train_df = df.iloc[:train_idx]
        val_df = df.iloc[train_idx:val_idx]
        test_df = df.iloc[val_idx:]
        
        print(f"📦 数据切分: Train {len(train_df)} | Val {len(val_df)} | Test {len(test_df)}")
        
        X_train = self._normalize_features(train_df, is_train=True)
        X_val = self._normalize_features(val_df, is_train=False)
        X_test = self._normalize_features(test_df, is_train=False)
        
        return (X_train, train_df), (X_val, val_df), (X_test, test_df)

# ==========================================
# 4. Dataset & Evaluation
# ==========================================
class LOBDataset(Dataset):
    def __init__(self, X, df, lookback):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(df["label"].values)
        self.ret = torch.FloatTensor(df["future_ret"].values) 
        self.lookback = lookback
        
    def __len__(self):
        return len(self.X) - self.lookback - 1
        
    def __getitem__(self, idx):
        x_window = self.X[idx : idx + self.lookback]
        y_label = self.y[idx + self.lookback]
        ret_val = self.ret[idx + self.lookback]
        return x_window, y_label, ret_val

def validate_cio_standard(model, loader, threshold=0.55):
    model.eval()
    all_preds, all_labels, all_rets = [], [], []
    
    with torch.no_grad():
        for x, y, ret in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_rets.extend(ret.cpu().numpy())
            
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    trades = 0
    pnl = 0.0
    cost = CONFIG["TRADE_COST"] * 2 
    
    for p, r in zip(all_preds, all_rets):
        if p == 1: 
            pnl += (r - cost)
            trades += 1
        elif p == 2: 
            pnl += (-r - cost)
            trades += 1
            
    avg_pnl_bps = (pnl / trades * 10000) if trades > 0 else 0
    
    if "1" in report: buy_prec = report['1']['precision']
    else: buy_prec = 0
    if "2" in report: sell_prec = report['2']['precision']
    else: sell_prec = 0
    
    if trades == 0: return 0, "No Trades"
    if buy_prec < threshold and sell_prec < threshold: return 0, f"Low Prec (B:{buy_prec:.2f})"
    
    score = (buy_prec + sell_prec) + (avg_pnl_bps * 0.1)
    msg = f"Score:{score:.2f} | Prec B:{buy_prec:.2f} S:{sell_prec:.2f} | PnL:{avg_pnl_bps:.1f}bps | Trades:{trades}"
    return score, msg

# ==========================================
# 5. 主训练循环
# ==========================================
def main():
    print(f"🚀 启动 CIO 级训练流程 | 设备: {DEVICE}")
    print(f"🎯 目标: {CONFIG['PREDICT_HORIZON']} Bars (30分钟) 趋势跟踪")
    
    forge = AlphaForgeV2(CONFIG)
    try:
        (X_tr, df_tr), (X_val, df_val), (X_te, df_te) = forge.prepare_data()
    except ValueError as e:
        print(e)
        return

    ds_train = LOBDataset(X_tr, df_tr, CONFIG["LOOKBACK"])
    ds_val = LOBDataset(X_val, df_val, CONFIG["LOOKBACK"])
    ds_test = LOBDataset(X_te, df_te, CONFIG["LOOKBACK"])
    
    labels = df_tr["label"].values[CONFIG["LOOKBACK"]:-1]
    class_counts = np.bincount(labels)
    # 防止除零
    if len(class_counts) < 3:
        class_counts = np.array([1, 1, 1]) 
    class_weights = 1. / (class_counts + 1e-6)
    
    # 简单的权重映射
    weights_map = np.zeros(len(labels))
    for i in range(len(class_counts)):
        if i < len(class_counts):
             weights_map[labels == i] = class_weights[i]
             
    sampler = WeightedRandomSampler(weights_map, len(weights_map))
    
    dl_train = DataLoader(ds_train, batch_size=CONFIG["BATCH_SIZE"], sampler=sampler)
    dl_val = DataLoader(ds_val, batch_size=CONFIG["BATCH_SIZE"], shuffle=False)
    
    model = DeepLOBTransformer().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["LR"], weight_decay=1e-4)
    criterion = AlphaFocalLoss(alpha=[1.0, 3.0, 3.0], gamma=2.0)
    
    best_score = -float('inf')
    patience_cnt = 0
    
    for epoch in range(CONFIG["EPOCHS"]):
        model.train()
        train_loss = []
        for x, y, _ in dl_train:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            
        score, msg = validate_cio_standard(model, dl_val)
        print(f"Epoch {epoch+1:02d} | Loss {np.mean(train_loss):.4f} | Val: {msg}")
        
        if score > best_score:
            best_score = score
            patience_cnt = 0
            torch.save(model.state_dict(), "best_cio_model.pth")
            print("   >>> ✅ 新纪录! 模型已保存")
        else:
            patience_cnt += 1
            if patience_cnt >= CONFIG["PATIENCE"]:
                print("⏹️ 早停触发")
                break
                
    print("\n" + "="*50)
    print("🔮 最终实盘演习 (Test Set)")
    if os.path.exists("best_cio_model.pth"):
        model.load_state_dict(torch.load("best_cio_model.pth"))
        dl_test = DataLoader(ds_test, batch_size=CONFIG["BATCH_SIZE"], shuffle=False)
        score, msg = validate_cio_standard(model, dl_test)
        print(f"TEST RESULT: {msg}")
    else:
        print("未保存任何有效模型")
    print("="*50)

if __name__ == "__main__":
    main()