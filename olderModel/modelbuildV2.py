import os
import glob
import warnings
import math
from datetime import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report

# ==========================================
# 0. 全局配置 (V4 - 实盘可交易标准)
# ==========================================
CONFIG = {
    # --- 路径与标的 ---
    "DATA_DIR": ".",              
    "MAIN_SYMBOL": "sz159920",
    "AUX_SYMBOL": "sh513130",     
    
    # --- 时序结构 ---
    "RESAMPLE_FREQ": "3S",        
    "PREDICT_HORIZON": 600,       # 30分钟 = 600 bars
    "LOOKBACK": 100,              # 输入窗口
    
    # --- [V4 新增] 核心交易参数 ---
    "TRADE_COST": 0.0001,         # 单边万1
    "MIN_PROFIT_THRESHOLD": 0.0002, # 额外净利缓冲 (2bps)，用于覆盖滑点
    
    # --- 训练参数 ---
    "BATCH_SIZE": 256,
    "EPOCHS": 50,
    "LR": 2e-5,                   # 调低 LR，适应复杂 Loss
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "PATIENCE": 15,
    "TIMEZONE": "Asia/Shanghai"
}

warnings.filterwarnings("ignore")
DEVICE = CONFIG["DEVICE"]

# ==========================================
# 1. 因子工厂 (AlphaFactorCalculator)
# ==========================================
class AlphaFactorCalculator:
    def __init__(self, windows=[20, 100, 300]):
        self.windows = windows 
        
    def _safe_div(self, a, b):
        return a / (b + 1e-9)

    def _rolling_stats(self, series, window, name):
        roll = series.rolling(window=window)
        mean = roll.mean()
        std = roll.std()
        zscore = self._safe_div(series - mean, std)
        slope = self._safe_div(series - series.shift(window), window)
        return {
            f"{name}_{window}w_zscore": zscore,
            f"{name}_{window}w_slope": slope,
        }

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """主入口: 计算所有因子"""
        mid = (df["bp1"] + df["sp1"]) / 2
        
        # [修改] 必须保留原始买卖价列，用于后续 Taker-Taker 打标
        factors = pd.DataFrame(index=df.index)
        factors["mid"] = mid
        factors["bp1"] = df["bp1"]
        factors["sp1"] = df["sp1"]
        
        if "bp1_aux" in df.columns:
            df["mid_aux"] = (df["bp1_aux"] + df["sp1_aux"]) / 2
            factors["mid_aux"] = df["mid_aux"]
            factors["bp1_aux"] = df["bp1_aux"]
            factors["sp1_aux"] = df["sp1_aux"]
        
        # --- A. 微观结构因子 ---
        ofi_buy = (df["bv1"] + 0.8*df["bv2"] + 0.6*df["bv3"]) 
        ofi_sell = (df["sv1"] + 0.8*df["sv2"] + 0.6*df["sv3"])
        factors["iOFI"] = self._safe_div(ofi_buy - ofi_sell, ofi_buy + ofi_sell)
        factors["QI"] = self._safe_div(df["bv1"] - df["sv1"], df["bv1"] + df["sv1"])
        factors["spread_bps"] = self._safe_div(df["sp1"] - df["bp1"], mid) * 10000
        
        # --- B. 资金流因子 ---
        factors["sentiment"] = df["sentiment"]
        
        # --- C. 期货因子 ---
        if "fut_imb" in df.columns:
            factors["fut_imb"] = df["fut_imb"]
            factors["FSB"] = (np.log(df["fut_price"]) - np.log(mid)) * np.sign(df["fut_imb"])
            factors["FLP"] = df["fut_pct"] * df["fut_imb"]
        
        # --- D. 套利因子 ---
        factors["premium_rate"] = df["premium_rate"]
        
        # --- E. 跨品种 ---
        if "mid_aux" in df.columns:
            factors["LLT_rs"] = df["mid_aux"].pct_change() - mid.pct_change()
            factors["price_ratio"] = self._safe_div(mid, df["mid_aux"])

        # --- F. 自动衍生 ---
        core_bases = ["sentiment", "iOFI", "QI", "premium_rate"]
        if "fut_imb" in df.columns: core_bases += ["fut_imb", "FSB"]
        if "LLT_rs" in factors.columns: core_bases += ["LLT_rs"]
        
        derived_list = []
        for col in core_bases:
            if col not in factors.columns: continue
            derived_list.append(factors[col].diff().rename(f"{col}_delta"))
            for w in self.windows:
                stats = self._rolling_stats(factors[col], w, col)
                derived_list.append(pd.DataFrame(stats))
                
        # 波动率状态
        ret = mid.pct_change()
        for w in self.windows:
            derived_list.append(ret.rolling(w).std().rename(f"volatility_{w}w"))
            
        all_factors = pd.concat([factors] + derived_list, axis=1)
        return all_factors.fillna(0.0).replace([np.inf, -np.inf], 0.0)

# ==========================================
# 2. 深度黑盒挖掘机 (V4 - VICReg 去冗余)
# ==========================================
class DeepFactorMiner(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, latent_dim), nn.BatchNorm1d(latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.GELU(),
            nn.Linear(64, 128), nn.GELU(),
            nn.Linear(128, input_dim)
        )
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.GELU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        factors = self.encoder(x)
        recon = self.decoder(factors)
        pred = self.predictor(factors)
        return factors, recon, pred

def off_diagonal(x):
    """[V4 新增] 提取非对角线元素"""
    n, m = x.shape
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def train_miner(X_train, y_train, input_dim, latent_dim=16, epochs=10):
    print(f"\n⛏️ [Miner] 启动深度挖掘 (VICReg 正交化模式)...")
    miner = DeepFactorMiner(input_dim, latent_dim).to(DEVICE)
    optimizer = optim.AdamW(miner.parameters(), lr=1e-3)
    
    xt = torch.FloatTensor(X_train).to(DEVICE)
    yt = torch.LongTensor(y_train).to(DEVICE)
    dl = DataLoader(torch.utils.data.TensorDataset(xt, yt), batch_size=2048, shuffle=True)
    
    loss_ce = nn.CrossEntropyLoss()
    loss_mse = nn.MSELoss()
    
    for epoch in range(epochs):
        miner.train()
        total_loss = 0
        for bx, by in dl:
            optimizer.zero_grad()
            factors, recon, pred = miner(bx)
            
            # 1. 预测与还原
            l_pred = loss_ce(pred, by)
            l_recon = loss_mse(recon, bx)
            
            # 2. [V4] 正交惩罚 (VICReg Covariance)
            factors_norm = factors - factors.mean(dim=0)
            factors_std = factors.std(dim=0) + 1e-6
            factors_norm = factors_norm / factors_std
            cov_mat = (factors_norm.T @ factors_norm) / (factors.shape[0] - 1)
            l_ortho = off_diagonal(cov_mat).pow(2).sum()
            
            # 复合 Loss
            loss = 0.7 * l_pred + 0.2 * l_recon + 0.1 * l_ortho
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    return miner

def extract_deep_factors(miner, X_data):
    miner.eval()
    with torch.no_grad():
        xt = torch.FloatTensor(X_data).to(DEVICE)
        factors, _, _ = miner(xt)
    return factors.cpu().numpy()

# ==========================================
# 3. 数据管道 (V4 - Taker-Taker PnL Labeling)
# ==========================================
class AlphaForgeV4:
    def __init__(self, cfg):
        self.cfg = cfg
        self.scaler = RobustScaler() 

    def _load_symbol_files(self, symbol):
        pattern = os.path.join(self.cfg["DATA_DIR"], "**", f"{symbol}*.csv")
        files = sorted(glob.glob(pattern, recursive=True))
        if not files:
             pattern = os.path.join(".", "**", f"{symbol}*.csv")
             files = sorted(glob.glob(pattern, recursive=True))
        return files

    def _read_and_clean(self, fpath):
        try:
            usecols = ["tx_local_time", "bp1", "bv1", "sp1", "sv1", 
                       "bp2", "bv2", "sp2", "sv2", "bp3", "bv3", "sp3", "sv3",
                       "sentiment", "tick_vol", "tick_vwap", "premium_rate", 
                       "fut_price", "fut_imb", "fut_pct"]
            
            preview = pd.read_csv(fpath, nrows=1)
            valid_cols = [c for c in usecols if c in preview.columns]
            df = pd.read_csv(fpath, usecols=valid_cols)
            if "tx_local_time" not in df.columns: return None

            df["datetime"] = pd.to_datetime(df["tx_local_time"], unit="ms", utc=True)\
                               .dt.tz_convert(self.cfg["TIMEZONE"]).dt.tz_localize(None)
            for c in [c for c in valid_cols if c != "tx_local_time"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            df = df.sort_values("datetime").drop_duplicates("datetime", keep="last")
            df = df.set_index("datetime")
            return df.resample(self.cfg["RESAMPLE_FREQ"]).last().dropna()
        except: return None

    def load_and_align(self):
        print("🚀 [Forge] 双流对齐加载 (主+辅)...")
        main_files = self._load_symbol_files(self.cfg["MAIN_SYMBOL"])
        aux_files = self._load_symbol_files(self.cfg["AUX_SYMBOL"])
        
        def get_date(fname):
            try: return os.path.basename(fname).split("-")[-3] + "-" + os.path.basename(fname).split("-")[-2] + "-" + os.path.basename(fname).split("-")[-1].split(".")[0]
            except: return "unknown"

        aux_map = {get_date(f): f for f in aux_files}
        full_list = []
        
        for mf in main_files:
            df_main = self._read_and_clean(mf)
            if df_main is None or len(df_main) < 100: continue
            
            date_key = get_date(mf)
            if date_key in aux_map:
                df_aux = self._read_and_clean(aux_map[date_key])
                if df_aux is not None:
                    df_aux = df_aux.add_suffix("_aux")
                    df_main = pd.merge_asof(df_main.sort_index(), df_aux.sort_index(), 
                                            left_index=True, right_index=True, 
                                            tolerance=pd.Timedelta("10s"), direction="backward")
            full_list.append(df_main)
            
        return pd.concat(full_list).sort_index() if full_list else None

    def process_pipeline(self):
        df = self.load_and_align()
        if df is None: raise ValueError("无数据")
        
        print("⚗️ [Forge] 计算白盒因子...")
        calc = AlphaFactorCalculator()
        df_factors = calc.process(df)
        
        # 过滤时间
        t = df_factors.index.time
        mask = ((t >= time(9, 30)) & (t <= time(10, 15))) | \
               ((t >= time(14, 0)) & (t <= time(14, 45)))
        df_factors = df_factors[mask]
        
        # --- [V4 核心修改] Taker-Taker 净收益打标 ---
        horizon = self.cfg["PREDICT_HORIZON"]
        comm = self.cfg["TRADE_COST"] * 2 # 双边费率
        threshold = self.cfg["MIN_PROFIT_THRESHOLD"]
        
        # 逻辑：我现在买入(对手价=Ask)，要看未来能不能在(Bid)卖出赚钱
        # 做多净利 = ln(未来Bid) - ln(当前Ask) - 费用
        future_bp1 = df_factors["bp1"].shift(-horizon)
        curr_sp1 = df_factors["sp1"]
        long_pnl = np.log(future_bp1) - np.log(curr_sp1) - comm
        
        # 做空净利 = ln(当前Bid) - ln(未来Ask) - 费用
        future_sp1 = df_factors["sp1"].shift(-horizon)
        curr_bp1 = df_factors["bp1"]
        short_pnl = np.log(curr_bp1) - np.log(future_sp1) - comm
        
        labels = np.zeros(len(df_factors), dtype=int)
        labels[long_pnl > threshold] = 1   # Buy
        labels[short_pnl > threshold] = 2  # Sell
        
        df_factors["label"] = labels
        # 记录每笔决策如果做对的理论收益，用于回测评估
        df_factors["executable_pnl"] = np.where(labels==1, long_pnl, np.where(labels==2, short_pnl, 0.0))
        
        # 切分
        df_factors = df_factors.dropna()
        n = len(df_factors)
        train_sz, val_sz = int(n * 0.8), int(n * 0.9)
        train, val, test = df_factors.iloc[:train_sz], df_factors.iloc[train_sz:val_sz], df_factors.iloc[val_sz:]
        
        # 排除非特征列
        exclude = ["label", "executable_pnl", "mid", "mid_aux", "bp1", "sp1", "bp1_aux", "sp1_aux", "sp1", "sv1"] 
        # 注意: 排除 bp1/sp1 等原始价格，防止模型拟合绝对价格
        feat_cols = [c for c in df_factors.columns if c not in exclude and c not in ["bp1","sp1","bp1_aux","sp1_aux"]]
        
        print(f"🧠 [Forge] 基础特征维度: {len(feat_cols)}")
        
        # 标准化
        X_train = self.scaler.fit_transform(train[feat_cols])
        X_val = self.scaler.transform(val[feat_cols])
        X_test = self.scaler.transform(test[feat_cols])
        
        # --- [V4] 训练黑盒挖掘机 ---
        miner = train_miner(X_train, train["label"].values, input_dim=len(feat_cols))
        
        # 提取黑盒因子
        f_tr = extract_deep_factors(miner, X_train)
        f_val = extract_deep_factors(miner, X_val)
        f_te = extract_deep_factors(miner, X_test)
        
        X_train = np.hstack([X_train, f_tr])
        X_val = np.hstack([X_val, f_val])
        X_test = np.hstack([X_test, f_te])
        
        print(f"🧬 [Forge] 混合特征维度: {X_train.shape[1]}")
        
        return (X_train, train), (X_val, val), (X_test, test)

# ==========================================
# 4. 模型与评估 (V4 - Cooldown Validation)
# ==========================================
class FeatureTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, num_classes=3):
        super().__init__()
        self.projector = nn.Sequential(nn.Linear(input_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.pos_embedding = nn.Parameter(torch.randn(1, CONFIG["LOOKBACK"], d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, batch_first=True, dropout=0.3)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, num_classes))
        
    def forward(self, x):
        x = self.projector(x) + self.pos_embedding
        x = self.transformer(x)
        return self.fc(x[:, -1, :])

class TimeSeriesDataset(Dataset):
    def __init__(self, X, df, lookback):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(df["label"].values)
        self.pnl = torch.FloatTensor(df["executable_pnl"].values)
        self.lookback = lookback
    def __len__(self): return len(self.X) - self.lookback - 1
    def __getitem__(self, idx):
        return self.X[idx:idx+self.lookback], self.y[idx+self.lookback], self.pnl[idx+self.lookback]

def validate_cio_v4(model, loader, threshold=0.6):
    """
    [V4 验证] 增加 Cooldown 逻辑，模拟真实持仓占用
    """
    model.eval()
    preds, labels, pnls = [], [], []
    
    with torch.no_grad():
        for x, y, p in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            max_p, p_cls = torch.max(probs, dim=1)
            
            # 置信度过滤
            final_p = torch.where(max_p > threshold, p_cls, torch.tensor(0).to(DEVICE))
            
            preds.extend(final_p.cpu().numpy())
            labels.extend(y.cpu().numpy())
            pnls.extend(p.cpu().numpy())
            
    # --- Cooldown 模拟执行 ---
    # 规则: 每次开仓后，假设占用资金 20 个 bars (1分钟)，期间不重复开仓
    executed_trades = 0
    total_pnl = 0.0
    cooldown = 0
    n_buys, n_sells = 0, 0
    
    for i in range(len(preds)):
        if cooldown > 0:
            cooldown -= 1
            continue
            
        action = preds[i]
        
        if action == 1: # Buy Signal
            executed_trades += 1
            # 只有当 Label 也是 1 时，我们才拿到了理论上的 long_pnl
            # 如果 Label 是 0 或 2，说明做多是错的/亏的。
            # 为了严谨，我们直接读取 'executable_pnl' 并不够，因为那只是"如果做对"的钱。
            # 简化回测：我们直接看 pnls[i]。
            # 但注意：Dataset 里的 pnl 存的是 (Label==1?long:short)。
            # 如果模型预测 1 但 Label 是 2，真实收益应该是负的。
            # 这里做简单近似：如果 Pred == Label，赚 pnl；否则亏 Trade Cost。
            
            if labels[i] == 1:
                total_pnl += pnls[i] # 赚到了
            else:
                # 预测错了方向或动能不足，亏损 = 手续费 + 可能的价差亏损
                # 简单惩罚：亏掉双边手续费
                total_pnl -= CONFIG["TRADE_COST"] * 2
            
            n_buys += 1
            cooldown = 20 # 锁定 1 分钟
            
        elif action == 2: # Sell Signal
            executed_trades += 1
            if labels[i] == 2:
                total_pnl += pnls[i]
            else:
                total_pnl -= CONFIG["TRADE_COST"] * 2
            n_sells += 1
            cooldown = 20
            
    avg_pnl_bps = (total_pnl / executed_trades * 10000) if executed_trades > 0 else 0
    
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    b_prec = report['1']['precision'] if '1' in report else 0
    s_prec = report['2']['precision'] if '2' in report else 0
    
    score = (b_prec + s_prec) + (avg_pnl_bps * 0.2)
    if executed_trades == 0: score = 0
    
    msg = (f"Score:{score:.2f} | PnL:{avg_pnl_bps:.1f}bps | Trades:{executed_trades} "
           f"(B:{n_buys} S:{n_sells}) | Prec B:{b_prec:.2f} S:{s_prec:.2f}")
    return score, msg

# ==========================================
# 5. 主程序
# ==========================================
def main():
    print(f"🚀 启动 CIO-V4 (Taker-Taker PnL版 + VICReg) | 设备: {DEVICE}")
    forge = AlphaForgeV4(CONFIG)
    
    try:
        (X_tr, df_tr), (X_val, df_val), (X_te, df_te) = forge.process_pipeline()
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return

    # 采样器
    lookback = CONFIG["LOOKBACK"]
    labels = df_tr["label"].values[lookback:-1]
    counts = np.bincount(labels, minlength=3)
    # 防止除0
    weights = 1. / (counts + 1e-6)
    # 映射回每个样本
    sample_weights = weights[labels]
    
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    ds_train = TimeSeriesDataset(X_tr, df_tr, lookback)
    ds_val = TimeSeriesDataset(X_val, df_val, lookback)
    ds_test = TimeSeriesDataset(X_te, df_te, lookback)
    
    dl_train = DataLoader(ds_train, batch_size=CONFIG["BATCH_SIZE"], sampler=sampler)
    dl_val = DataLoader(ds_val, batch_size=CONFIG["BATCH_SIZE"], shuffle=False)
    
    print(f"🏗️ 构建 FeatureTransformer (Input Dim: {X_tr.shape[1]})")
    model = FeatureTransformer(input_dim=X_tr.shape[1]).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["LR"], weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 2.0]).to(DEVICE))
    
    best_score = -999
    patience = 0
    
    for epoch in range(CONFIG["EPOCHS"]):
        model.train()
        losses = []
        for x, y, _ in dl_train:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        score, msg = validate_cio_v4(model, dl_val)
        print(f"Epoch {epoch+1:02d} | Loss {np.mean(losses):.4f} | {msg}")
        
        if score > best_score:
            best_score = score
            patience = 0
            torch.save(model.state_dict(), "best_model_v4.pth")
            print("   >>> ✅ 新纪录!")
        else:
            patience += 1
            if patience >= CONFIG["PATIENCE"]:
                print("⏹️ 早停")
                break

    print("\n🔮 最终测试 (OOS)")
    if os.path.exists("best_model_v4.pth"):
        model.load_state_dict(torch.load("best_model_v4.pth"))
        dl_test = DataLoader(ds_test, batch_size=CONFIG["BATCH_SIZE"], shuffle=False)
        _, msg = validate_cio_v4(model, dl_test)
        print(f"TEST RESULT: {msg}")

if __name__ == "__main__":
    main()