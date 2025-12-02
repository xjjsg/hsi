# -*- coding: utf-8 -*-
"""
文件名: Hybrid_Miner_Final.py
版本: Final (Fixed KeyError: ctx_ret)
功能: 
    1. 修复了重复前缀导致的 KeyError 问题
    2. 集成 5 大类基础知识特征
    3. 双品种融合 (159920 + 513130)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib
import math
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 计算设备: {DEVICE} (Final Mode)")

# ==============================================================================
# 1. 终极特征工程处理器 (Knowledge-Based Processor)
# ==============================================================================
class DualDataProcessor:
    def __init__(self, main_file, aux_file, lookback=60, horizon=20):
        self.main_file = main_file
        self.aux_file = aux_file
        self.lookback = lookback
        self.horizon = horizon
        self.scaler = StandardScaler()

    def _process_single(self, filepath, is_aux=False):
        print(f"  -> 处理数据: {filepath} (辅助: {is_aux})")
        try:
            raw = pd.read_csv(filepath)
            if 'tx_local_time' in raw.columns:
                raw['datetime'] = pd.to_datetime(raw['tx_local_time'], unit='ms')
                if raw['datetime'].dt.tz is None:
                    raw['datetime'] = raw['datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
                df = raw.set_index('datetime').sort_index()
            else:
                df = raw

            # 3s 重采样
            agg_rules = {
                'price': 'last', 'tick_vol': 'sum', 'tick_amt': 'sum', 'tick_vwap': 'mean',
                'premium_rate': 'last', 'sentiment': 'last',
                'bp1':'last', 'bv1':'last', 'sp1':'last', 'sv1':'last'
            }
            for i in range(2, 6):
                if f'bp{i}' in df.columns:
                    agg_rules[f'bp{i}'] = 'last'; agg_rules[f'bv{i}'] = 'last'
                    agg_rules[f'sp{i}'] = 'last'; agg_rules[f'sv{i}'] = 'last'
            
            # 期货数据仅主标的有
            if not is_aux and 'fut_price' in df.columns:
                agg_rules['fut_price'] = 'last'

            df = df.resample('3s').agg(agg_rules).ffill().dropna()

            # === 特征工程核心 (修复: 内部不加前缀，最后统一加) ===
            
            # 1. 基础物理 (Microstructure)
            df['mid'] = (df['bp1'] + df['sp1']) / 2
            df['spread'] = (df['sp1'] - df['bp1']) / df['mid']
            
            # OFI
            df['ofi'] = (df['bv1'] - df['sv1']) / (df['bv1'] + df['sv1'] + 1e-6)
            
            # Depth Slope
            if 'bp5' in df.columns:
                df['bid_slope'] = (df['bp1'] - df['bp5']) / 5
                df['ask_slope'] = (df['sp5'] - df['sp1']) / 5
            
            # WMP Bias
            wmp = (df['bp1'] * df['sv1'] + df['sp1'] * df['bv1']) / (df['bv1'] + df['sv1'] + 1e-6)
            df['wmp_bias'] = (wmp - df['mid']) / df['mid']

            # 2. 基础数学 (Stationarity)
            df['ret'] = np.log(df['mid'] / df['mid'].shift(1)).fillna(0)
            df['vol_chg'] = np.log(df['tick_vol'] + 1).diff().fillna(0)
            
            ma_20 = df['mid'].rolling(20).mean()
            std_20 = df['mid'].rolling(20).std()
            df['z_score'] = (df['mid'] - ma_20) / (std_20 + 1e-6)

            # 3. 基础博弈 (Aggressiveness)
            df['vwap_bias'] = (df['tick_vwap'] - df['mid']) / df['mid']
            depth_amt = (df['bv1'] * df['bp1']) + (df['sv1'] * df['sp1'])
            df['trade_intensity'] = df['tick_amt'] / (depth_amt + 1e-6)

            # 4. 情绪与期货 (仅主标的计算期货)
            # sentiment 是都有的
            
            if not is_aux and 'fut_price' in df.columns:
                df['fut_basis'] = df['fut_price'] / df['price'] - 1
                df['fut_lead'] = df['fut_price'].pct_change() - df['price'].pct_change()

            # === 关键修复: 最后统一重命名 ===
            if is_aux:
                df = df.add_prefix('ctx_')
                
            return df
        except Exception as e:
            print(f"处理失败 {filepath}: {e}")
            return None

    def load_and_merge(self):
        print("⚡ [Step 1] 特征工程与融合...")
        df_main = self._process_single(self.main_file, is_aux=False)
        df_aux = self._process_single(self.aux_file, is_aux=True)
        
        if df_main is None: return None, []

        # 合并
        if df_aux is not None:
            # Inner Join 对齐时间
            df = df_main.join(df_aux, how='inner')
            
            # === 5. 基础关系知识 (Cross-Sectional) ===
            # 现在 df 中有 'ret' 和 'ctx_ret'，可以直接相减
            df['rel_ret'] = df['ret'] - df['ctx_ret']
            df['sent_gap'] = df['sentiment'] - df['ctx_sentiment']
            df['ofi_diff'] = df['ofi'] - df['ctx_ofi']
        else:
            df = df_main

        # === 6. 基础时间知识 ===
        minutes = df.index.hour * 60 + df.index.minute
        df['time_progress'] = (minutes - 570) / (900 - 570)
        
        # 目标: 未来波动率 (使用主标的 ret)
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=self.horizon)
        df['target_vol'] = df['ret'].rolling(window=indexer).std() * 10000
        
        df = df.replace([np.inf, -np.inf], 0).fillna(0).dropna()
        
        # 筛选特征
        exclude = ['tx_server_time', 'target_vol', 'price', 'bp1', 'sp1', 'ctx_price', 'ctx_bp1', 'ctx_sp1']
        feat_cols = [c for c in df.columns if c not in exclude and df[c].dtype != 'object']
        
        self.feature_cols = feat_cols
        return df, feat_cols

    def get_tensors(self, df, feature_cols, fit_scaler=False):
        data = df[feature_cols].values
        target = df['target_vol'].values
        
        if fit_scaler: data = self.scaler.fit_transform(data)
        else: data = self.scaler.transform(data)
            
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i-self.lookback : i])
            y.append(target[i])
        return np.array(X), np.array(y)

# ==============================================================================
# 2. 模型架构 (AlphaNet + TCN + Transformer)
# ==============================================================================
class AlphaLayer(nn.Module):
    def __init__(self, input_dim, window=20):
        super(AlphaLayer, self).__init__()
        self.window = window
        self.pool = nn.AvgPool1d(kernel_size=window, stride=1, padding=0)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        pad = torch.zeros(x.shape[0], x.shape[1], self.window-1).to(x.device)
        x_pad = torch.cat([pad, x], dim=2)
        mean = self.pool(x_pad)
        x2_pad = torch.cat([pad, x**2], dim=2)
        mean_sq = self.pool(x2_pad)
        std = torch.sqrt(torch.clamp(mean_sq - mean**2, min=1e-6))
        return torch.cat([x, mean, std], dim=1).permute(0, 2, 1)

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        self.relu1 = nn.GELU()
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        self.relu2 = nn.GELU()
        self.chomp = padding
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
    def forward(self, x):
        out = self.relu1(self.conv1(x)[:, :, :-self.chomp])
        out = self.relu2(self.conv2(out)[:, :, :-self.chomp])
        res = x if self.downsample is None else self.downsample(x)
        return out + res

class HybridMinerNet(nn.Module):
    def __init__(self, input_dim, d_model=128, n_factors=128):
        super(HybridMinerNet, self).__init__()
        self.alpha_layer = AlphaLayer(input_dim, window=20)
        self.tcn = TemporalBlock(input_dim*3, d_model, kernel_size=3, dilation=1)
        self.pos_encoder = self._make_pos_encoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=256, dropout=0.1)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=3)
        self.factor_head = nn.Sequential(nn.Linear(d_model, 256), nn.GELU(), nn.Linear(256, n_factors), nn.Tanh())
        self.predictor = nn.Linear(n_factors, 1)

    def _make_pos_encoding(self, d_model, max_len=5000):
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return nn.Parameter(pe.unsqueeze(0).transpose(0, 1), requires_grad=False)

    def forward(self, x):
        x = self.alpha_layer(x)
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = x.permute(2, 0, 1)
        x = x + self.pos_encoder[:x.size(0), :]
        x = self.transformer(x)
        last_step = x[-1, :, :]
        factors = self.factor_head(last_step)
        pred = self.predictor(factors)
        return pred, factors

# ==============================================================================
# 3. 训练与挖掘
# ==============================================================================
def run_final_miner():
    MAIN = 'sz159920.csv'
    AUX = 'sh513130.csv'
    
    # 1. 智能数据处理
    proc = DualDataProcessor(MAIN, AUX, lookback=60, horizon=20)
    df, feat_cols = proc.load_and_merge()
    if df is None: return

    print(f"最终输入特征数: {len(feat_cols)}")
    print(f"包含: OFI, Spread, WMP Bias, Cross-RS, Sentiment Gap 等")

    split = int(len(df) * 0.8)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]
    
    X_train, y_train = proc.get_tensors(train_df, feat_cols, fit_scaler=True)
    X_test, y_test = proc.get_tensors(test_df, feat_cols, fit_scaler=False)
    
    joblib.dump(proc.scaler, 'final_scaler.pkl')
    
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=128, shuffle=True
    )
    
    # 2. 模型
    model = HybridMinerNet(input_dim=len(feat_cols), d_model=128, n_factors=128).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # 3. 训练
    print("\n🚀 开始训练 Final Miner (Knowledge Enhanced)...")
    model.train()
    for epoch in range(15):
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred, _ = model(X)
            loss = criterion(pred.squeeze(), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss {total_loss/len(train_loader):.6f}")
        
    torch.save(model.state_dict(), 'final_miner_model.pth')
    
    # 4. 导出因子
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
        _, factors = model(X_test_tensor)
        
        cols = [f'Latent_{i:03d}' for i in range(128)]
        res_df = pd.DataFrame(factors.cpu().numpy(), columns=cols, index=test_df.index[60:])
        res_df['target_vol'] = y_test
        
        # 简单验证
        ic_scores = [abs(res_df[c].corr(res_df['target_vol'])) for c in cols]
        print(f"平均 Top 10 因子 IC: {np.mean(sorted(ic_scores)[-10:]):.4f}")
        
        res_df.to_csv("final_mined_factors.csv")
        print("✅ 因子表已导出 -> final_mined_factors.csv")

if __name__ == "__main__":
    run_final_miner()