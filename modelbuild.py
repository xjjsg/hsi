# -*- coding: utf-8 -*-
"""
Alpha System v8.1 - Final Stable Edition
----------------------------------------
功能全集：
1. [修复] 彻底解决 classification_report 的 KeyError 问题。
2. [参数] 适配单边万1成本，训练门槛 0.0015。
3. [模型] 集成 SE-Block + Inception + Temporal Attention。
4. [风控] 资金管理回测，置信度 > 0.7 才开仓。

@Ver: 8.1 Stable
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

warnings.filterwarnings('ignore')

# ==========================================
# 1. 全局配置 (Configuration)
# ==========================================
CONFIG = {
    # --- 路径配置 ---
    'DATA_DIR': './data',          
    'MAIN_SYMBOL': 'sz159920',     
    'AUX_SYMBOL': 'sh513130',      
    
    # --- 因子与数据 ---
    'RESAMPLE_FREQ': '3S',         # 3秒重采样
    'PREDICT_HORIZON': 60,         # 预测窗口 180秒
    'LOOKBACK': 60,                # 回看窗口 180秒
    
    # [参数调整] 训练打标门槛 0.0015
    # 成本极低(0.0002)，0.0015 的波动足以产生丰厚利润，且样本量比 0.002 多
    'COST_THRESHOLD': 0.0015,   
    
    # --- 资金管理回测 ---
    'TRADE_COST': 0.0001,          # [关键] 单边万1成本
    'INITIAL_CAPITAL': 20000,      # 初始本金 2万
    'CONF_THRESHOLD': 0.70,        # [关键] 70% 把握即开仓
    'MAX_POSITION': 0.8,           # 最大仓位 80%
    
    # --- 训练参数 ---
    'BATCH_SIZE': 256,             # 显存允许可调大
    'EPOCHS': 50,
    'LR': 1e-4,
    'WEIGHT_DECAY': 1e-4,          # 强正则化防止过拟合
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'PATIENCE': 20,                # 早停耐心
    'WARMUP_EPOCHS': 10,           # 热身轮数
}

# ==========================================
# 2. 数据工厂：Alpha Forge
# ==========================================
class AlphaForge:
    def __init__(self, cfg):
        self.cfg = cfg
        self.weights = np.array([1.0, 0.8, 0.6, 0.4, 0.2])

    def load_and_split(self):
        print(f"🚀 [AlphaForge] 启动... 扫描: {self.cfg['DATA_DIR']}")
        pairs = self._match_files()
        pairs.sort(key=lambda x: x[0])
        
        if len(pairs) < 2: raise ValueError("数据不足2天")
        
        train_pairs = pairs[:-1]
        test_pair = pairs[-1]
        
        print(f"📅 训练集: {train_pairs[0][0]} ~ {train_pairs[-1][0]}")
        print(f"📅 测试集: {test_pair[0]}")
        
        train_df = self._process_batch(train_pairs)
        test_df = self._process_batch([test_pair])
        
        return train_df, test_df

    def _process_batch(self, pairs):
        dfs = []
        for date, mf, af in pairs:
            try:
                df = self._load_pair(mf, af, date)
                if df is None or len(df) < 200: continue
                df = self._calc_factors(df)
                df = self._make_labels(df)
                df = df.replace([np.inf, -np.inf], np.nan).dropna()
                dfs.append(df)
            except Exception as e:
                print(f"⚠️ 跳过 {date}: {e}")
        return pd.concat(dfs).sort_index() if dfs else pd.DataFrame()

    def _match_files(self):
        m_pattern = os.path.join(self.cfg['DATA_DIR'], "**", f"{self.cfg['MAIN_SYMBOL']}*.csv")
        a_pattern = os.path.join(self.cfg['DATA_DIR'], "**", f"{self.cfg['AUX_SYMBOL']}*.csv")
        m_files = glob.glob(m_pattern, recursive=True)
        a_files = glob.glob(a_pattern, recursive=True)
        
        def get_date(p):
            try:
                base = os.path.basename(p)
                parts = base.replace('.csv','').split('-')
                if len(parts) >= 3: return f"{parts[-3]}-{parts[-2]}-{parts[-1]}"
            except: pass
            return None

        m_map = {get_date(f): f for f in m_files if get_date(f)}
        a_map = {get_date(f): f for f in a_files if get_date(f)}
        common = sorted(list(set(m_map.keys()) & set(a_map.keys())))
        return [(d, m_map[d], a_map[d]) for d in common]

    def _load_pair(self, m_path, a_path, date_str):
        def _read(p):
            d = pd.read_csv(p)
            d['datetime'] = pd.to_datetime(date_str + ' ' + d['tx_server_time'])
            return d.set_index('datetime').sort_index().groupby(level=0).last()
        
        df_m, df_a = _read(m_path), _read(a_path)
        
        agg = {
            'price': 'last', 'tick_vol': 'sum',
            'bp1': 'last', 'sp1': 'last', 'bp2': 'last', 'sp2': 'last', 
            'bp3': 'last', 'sp3': 'last', 'bp4': 'last', 'sp4': 'last', 
            'bp5': 'last', 'sp5': 'last', 'bv1': 'last', 'sv1': 'last',
            'bv2': 'last', 'sv2': 'last', 'bv3': 'last', 'sv3': 'last',
            'bv4': 'last', 'sv4': 'last', 'bv5': 'last', 'sv5': 'last',
        }
        for c in ['index_price', 'fut_price', 'fut_imb']:
            if c in df_m.columns: agg[c] = 'last'
            
        df_m = df_m.resample(self.cfg['RESAMPLE_FREQ']).agg(agg)
        df_a = df_a.resample(self.cfg['RESAMPLE_FREQ']).agg({'price': 'last', 'tick_vol': 'sum'})
        df_a.columns = ['peer_price', 'peer_vol']
        return df_m.join(df_a, how='inner')

    def _calc_factors(self, df):
        # 1. Meta Factors
        sec = df.index.hour * 3600 + df.index.minute * 60 + df.index.second
        df['meta_time'] = np.clip(np.where(sec <= 41400, (sec-34200)/14400, 0.5+(sec-46800)/14400), 0, 1)
        
        # 2. Micro Factors
        mid = (df['bp1'] + df['sp1']) / 2
        df['mid'] = mid
        safe_mid = mid.replace(0, np.nan).fillna(method='ffill')
        
        wb = sum(df[f'bv{i}']*self.weights[i-1] for i in range(1,6))
        wa = sum(df[f'sv{i}']*self.weights[i-1] for i in range(1,6))
        df['feat_micro_pressure'] = (wb - wa) / (wb + wa + 1e-8)
        
        # 3. Oracle Factors
        if 'index_price' in df.columns:
            df['feat_oracle_basis'] = (df['index_price'] - safe_mid) / safe_mid
            df['feat_oracle_idx_mom'] = df['index_price'].pct_change(2)
        if 'fut_price' in df.columns:
            df['feat_oracle_fut_lead'] = df['fut_price'].pct_change()
            
        # 4. Peer Factors
        df['feat_peer_diff'] = df['price'].pct_change() - df['peer_price'].pct_change()
        return df

    def _make_labels(self, df):
        """Triple Barrier Method"""
        mid = df['mid']
        horizon = self.cfg['PREDICT_HORIZON']
        threshold = self.cfg['COST_THRESHOLD']
        
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=horizon)
        future_max = mid.rolling(window=indexer).max()
        future_min = mid.rolling(window=indexer).min()
        
        max_ret = future_max / mid - 1
        min_ret = future_min / mid - 1
        
        labels = np.zeros(len(df))
        mask_buy = max_ret > threshold
        mask_sell = min_ret < -threshold
        
        labels[mask_buy] = 1
        labels[mask_sell] = 2
        
        conflict = mask_buy & mask_sell
        if conflict.any():
            c_max = max_ret[conflict]
            c_min = min_ret[conflict].abs()
            labels[conflict] = np.where(c_max > c_min, 1, 2)
            
        df['label'] = labels
        # 保留真实未来收益 (保守回测用)
        df['real_future_ret'] = mid.shift(-horizon) / mid - 1
        return df

# ==========================================
# 3. 高级模型组件 (Attention & SE)
# ==========================================
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class InceptionBlock(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(in_chan, out_chan, 1), nn.LeakyReLU(0.01), nn.BatchNorm2d(out_chan))
        self.b2 = nn.Sequential(nn.Conv2d(in_chan, out_chan, 1), nn.LeakyReLU(0.01), 
                                nn.Conv2d(out_chan, out_chan, (3,1), padding=(1,0)), nn.LeakyReLU(0.01), nn.BatchNorm2d(out_chan))
        self.b3 = nn.Sequential(nn.Conv2d(in_chan, out_chan, 1), nn.LeakyReLU(0.01),
                                nn.Conv2d(out_chan, out_chan, (5,1), padding=(2,0)), nn.LeakyReLU(0.01), nn.BatchNorm2d(out_chan))
        self.b4 = nn.Sequential(nn.MaxPool2d((3,1), stride=1, padding=(1,0)),
                                nn.Conv2d(in_chan, out_chan, 1), nn.LeakyReLU(0.01), nn.BatchNorm2d(out_chan))
        self.se = SEBlock(out_chan * 4)
    def forward(self, x):
        out = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        return self.se(out)

class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.scale = float(hidden_size) ** 0.5

    def forward(self, lstm_output):
        last_step = lstm_output[:, -1, :].unsqueeze(1)
        scores = torch.bmm(self.query(last_step), self.key(lstm_output).transpose(1, 2)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights, self.value(lstm_output))
        return context.squeeze(1)

class HybridDeepLOB(nn.Module):
    def __init__(self, num_expert):
        super().__init__()
        c_chan = 32
        m_hid = 64
        l_hid = 128
        
        # A. Visual Stream (Deep Compression)
        self.compress = nn.Sequential(
            nn.Conv2d(1, c_chan, (1, 2), stride=(1, 2)), nn.LeakyReLU(), nn.BatchNorm2d(c_chan),
            nn.Conv2d(c_chan, c_chan, (4, 1), padding='same'), nn.LeakyReLU(), nn.BatchNorm2d(c_chan),
            nn.Conv2d(c_chan, c_chan, (1, 2), stride=(1, 2)), nn.LeakyReLU(), nn.BatchNorm2d(c_chan),
            nn.Conv2d(c_chan, c_chan, (1, 5), stride=(1, 5)), nn.LeakyReLU(), nn.BatchNorm2d(c_chan),
        )
        self.inception1 = InceptionBlock(c_chan, c_chan)
        self.inception2 = InceptionBlock(128, 64)
        
        # B. Expert Stream
        self.expert = nn.Sequential(
            nn.Linear(num_expert, m_hid), nn.LeakyReLU(), nn.BatchNorm1d(m_hid), nn.Dropout(0.2)
        )
        
        # C. Fusion & Temporal
        fusion_dim = 256 + m_hid
        self.lstm = nn.LSTM(fusion_dim, l_hid, num_layers=2, batch_first=True, dropout=0.5)
        self.attention = TemporalAttention(l_hid)
        
        self.dropout = nn.Dropout(0.5)
        self.head = nn.Sequential(
            nn.Linear(l_hid, 64), nn.LeakyReLU(), nn.Linear(64, 3)
        )

    def forward(self, x_lob, x_exp):
        x = x_lob.unsqueeze(1)
        feat = self.compress(x)
        feat = self.inception1(feat)
        feat = self.inception2(feat)
        feat = feat.squeeze(-1).permute(0, 2, 1)
        
        if feat.shape[1] != x_exp.shape[1]:
            feat = feat.permute(0, 2, 1)
            feat = nn.functional.adaptive_avg_pool1d(feat, x_exp.shape[1])
            feat = feat.permute(0, 2, 1)
            
        B, T, F = x_exp.shape
        exp = self.expert(x_exp.reshape(-1, F)).reshape(B, T, -1)
        
        combined = torch.cat([feat, exp], dim=2)
        lstm_out, _ = self.lstm(combined)
        context = self.attention(lstm_out)
        return self.head(self.dropout(context))

# ==========================================
# 4. 训练引擎
# ==========================================
class ETFDataset(Dataset):
    def __init__(self, df, lookback, scaler=None):
        self.lookback = lookback
        lob_cols = [f'{s}{i}' for i in range(1,6) for s in ['bp','sp']] + \
                   [f'{s}{i}' for i in range(1,6) for s in ['bv','sv']]
        exp_cols = [c for c in df.columns if c.startswith('feat_') or c.startswith('meta_')]
        
        mid = df['mid'].values.reshape(-1, 1)
        safe_mid = np.where(mid==0, 1.0, mid)
        lob_data = df[lob_cols].values
        lob_data[:, :10] = (lob_data[:, :10] - mid) / safe_mid * 10000
        lob_data[:, 10:] = np.log1p(lob_data[:, 10:])
        self.X_lob = np.nan_to_num(lob_data).astype(np.float32)
        
        exp_data = np.nan_to_num(df[exp_cols].values)
        if scaler is None:
            self.scaler = StandardScaler()
            self.X_exp = self.scaler.fit_transform(exp_data).astype(np.float32)
        else:
            self.scaler = scaler
            self.X_exp = self.scaler.transform(exp_data).astype(np.float32)
            
        self.Y = df['label'].values.astype(np.int64)
        self.raw_ret = df['real_future_ret'].values
        
    def __len__(self): return len(self.Y) - self.lookback
    def __getitem__(self, i):
        s, e = i, i + self.lookback
        return self.X_lob[s:e], self.X_exp[s:e], self.Y[e-1], self.raw_ret[e-1]

def backtest_evaluate(model, dataloader, cfg):
    model.eval()
    cash = float(cfg['INITIAL_CAPITAL'])
    initial_cap = cash
    cost = cfg['TRADE_COST']
    conf_thresh = cfg['CONF_THRESHOLD']
    max_pos = cfg['MAX_POSITION']
    
    total_trades = 0; wins = 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for x_lob, x_exp, y, real_ret in dataloader:
            x_lob, x_exp = x_lob.to(cfg['DEVICE']), x_exp.to(cfg['DEVICE'])
            logits = model(x_lob, x_exp)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            y = y.numpy(); real_ret = real_ret.numpy()
            
            for i in range(len(probs)):
                p_hold, p_buy, p_sell = probs[i]
                signal = 0; confidence = 0.0
                
                if p_buy > p_hold and p_buy > p_sell and p_buy > conf_thresh:
                    signal = 1; confidence = p_buy
                elif p_sell > p_hold and p_sell > p_buy and p_sell > conf_thresh:
                    signal = 2; confidence = p_sell
                
                all_preds.append(signal)
                all_labels.append(y[i])
                
                if signal == 0: continue
                
                scale = (confidence - conf_thresh) / (1 - conf_thresh)
                scale = min(scale, max_pos)
                trade_val = cash * scale
                if trade_val < 2000: continue
                
                direction = 1 if signal == 1 else -1
                pnl = trade_val * (direction * real_ret[i] - 2 * cost)
                cash += pnl
                total_trades += 1
                if pnl > 0: wins += 1
                
    pnl_abs = cash - initial_cap
    print("\n" + "="*40)
    print(f"💰 [资金回测] 初始: {initial_cap:.0f} (成本万{int(cost*10000)})")
    if total_trades == 0:
        print("⚠️ 无交易 (信号太弱)")
        return 0.0
        
    print(f"最终净值: {cash:.2f}")
    print(f"交易次数: {total_trades} | 胜率: {wins/total_trades:.2%}")
    
    # [修复] 安全获取 Precision，防止 Key Error
    rep = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    buy_prec = rep.get('1', {}).get('precision', 0.0)
    sell_prec = rep.get('2', {}).get('precision', 0.0)
    
    print(f"Buy Precision: {buy_prec:.2f}")
    print(f"Sell Precision: {sell_prec:.2f}")
    print("="*40)
    return pnl_abs

def train_system():
    forge = AlphaForge(CONFIG)
    try:
        train_df, test_df = forge.load_and_split()
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # 标签分布 & 健壮权重计算
    c = np.bincount(train_df['label'].astype(int), minlength=3)
    print(f"📊 标签分布: {c}")
    
    ds_train = ETFDataset(train_df, CONFIG['LOOKBACK'])
    ds_test = ETFDataset(test_df, CONFIG['LOOKBACK'], scaler=ds_train.scaler)
    dl_train = DataLoader(ds_train, CONFIG['BATCH_SIZE'], shuffle=True)
    dl_test = DataLoader(ds_test, CONFIG['BATCH_SIZE'], shuffle=False)
    
    model = HybridDeepLOB(ds_train.X_exp.shape[1]).to(CONFIG['DEVICE'])
    
    # 权重策略
    w_hold = 1.0
    w_buy = min((c[0]/(c[1]+1)) * 0.5, 10.0) 
    w_sell = min((c[0]/(c[2]+1)) * 0.5, 10.0)
    weights = torch.tensor([w_hold, w_buy, w_sell], dtype=torch.float32).to(CONFIG['DEVICE'])
    print(f"⚖️ 使用权重: {weights.cpu().numpy()}")
    
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['LR'], weight_decay=CONFIG['WEIGHT_DECAY'])
    
    best_pnl = -np.inf
    patience = 0
    max_patience = CONFIG['PATIENCE']
    warmup = CONFIG['WARMUP_EPOCHS']
    
    print("\n🔥 开始训练...")
    for epoch in range(CONFIG['EPOCHS']):
        model.train()
        loss_sum = 0
        for x_lob, x_exp, y, _ in dl_train:
            x_lob, x_exp, y = x_lob.to(CONFIG['DEVICE']), x_exp.to(CONFIG['DEVICE']), y.to(CONFIG['DEVICE'])
            optimizer.zero_grad()
            out = model(x_lob, x_exp)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            
        print(f"Epoch {epoch+1} | Loss: {loss_sum/len(dl_train):.4f}")
        pnl = backtest_evaluate(model, dl_test, CONFIG)
        
        if pnl > best_pnl:
            best_pnl = pnl
            patience = 0
            torch.save(model.state_dict(), 'alpha_model_v8_stable.pth')
            print(">>> 新高! 模型保存.")
        else:
            if epoch >= warmup:
                patience += 1
                print(f"   -> 未提升 ({patience}/{max_patience})")
                if patience >= max_patience:
                    print("🛑 早停.")
                    break

if __name__ == "__main__":
    train_system()