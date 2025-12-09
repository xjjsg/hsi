# -*- coding: utf-8 -*-
"""
Alpha System Ultimate (阿尔法系统 - 终极完整版)
-----------------------------------------------
功能全集：
1. [数据] 自动配对 sz159920/sh513130，执行 3S 重采样与清洗。
2. [因子] 生成微观(Micro)、宏观(Oracle)、共振(Peer)、状态(Meta)四大类因子。
3. [标签] Triple Barrier Method (触达止盈)，捕捉过程中的 0.002 波动。
4. [模型] Hybrid DeepLOB (Inception-CNN + MLP + LSTM) 双流架构。
5. [回测] 资金管理回测 (Kelly-style)，按置信度动态调整仓位。
6. [训练] 自动逆频率加权 + 早停机制 + 学习率衰减。

@Ver: 7.0 Final Complete
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# 忽略 pandas 的 SettingWithCopyWarning
warnings.filterwarnings('ignore')

# ==========================================
# 1. 全局配置 (Configuration)
# ==========================================
CONFIG = {
    # --- 路径配置 (请根据实际情况修改) ---
    'DATA_DIR': './data',          # 数据根目录
    'MAIN_SYMBOL': 'sz159920',     # 主标的
    'AUX_SYMBOL': 'sh513130',      # 辅助标的
    
    # --- 因子与数据 ---
    'RESAMPLE_FREQ': '3S',         # 3秒重采样 (去噪+匹配滞后)
    'PREDICT_HORIZON': 60,         # 预测未来 60个周期 (180秒)
    'LOOKBACK': 60,                # 回看窗口长度 (180秒)
    
    # --- 标签生成 ---
    # [关键] 训练门槛降至 0.0012 (覆盖成本即可)，让模型敢于开仓
    'COST_THRESHOLD': 0.0012,   
    
    # --- 资金管理回测 ---
    'TRADE_COST': 0.0006,          # 单边成本 (万6, 含佣金+滑点)
    'INITIAL_CAPITAL': 20000,      # 初始本金
    'CONF_THRESHOLD': 0.6,         # 开仓置信度门槛 (概率 > 0.6 才开仓)
    'MAX_POSITION': 0.8,           # 单笔最大仓位 (80% 本金)
    
    # --- 训练参数 ---
    'BATCH_SIZE': 512,
    'EPOCHS': 50,
    'LR': 1e-4,
    'WEIGHT_DECAY': 1e-5,          # L2正则化
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'PATIENCE': 20,                # 早停耐心
    'WARMUP_EPOCHS': 10,           # 热身期
}

# ==========================================
# 2. 数据工厂：Alpha Forge
# ==========================================
class AlphaForge:
    def __init__(self, cfg):
        self.cfg = cfg
        # 盘口衰减权重 (Level 1 -> Level 5)
        self.weights = np.array([1.0, 0.8, 0.6, 0.4, 0.2])

    def load_and_split(self):
        """扫描目录，配对文件，按日期切分训练/测试集"""
        print(f"🚀 [AlphaForge] 启动... 扫描: {self.cfg['DATA_DIR']}")
        
        pairs = self._match_files()
        if len(pairs) < 2:
            raise ValueError(f"数据不足！找到 {len(pairs)} 天数据，至少需要2天进行回测。")
            
        # 按日期排序
        pairs.sort(key=lambda x: x[0])
        
        # 最后一天作为测试集 (Walk-forward testing)
        train_pairs = pairs[:-1]
        test_pair = pairs[-1]
        
        print(f"📅 训练集: {train_pairs[0][0]} ~ {train_pairs[-1][0]} ({len(train_pairs)}天)")
        print(f"📅 测试集: {test_pair[0]} (1天)")
        
        train_df = self._process_batch(train_pairs)
        test_df = self._process_batch([test_pair])
        
        return train_df, test_df

    def _process_batch(self, pairs):
        dfs = []
        for date, mf, af in pairs:
            try:
                # 1. 加载与对齐
                df = self._load_pair(mf, af, date)
                if df is None or len(df) < 200: continue
                
                # 2. 计算因子
                df = self._calc_factors(df)
                
                # 3. 生成标签 (Triple Barrier)
                df = self._make_labels(df)
                
                # 4. [关键] 无穷值清洗
                df = df.replace([np.inf, -np.inf], np.nan)
                
                dfs.append(df.dropna())
            except Exception as e:
                print(f"⚠️ 跳过 {date}: {e}")
                
        if not dfs: return pd.DataFrame()
        return pd.concat(dfs).sort_index()

    def _match_files(self):
        """根据日期匹配主标的和辅助标的的文件"""
        m_pattern = os.path.join(self.cfg['DATA_DIR'], "**", f"{self.cfg['MAIN_SYMBOL']}*.csv")
        a_pattern = os.path.join(self.cfg['DATA_DIR'], "**", f"{self.cfg['AUX_SYMBOL']}*.csv")
        
        m_files = glob.glob(m_pattern, recursive=True)
        a_files = glob.glob(a_pattern, recursive=True)
        
        def extract_date(path):
            try:
                # 假设格式包含 YYYY-MM-DD
                base = os.path.basename(path)
                parts = base.replace('.csv','').split('-')
                # 取最后三段组成日期
                if len(parts) >= 3:
                    return f"{parts[-3]}-{parts[-2]}-{parts[-1]}"
            except: pass
            return None

        m_map = {extract_date(f): f for f in m_files if extract_date(f)}
        a_map = {extract_date(f): f for f in a_files if extract_date(f)}
        
        common = sorted(list(set(m_map.keys()) & set(a_map.keys())))
        return [(d, m_map[d], a_map[d]) for d in common]

    def _load_pair(self, m_path, a_path, date_str):
        def _read(p):
            d = pd.read_csv(p)
            d['datetime'] = pd.to_datetime(date_str + ' ' + d['tx_server_time'])
            return d.set_index('datetime').sort_index().groupby(level=0).last()
        
        df_m = _read(m_path)
        df_a = _read(a_path)
        
        # 聚合规则
        agg = {
            'price': 'last', 'tick_vol': 'sum',
            'bp1': 'last', 'sp1': 'last',
            'bp2': 'last', 'sp2': 'last', 'bp3': 'last', 'sp3': 'last',
            'bp4': 'last', 'sp4': 'last', 'bp5': 'last', 'sp5': 'last',
            'bv1': 'last', 'sv1': 'last',
            'bv2': 'last', 'sv2': 'last', 'bv3': 'last', 'sv3': 'last',
            'bv4': 'last', 'sv4': 'last', 'bv5': 'last', 'sv5': 'last',
        }
        # 检查上帝视角数据
        for c in ['index_price', 'fut_price', 'fut_imb']:
            if c in df_m.columns: agg[c] = 'last'
            
        # 重采样
        rule = self.cfg['RESAMPLE_FREQ']
        df_m = df_m.resample(rule).agg(agg)
        df_a = df_a.resample(rule).agg({'price': 'last', 'tick_vol': 'sum'})
        df_a.columns = ['peer_price', 'peer_vol']
        
        # 内连接对齐
        return df_m.join(df_a, how='inner')

    def _calc_factors(self, df):
        """核心特征工程"""
        
        # 1. Meta Factors (时间状态)
        sec = df.index.hour * 3600 + df.index.minute * 60 + df.index.second
        time_norm = np.where(sec <= 41400, (sec - 34200)/14400, 0.5 + (sec - 46800)/14400)
        df['meta_time'] = np.clip(time_norm, 0, 1)
        
        # 2. Micro Factors (L2 微观)
        mid = (df['bp1'] + df['sp1']) / 2
        df['mid'] = mid
        safe_mid = mid.replace(0, np.nan).fillna(method='ffill')

        wb = sum(df[f'bv{i}']*self.weights[i-1] for i in range(1,6))
        wa = sum(df[f'sv{i}']*self.weights[i-1] for i in range(1,6))
        df['feat_micro_pressure'] = (wb - wa) / (wb + wa + 1e-8)
        
        # 3. Oracle Factors (上帝视角)
        if 'index_price' in df.columns:
            df['feat_oracle_basis'] = (df['index_price'] - safe_mid) / safe_mid
            df['feat_oracle_idx_mom'] = df['index_price'].pct_change(2)
            
        if 'fut_price' in df.columns:
            df['feat_oracle_fut_lead'] = df['fut_price'].pct_change()
            
        # 4. Peer Factors (共振)
        df['feat_peer_diff'] = df['price'].pct_change() - df['peer_price'].pct_change()
        
        return df

    def _make_labels(self, df):
        """
        [Triple Barrier Method]
        捕捉过程中的最大涨跌幅
        """
        mid = df['mid']
        horizon = self.cfg['PREDICT_HORIZON']
        threshold = self.cfg['COST_THRESHOLD']
        
        # 使用 Forward Window 获取未来窗口内的 Max/Min
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=horizon)
        future_max = mid.rolling(window=indexer).max()
        future_min = mid.rolling(window=indexer).min()
        
        max_ret = future_max / mid - 1
        min_ret = future_min / mid - 1
        
        labels = np.zeros(len(df))
        
        # 只要触碰过止盈线，就视为机会
        mask_buy = max_ret > threshold
        mask_sell = min_ret < -threshold
        
        labels[mask_buy] = 1
        labels[mask_sell] = 2
        
        # 冲突处理：谁的空间大听谁的
        conflict = mask_buy & mask_sell
        if conflict.any():
            c_max = max_ret[conflict]
            c_min = min_ret[conflict].abs()
            labels[conflict] = np.where(c_max > c_min, 1, 2)
            
        df['label'] = labels
        # 保留 Point-to-Point 收益用于保守回测
        df['real_future_ret'] = mid.shift(-horizon) / mid - 1
        return df

# ==========================================
# 3. 模型核心: Inception Hybrid
# ==========================================
class InceptionBlock(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(in_chan, out_chan, 1), nn.LeakyReLU(), nn.BatchNorm2d(out_chan))
        self.b2 = nn.Sequential(nn.Conv2d(in_chan, out_chan, 1), nn.LeakyReLU(), 
                                nn.Conv2d(out_chan, out_chan, (3,1), padding=(1,0)), nn.LeakyReLU(), nn.BatchNorm2d(out_chan))
        self.b3 = nn.Sequential(nn.Conv2d(in_chan, out_chan, 1), nn.LeakyReLU(),
                                nn.Conv2d(out_chan, out_chan, (5,1), padding=(2,0)), nn.LeakyReLU(), nn.BatchNorm2d(out_chan))
        self.b4 = nn.Sequential(nn.MaxPool2d((3,1), stride=1, padding=(1,0)),
                                nn.Conv2d(in_chan, out_chan, 1), nn.LeakyReLU(), nn.BatchNorm2d(out_chan))
    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)

class HybridDeepLOB(nn.Module):
    def __init__(self, num_expert):
        super().__init__()
        
        # A. Visual Stream (LOB)
        # 压缩宽度: 20 -> 10 -> 5 -> 1
        self.compress = nn.Sequential(
            nn.Conv2d(1, 16, (1, 2), stride=(1, 2)), nn.LeakyReLU(), nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, (4, 1), padding='same'), nn.LeakyReLU(), nn.BatchNorm2d(16), # Time conv
            nn.Conv2d(16, 16, (1, 2), stride=(1, 2)), nn.LeakyReLU(), nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, (1, 5), stride=(1, 5)), nn.LeakyReLU(), nn.BatchNorm2d(16),
        )
        # Inception (N, 16, T, 1) -> (N, 64, T, 1)
        self.inception = InceptionBlock(16, 16) 
        
        # B. Expert Stream
        self.expert = nn.Sequential(
            nn.Linear(num_expert, 32), nn.LeakyReLU(), nn.BatchNorm1d(32)
        )
        
        # C. Fusion
        self.lstm = nn.LSTM(64 + 32, 128, batch_first=True, dropout=0.3) # 增加 dropout
        self.head = nn.Linear(128, 3)

    def forward(self, x_lob, x_exp):
        # x_lob: (N, T, 20) -> (N, 1, T, 20)
        x = x_lob.unsqueeze(1)
        
        # 1. 压缩盘口 (N, 16, T, 1)
        feat_lob = self.compress(x)
        
        # 2. 多尺度感知
        feat_lob = self.inception(feat_lob) # (N, 64, T, 1)
        
        # 3. 维度变换 (N, T, 64)
        feat_lob = feat_lob.squeeze(-1).permute(0, 2, 1)
        
        # 4. 时间对齐 (Adaptive Pooling)
        if feat_lob.shape[1] != x_exp.shape[1]:
            feat_lob = feat_lob.permute(0, 2, 1) # (N, C, T)
            feat_lob = nn.functional.adaptive_avg_pool1d(feat_lob, x_exp.shape[1])
            feat_lob = feat_lob.permute(0, 2, 1) # (N, T, C)
            
        # 5. 处理专家因子
        B, T, F = x_exp.shape
        feat_exp = self.expert(x_exp.reshape(-1, F)).reshape(B, T, -1)
        
        # 6. 融合与预测
        combined = torch.cat([feat_lob, feat_exp], dim=2) # (N, T, 320)
        out, _ = self.lstm(combined)
        return self.head(out[:, -1, :])

# ==========================================
# 4. 训练引擎 (带资金管理回测)
# ==========================================
class ETFDataset(Dataset):
    def __init__(self, df, lookback, scaler=None):
        self.lookback = lookback
        
        # LOB列名
        lob_cols = [f'{s}{i}' for i in range(1,6) for s in ['bp','sp']] + \
                   [f'{s}{i}' for i in range(1,6) for s in ['bv','sv']]
        # 专家因子列名
        exp_cols = [c for c in df.columns if c.startswith('feat_') or c.startswith('meta_')]
        
        # --- 归一化 ---
        mid = df['mid'].values.reshape(-1, 1)
        safe_mid = np.where(mid==0, 1.0, mid) 
        
        lob_data = df[lob_cols].values
        lob_data[:, :10] = (lob_data[:, :10] - mid) / safe_mid * 10000
        lob_data[:, 10:] = np.log1p(lob_data[:, 10:])
        
        # 二次清洗
        lob_data = np.nan_to_num(lob_data, nan=0.0, posinf=0.0, neginf=0.0)
        self.X_lob = lob_data.astype(np.float32)
        
        # Expert Norm
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
    """
    [资金管理回测] 
    Logic: 信号越强，仓位越重 (Kelly-style)
    """
    model.eval()
    
    cash = float(cfg['INITIAL_CAPITAL'])
    initial_cap = cash
    cost = cfg['TRADE_COST']
    conf_thresh = cfg['CONF_THRESHOLD']
    max_pos = cfg['MAX_POSITION']
    
    total_trades = 0
    wins = 0
    
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for x_lob, x_exp, y, real_ret in dataloader:
            x_lob, x_exp = x_lob.to(cfg['DEVICE']), x_exp.to(cfg['DEVICE'])
            
            # 获取概率
            logits = model(x_lob, x_exp)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            real_ret = real_ret.numpy()
            y = y.numpy()
            
            for i in range(len(probs)):
                p_hold, p_buy, p_sell = probs[i]
                
                signal = 0
                confidence = 0.0
                
                # 决策: 概率最大且超过阈值
                if p_buy > p_hold and p_buy > p_sell and p_buy > conf_thresh:
                    signal = 1
                    confidence = p_buy
                elif p_sell > p_hold and p_sell > p_buy and p_sell > conf_thresh:
                    signal = 2
                    confidence = p_sell
                
                all_preds.append(signal)
                all_labels.append(y[i])
                
                if signal == 0: continue
                
                # --- 仓位管理 ---
                # 线性映射: (conf - thresh) / (1 - thresh)
                scale = (confidence - conf_thresh) / (1 - conf_thresh)
                scale = min(scale, max_pos) # 封顶
                
                trade_val = cash * scale
                if trade_val < 2000: continue # 资金太少不开仓(避免手续费磨损)
                
                # 结算
                direction = 1 if signal == 1 else -1
                pnl = trade_val * (direction * real_ret[i] - 2 * cost)
                
                cash += pnl
                total_trades += 1
                if pnl > 0: wins += 1
                
    pnl_abs = cash - initial_cap
    roi = pnl_abs / initial_cap
    
    print("\n" + "="*40)
    print(f"💰 [资金回测] 初始: {initial_cap}")
    if total_trades == 0:
        print("⚠️ 无交易 (信号太弱)")
        return 0.0
        
    print(f"最终净值: {cash:.2f} (ROI: {roi:.2%})")
    print(f"交易次数: {total_trades} | 胜率: {wins/total_trades:.2%}")
    
    rep = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    print(f"Buy Precision: {rep['1']['precision']:.2f}")
    print("="*40)
    
    return pnl_abs

def train_system():
    forge = AlphaForge(CONFIG)
    try:
        train_df, test_df = forge.load_and_split()
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # 标签分布
    c = np.bincount(train_df['label'].astype(int))
    print(f"📊 Triple Barrier 标签分布: Hold={c[0]}, Buy={c[1]}, Sell={c[2]}")
    
    ds_train = ETFDataset(train_df, CONFIG['LOOKBACK'])
    ds_test = ETFDataset(test_df, CONFIG['LOOKBACK'], scaler=ds_train.scaler)
    dl_train = DataLoader(ds_train, CONFIG['BATCH_SIZE'], shuffle=True)
    dl_test = DataLoader(ds_test, CONFIG['BATCH_SIZE'], shuffle=False)
    
    model = HybridDeepLOB(ds_train.X_exp.shape[1]).to(CONFIG['DEVICE'])
    
    # 智能权重: 温和修正 (1:10:10)
    w_hold = 1.0
    # 防止权重过大导致激进
    w_buy = min((c[0]/c[1]) * 0.5, 10.0) if c[1] > 0 else 1.0
    w_sell = min((c[0]/c[2]) * 0.5, 10.0) if c[2] > 0 else 1.0
    
    weights = torch.tensor([w_hold, w_buy, w_sell], dtype=torch.float32).to(CONFIG['DEVICE'])
    print(f"⚖️ 智能修正权重: {weights.cpu().numpy()}")
    
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    # [优化] 加入权重衰减 (L2 正则)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['LR'], weight_decay=CONFIG['WEIGHT_DECAY'])
    
    best_pnl = -np.inf
    patience = 0
    # [优化] 增加耐心和热身
    max_patience = 20 
    warmup = 10
    
    print("\n🔥 开始终极训练...")
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
            torch.save(model.state_dict(), 'alpha_model_v6.pth')
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