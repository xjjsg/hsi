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
    'COST_THRESHOLD': 0.0012,   
    
    # --- 资金管理回测 ---
    'TRADE_COST': 0.0001,          # 单边万1成本
    'INITIAL_CAPITAL': 200000,      # 初始本金 20万
    'CONF_THRESHOLD': 0.55,        # 75% 把握即开仓
    'MAX_POSITION': 0.9,           # 最大仓位 90%
    
    # --- 训练参数 ---
    'BATCH_SIZE': 512,             
    'EPOCHS': 100,
    'LR': 1e-4,
    'WEIGHT_DECAY': 1e-4,          # 强正则化
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'PATIENCE': 20,                # 早停耐心
    'WARMUP_EPOCHS': 10,           # 热身轮数
}

# ==========================================
# 2. 数据工厂：Alpha Forge (含新增因子)
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
        
        print(f"训练集: {train_pairs[0][0]} ~ {train_pairs[-1][0]}")
        print(f"测试集: {test_pair[0]}")
        
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
        
        # [新增] 确保核心 Alpha 因子源数据被读入
        extra_cols = ['index_price', 'fut_price', 'fut_imb', 'premium_rate', 'sentiment', 'tick_vwap']
        for c in extra_cols:
            if c in df_m.columns: 
                agg[c] = 'last'
            
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
        
        # [新增] 四大核心 Alpha 因子
        # A. 折溢价率
        if 'premium_rate' in df.columns:
            df['feat_premium_rate'] = df['premium_rate']
        elif 'index_price' in df.columns:
            df['feat_premium_rate'] = (df['index_price'] - safe_mid) / safe_mid

        # B. 情绪因子
        if 'sentiment' in df.columns:
            df['feat_sentiment'] = df['sentiment']
            
        # C. 期货盘口失衡
        if 'fut_imb' in df.columns:
            df['feat_fut_imb'] = df['fut_imb']
            
        # D. 资金流向力度 (Flow Force)
        if 'tick_vwap' in df.columns and 'tick_vol' in df.columns:
            df['feat_flow_force'] = (df['tick_vwap'] - df['mid']) * np.log1p(df['tick_vol'])
            
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
        # 保留真实未来收益 (旧版回测用，新版用全量数据模拟)
        df['real_future_ret'] = mid.shift(-horizon) / mid - 1
        return df

# ==========================================
# 3. 高级模型组件 (Unchanged)
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
        
        self.compress = nn.Sequential(
            nn.Conv2d(1, c_chan, (1, 2), stride=(1, 2)), nn.LeakyReLU(), nn.BatchNorm2d(c_chan),
            nn.Conv2d(c_chan, c_chan, (4, 1), padding='same'), nn.LeakyReLU(), nn.BatchNorm2d(c_chan),
            nn.Conv2d(c_chan, c_chan, (1, 2), stride=(1, 2)), nn.LeakyReLU(), nn.BatchNorm2d(c_chan),
            nn.Conv2d(c_chan, c_chan, (1, 5), stride=(1, 5)), nn.LeakyReLU(), nn.BatchNorm2d(c_chan),
        )
        self.inception1 = InceptionBlock(c_chan, c_chan)
        self.inception2 = InceptionBlock(128, 64)
        
        self.expert = nn.Sequential(
            nn.Linear(num_expert, m_hid), nn.LeakyReLU(), nn.BatchNorm1d(m_hid), nn.Dropout(0.2)
        )
        
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
# 4. 训练与实盘模拟引擎
# ==========================================
class ETFDataset(Dataset):
    def __init__(self, df, lookback, scaler=None):
        self.lookback = lookback
        lob_cols = [f'{s}{i}' for i in range(1,6) for s in ['bp','sp']] + [f'{s}{i}' for i in range(1,6) for s in ['bv','sv']]
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

def backtest_evaluate(model, dataloader, cfg, raw_df=None):
    """
    [升级版] 真实实盘逻辑回测引擎
    特性：
    1. 交易日志输出到文件 (backtest_log.txt)，解决终端刷屏问题
    2. 对手价成交 (Ask买/Bid卖)
    3. 每日收盘前(14:55)强制清仓
    """
    model.eval()
    
    # --- 阶段一：全量推理 ---
    all_probs = []
    with torch.no_grad():
        for x_lob, x_exp, _, _ in dataloader:
            x_lob = x_lob.to(cfg['DEVICE'])
            x_exp = x_exp.to(cfg['DEVICE'])
            logits = model(x_lob, x_exp)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
    
    if not all_probs: return 0.0
    probs_stream = np.concatenate(all_probs, axis=0)
    
    valid_len = len(probs_stream)
    if raw_df is None: return 0.0
        
    # 对齐数据
    sim_df = raw_df.iloc[-valid_len:].copy()
    
    # 提取关键列 (对手价)
    ask_prices = sim_df['sp1'].values # 卖一价 (买入用)
    bid_prices = sim_df['bp1'].values # 买一价 (卖出用)
    last_prices = sim_df['price'].values
    
    # 时间处理
    if 'datetime' in sim_df.columns:
        raw_times = pd.to_datetime(sim_df['datetime'].values)
    else:
        raw_times = pd.to_datetime(sim_df.index.values)
        
    # 计算每日收盘强制平仓标志 (14:55 之后)
    eod_flags = (raw_times.hour == 14) & (raw_times.minute >= 55)
    
    times_str = raw_times.strftime('%Y-%m-%d %H:%M:%S').values
    
    cash = float(cfg['INITIAL_CAPITAL'])
    shares = 0.0
    initial_cap = cash
    cost_rate = cfg['TRADE_COST']
    conf_thresh = cfg['CONF_THRESHOLD']
    max_pos = cfg['MAX_POSITION']
    
    trades = 0; wins = 0
    entry_price = 0.0
    is_holding = False
    
    # [修改] 准备日志文件
    log_file = "backtest_log.txt"
    
    # 使用 'w' 模式每次覆盖，如果想追加可用 'a'
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"=== 回测交易日志 (Start: {times_str[0]}) ===\n")
        f.write(f"参数: Threshold={conf_thresh}, Cost={cost_rate}, EOD_Clear=14:55\n")
        f.write("-" * 60 + "\n")
        
        for t in range(valid_len):
            curr_ask = ask_prices[t]
            curr_bid = bid_prices[t]
            
            # 数据清洗：如果盘口缺失，暂用最新价代替
            if curr_ask <= 0: curr_ask = last_prices[t]
            if curr_bid <= 0: curr_bid = last_prices[t]
            
            current_time = times_str[t]
            p_hold, p_buy, p_sell = probs_stream[t]
            
            # ==========================================
            # 每日收盘前强制风控
            # ==========================================
            if eod_flags[t]:
                if is_holding:
                    # 强平
                    revenue = shares * curr_bid * (1 - cost_rate)
                    raw_pnl = revenue - (shares * entry_price / (1 - cost_rate))
                    pnl_pct = (curr_bid - entry_price) / entry_price
                    
                    cash += revenue
                    shares = 0.0
                    is_holding = False
                    trades += 1
                    if raw_pnl > 0: wins += 1
                    
                    f.write(f"[{current_time}] 🟢 SELL @ {curr_bid:.4f} | PnL: {raw_pnl:+.2f} ({pnl_pct:.2%}) | Reason: EOD_Clear\n")
                
                continue # 收盘时段跳过后续开仓判断
            
            # ==========================================
            # 正常交易逻辑
            # ==========================================
            signal = 0; confidence = 0.0
            if p_buy > p_hold and p_buy > p_sell and p_buy > conf_thresh:
                signal = 1; confidence = p_buy
            elif p_sell > p_hold and p_sell > p_buy and p_sell > conf_thresh:
                signal = 2; confidence = p_sell
                
            if not is_holding:
                if signal == 1: 
                    scale = min((confidence - conf_thresh) / (1 - conf_thresh), max_pos)
                    budget = cash * scale
                    if budget > 2000:
                        # 用卖一价买入
                        shares = budget / curr_ask * (1 - cost_rate)
                        cash -= budget
                        entry_price = curr_ask 
                        is_holding = True
                        
                        f.write(f"[{current_time}] 🔴 BUY  @ {curr_ask:.4f} | Conf: {confidence:.2f}\n")
            else:
                pnl_pct = (curr_bid - entry_price) / entry_price
                
                stop_loss = pnl_pct < -0.008
                signal_lost = p_buy < conf_thresh
                reverse_signal = signal == 2
                
                if stop_loss or signal_lost or reverse_signal:
                    # 用买一价卖出
                    revenue = shares * curr_bid * (1 - cost_rate)
                    raw_pnl = revenue - (shares * entry_price / (1 - cost_rate))
                    
                    cash += revenue
                    shares = 0.0
                    is_holding = False
                    trades += 1
                    if raw_pnl > 0: wins += 1
                    
                    reason = "StopLoss" if stop_loss else ("RevSignal" if reverse_signal else "SigLost")
                    f.write(f"[{current_time}] 🟢 SELL @ {curr_bid:.4f} | PnL: {raw_pnl:+.2f} ({pnl_pct:.2%}) | Reason: {reason}\n")

        # 模拟结束强制结算
        if is_holding:
            final_bid = bid_prices[-1] if bid_prices[-1] > 0 else last_prices[-1]
            cash += shares * final_bid * (1 - cost_rate)
            f.write(f"[{times_str[-1]}] 🟢 SELL @ {final_bid:.4f} | Reason: Simulation_End\n")
            
        final_pnl = cash - initial_cap
        win_rate = wins / trades if trades > 0 else 0
        
        # 写入摘要到文件末尾
        summary = (
            f"\n{'='*40}\n"
            f"[最终结果]\n"
            f"最终净值: {cash:.2f}\n"
            f"收益率  : {(cash/initial_cap - 1)*100:.2f}%\n"
            f"交易次数: {trades} | 胜率: {win_rate:.2%}\n"
            f"{'='*40}\n"
        )
        f.write(summary)

    # 终端只打印摘要
    print("\n" + "="*40)
    print(f"[实盘模拟引擎] 初始: {initial_cap:.0f}")
    print(f"最终净值: {cash:.2f}")
    print(f"收益率  : {(cash/initial_cap - 1)*100:.2f}%")
    print(f"交易次数: {trades} | 胜率: {win_rate:.2%}")
    print(f"👉 详细日志已保存至: {log_file}")
    print("="*40)
    
    return final_pnl
def train_system():
    forge = AlphaForge(CONFIG)
    try:
        train_df, test_df = forge.load_and_split()
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    c = np.bincount(train_df['label'].astype(int), minlength=3)
    print(f"标签分布: {c}")
    
    ds_train = ETFDataset(train_df, CONFIG['LOOKBACK'])
    # 注意：dl_test 需要 shuffle=False 才能保证时间顺序
    ds_test = ETFDataset(test_df, CONFIG['LOOKBACK'], scaler=ds_train.scaler)
    dl_train = DataLoader(ds_train, CONFIG['BATCH_SIZE'], shuffle=True)
    dl_test = DataLoader(ds_test, CONFIG['BATCH_SIZE'], shuffle=False)
    
    model = HybridDeepLOB(ds_train.X_exp.shape[1]).to(CONFIG['DEVICE'])
    
    # 类别权重
    w_hold = 1.0
    w_buy = c[0]/(c[1]+1)
    w_sell = c[0]/(c[2]+1)
    weights = torch.tensor([w_hold, w_buy, w_sell], dtype=torch.float32).to(CONFIG['DEVICE'])
    print(f"使用权重: {weights.cpu().numpy()}")
    
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
        
        # [修改] 传入 raw_df=test_df 供事件驱动回测使用
        pnl = backtest_evaluate(model, dl_test, CONFIG, raw_df=test_df)
        
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
                    print("早停.")
                    break

if __name__ == "__main__":
    train_system()