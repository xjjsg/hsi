import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

# ==========================================
# 1. 模型定义 (保持不变)
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
# 2. 实时流处理引擎 (修正版)
# ==========================================
class TradingEngine:
    def __init__(self, model_path, config):
        self.cfg = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model_path = model_path
        self.model = None 
        
        # 交易状态
        self.cash = self.cfg['INITIAL_CAPITAL']
        self.shares = 0.0 # 持仓份额 (float)
        self.initial_capital = self.cash
        self.cost_rate = self.cfg['TRADE_COST']
        
        # 数据处理
        self.scaler = StandardScaler()
        self.weights = np.array([1.0, 0.8, 0.6, 0.4, 0.2]) # 压力因子权重

    def warm_up_scaler(self, prev_date_files):
        """
        加载前一天的数据，计算特征，并拟合 StandardScaler。
        """
        print(f"🔥 系统预热: 使用历史数据拟合 Scaler... ({os.path.basename(prev_date_files[0])})")
        
        # 1. 加载原始数据
        df_raw = self._load_and_process_pair(prev_date_files[0], prev_date_files[1])
        
        # 2. 计算特征
        df_features = self._calc_factors_stream(df_raw)
        
        # 3. 清洗数据 (批量模式下，丢弃 pct_change 产生的 NaN)
        df_features = df_features.replace([np.inf, -np.inf], np.nan).dropna()
        
        # 4. 提取特征列
        exp_cols = [c for c in df_features.columns if c.startswith('feat_') or c.startswith('meta_')]
        self.exp_cols = exp_cols
        
        if len(exp_cols) == 0:
            raise ValueError("❌ 错误: 未找到特征列。请检查 _calc_factors_stream 是否正常工作。")
            
        print(f"   -> 找到 {len(exp_cols)} 个特征列")
        
        # 5. 拟合 Scaler
        self.scaler.fit(df_features[exp_cols].values)
        print("✅ Scaler 拟合完成")
        
        # 6. 初始化并加载模型
        self.model = HybridDeepLOB(num_expert=len(exp_cols)).to(self.device)
        try:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            print("✅ 模型权重加载成功")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            exit()

    def run_simulation(self, main_file, aux_file):
        """模拟实时交易主循环"""
        print(f"🎬 开始模拟交易: {os.path.basename(main_file)}")
        
        # 1. 加载数据流
        df_raw = self._load_and_process_pair(main_file, aux_file)
        
        # [关键修复] 为原始数据计算 mid，供 execute_trade 使用
        df_raw['mid'] = (df_raw['bp1'] + df_raw['sp1']) / 2
        
        timestamps = df_raw.index
        lookback = self.cfg['LOOKBACK']
        
        # 2. 循环回放
        buffer_size = lookback + 5
        if len(df_raw) < buffer_size:
            print("⚠️ 数据过短")
            return

        print(f"⏳ 数据流回放中... (共 {len(df_raw)} 个 Tick)")
        
        # 记录最后一个有效价格用于结算
        self.last_known_price = 0.0

        for i in range(buffer_size, len(df_raw)):
            current_time = timestamps[i]
            current_row = df_raw.iloc[i] 
            self.last_known_price = current_row['mid']
            
            # --- 模拟只看得到过去 ---
            slice_start = max(0, i - lookback - 10) 
            df_slice = df_raw.iloc[slice_start : i+1].copy()
            
            # --- 实时特征工程 ---
            df_features = self._calc_factors_stream(df_slice)
            
            if len(df_features) < lookback: continue
                
            # 取最后 lookback 行作为输入
            input_df = df_features.iloc[-lookback:]
            
            # --- 推理与执行 ---
            x_lob, x_exp = self._prepare_tensor(input_df)
            signal, confidence, probs = self._infer(x_lob, x_exp)
            
            # 这里传入的 current_row 此时已经包含了 'mid'
            self._execute_trade(signal, confidence, current_row, current_time)
            
        self._report_final()

    def _execute_trade(self, signal, confidence, row, time):
        """简单的执行逻辑"""
        # [修复] 这里的 row['mid'] 现在是安全的，因为我们在 run_simulation 里加上了
        current_price = row['mid']
        if current_price <= 0: return

        conf_thresh = self.cfg['CONF_THRESHOLD']
        max_pos = self.cfg['MAX_POSITION']
        
        market_value = self.shares * current_price
        total_asset = self.cash + market_value
        
        # Buy Signal (1)
        if signal == 1:
             # 计算目标仓位
             scale = min((confidence - conf_thresh) / (1 - conf_thresh), max_pos)
             target_val = total_asset * scale
             cost = target_val - market_value # 需要补多少钱
             
             if self.shares == 0 and cost > 2000: # 简化：只做从0开仓，或加仓
                 buy_shares = cost / current_price 
                 fee = cost * self.cost_rate
                 self.shares += buy_shares
                 self.cash -= (cost + fee)
                 print(f"[{time}] 🔴 BUY  @{current_price:.3f} | Conf:{confidence:.2f} | Cash:{self.cash:.0f}")

        # Sell Signal (2)
        elif signal == 2 and self.shares > 0:
             # 简化：清仓
             revenue = self.shares * current_price
             fee = revenue * self.cost_rate
             self.cash += (revenue - fee)
             self.shares = 0
             print(f"[{time}] 🟢 SELL @{current_price:.3f} | Conf:{confidence:.2f} | Cash:{self.cash:.0f}")

    def _infer(self, x_lob, x_exp):
        with torch.no_grad():
            x_lob = x_lob.unsqueeze(0).to(self.device)
            x_exp = x_exp.unsqueeze(0).to(self.device)
            
            logits = self.model(x_lob, x_exp)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
            p_hold, p_buy, p_sell = probs
            signal = 0; conf = 0.0
            
            if p_buy > self.cfg['CONF_THRESHOLD'] and p_buy > p_sell:
                signal = 1; conf = p_buy
            elif p_sell > self.cfg['CONF_THRESHOLD'] and p_sell > p_buy:
                signal = 2; conf = p_sell
                
            return signal, conf, probs

    def _prepare_tensor(self, df):
        lob_cols = [f'{s}{i}' for i in range(1,6) for s in ['bp','sp']] + [f'{s}{i}' for i in range(1,6) for s in ['bv','sv']]
        
        mid = df['mid'].values.reshape(-1, 1)
        safe_mid = np.where(mid==0, 1.0, mid)
        
        lob_data = df[lob_cols].values.copy()
        lob_data[:, :10] = (lob_data[:, :10] - mid) / safe_mid * 10000
        lob_data[:, 10:] = np.log1p(lob_data[:, 10:])
        
        exp_data = df[self.exp_cols].values
        exp_data = np.nan_to_num(exp_data) 
        exp_data_scaled = self.scaler.transform(exp_data)
        
        return torch.tensor(lob_data, dtype=torch.float32), torch.tensor(exp_data_scaled, dtype=torch.float32)

    def _calc_factors_stream(self, df):
        """计算因子"""
        df = df.copy()
        
        # 1. Meta Factors
        sec = df.index.hour * 3600 + df.index.minute * 60 + df.index.second
        df['meta_time'] = np.clip(np.where(sec <= 41400, (sec-34200)/14400, 0.5+(sec-46800)/14400), 0, 1)
        
        # 2. Micro Factors
        mid = (df['bp1'] + df['sp1']) / 2
        df['mid'] = mid
        
        wb = sum(df[f'bv{i}']*self.weights[i-1] for i in range(1,6))
        wa = sum(df[f'sv{i}']*self.weights[i-1] for i in range(1,6))
        df['feat_micro_pressure'] = (wb - wa) / (wb + wa + 1e-8)
        
        # 3. Oracle Factors
        if 'index_price' in df.columns:
            # [修复] 警告修复: method='ffill' -> ffill()
            safe_mid = mid.replace(0, np.nan).ffill()
            df['feat_oracle_basis'] = (df['index_price'] - safe_mid) / safe_mid
            df['feat_oracle_idx_mom'] = df['index_price'].pct_change(2)
            
        if 'fut_price' in df.columns:
            df['feat_oracle_fut_lead'] = df['fut_price'].pct_change()

        # 4. Peer Factors
        df['feat_peer_diff'] = df['price'].pct_change() - df['peer_price'].pct_change()
        
        # 流式处理中用0填充
        if len(df) < 200: 
             df = df.fillna(0)
             
        return df

    def _load_and_process_pair(self, m_path, a_path):
        """读取并按 3s 重采样对齐"""
        date_str = os.path.basename(m_path).split('sz159920-')[-1].replace('.csv', '')
        
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
            
        # [修复] 警告修复: '3S' -> '3s' (lowercase s)
        df_m = df_m.resample(self.cfg['RESAMPLE_FREQ']).agg(agg)
        df_a = df_a.resample(self.cfg['RESAMPLE_FREQ']).agg({'price': 'last', 'tick_vol': 'sum'})
        df_a.columns = ['peer_price', 'peer_vol']
        
        # [修复] 警告修复: fillna(method='ffill') -> ffill()
        df = df_m.join(df_a, how='inner').ffill().dropna()
        return df

    def _report_final(self):
        # 使用最后记录的价格计算净值
        last_price = self.last_known_price if hasattr(self, 'last_known_price') and self.last_known_price > 0 else 1.0
        
        market_val = self.shares * last_price
        total_asset = self.cash + market_val
        pnl = total_asset - self.initial_capital
        ret = (pnl / self.initial_capital) * 100
        
        print("\n" + "="*40)
        print(f"🏁 模拟结束")
        print(f"最终资金: {self.cash:.2f}")
        print(f"持仓份额: {self.shares:.2f} (市值: {market_val:.2f})")
        print(f"总资产  : {total_asset:.2f}")
        print(f"总收益  : {pnl:.2f} ({ret:.2f}%)")
        print("="*40)

# ==========================================
# 3. 运行入口
# ==========================================
CONFIG = {
    'DATA_DIR': './data',          
    'MAIN_SYMBOL': 'sz159920',     
    'AUX_SYMBOL': 'sh513130',      
    'RESAMPLE_FREQ': '3s',         # [修复] 使用小写 's'
    'LOOKBACK': 60,                
    'TRADE_COST': 0.0001,          
    'INITIAL_CAPITAL': 200000,      
    'CONF_THRESHOLD': 0.75,        
    'MAX_POSITION': 0.9,           
}

def find_files_recursive(data_dir, main_sym, aux_sym):
    """自动查找最新和次新的数据"""
    m_pattern = os.path.join(data_dir, "**", f"{main_sym}*.csv")
    m_files = sorted(glob.glob(m_pattern, recursive=True))
    
    dates = []
    date_map = {}
    
    for f in m_files:
        try:
            base = os.path.basename(f)
            d_str = base.split(f"{main_sym}-")[-1].replace('.csv', '')
            dates.append(d_str)
            date_map[d_str] = f
        except: pass
        
    dates.sort()
    if len(dates) < 2:
        raise ValueError(f"数据不足 2 天，找到的日期: {dates}")
        
    latest = dates[-1]
    prev = dates[-2]
    
    def get_aux(d):
        m_file = date_map[d]
        aux_name = os.path.basename(m_file).replace(main_sym, aux_sym)
        aux_path = os.path.join(os.path.dirname(m_file), aux_name)
        if not os.path.exists(aux_path):
            aux_pattern = os.path.join(data_dir, "**", aux_name)
            found = glob.glob(aux_pattern, recursive=True)
            if found: aux_path = found[0]
            else: raise FileNotFoundError(f"找不到辅助文件: {aux_name}")
        return aux_path

    return (date_map[latest], get_aux(latest)), (date_map[prev], get_aux(prev))

if __name__ == "__main__":
    try:
        latest_pair, prev_pair = find_files_recursive(CONFIG['DATA_DIR'], CONFIG['MAIN_SYMBOL'], CONFIG['AUX_SYMBOL'])
        print(f"📅 目标日: {os.path.basename(latest_pair[0])}")
        print(f"📅 预热日: {os.path.basename(prev_pair[0])}")
    except Exception as e:
        print(f"❌ 文件查找失败: {e}")
        exit()

    model_file = 'alpha_model_v8_stable.pth'
    if not os.path.exists(model_file):
        found = glob.glob(f"**/{model_file}", recursive=True)
        if found: model_file = found[0]
        else:
            print("❌ 找不到 .pth 模型文件")
            exit()

    engine = TradingEngine(model_file, CONFIG)
    engine.warm_up_scaler(prev_pair)
    engine.run_simulation(latest_pair[0], latest_pair[1])