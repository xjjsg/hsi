import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import sys
import os
import math
import warnings
from datetime import datetime, time as dt_time

warnings.filterwarnings('ignore')

# ==============================================================================
# 0. 配置区域 (请根据实际文件名调整)
# ==============================================================================
CONFIG = {
    'MAIN_FILE': './data/sz159920/sz159920-2025-12-04.csv', # 今天的csv
    'AUX_FILE':  './data/sh513130/sh513130-2025-12-04.csv', # 今天的辅助csv
    'ARTIFACT_PATH': 'FACTOR_STRATEGY_ARTIFACT.pth',        # 模型文件
    'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'MAX_LOOKBACK': 120,
    'RESAMPLE_FREQ': '3s',
}

print(f"🚀 模拟盘回测启动 (修复版) |日期: 2025-12-04 | 设备: {CONFIG['DEVICE']}")

# ==============================================================================
# 1. 核心类定义
# ==============================================================================

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(inplace=True), nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()
        self.b1 = nn.Conv1d(in_channels, out_channels//4, 1)
        self.b2 = nn.Sequential(nn.Conv1d(in_channels, out_channels//4, 1), nn.Conv1d(out_channels//4, out_channels//4, 3, padding=1))
        self.b3 = nn.Sequential(nn.Conv1d(in_channels, out_channels//4, 1), nn.Conv1d(out_channels//4, out_channels//4, 5, padding=2))
        self.b4 = nn.Sequential(nn.MaxPool1d(3, 1, 1), nn.Conv1d(in_channels, out_channels//4, 1))
    def forward(self, x): return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)

class SinusoidalPosEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer('pe', pe.unsqueeze(0).transpose(0, 1))
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class Direction(nn.Module):
    def __init__(self, input_dim, d_model=256, nhead=16, num_layers=6, num_classes=3):
        super(Direction, self).__init__()
        self.stem = nn.Sequential(nn.Conv1d(input_dim, 64, 1), nn.BatchNorm1d(64), nn.LeakyReLU())
        self.inception = InceptionBlock(64, d_model)
        self.se = SEBlock(d_model)
        self.pos_encoder = SinusoidalPosEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=512, dropout=0.1)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Sequential(nn.Linear(d_model, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, num_classes))
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.se(self.inception(self.stem(x))) 
        x = x.permute(2, 0, 1) 
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.fc(x.mean(dim=0))

class ManualFactorGenerator:
    def process(self, df):
        res = pd.DataFrame(index=df.index)
        db = df['bp1'].diff(); ds = df['sp1'].diff()
        dvb = df['bv1'].diff(); dvs = df['sv1'].diff()
        delta_vb = np.select([db > 0, db < 0], [df['bv1'], 0], default=dvb)
        delta_va = np.select([ds > 0, ds < 0], [0, df['sv1']], default=dvs)
        voi = delta_vb - delta_va
        res['alpha_voi_raw'] = voi
        vol_ma = df['tick_vol'].rolling(10).mean().replace(0, 1)
        res['alpha_voi_smart'] = voi / vol_ma
        if 'tick_vwap' in df.columns:
            res['alpha_vwap_bias'] = (df['tick_vwap'] - df['mid_price']) / df['mid_price']
        if 'bv5' in df.columns:
            sum_bid = sum(df[f'bv{i}'] for i in range(1, 6))
            sum_ask = sum(df[f'sv{i}'] for i in range(1, 6))
            total_depth = sum_bid + sum_ask + 1e-6
            res['alpha_depth_imb_l5'] = (sum_bid - sum_ask) / total_depth
            res['alpha_wall_bid'] = df['bv5'] / (df['bv1'] + 1)
            res['alpha_wall_ask'] = df['sv5'] / (df['sv1'] + 1)
        l1_imb = df['bv1'] / (df['bv1'] + df['sv1'] + 1e-6)
        micro_price = df['bp1'] * (1 - l1_imb) + df['sp1'] * l1_imb
        res['alpha_micro_dev'] = (micro_price - df['mid_price']) / df['mid_price']
        if 'ctx_mid_price' in df.columns:
            res['alpha_cross_rs'] = df['log_ret'] - df['ctx_log_ret']
            ctx_lag_2 = df['ctx_log_ret'].shift(2).fillna(0)
            res['alpha_cross_lead_lag'] = ctx_lag_2 - df['log_ret']
            spread = np.log(df['mid_price']) - np.log(df['ctx_mid_price'])
            spread_mean = spread.rolling(120).mean()
            spread_std = spread.rolling(120).std()
            res['alpha_cross_arb_z'] = (spread - spread_mean) / (spread_std + 1e-6)
            if 'sentiment' in df.columns and 'ctx_sentiment' in df.columns:
                res['alpha_sent_gap'] = df['sentiment'] - df['ctx_sentiment']
        minutes = df.index.hour * 60 + df.index.minute
        res['feat_time_norm'] = (minutes - 570) / (900 - 570)
        mask_late = minutes >= 890 
        res['logic_market_closing'] = 0.0
        res.loc[mask_late, 'logic_market_closing'] = 1.0
        mask_open = (minutes >= 570) & (minutes <= 575)
        res['logic_market_opening'] = 0.0
        res.loc[mask_open, 'logic_market_opening'] = 1.0
        if 'fut_price' in df.columns:
            fut_ret = df['fut_price'].pct_change().fillna(0)
            res['alpha_fut_lead'] = fut_ret - df['log_ret']
        return res.fillna(0)

class OrderExecutor:
    def __init__(self, initial_cash=100000.0):
        self.initial_capital = initial_cash
        self.cash = initial_cash
        self.position = 0
        self.entry_price = 0.0
        self.entry_cost_total = 0.0 
        self.fixed_cost = 5.0
        self.trade_val = 100000.0 
        self.trade_records = [] 
        self.last_buy_time = None

    def generate_order(self, score, current_price, ask1, bid1, is_noon_shock=False):
        direction = 1 if score > 0 else -1
        order_type = "LIMIT"
        order_price = 0.0
        urgency = "LOW"
        abs_score = abs(score)
        if is_noon_shock or abs_score > 0.7:
            urgency = "HIGH"
            order_type = "MARKET_TAKER"
            order_price = ask1 if direction == 1 else bid1
        elif abs_score > 0.4:
            urgency = "MID"
            order_type = "LIMIT_MAKER"
            order_price = bid1 if direction == 1 else ask1
        else:
            return None 
        vol = int(self.trade_val / current_price / 100) * 100
        if vol == 0: return None
        return {
            'action': 'OPEN' if self.position == 0 else 'CLOSE',
            'direction': direction,
            'price': order_price,
            'vol': vol,
            'type': order_type,
            'urgency': urgency
        }

    def match_order(self, order, market_snapshot):
        if not order: return False
        ask1 = market_snapshot['sp1']
        bid1 = market_snapshot['bp1']
        current_ts = market_snapshot.name 
        is_filled = False
        fill_price = order['price']
        if 'TAKER' in order['type'] or 'FORCE' in order['type']:
            is_filled = True
            fill_price = order['price'] 
        elif 'MAKER' in order['type']:
            if order['direction'] == 1: 
                if ask1 <= order['price']: is_filled = True; fill_price = ask1 
            else: 
                if bid1 >= order['price']: is_filled = True; fill_price = bid1
        if is_filled:
            self._execute_accounting(order, fill_price, current_ts)
            return True
        return False

    def _execute_accounting(self, order, price, ts):
        cost = self.fixed_cost
        val = order['vol'] * price
        if order['direction'] == 1:
            actual_cost = val + cost
            self.cash -= actual_cost
            self.position += order['vol']
            self.entry_price = price
            self.entry_cost_total = actual_cost
            self.last_buy_time = ts
        else:
            actual_revenue = val - cost
            self.cash += actual_revenue
            net_profit = actual_revenue - self.entry_cost_total
            ret_pct = (net_profit / self.entry_cost_total) * 100 if self.entry_cost_total > 0 else 0
            self.trade_records.append({
                'buy_time': self.last_buy_time,
                'sell_time': ts,
                'net_pnl': net_profit,
                'ret_pct': ret_pct,
                'hold_time': (ts - self.last_buy_time).total_seconds() if self.last_buy_time else 0
            })
            color = "\033[91m" if net_profit > 0 else "\033[92m"
            print(f"🔴 [成交] SELL {order['vol']} @ {price:.3f} | 净利: {color}{net_profit:.2f}元\033[0m | {ts.time()}")
            self.position = 0; self.entry_price = 0; self.entry_cost_total = 0

    def print_daily_report(self):
        print("\n" + "="*50)
        print(f"📊 模拟盘回测报告")
        print("="*50)
        if not self.trade_records:
            print("⚠️ 今日无成交记录")
            return
        df_trades = pd.DataFrame(self.trade_records)
        total_trades = len(df_trades)
        wins = df_trades[df_trades['net_pnl'] > 0]
        total_pnl = df_trades['net_pnl'].sum()
        win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
        print(f"💰 最终权益:     {self.cash:.2f}")
        print(f"📈 累计盈亏:     {total_pnl:+.2f} 元")
        print(f"-"*30)
        print(f"🎲 交易笔数:     {total_trades}")
        print(f"🏆 胜率:         {win_rate:.1f}%")
        print(f"📉 最大亏损:     {df_trades['net_pnl'].min():.2f} 元")
        print("="*50 + "\n")

# ==============================================================================
# 2. 数据处理与加载 (关键修复: 补全特征计算)
# ==============================================================================
def load_and_merge_data():
    def load_one(path, is_aux=False):
        if not os.path.exists(path):
            print(f"❌ 找不到文件: {path}"); sys.exit(1)
        
        df = pd.read_csv(path)
        if 'tx_local_time' in df.columns:
            df['datetime'] = pd.to_datetime(df['tx_local_time'], unit='ms')
            if df['datetime'].dt.tz is None:
                df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
            df.set_index('datetime', inplace=True)
        
        agg_rules = {
            'price': 'last', 'tick_vol': 'sum', 'tick_amt': 'sum', 'tick_vwap': 'mean',
            'premium_rate': 'last', 'sentiment': 'last',
            'bp1':'last', 'bv1':'last', 'sp1':'last', 'sv1':'last'
        }
        if 'index_price' in df.columns: agg_rules['index_price'] = 'last' # 修复 Missing index_price
        
        for i in range(2, 6):
            if f'bp{i}' in df.columns:
                agg_rules[f'bp{i}'] = 'last'; agg_rules[f'bv{i}'] = 'last'
                agg_rules[f'sp{i}'] = 'last'; agg_rules[f'sv{i}'] = 'last'
                
        if 'fut_price' in df.columns and not is_aux: agg_rules['fut_price'] = 'last'

        df = df.resample(CONFIG['RESAMPLE_FREQ']).agg(agg_rules).ffill().dropna()
        
        # --- 补全特征计算 (与 getfactor.py 对齐) ---
        df['mid_price'] = (df['bp1'] + df['sp1']) / 2
        df['log_ret'] = np.log(df['mid_price'] / df['mid_price'].shift(1)).fillna(0)
        
        depth_l1 = df['bv1'] + df['sv1'] + 1e-6
        df['feat_imb'] = (df['bv1'] - df['sv1']) / depth_l1
        df['feat_spread'] = (df['sp1'] - df['bp1']) / df['mid_price']
        
        if 'bv5' in df.columns:
            bv_all = sum(df[f'bv{i}'] for i in range(1, 6))
            sv_all = sum(df[f'sv{i}'] for i in range(1, 6))
            df['feat_depth_total'] = bv_all + sv_all
            df['feat_imb_5'] = (bv_all - sv_all) / (df['feat_depth_total'] + 1e-6)
        
        if 'bp5' in df.columns:
            df['feat_bid_slope'] = (df['bp1'] - df['bp5']) / 5
            df['feat_ask_slope'] = (df['sp5'] - df['sp1']) / 5
        else:
            df['feat_bid_slope'] = 0; df['feat_ask_slope'] = 0
        
        depth_amt = (df['bv1'] * df['bp1']) + (df['sv1'] * df['sp1'])
        df['feat_trade_intensity'] = df['tick_amt'] / (depth_amt + 1e-6)

        df['feat_vol_chg'] = np.log(df['tick_vol'] + 1).diff().fillna(0)
        ma_20 = df['mid_price'].rolling(20).mean()
        std_20 = df['mid_price'].rolling(20).std()
        df['feat_z_score'] = (df['mid_price'] - ma_20) / (std_20 + 1e-6)
        # ----------------------------------------
        
        if is_aux: df = df.add_prefix('ctx_')
        return df

    print("⏳ 正在加载并合并数据...")
    df_m = load_one(CONFIG['MAIN_FILE'], is_aux=False)
    df_a = load_one(CONFIG['AUX_FILE'], is_aux=True)
    
    df_final = df_m.join(df_a, how='inner').dropna()
    print(f"✅ 数据加载完毕，共 {len(df_final)} 个 3s 周期数据点")
    return df_final

# ==============================================================================
# 3. 主回测逻辑
# ==============================================================================
# ==============================================================================
# 3. 主回测逻辑 (诊断版 - 带Debug输出)
# ==============================================================================
def run_simulation():
    df = load_and_merge_data()
    manual_gen = ManualFactorGenerator()
    
    if not os.path.exists(CONFIG['ARTIFACT_PATH']):
        print(f"❌ 模型文件未找到: {CONFIG['ARTIFACT_PATH']}"); return

    try:
        artifact = torch.load(CONFIG['ARTIFACT_PATH'], map_location=CONFIG['DEVICE'], weights_only=False)
    except Exception as e:
        print(f"⚠️ 加载失败，尝试旧方式: {e}")
        artifact = torch.load(CONFIG['ARTIFACT_PATH'], map_location=CONFIG['DEVICE'])

    feature_names = artifact['features']['input_names']
    scaler = artifact['models']['direction_scaler']
    
    model = Direction(len(feature_names), d_model=256, num_layers=6).to(CONFIG['DEVICE'])
    model.load_state_dict(artifact['models']['direction_state_dict'])
    model.eval()
    
    executor = OrderExecutor(initial_cash=100000.0)
    
    print("\n▶️ 开始逐行回放 (Simulating)...")
    print(f"🔍 [诊断信息] 数据长度: {len(df)} | 特征数量: {len(feature_names)}")
    
    # 特征工程
    df_feat = manual_gen.process(df)
    df_all = df.join(df_feat, how='inner')
    
    input_data_raw = df_all[feature_names].values
    input_data_norm = scaler.transform(np.nan_to_num(input_data_raw))
    full_tensor = torch.FloatTensor(input_data_norm).to(CONFIG['DEVICE'])
    
    lookback = CONFIG['MAX_LOOKBACK']
    
    # 统计变量
    max_score = -999.0
    min_score = 999.0
    debug_print_count = 0
    
    for t in range(lookback, len(df_all)):
        # 1. 推理
        slice_tensor = full_tensor[t-lookback : t].unsqueeze(0)
        with torch.no_grad():
            prob = torch.softmax(model(slice_tensor), dim=1)
            # score = P(涨) - P(跌)
            score = (prob[0, 1] - prob[0, 2]).item()
            
        # 记录极值
        max_score = max(max_score, score)
        min_score = min(min_score, score)
        
        snapshot = df_all.iloc[t]
        ts_time = snapshot.name.time()

        # 2. 诊断打印 (当信号稍强时打印，看看为何没成交)
        if abs(score) > 0.2 and debug_print_count < 10: # 只打印前10次，避免刷屏
            print(f"   [DEBUG {ts_time}] Score:{score:.4f} | Pos:{executor.position} | 阈值:0.4")
            debug_print_count += 1
            
        # 3. 尾盘清仓
        if ts_time >= dt_time(14, 57):
            if executor.position > 0:
                print(f"⏰ [尾盘] 强制清仓 @ {ts_time}")
                order = {
                    'action': 'CLOSE', 'direction': -1, 'vol': executor.position,
                    'price': snapshot['bp1'], 'type': 'MARKET_FORCE', 'urgency': 'HIGH'
                }
                executor.match_order(order, snapshot)
            continue
            
        is_noon = dt_time(13,0) <= ts_time <= dt_time(13,5)
        
        # 4. 信号逻辑
        target_dir = 0
        # 逻辑：只有空仓时才做多，持仓时才平仓。不做空。
        if executor.position == 0 and score > 0: target_dir = 1
        elif executor.position > 0 and score < 0: target_dir = -1
        
        if target_dir != 0:
            price = snapshot['price']
            ask1 = snapshot['sp1']
            bid1 = snapshot['bp1']
            
            # 生成订单 (内部有 abs_score > 0.4 的检查)
            order = executor.generate_order(score, price, ask1, bid1, is_noon)
            if order:
                if order['action'] == 'OPEN' and executor.position == 0:
                    executor.match_order(order, snapshot)
                elif order['action'] == 'CLOSE' and executor.position > 0:
                    order['vol'] = executor.position
                    executor.match_order(order, snapshot)
    
    print("\n" + "-"*50)
    print(f"🔍 [诊断总结]")
    print(f"   最大 Score: {max_score:.4f}")
    print(f"   最小 Score: {min_score:.4f}")
    if max_score < 0.4 and min_score > -0.4:
        print("   ❌ 原因确认: 全天 Score 均未突破 +/- 0.4 阈值。")
        print("   💡 建议: 在 generate_order 中降低阈值 (如 0.4 -> 0.1) 进行测试。")
    elif max_score < 0.4 and min_score < -0.4:
        print("   ❌ 原因确认: 模型有看跌信号(Short)，但策略逻辑不允许裸做空(Pos=0时不做空)。")
    else:
        print("   ❓ 信号足够，但未成交，可能是撮合价格未满足 Maker 条件。")
    print("-"*50)

    executor.print_daily_report()
if __name__ == "__main__":
    run_simulation()