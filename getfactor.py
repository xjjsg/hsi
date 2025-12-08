import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import glob
import re
import warnings
import math
from datetime import datetime, time as dt_time

warnings.filterwarnings('ignore')

# ==============================================================================
# 0. 全局配置 (v5.2 维度修复版)
# ==============================================================================
CONFIG = {
    # --- 路径配置 ---
    'DATA_DIR': './data',   
    'MAIN_SYMBOL': 'sz159920', 
    'AUX_SYMBOL':  'sh513130', 
    # --- 计算设备 ---
    'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
    
    # --- 训练核心参数 ---
    'MAX_LOOKBACK': 60,      # 输入窗口：3分钟
    'HORIZON': 60,           # 预测窗口：3分钟
    'RESAMPLE_FREQ': '3s',   
    'TRAIN_EPOCHS': 50,      
    'BATCH_SIZE': 512,       
    'LEARNING_RATE': 1e-4,   
    
    # --- 交易成本参数 ---
    'COST_THRESHOLD': 0.0008, 
    'SIM_COST_TAKER': 0.0006, 
    'SIM_COST_MAKER': 0.0001, 
    
    # --- 输出 ---
    'ARTIFACT_NAME': 'FACTOR_STRATEGY_ARTIFACT.pth',
    'FACTOR_LIB_NAME': 'factor_lib_final.csv'
}

print(f"🚀 Factor Factory v5.2 | Dimension Fixed | 256-Dim Model")

# ==============================================================================
# 1. 核心组件：模型与损失
# ==============================================================================

class WeightedMSELoss(nn.Module):
    def __init__(self, penalty_factor=1.0): 
        super().__init__()
        self.penalty = penalty_factor
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, target):
        loss = self.mse(pred.view(-1), target.view(-1))
        abs_target = torch.abs(target.view(-1))
        weights = 1.0 + self.penalty * (abs_target / CONFIG['COST_THRESHOLD'])
        weighted_loss = (loss * weights).mean()
        return weighted_loss

class Time2Vec(nn.Module):
    def __init__(self, output_dim=8):
        super().__init__()
        self.w = nn.Parameter(torch.randn(1, output_dim))
        self.b = nn.Parameter(torch.randn(1, output_dim))
    
    def forward(self, x):
        return torch.sin(x @ self.w + self.b)

class QuantModel(nn.Module):
    def __init__(self, input_dim, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        
        # input_dim 必须是剔除时间特征后的维度
        self.stem = nn.Sequential(
            nn.Conv1d(input_dim, d_model, 1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.time_emb = Time2Vec(output_dim=d_model)
        
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=0.3, 
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        
        self.attn_pool = nn.Linear(d_model, 1)
        
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 1) 
        )

    def forward(self, x):
        # x: [Batch, Seq, Feat]
        # 最后一维是 feat_time_norm，单独提取
        x_time = x[:, :, -1:] 
        x_feat = x[:, :, :-1] # 这里的维度是 input_dim
        
        h = x_feat.permute(0, 2, 1) 
        h = self.stem(h)
        h = h.permute(0, 2, 1)      
        
        t_emb = self.time_emb(x_time)
        h = h + t_emb
        
        h = self.transformer(h) 
        last_step = h[:, -1, :]
        
        return self.head(last_step)

# ==============================================================================
# 2. 增强版模拟交易引擎
# ==============================================================================
class SimTrader:
    @staticmethod
    def calculate_metrics(equity_curve, trade_logs):
        if not trade_logs:
            return {'total_ret': 0, 'sharpe': 0, 'mdd': 0, 'win_rate': 0, 'score': -999, 'pl_ratio': 0}

        initial_capital = 100000.0
        final_equity = equity_curve[-1]
        total_ret = (final_equity - initial_capital) / initial_capital * 100

        equity_arr = np.array(equity_curve)
        peak = np.maximum.accumulate(equity_arr)
        drawdown = (peak - equity_arr) / peak
        max_dd = drawdown.max() * 100

        pnls = [t['net_pnl'] for t in trade_logs]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        win_rate = len(wins) / len(pnls) if len(pnls) > 0 else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 1e-6
        pl_ratio = avg_win / avg_loss

        if np.std(pnls) == 0: sharpe = 0
        else: sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(len(pnls)) 

        score = (total_ret * 1.0) + (sharpe * 2.0) + (win_rate * 10.0) - (max_dd * 3.0)

        return {
            'total_ret': round(total_ret, 3),
            'sharpe': round(sharpe, 3),
            'mdd': round(max_dd, 3),
            'win_rate': round(win_rate * 100, 1),
            'pl_ratio': round(pl_ratio, 2),
            'trade_count': len(pnls),
            'score': round(score, 3)
        }

    @staticmethod
    def run_simulation(model, df, feature_cols, verbose=False):
        model.eval()
        
        raw_vals = df[feature_cols].values
        raw_vals = np.clip(raw_vals, -10, 10)
        raw_vals = np.nan_to_num(raw_vals)
        
        prices = df['mid_price'].values
        if 'sp1' in df.columns:
            ask_prices = df['sp1'].values
            bid_prices = df['bp1'].values
            ask_vols = df['sv1'].values
            bid_vols = df['bv1'].values
        else:
            ask_prices = prices * 1.0005
            bid_prices = prices * 0.9995
            ask_vols = np.ones_like(prices) * 10000
            bid_vols = np.ones_like(prices) * 10000
            
        iopvs = df['iopv'].values if 'iopv' in df.columns else prices
        times = df.index.time
        
        lookback = CONFIG['MAX_LOOKBACK']
        X_batch = []
        for i in range(lookback, len(raw_vals)):
            X_batch.append(raw_vals[i-lookback : i])
            
        if not X_batch: return {'total_ret': 0, 'score': -999}
        
        preds = []
        batch_size = 512
        with torch.no_grad():
            for i in range(0, len(X_batch), batch_size):
                bx = np.array(X_batch[i:i+batch_size])
                tensor = torch.FloatTensor(bx).to(CONFIG['DEVICE'])
                out = model(tensor)
                preds.extend(out.cpu().view(-1).numpy())

        cash = 100000.0
        position = 0
        entry_price = 0.0
        pending_order = None 
        trade_logs = []
        equity_curve = []
        
        if verbose:
            print(f"\n{'='*20} 实盘模拟 (Smart Execution) {'='*20}")
        
        for i, pred_ret in enumerate(preds):
            idx = lookback + i
            curr_time = times[idx]
            curr_mid = prices[idx]
            curr_ask = ask_prices[idx]
            curr_bid = bid_prices[idx]
            curr_iopv = iopvs[idx]
            v_ask1 = ask_vols[idx]
            v_bid1 = bid_vols[idx]
            
            curr_equity = cash + (position * curr_mid if position > 0 else 0)
            equity_curve.append(curr_equity)
            
            # 14:55 强制清仓
            if curr_time >= dt_time(14, 55):
                if position > 0:
                    revenue = position * curr_bid * (1 - CONFIG['SIM_COST_TAKER'])
                    net_pnl = revenue - (position * entry_price)
                    trade_logs.append({'net_pnl': net_pnl})
                    cash += revenue
                    position = 0
                    if verbose: print(f"[{curr_time}] 🏁 FORCE CLOSE PnL: {net_pnl:.2f}")
                pending_order = None
                continue

            # 信号增强
            discount = (curr_mid / curr_iopv) - 1
            if discount < -0.0015: pred_ret += 0.0005 

            # 交易执行 (Maker)
            if pending_order:
                filled = False
                if pending_order['direction'] == 1: 
                    if curr_ask <= pending_order['price']:
                        filled = True
                    elif curr_bid == pending_order['price'] and v_bid1 < 5000:
                        filled = True 
                    if filled:
                        vol = int(50000 / pending_order['price'] / 100) * 100
                        cost = vol * pending_order['price'] * (1 + CONFIG['SIM_COST_MAKER'])
                        if cash >= cost:
                            cash -= cost
                            position = vol
                            entry_price = pending_order['price']
                            if verbose: print(f"[{curr_time}] 🔵 MAKER BUY {vol} @ {entry_price:.3f}")
                            pending_order = None

                elif pending_order['direction'] == -1:
                    if curr_bid >= pending_order['price'] or (curr_ask == pending_order['price'] and v_ask1 < 5000):
                        revenue = position * pending_order['price'] * (1 - CONFIG['SIM_COST_MAKER'])
                        net_pnl = revenue - (position * entry_price)
                        trade_logs.append({'net_pnl': net_pnl})
                        cash += revenue
                        position = 0
                        if verbose: print(f"[{curr_time}] 🟢 MAKER SELL PnL: {net_pnl:.2f}")
                        pending_order = None
                
                # 撤单
                if pending_order:
                    dist = abs(curr_mid - pending_order['price']) / curr_mid
                    if dist > 0.002: pending_order = None
                    elif (pending_order['direction']==1 and pred_ret < -0.0002): pending_order = None
                    elif (pending_order['direction']==-1 and pred_ret > 0.0002): pending_order = None

            # 开仓
            if position == 0 and not pending_order:
                if pred_ret > CONFIG['COST_THRESHOLD']:
                    # Smart Execution
                    if v_ask1 < 80000:
                        vol = int(50000 / curr_ask / 100) * 100
                        cost = vol * curr_ask * (1 + CONFIG['SIM_COST_TAKER'])
                        if cash >= cost:
                            cash -= cost
                            position = vol
                            entry_price = curr_ask
                            if verbose: print(f"[{curr_time}] ⚡ TAKER BUY (Thin Wall) @ {entry_price:.3f} (Sig:{pred_ret:.4f})")
                    else:
                        pending_order = {'direction': 1, 'price': curr_bid, 'start_tick': idx}
            
            # 持仓
            elif position > 0 and not pending_order:
                current_pnl_pct = (curr_bid / entry_price) - 1
                if current_pnl_pct < -0.003 and pred_ret < 0:
                    revenue = position * curr_bid * (1 - CONFIG['SIM_COST_TAKER'])
                    net_pnl = revenue - (position * entry_price)
                    trade_logs.append({'net_pnl': net_pnl})
                    cash += revenue
                    position = 0
                    if verbose: print(f"[{curr_time}] ⚠️ STOP LOSS PnL: {net_pnl:.2f}")
                elif pred_ret < -0.0001: 
                    pending_order = {'direction': -1, 'price': curr_ask, 'start_tick': idx}

        return SimTrader.calculate_metrics(equity_curve, trade_logs)

# ==============================================================================
# 3. 数据工厂
# ==============================================================================

class DataLoaderService:
    @staticmethod
    def get_daily_files(symbol):
        dir_path = os.path.join(CONFIG['DATA_DIR'], symbol)
        if not os.path.exists(dir_path): return {}
        files = glob.glob(os.path.join(dir_path, f"{symbol}-*.csv"))
        date_map = {}
        date_pattern = re.compile(r"(\d{4}-\d{2}-\d{2})")
        for f in files:
            match = date_pattern.search(os.path.basename(f))
            if match: date_map[match.group(1)] = f
        return date_map

    @staticmethod
    def load_single_day(filepath, is_aux=False):
        try:
            raw = pd.read_csv(filepath)
            if 'tx_local_time' in raw.columns:
                raw['datetime'] = pd.to_datetime(raw['tx_local_time'], unit='ms')
                raw['datetime'] = raw['datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
                df = raw.set_index('datetime').sort_index()
            else: return None

            agg_rules = {
                'price': 'last', 'tick_vol': 'sum', 'tick_amt': 'sum',
                'bp1':'last', 'bv1':'last', 'sp1':'last', 'sv1':'last'
            }
            if 'index_price' in df.columns: agg_rules['index_price'] = 'last'
            if 'iopv' in df.columns: agg_rules['iopv'] = 'last'
            if 'sentiment' in df.columns: agg_rules['sentiment'] = 'last'
            if 'fut_price' in df.columns: agg_rules['fut_price'] = 'last'
            
            for i in range(2, 6):
                for side in ['bp','bv','sp','sv']:
                    col = f'{side}{i}'
                    if col in df.columns: agg_rules[col] = 'last'

            df = df.resample(CONFIG['RESAMPLE_FREQ']).agg(agg_rules).ffill().dropna()
            df['mid_price'] = (df['bp1'] + df['sp1']) / 2
            
            # 计算 log_ret，修复 KeyError
            df['log_ret'] = np.log(df['mid_price'] / df['mid_price'].shift(1)).fillna(0)
            
            if is_aux: df = df.add_prefix('ctx_')
            return df
        except Exception:
            return None

class RobustFactorGenerator:
    """
    Target: 200+ Dimensions
    Logic: User's Sentiment Algo + Microstructure + Multi-scale Expansion
    """
    def process(self, df):
        res = pd.DataFrame(index=df.index)
        
        price = df['price']
        mid = (df['bp1'] + df['sp1']) / 2
        tick_vol = df['tick_vol'].replace(0, 1e-6)
        tick_amt = df['tick_amt']
        
        # --- A. 移植用户的 Sentiment 逻辑 (Self-Sentiment) ---
        vwap_3s = tick_amt / tick_vol
        diff_bp = (price - vwap_3s) / vwap_3s * 10000
        score_mom = diff_bp.clip(-4, 4) * 1.5
        
        score_struct = pd.Series(0, index=df.index)
        score_struct[price >= df['sp1']] = 2  
        score_struct[price <= df['bp1']] = -2 
        
        rolling_high = price.rolling(300).max()
        rolling_low = price.rolling(300).min()
        ctx_factor = pd.Series(1.0, index=df.index)
        cond_high = (abs(price - rolling_high)/rolling_high < 0.002) & (score_mom > 0)
        cond_low = (abs(price - rolling_low)/rolling_low < 0.002) & (score_mom < 0)
        ctx_factor[cond_high] = 1.2
        ctx_factor[cond_low] = 1.5
        
        vol_avg = tick_vol.rolling(20).mean() + 100
        vol_ratio = (tick_vol / vol_avg).clip(0, 4)
        
        raw_self_sent = (score_mom + score_struct) * vol_ratio * ctx_factor
        res['feat_self_sentiment'] = raw_self_sent.ewm(alpha=0.3).mean()
        
        # --- B. 核心基因子 (Base Factors) ---
        if 'sentiment' in df.columns: res['feat_mkt_sentiment'] = df['sentiment']
        else: res['feat_mkt_sentiment'] = 0

        if 'iopv' in df.columns:
            res['feat_premium'] = (price - df['iopv']) / (df['iopv'] + 1e-6) * 1000
        else: res['feat_premium'] = 0
            
        # OIR
        if 'bv1' in df.columns and 'sv1' in df.columns:
            depth_l1 = df['bv1'] + df['sv1'] + 1e-6
            res['feat_oir_l1'] = (df['bv1'] - df['sv1']) / depth_l1
            
            sum_bid = sum(df[f'bv{i}'] for i in range(1, 6) if f'bv{i}' in df.columns)
            sum_ask = sum(df[f'sv{i}'] for i in range(1, 6) if f'sv{i}' in df.columns)
            res['feat_oir_total'] = (sum_bid - sum_ask) / (sum_bid + sum_ask + 1e-6)
        
        # VOI
        db = df['bp1'].diff()
        ds = df['sp1'].diff()
        dvb = df['bv1'].diff()
        dvs = df['sv1'].diff()
        bid_chg = np.where(db > 0, df['bv1'], np.where(db == 0, dvb, 0))
        ask_chg = np.where(ds < 0, df['sv1'], np.where(ds == 0, dvs, 0))
        res['feat_voi'] = bid_chg - ask_chg

        if 'fut_price' in df.columns:
            res['feat_fut_basis'] = (df['fut_price'] - price) / price * 1000
        else: res['feat_fut_basis'] = 0

        # --- C. 因子裂变 (Expansion) ---
        base_factors = [c for c in res.columns]
        windows = [10, 30, 60] 
        
        for col in base_factors:
            series = res[col]
            for w in windows:
                res[f'{col}_diff_{w}'] = series.diff(w)
            
            res[f'{col}_std_20'] = series.rolling(20).std()
            
            z_score = (series - series.rolling(60).mean()) / (series.rolling(60).std() + 1e-6)
            res[f'{col}_z_60'] = z_score.clip(-3, 3)
            
            res[f'{col}_madist_60'] = series - series.rolling(60).mean()

        # --- D. Price Action ---
        # 复用已有的 log_ret
        res['feat_log_ret'] = np.log(mid / mid.shift(1)).fillna(0)
        for w in [10, 30, 60]:
            res[f'feat_vol_{w}'] = res['feat_log_ret'].rolling(w).std() * np.sqrt(w)

        # --- E. Interactions & Lags ---
        res['feat_sent_resonance'] = res['feat_self_sentiment'] * res['feat_mkt_sentiment']
        res['feat_pv_div'] = res['feat_log_ret'] * res['feat_voi']
        
        cols = res.columns.tolist()
        for col in cols:
            res[f'{col}_lag1'] = res[col].shift(1)

        res = res.replace([np.inf, -np.inf], np.nan).fillna(0)
        res = res.clip(lower=res.quantile(0.001), upper=res.quantile(0.999), axis=1)
        
        return res

# ==============================================================================
# 4. 训练管理
# ==============================================================================
class DeepModelManager:
    def __init__(self, feat_cols):
        self.feat_cols = feat_cols
        self.model = None

    def prepare_xy(self, df_list, is_train=True):
        X, y = [], []
        threshold = CONFIG['COST_THRESHOLD']
        
        for df in df_list:
            raw = df[self.feat_cols].values
            raw = np.nan_to_num(raw)
            
            target = df['log_ret'].rolling(CONFIG['HORIZON']).sum().shift(-CONFIG['HORIZON'])
            target = target.fillna(0).values
            
            if is_train:
                mask = np.abs(target) < threshold
                target[mask] = 0.0
            
            lookback = CONFIG['MAX_LOOKBACK']
            for i in range(lookback, len(raw) - CONFIG['HORIZON']):
                X.append(raw[i-lookback : i])
                y.append(target[i])
                
        return np.array(X), np.array(y)

    def run(self, train_dfs, valid_dfs):
        print(f"🔄 构建训练集 ({len(train_dfs)}天)...")
        X_train, y_train = self.prepare_xy(train_dfs, is_train=True)
        if len(X_train) == 0: return 0.0
        
        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        train_dl = DataLoader(train_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=True)
        
        input_dim = len(self.feat_cols)
        print(f"🔹 模型输入维度: {input_dim}")
        
        # [关键修复] input_dim - 1，减去时间维
        self.model = QuantModel(input_dim=input_dim - 1).to(CONFIG['DEVICE'])
        criterion = WeightedMSELoss(penalty_factor=1.0) 
        optimizer = optim.AdamW(self.model.parameters(), lr=CONFIG['LEARNING_RATE'])
        
        best_score = -999.0
        
        print(f"\n🔥 开始训练 | 目标: 复合因子最大化 (Score)")
        print(f"{'Epoch':<6} | {'Loss':<8} | {'Score':<8} | {'Ret(%)':<8} | {'Sharpe':<6} | {'MDD(%)':<6} | {'WinRate':<7}")
        print("-" * 80)
        
        for epoch in range(CONFIG['TRAIN_EPOCHS']):
            self.model.train()
            total_loss = 0
            for bx, by in train_dl:
                bx, by = bx.to(CONFIG['DEVICE']), by.to(CONFIG['DEVICE'])
                optimizer.zero_grad()
                pred = self.model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            metrics = SimTrader.run_simulation(self.model, valid_dfs[-1], self.feat_cols, verbose=False)
            valid_score = metrics['score']
            
            save_mark = ""
            if valid_score > best_score:
                best_score = valid_score
                torch.save(self.model.state_dict(), "best_model.pth")
                save_mark = "🏆"
            
            avg_loss = total_loss / len(train_dl)
            print(f"{epoch+1:02d}     | {avg_loss:.5f}  | {valid_score:.2f}     | {metrics['total_ret']:<8} | {metrics['sharpe']:<6} | {metrics['mdd']:<6} | {metrics['win_rate']}% {save_mark}")
            
        print("\n✨ 最佳模型详细回测 (验证集):")
        if os.path.exists("best_model.pth"):
            self.model.load_state_dict(torch.load("best_model.pth"))
            final_metrics = SimTrader.run_simulation(self.model, valid_dfs[-1], self.feat_cols, verbose=True)
            print("-" * 50)
            print(f"最终评分: {final_metrics['score']}")
            print(f"累计收益: {final_metrics['total_ret']}%")
            print(f"最大回撤: {final_metrics['mdd']}%")
            print(f"夏普比率: {final_metrics['sharpe']}")
            print(f"交易胜率: {final_metrics['win_rate']}%")
            print(f"盈亏比:   {final_metrics['pl_ratio']}")
            os.remove("best_model.pth")
            
        return best_score

# ==============================================================================
# 5. 主程序
# ==============================================================================
def main():
    print(f"📂 扫描数据目录: {CONFIG['DATA_DIR']} ...")
    main_files_map = DataLoaderService.get_daily_files(CONFIG['MAIN_SYMBOL'])
    aux_files_map = DataLoaderService.get_daily_files(CONFIG['AUX_SYMBOL'])
    
    common_dates = sorted(list(set(main_files_map.keys()) & set(aux_files_map.keys())))
    if len(common_dates) < 2:
        print(f"❌ 数据不足! 仅找到 {len(common_dates)} 个有效日期对。")
        return

    split_idx = len(common_dates) - 1 
    train_dates = common_dates[:split_idx]
    valid_dates = common_dates[split_idx:]
    print(f"📅 训练集: {len(train_dates)} 天 | 验证集: {len(valid_dates)} 天 ({valid_dates[0]}~{valid_dates[-1]})")
    
    gen = RobustFactorGenerator()
    train_dfs = []
    valid_dfs = []
    first_df = None
    
    print("⚡ 加载数据中...")
    for date in common_dates:
        f_m = main_files_map[date]
        f_a = aux_files_map[date]
        df_m = DataLoaderService.load_single_day(f_m, False)
        df_a = DataLoaderService.load_single_day(f_a, True)
        if df_m is not None and df_a is not None:
            df = df_m.join(df_a, how='inner')
            df_feat = gen.process(df)
            df_final = df.join(df_feat, how='inner').dropna()
            
            if date in train_dates: train_dfs.append(df_final)
            else: valid_dfs.append(df_final)
            if first_df is None: first_df = df_final

    feat_cols = [c for c in df_final.columns if c.startswith('feat_')]
    if 'feat_time_norm' in feat_cols:
        feat_cols.remove('feat_time_norm')
        feat_cols.append('feat_time_norm')

    print(f"🔹 最终特征维度: {len(feat_cols)}")
    mgr = DeepModelManager(feat_cols)
    mgr.run(train_dfs, valid_dfs)
    
    artifact = {
        'meta': {
            'features': feat_cols,
            'horizon': CONFIG['HORIZON'],
            'threshold': CONFIG['COST_THRESHOLD']
        },
        'state_dict': mgr.model.state_dict()
    }
    torch.save(artifact, CONFIG['ARTIFACT_NAME'])
    pd.Series(feat_cols).to_csv(CONFIG['FACTOR_LIB_NAME'], index=False, header=False)

if __name__ == "__main__":
    main()