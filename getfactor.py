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
# 0. 全局配置
# ==============================================================================
CONFIG = {
    # --- 路径配置 ---
    'DATA_DIR': './data',   
    'MAIN_SYMBOL': 'sz159920', # 主力标的
    'AUX_SYMBOL':  'sh513130', # 辅助标的 (HSTECH)
    # --- 计算设备 ---

    'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
    # --- 训练核心参数 ---
    'MAX_LOOKBACK': 60,      # 输入窗口：3分钟 (60 tick)
    'HORIZON': 60,           # 预测窗口：3分钟 (60 tick) [基于波动率分析结论]
    'RESAMPLE_FREQ': '3s',   
    'TRAIN_EPOCHS': 30,      
    'BATCH_SIZE': 512,       
    'LEARNING_RATE': 1e-4,
    
    # --- 交易成本参数 ---
    'COST_THRESHOLD': 0.001, # 刚性成本 0.1% (用于标签过滤)
    'SIM_COST_TAKER': 0.0006, # Taker 费率 (手续费+冲击)
    'SIM_COST_MAKER': 0.0001, # Maker 费率 (仅手续费)
    
    # --- 输出 ---
    'ARTIFACT_NAME': 'FACTOR_STRATEGY_ARTIFACT.pth',
    'FACTOR_LIB_NAME': 'factor_lib_final.csv'
}

print(f"🚀 Factor Factory Final | Horizon: 3min | Valid: SimTrader PnL")

# ==============================================================================
# 1. 核心组件：损失函数与模型
# ==============================================================================

class WeightedMSELoss(nn.Module):
    """
    波动率加权损失：重罚错过大波动的行为
    """
    def __init__(self, penalty_factor=5.0):
        super().__init__()
        self.penalty = penalty_factor
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, target):
        loss = self.mse(pred.view(-1), target.view(-1))
        abs_target = torch.abs(target.view(-1))
        # 权重 = 1.0 + Penalty * (波动幅度 / 成本阈值)
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
    """
    轻量化模型：Stem + TimeEmb + 2-Layer Transformer
    """
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=3):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv1d(input_dim, d_model, 1),
            nn.BatchNorm1d(d_model),
            nn.LeakyReLU()
        )
        self.time_emb = Time2Vec(output_dim=d_model)
        
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=128, dropout=0.3, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.se_fc = nn.Sequential(
            nn.Linear(d_model, d_model // 4, bias=False),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model, bias=False),
            nn.Sigmoid()
        )
        
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1) 
        )

    def forward(self, x):
        x_time = x[:, :, -1:] 
        x_feat = x[:, :, :-1]
        
        t_emb = self.time_emb(x_time)
        
        h = x_feat.permute(0, 2, 1) 
        h = self.stem(h)
        
        b, c, t = h.size()
        y = self.avg_pool(h).view(b, c)
        y = self.se_fc(y).view(b, c, 1)
        h = h * y.expand_as(h)
        
        h = h.permute(0, 2, 1)
        h = h + t_emb
        
        h = self.transformer(h)
        return self.head(h[:, -1, :])

# ==============================================================================
# 2. 模拟交易引擎 (Validation Engine)
# ==============================================================================
class SimTrader:
    """
    集成了动态挂单管理和午盘套利逻辑的验证引擎
    """
    @staticmethod
    def run_simulation(model, df, feature_cols):
        model.eval()
        
        # 1. 准备数据
        raw_vals = df[feature_cols].values
        raw_vals = np.clip(raw_vals, -10, 10)
        raw_vals = np.nan_to_num(raw_vals)
        
        prices = df['mid_price'].values
        # 尝试获取买一卖一，如果没有则用 Mid 模拟
        if 'sp1' in df.columns:
            ask_prices = df['sp1'].values
            bid_prices = df['bp1'].values
        else:
            ask_prices = prices * 1.0005
            bid_prices = prices * 0.9995
            
        index_prices = df['index_price'].values if 'index_price' in df.columns else np.zeros_like(prices)
        times = df.index.time
        
        # 2. 批量推理
        lookback = CONFIG['MAX_LOOKBACK']
        X_batch = []
        for i in range(lookback, len(raw_vals)):
            X_batch.append(raw_vals[i-lookback : i])
            
        if not X_batch: return 0.0
        
        preds = []
        batch_size = 1024
        with torch.no_grad():
            for i in range(0, len(X_batch), batch_size):
                bx = np.array(X_batch[i:i+batch_size])
                tensor = torch.FloatTensor(bx).to(CONFIG['DEVICE'])
                out = model(tensor)
                preds.extend(out.cpu().view(-1).numpy())

        # 3. 逐 Tick 事件循环
        cash = 100000.0
        position = 0
        pending_order = None # {'direction': 1/-1, 'price': float, 'start_tick': int}
        
        noon_close_idx = -1
        noon_open_checked = False
        
        taker_fee = CONFIG['SIM_COST_TAKER']
        maker_fee = CONFIG['SIM_COST_MAKER']
        
        for i, pred_ret in enumerate(preds):
            idx = lookback + i
            curr_time = times[idx]
            curr_mid = prices[idx]
            curr_ask = ask_prices[idx]
            curr_bid = bid_prices[idx]
            curr_idx_price = index_prices[idx]
            
            # --- 日内风控: 14:55 清仓 ---
            if curr_time >= dt_time(14, 55):
                if position > 0:
                    cash += position * curr_bid * (1 - taker_fee)
                    position = 0
                pending_order = None
                continue
            
            # --- 策略: 午盘套利 ---
            if curr_time <= dt_time(11, 30, 5) and curr_time >= dt_time(11, 29, 0):
                noon_close_idx = idx
            
            if not noon_open_checked and curr_time >= dt_time(13, 0, 0):
                noon_open_checked = True
                if noon_close_idx != -1:
                    p_close_etf = prices[noon_close_idx]
                    p_close_idx = index_prices[noon_close_idx]
                    p_open_etf = curr_mid 
                    p_open_idx = curr_idx_price 
                    if p_close_idx > 0 and p_close_etf > 0:
                        idx_ret = p_open_idx / p_close_idx - 1
                        etf_ret = p_open_etf / p_close_etf - 1
                        gap = idx_ret - etf_ret
                        # 强力买入/卖出信号
                        if gap > 0.002: pred_ret = max(pred_ret, 0.01)
                        elif gap < -0.002: pred_ret = min(pred_ret, -0.01)

            # --- 策略: 挂单成交判定 ---
            if pending_order:
                if pending_order['direction'] == 1: # 买单
                    if curr_ask <= pending_order['price']:
                        vol = int(50000 / pending_order['price'] / 100) * 100
                        if cash >= vol * pending_order['price'] * (1 + maker_fee):
                            cash -= vol * pending_order['price'] * (1 + maker_fee)
                            position = vol
                            pending_order = None 
                elif pending_order['direction'] == -1: # 卖单
                    if curr_bid >= pending_order['price']:
                        cash += position * pending_order['price'] * (1 - maker_fee)
                        position = 0
                        pending_order = None 
            
            # --- 策略: 订单生成与动态调整 ---
            if position > 0:
                # 动态调整卖出价
                target_sell_price = curr_mid * (1 + max(0, pred_ret))
                if pred_ret < -0.0005: # 预测反转，市价止损
                    cash += position * curr_bid * (1 - taker_fee)
                    position = 0
                    pending_order = None
                else:
                    # 改单逻辑
                    if pending_order is None or abs(target_sell_price - pending_order['price']) / curr_mid > 0.0002:
                        pending_order = {'direction': -1, 'price': target_sell_price, 'start_tick': idx}
            
            elif position == 0:
                # 开仓逻辑
                if pred_ret > CONFIG['COST_THRESHOLD'] * 1.2:
                    target_buy_price = curr_mid 
                    if pending_order is None or abs(target_buy_price - pending_order['price']) / curr_mid > 0.0002:
                        pending_order = {'direction': 1, 'price': target_buy_price, 'start_tick': idx}
                else:
                    if pending_order and pending_order['direction'] == 1:
                        pending_order = None

            # 超时撤单
            if pending_order and (idx - pending_order['start_tick']) > 60:
                pending_order = None
                
        # 最终结算
        if position > 0:
            cash += position * prices[-1] * (1 - taker_fee)
            
        return cash - 100000.0

# ==============================================================================
# 3. 数据工厂与文件提取 (恢复原始稳健逻辑)
# ==============================================================================

class DataLoaderService:
    @staticmethod
    def get_daily_files(symbol):
        """
        扫描 HSI/data/{symbol}/ 下的所有 csv 文件
        返回字典: { '2025-11-26': '完整路径', ... }
        参考原始代码逻辑
        """
        dir_path = os.path.join(CONFIG['DATA_DIR'], symbol)
        if not os.path.exists(dir_path):
            print(f"❌ 目录未找到: {dir_path}")
            return {}
        
        pattern = os.path.join(dir_path, f"{symbol}-*.csv")
        files = glob.glob(pattern)
        
        date_map = {}
        # 兼容性修复：使用 Regex 提取完整日期 YYYY-MM-DD
        date_pattern = re.compile(r"(\d{4}-\d{2}-\d{2})")
        
        for f in files:
            match = date_pattern.search(os.path.basename(f))
            if match:
                date_str = match.group(1)
                date_map[date_str] = f
        
        return date_map

    @staticmethod
    def load_single_day(filepath, is_aux=False):
        """
        加载单日数据并进行聚合
        """
        try:
            raw = pd.read_csv(filepath)
            
            # 时间处理
            if 'tx_local_time' in raw.columns:
                raw['datetime'] = pd.to_datetime(raw['tx_local_time'], unit='ms')
                raw['datetime'] = raw['datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
                df = raw.set_index('datetime').sort_index()
            else:
                return None

            # 聚合规则
            agg_rules = {
                'price': 'last', 'tick_vol': 'sum', 'tick_amt': 'sum',
                'bp1':'last', 'bv1':'last', 'sp1':'last', 'sv1':'last'
            }
            if 'index_price' in df.columns: agg_rules['index_price'] = 'last'
            for c in ['bp2','bv2','sp2','sv2','bp5','bv5','sp5','sv5']:
                if c in df.columns: agg_rules[c] = 'last'

            df = df.resample(CONFIG['RESAMPLE_FREQ']).agg(agg_rules).ffill().dropna()

            # 基础特征
            df['mid_price'] = (df['bp1'] + df['sp1']) / 2
            df['log_ret'] = np.log(df['mid_price'] / df['mid_price'].shift(1)).fillna(0)
            
            if is_aux: df = df.add_prefix('ctx_')
            return df
        except Exception as e:
            print(f"❌ 读取错误 {filepath}: {e}")
            return None

class ManualFactorGenerator:
    """
    特征工程：使用 Rolling Z-Score 进行动态归一化
    """
    def process(self, df):
        res = pd.DataFrame(index=df.index)
        
        # 1. 价格相对特征 (动态 Z-Score)
        ma_20 = df['mid_price'].rolling(20).mean()
        std_20 = df['mid_price'].rolling(20).std()
        res['feat_dist_ma20'] = (df['mid_price'] - ma_20) / (std_20 + 1e-6)
        res['feat_vol_20'] = std_20 / ma_20
        
        # 2. 资金流 VOI (比率归一化)
        db = df['bp1'].diff(); ds = df['sp1'].diff()
        dvb = df['bv1'].diff(); dvs = df['sv1'].diff()
        delta_vb = np.select([db > 0, db < 0], [df['bv1'], 0], default=dvb)
        delta_va = np.select([ds > 0, ds < 0], [0, df['sv1']], default=dvs)
        voi = delta_vb - delta_va
        
        vol_ma = df['tick_vol'].rolling(60).mean().replace(0, 1)
        res['feat_voi_norm'] = voi / (vol_ma + 1e-6)
        
        # 3. 深度不平衡
        depth = df['bv1'] + df['sv1'] + 1e-6
        res['feat_imb'] = (df['bv1'] - df['sv1']) / depth
        res['feat_spread'] = (df['sp1'] - df['bp1']) / df['mid_price']
        
        # 4. 辅助数据增强 (HSTECH)
        if 'ctx_mid_price' in df.columns:
            res['feat_cross_rs'] = df['log_ret'] - df['ctx_log_ret']
            
            ctx_depth = df['ctx_bv1'] + df['ctx_sv1'] + 1e-6
            ctx_imb = (df['ctx_bv1'] - df['ctx_sv1']) / ctx_depth
            res['feat_imb_div'] = ctx_imb - res['feat_imb']
            
            spread = np.log(df['mid_price']) - np.log(df['ctx_mid_price'])
            sp_mean = spread.rolling(120).mean()
            sp_std = spread.rolling(120).std()
            res['feat_cross_z'] = (spread - sp_mean) / (sp_std + 1e-6)

        # 5. 时间特征
        minutes = df.index.hour * 60 + df.index.minute
        res['feat_time_norm'] = (minutes - 570) / (900 - 570)
        
        return res.fillna(0)

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
            raw = np.clip(raw, -10, 10)
            raw = np.nan_to_num(raw)
            
            # Target: 预测 Horizon 后的累计收益
            target = df['log_ret'].rolling(CONFIG['HORIZON']).sum().shift(-CONFIG['HORIZON'])
            target = target.fillna(0).values
            
            # 标签过滤 (Soft Threshold)：仅在训练时应用
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
        
        self.model = QuantModel(input_dim=len(self.feat_cols)-1).to(CONFIG['DEVICE'])
        criterion = WeightedMSELoss(penalty_factor=10.0) 
        optimizer = optim.AdamW(self.model.parameters(), lr=CONFIG['LEARNING_RATE'])
        
        best_pnl = -99999.0
        
        print(f"\n🔥 开始训练 | 目标: SimTrader PnL (Dynamic Orders)")
        print("-" * 65)
        
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
            
            # 验证环节：直接跑模拟交易 (使用最新的一天数据)
            valid_pnl = SimTrader.run_simulation(self.model, valid_dfs[-1], self.feat_cols)
            
            save_mark = ""
            if valid_pnl > best_pnl:
                best_pnl = valid_pnl
                torch.save(self.model.state_dict(), "best_model.pth")
                save_mark = "🏆 New Best!"
            
            avg_loss = total_loss / len(train_dl)
            print(f"Epoch {epoch+1:02d} | W-Loss: {avg_loss:.6f} | Sim PnL: {valid_pnl:.2f} {save_mark}")
            
        if os.path.exists("best_model.pth"):
            self.model.load_state_dict(torch.load("best_model.pth"))
            os.remove("best_model.pth")
            
        return best_pnl

# ==============================================================================
# 5. 主程序 (Main Pipeline)
# ==============================================================================
def main():
    # 1. 扫描所有日期文件 (恢复原始的 get_daily_files 逻辑)
    print(f"📂 扫描数据目录: {CONFIG['DATA_DIR']} ...")
    main_files_map = DataLoaderService.get_daily_files(CONFIG['MAIN_SYMBOL'])
    aux_files_map = DataLoaderService.get_daily_files(CONFIG['AUX_SYMBOL'])
    
    # 找交集日期
    common_dates = sorted(list(set(main_files_map.keys()) & set(aux_files_map.keys())))
    
    if len(common_dates) < 2:
        print(f"❌ 数据不足! 仅找到 {len(common_dates)} 个有效日期对。需要至少 2 天 (1 Train + 1 Valid)。")
        print(f"   Debug: Main Keys: {list(main_files_map.keys())[:5]}")
        return

    # 2. N-1 滚动切分
    if len(common_dates) < 60:
        split_idx = len(common_dates) - 1 
    else:
        split_idx = len(common_dates) - 5 
        
    train_dates = common_dates[:split_idx]
    valid_dates = common_dates[split_idx:]
    
    print(f"📅 训练集: {len(train_dates)} 天 | 验证集: {len(valid_dates)} 天 ({valid_dates[0]}~{valid_dates[-1]})")
    
    gen = ManualFactorGenerator()
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

    # 3. 确定特征列表
    feat_cols = [c for c in first_df.columns if c.startswith('feat_')]
    if 'feat_time_norm' in feat_cols:
        feat_cols.remove('feat_time_norm')
        feat_cols.append('feat_time_norm') # 确保时间特征在最后

    print(f"🔹 特征维度: {len(feat_cols)}")

    # 4. 训练与验证
    mgr = DeepModelManager(feat_cols)
    final_pnl = mgr.run(train_dfs, valid_dfs)
    
    print(f"\n✅ 最终模型 PnL (验证集): {final_pnl:.2f}")

    # 5. 保存成果
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