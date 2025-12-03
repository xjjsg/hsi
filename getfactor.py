import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import os
import math
import glob
import re
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 0. 项目环境配置 (Project Configuration)
# ==============================================================================
CONFIG = {
    # --- 路径配置 ---
    'DATA_DIR': './data',   # 数据根目录 HSI/data/
    'MAIN_SYMBOL': 'sz159920', # 主力标的代码 (文件夹名)
    'AUX_SYMBOL':  'sh513130', # 辅助标的代码 (文件夹名)
    
    # --- 训练参数 ---
    'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'MAX_LOOKBACK': 120,     
    'HORIZON': 20,           
    'RESAMPLE_FREQ': '3s',   
    'TRAIN_EPOCHS': 30,      
    'BARRIER_THRESHOLD': 0.002, 
    
    # --- 输出 ---
    'ARTIFACT_NAME': 'FACTOR_STRATEGY_ARTIFACT.pth',
    'FACTOR_LIB_NAME': 'factor_lib_final.csv'
}

print(f"🚀 Factor Factory Multi-Day Engine | Device: {CONFIG['DEVICE']}")

# ==============================================================================
# 1. 多日数据加载服务 (Multi-Day Data Loader)
# ==============================================================================
class DataLoaderService:
    @staticmethod
    def get_daily_files(symbol):
        """
        扫描 HSI/data/{symbol}/ 下的所有 csv 文件
        返回字典: { '2025-11-26': '完整路径', ... }
        """
        dir_path = os.path.join(CONFIG['DATA_DIR'], symbol)
        if not os.path.exists(dir_path):
            print(f"❌ 目录未找到: {dir_path}")
            return {}
        
        # 匹配 symbol-日期.csv 的模式
        pattern = os.path.join(dir_path, f"{symbol}-*.csv")
        files = glob.glob(pattern)
        
        date_map = {}
        for f in files:
            # 提取日期 (假设格式为 *-YYYY-MM-DD.csv)
            match = re.search(r'(\d{4}-\d{2}-\d{2})', os.path.basename(f))
            if match:
                date_str = match.group(1)
                date_map[date_str] = f
        
        return date_map

    @staticmethod
    def load_single_day(filepath, is_aux=False):
        """加载单日单文件并进行基础清洗"""
        try:
            raw = pd.read_csv(filepath)
            
            # 1. 时间处理
            if 'tx_local_time' in raw.columns:
                raw['datetime'] = pd.to_datetime(raw['tx_local_time'], unit='ms')
                if raw['datetime'].dt.tz is None:
                    raw['datetime'] = raw['datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
                df = raw.set_index('datetime').sort_index()
            else:
                df = raw

            # 2. 动态聚合 (L1-L5)
            agg_rules = {
                'price': 'last', 'tick_vol': 'sum', 'tick_amt': 'sum', 'tick_vwap': 'mean',
                'premium_rate': 'last', 'sentiment': 'last',
                'bp1':'last', 'bv1':'last', 'sp1':'last', 'sv1':'last'
            }
            if 'index_price' in df.columns: agg_rules['index_price'] = 'last'
            if not is_aux and 'fut_price' in df.columns: agg_rules['fut_price'] = 'last'
            
            for i in range(2, 6):
                if f'bp{i}' in df.columns:
                    agg_rules[f'bp{i}'] = 'last'; agg_rules[f'bv{i}'] = 'last'
                    agg_rules[f'sp{i}'] = 'last'; agg_rules[f'sv{i}'] = 'last'

            # 3. 重采样
            df = df.resample(CONFIG['RESAMPLE_FREQ']).agg(agg_rules).ffill().dropna()

            # 4. 基础特征 (日内独立计算)
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

            if is_aux: df = df.add_prefix('ctx_')
            
            return df
        except Exception as e:
            print(f"❌ 读取错误 {filepath}: {e}")
            return None

# ==============================================================================
# 2. 神经网络组件 (Model Components - 保持不变)
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
        std = torch.sqrt(torch.clamp(self.pool(torch.cat([pad, x**2], dim=2)) - mean**2, min=1e-6))
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
        out = self.conv1(x)[:, :, :-self.chomp] 
        out = self.relu1(out)
        out = self.conv2(out)[:, :, :-self.chomp]
        out = self.relu2(out)
        res = x if self.downsample is None else self.downsample(x)
        return out + res

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

class HybridMinerNet(nn.Module):
    def __init__(self, input_dim, d_model=128, n_factors=128):
        super(HybridMinerNet, self).__init__()
        self.alpha_layer = AlphaLayer(input_dim)
        self.tcn = TemporalBlock(input_dim*3, d_model, kernel_size=3, dilation=1)
        self.pos_encoder = SinusoidalPosEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=256, dropout=0.1)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=3)
        self.factor_head = nn.Sequential(nn.Linear(d_model, 256), nn.GELU(), nn.Linear(256, n_factors), nn.Tanh())
        self.predictor = nn.Linear(n_factors, 1)
    def forward(self, x):
        x = self.alpha_layer(x)
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = x.permute(2, 0, 1)
        x = self.pos_encoder(x)
        factors = self.factor_head(self.transformer(x)[-1]) 
        return self.predictor(factors), factors

# ==============================================================================
# 3. 手工因子生成器 (Manual Factor Injection)
# ==============================================================================
class ManualFactorGenerator:
    def process(self, df):
        # 针对单日 DataFrame 进行处理，无需担心跨日问题
        res = pd.DataFrame(index=df.index)
        
        # 1. 资金流
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

        # 2. 深度微观 (L5)
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

        # 3. 跨品种
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

        # 4. 场景
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

# ==============================================================================
# 4. 深度模型管理器 (支持多日列表输入)
# ==============================================================================
class DeepModelManager:
    def __init__(self, name, model_cls, input_cols, lookback, n_factors=128, is_cls=False):
        self.name = name
        self.model_cls = model_cls
        self.input_cols = input_cols
        self.lookback = lookback
        self.n_factors = n_factors
        self.is_cls = is_cls
        self.trained_model = None
        self.trained_scaler = None

    def _prepare_single_day(self, df, fit=False, scaler=None):
        """对单日数据进行标准化和滑窗切片，避免跨日污染"""
        raw = df[self.input_cols].values
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 标准化
        if fit:
            # 如果是 Fit 阶段，先只 Transform (Fit 在外部统一做)
            # 或者这里假设 scaler 已经 ready
            data = scaler.transform(raw)
        else:
            data = scaler.transform(raw)
            
        # 滑窗切片
        X_list = []
        # 注意: 每一天前 lookback 个数据是无效的，丢弃
        for i in range(self.lookback, len(data)):
            X_list.append(data[i-self.lookback : i])
        
        if len(X_list) == 0: return None, None
        X = np.array(X_list)
        
        # 生成标签
        y = None
        if fit:
            if self.is_cls:
                # 三势垒 (锁定 0.002)
                prices = df['mid_price'].values
                labels = np.zeros(len(data))
                horizon = CONFIG['HORIZON']
                threshold = CONFIG['BARRIER_THRESHOLD']
                
                valid_len = len(prices) - horizon
                for i in range(valid_len):
                    curr = prices[i]
                    future_window = prices[i+1 : i+horizon+1]
                    if np.any(future_window >= curr * (1 + threshold)):
                        labels[i] = 1 # 涨
                    elif np.any(future_window <= curr * (1 - threshold)):
                        labels[i] = 2 # 跌
                y = labels[self.lookback:] # 对齐 X
            else:
                # 波动率
                vol = df['log_ret'].rolling(20).std().shift(-20).values * 10000
                y = np.nan_to_num(vol[self.lookback:], nan=0)
        
        return X, y

    def train(self, df_list):
        print(f"\n🔄 [Training] 开始训练模型: {self.name}")
        
        # 1. 全局 Scaler Fit (抽取所有数据计算均值方差)
        print("   -> 计算全局统计量 (Scaler Fit)...")
        all_raw_data = []
        # 采样部分数据以节省内存，或者全部
        for df in df_list:
            all_raw_data.append(df[self.input_cols].values)
        
        full_matrix = np.concatenate(all_raw_data, axis=0)
        full_matrix = np.nan_to_num(full_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.trained_scaler = StandardScaler()
        self.trained_scaler.fit(full_matrix)
        del full_matrix, all_raw_data # 释放内存

        # 2. 逐日生成训练集 (X, y)
        print("   -> 逐日生成张量 (避免跨日污染)...")
        X_all, y_all = [], []
        
        # 留最后几天做验证 (80/20 Split by Days)
        train_days = int(len(df_list) * 0.8)
        train_df_list = df_list[:train_days]
        
        for df in train_df_list:
            X_day, y_day = self._prepare_single_day(df, fit=True, scaler=self.trained_scaler)
            if X_day is not None:
                # 再次对齐长度
                min_len = min(len(X_day), len(y_day))
                X_all.append(X_day[:min_len])
                y_all.append(y_day[:min_len])
        
        X_train = np.concatenate(X_all, axis=0)
        y_train = np.concatenate(y_all, axis=0)
        
        print(f"   -> 训练集形状: {X_train.shape}")
        
        ds = TensorDataset(torch.FloatTensor(X_train), 
                           torch.LongTensor(y_train) if self.is_cls else torch.FloatTensor(y_train))
        dl = DataLoader(ds, batch_size=64, shuffle=True)
        
        # 初始化模型
        input_dim = len(self.input_cols)
        if self.is_cls:
            model = self.model_cls(input_dim, d_model=256, num_layers=6).to(CONFIG['DEVICE'])
            loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.2, 1.0, 1.0]).to(CONFIG['DEVICE']))
            lr = 5e-5
        else:
            model = self.model_cls(input_dim, d_model=128, num_layers=3).to(CONFIG['DEVICE'])
            loss_fn = nn.MSELoss()
            lr = 1e-4
        
        opt = optim.AdamW(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG['TRAIN_EPOCHS'], eta_min=1e-6)
        
        model.train()
        for epoch in range(CONFIG['TRAIN_EPOCHS']):
            total_loss = 0
            for bx, by in dl:
                bx, by = bx.to(CONFIG['DEVICE']), by.to(CONFIG['DEVICE'])
                opt.zero_grad()
                out = model(bx)
                loss = loss_fn(out if self.is_cls else out[0].squeeze(), by)
                loss.backward()
                opt.step()
                total_loss += loss.item()
            
            scheduler.step()
            if (epoch+1) % 5 == 0:
                print(f"   Epoch {epoch+1:02d}/{CONFIG['TRAIN_EPOCHS']} | Loss: {total_loss/len(dl):.6f} | LR: {opt.param_groups[0]['lr']:.2e}")
        
        self.trained_model = model
        return model

    def process(self, df_list):
        # 训练
        model = self.train(df_list)
        model.eval()
        
        # 推理 (对每一天分别推理，然后拼接结果)
        print("   -> 开始推理...")
        all_results_df = []
        
        for df in df_list:
            X_day, _ = self._prepare_single_day(df, fit=False, scaler=self.trained_scaler)
            if X_day is None: continue
            
            X_tensor = torch.FloatTensor(X_day).to(CONFIG['DEVICE'])
            outs = []
            with torch.no_grad():
                for i in range(0, len(X_tensor), 256):
                    batch = X_tensor[i:i+256]
                    res = model(batch)
                    if self.is_cls:
                        prob = torch.softmax(res, dim=1)
                        outs.append((prob[:,1]-prob[:,2]).cpu().numpy())
                    else:
                        outs.append(res[1].cpu().numpy())
            
            vals = np.concatenate(outs)
            # 生成该日的 Result DataFrame (对齐索引)
            # 注意: X_day 是从 lookback 开始的
            day_res = pd.DataFrame(index=df.index[self.lookback:])
            # 裁剪以防万一
            min_l = min(len(vals), len(day_res))
            day_res = day_res.iloc[:min_l]
            vals = vals[:min_l]
            
            if self.is_cls:
                day_res[f'alpha_{self.name}_score'] = vals
            else:
                for k in range(vals.shape[1]): day_res[f'alpha_{self.name}_{k:03d}'] = vals[:, k]
            
            all_results_df.append(day_res)
            
        return pd.concat(all_results_df, axis=0)

# ==============================================================================
# 5. 主流程 (Multi-Day Pipeline)
# ==============================================================================
def main():
    # 1. 扫描所有日期文件
    print(f"📂 扫描数据目录: {CONFIG['DATA_DIR']} ...")
    main_files = DataLoaderService.get_daily_files(CONFIG['MAIN_SYMBOL'])
    aux_files = DataLoaderService.get_daily_files(CONFIG['AUX_SYMBOL'])
    
    # 找交集日期 (确保同一天都有数据)
    common_dates = sorted(list(set(main_files.keys()) & set(aux_files.keys())))
    print(f"✅ 找到 {len(common_dates)} 个共同交易日: {common_dates[:3]} ...")
    
    if not common_dates:
        print("❌ 没有找到共同日期的文件，请检查文件名格式 (symbol-YYYY-MM-DD.csv)")
        return

    # 2. 逐日加载、合并、注入因子 (预处理)
    daily_df_list = []
    manual_gen = ManualFactorGenerator()
    
    print("\n⚡ [Preprocessing] 逐日清洗与因子注入...")
    for date in common_dates:
        # 加载
        f_main = main_files[date]
        f_aux = aux_files[date]
        
        df_m = DataLoaderService.load_single_day(f_main, is_aux=False)
        df_a = DataLoaderService.load_single_day(f_aux, is_aux=True)
        
        if df_m is None or df_a is None: continue
        
        # 合并
        df_day = df_m.join(df_a, how='inner')
        if len(df_day) < 200: continue # 跳过数据太少的日子
        
        # 注入手工因子 (Day-by-Day, safe from cross-day leakage)
        df_manual = manual_gen.process(df_day)
        df_final = df_day.join(df_manual, how='inner')
        
        daily_df_list.append(df_final)
        
    print(f"📊 预处理完成，有效天数: {len(daily_df_list)}")
    if not daily_df_list: return

    # 3. 确定特征列表 (取第一天的数据来获取列名)
    sample_df = daily_df_list[0]
    excludes = ['tx_server_time', 'datetime']
    feats = [c for c in sample_df.columns if c not in excludes and np.issubdtype(sample_df[c].dtype, np.number)]
    print(f"🔹 模型特征维度: {len(feats)}")

    # 4. 初始化模型管理器
    dir_mgr = DeepModelManager("direction", Direction, feats, lookback=CONFIG['MAX_LOOKBACK'], is_cls=True)
    miner_mgr = DeepModelManager("miner", HybridMinerNet, feats, lookback=60, n_factors=128, is_cls=False)

    # 5. 训练与推理 (传入列表，内部处理)
    # Direction
    res_dir = dir_mgr.process(daily_df_list)
    # Miner
    res_miner = miner_mgr.process(daily_df_list)
    
    # 手工因子结果 (拼接所有天)
    # 注意: 需要先把 daily_df_list 里的手工因子提取出来拼在一起，方便最后导出
    # 我们只需保留手工因子列
    manual_cols = [c for c in sample_df.columns if c.startswith('alpha_') or c.startswith('logic_')]
    res_manual_list = [day[manual_cols].iloc[CONFIG['MAX_LOOKBACK']:] for day in daily_df_list] # 简单截断对齐
    res_manual = pd.concat(res_manual_list, axis=0)

    # 6. 最终合并与筛选
    # 对齐索引 (Inner Join)
    final_df = pd.concat([res_dir, res_miner, res_manual], axis=1).dropna()
    
    # 构造 Target (用于 IC 计算)
    # 需要对每一天分别算 ret shift，再拼起来
    target_list = []
    for day in daily_df_list:
        # Shift -20
        t = day['log_ret'].shift(-20).iloc[CONFIG['MAX_LOOKBACK']:]
        target_list.append(t)
    target = pd.concat(target_list, axis=0).reindex(final_df.index).fillna(0)
    
    print("\n🔍 计算 IC 并筛选...")
    ic_map = {}
    for c in final_df.columns:
        if final_df[c].std() == 0: continue
        corr = spearmanr(final_df[c].values, target.values)[0]
        if not np.isnan(corr): ic_map[c] = abs(corr)
    
    selected_factors = sorted(ic_map.keys(), key=lambda x: ic_map[x], reverse=True)[:135]
    final_output = final_df[selected_factors]

    # 7. 打包
    strategy_artifact = {
        'meta': {
            'description': 'Multi-Day Hybrid Strategy',
            'train_dates': common_dates,
            'input_feature_count': len(feats),
            'output_factor_count': len(selected_factors),
            'lookback_direction': CONFIG['MAX_LOOKBACK'],
            'threshold': CONFIG['BARRIER_THRESHOLD']
        },
        'features': {'input_names': feats, 'output_names': selected_factors},
        'models': {
            'direction_state_dict': dir_mgr.trained_model.state_dict(),
            'direction_scaler': dir_mgr.trained_scaler,
            'miner_state_dict': miner_mgr.trained_model.state_dict(),
            'miner_scaler': miner_mgr.trained_scaler,
        }
    }
    
    torch.save(strategy_artifact, CONFIG['ARTIFACT_NAME'])
    final_output.to_csv(CONFIG['FACTOR_LIB_NAME'])
    
    print(f"\n✅ 全部完成! 包含 {len(common_dates)} 天的训练成果已保存。")

if __name__ == "__main__":
    main()