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
import warnings

warnings.filterwarnings('ignore')

# ================= 配置区域 =================
CONFIG = {
    'MAIN_FILE': 'sz159920.csv',
    'AUX_FILE':  'sh513130.csv',
    'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'MAX_LOOKBACK': 120,   # 模型回看窗口
    'HORIZON': 20,         # 预测未来窗口
    'RESAMPLE': '3s',      # 数据重采样频率
    'TRAIN_EPOCHS': 15,    # 训练轮数
    'ARTIFACT_NAME': 'FACTOR_STRATEGY_ARTIFACT.pth' # 最终输出给交易模型的文件名
}

print(f"🚀 Factor Factory Engine | Device: {CONFIG['DEVICE']}")

# ==============================================================================
# 1. 数据加载与特征工程 (完整版)
# ==============================================================================
class DataLoaderService:
    @staticmethod
    def load_and_process(filepath, is_aux=False):
        if not os.path.exists(filepath): return None
        try:
            raw = pd.read_csv(filepath)
            if 'tx_local_time' in raw.columns:
                raw['datetime'] = pd.to_datetime(raw['tx_local_time'], unit='ms')
                if raw['datetime'].dt.tz is None:
                    raw['datetime'] = raw['datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
                df = raw.set_index('datetime').sort_index()
            else:
                df = raw
            
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

            df = df.resample(CONFIG['RESAMPLE']).agg(agg_rules).ffill().dropna()

            # --- 特征计算 ---
            df['mid_price'] = (df['bp1'] + df['sp1']) / 2
            df['log_ret'] = np.log(df['mid_price'] / df['mid_price'].shift(1)).fillna(0)
            
            total_depth = df['bv1'] + df['sv1'] + 1e-6
            df['feat_imb'] = (df['bv1'] - df['sv1']) / total_depth
            df['feat_spread'] = (df['sp1'] - df['bp1']) / df['mid_price']
            
            # 高级微观特征
            if 'bp5' in df.columns:
                df['feat_bid_slope'] = (df['bp1'] - df['bp5']) / 5
                df['feat_ask_slope'] = (df['sp5'] - df['sp1']) / 5
            else:
                df['feat_bid_slope'] = 0; df['feat_ask_slope'] = 0
            
            depth_amt = (df['bv1'] * df['bp1']) + (df['sv1'] * df['sp1'])
            df['feat_trade_intensity'] = df['tick_amt'] / (depth_amt + 1e-6)

            wmp = (df['bp1'] * df['sv1'] + df['sp1'] * df['bv1']) / total_depth
            df['feat_wmp_bias'] = (wmp - df['mid_price']) / df['mid_price']
            
            if 'tick_vwap' in df.columns:
                df['feat_vwap_bias'] = (df['tick_vwap'] - df['mid_price']) / df['mid_price']
            else:
                df['feat_vwap_bias'] = 0
                
            df['feat_vol_chg'] = np.log(df['tick_vol'] + 1).diff().fillna(0)
            ma_20 = df['mid_price'].rolling(20).mean()
            std_20 = df['mid_price'].rolling(20).std()
            df['feat_z_score'] = (df['mid_price'] - ma_20) / (std_20 + 1e-6)

            if is_aux: df = df.add_prefix('ctx_')
            return df
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

    @staticmethod
    def get_merged_data():
        df_main = DataLoaderService.load_and_process(CONFIG['MAIN_FILE'], is_aux=False)
        df_aux = DataLoaderService.load_and_process(CONFIG['AUX_FILE'], is_aux=True)
        if df_main is None: return None
        if df_aux is not None: df = df_main.join(df_aux, how='inner')
        else: df = df_main
        return df.replace([np.inf, -np.inf], np.nan).fillna(0)

# ==============================================================================
# 2. 神经网络组件 (含 TemporalBlock 修复)
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
        # 动态切除右侧 Padding (防止未来函数)
        out = self.conv1(x)[:, :, :-self.chomp]
        out = self.relu1(out)
        out = self.conv2(out)[:, :, :-self.chomp]
        out = self.relu2(out)
        res = x if self.downsample is None else self.downsample(x)
        return out + res

# --- 模型定义 ---
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()
        self.b1 = nn.Conv1d(in_channels, out_channels//4, 1)
        self.b2 = nn.Sequential(nn.Conv1d(in_channels, out_channels//4, 1), nn.Conv1d(out_channels//4, out_channels//4, 3, padding=1))
        self.b3 = nn.Sequential(nn.Conv1d(in_channels, out_channels//4, 1), nn.Conv1d(out_channels//4, out_channels//4, 5, padding=2))
        self.b4 = nn.Sequential(nn.MaxPool1d(3, 1, 1), nn.Conv1d(in_channels, out_channels//4, 1))
    def forward(self, x): return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)

class DeepLOB_Ultimate(nn.Module):
    def __init__(self, input_dim, d_model=256, nhead=16, num_layers=6, num_classes=3):
        super(DeepLOB_Ultimate, self).__init__()
        self.stem = nn.Sequential(nn.Conv1d(input_dim, 64, 1), nn.BatchNorm1d(64), nn.LeakyReLU())
        self.inception = InceptionBlock(64, d_model)
        self.se = SEBlock(d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, d_model, 5000))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=512, dropout=0.1)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Sequential(nn.Linear(d_model, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, num_classes))
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.se(self.inception(self.stem(x)))
        x = x.permute(2, 0, 1)
        x = x + self.pos_encoder[:, :, :x.size(0)].permute(2,0,1)
        x = self.transformer(x)
        return self.fc(x.mean(dim=0))

class HybridMinerNet(nn.Module):
    def __init__(self, input_dim, d_model=128, n_factors=128):
        super(HybridMinerNet, self).__init__()
        self.alpha_layer = AlphaLayer(input_dim)
        self.tcn = TemporalBlock(input_dim*3, d_model, kernel_size=3, dilation=1)
        self.pos_encoder = nn.Parameter(torch.randn(1, d_model, 5000))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=256, dropout=0.1)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=3)
        self.factor_head = nn.Sequential(nn.Linear(d_model, 256), nn.GELU(), nn.Linear(256, n_factors), nn.Tanh())
        self.predictor = nn.Linear(n_factors, 1)
    def forward(self, x):
        x = self.alpha_layer(x)
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = x.permute(2, 0, 1)
        x = x + self.pos_encoder[:, :, :x.size(0)].permute(2,0,1)
        factors = self.factor_head(self.transformer(x)[-1])
        return self.predictor(factors), factors

# ==============================================================================
# 3. 因子管理器
# ==============================================================================
class FactorGenerator:
    def process(self, df): raise NotImplementedError

class ManualFactorGenerator(FactorGenerator):
    def process(self, df):
        print("🛠️ 计算手工因子...")
        res = pd.DataFrame(index=df.index)
        # Micro
        db = df['bp1'].diff(); ds = df['sp1'].diff()
        dvb = df['bv1'].diff(); dvs = df['sv1'].diff()
        delta_vb = np.select([db > 0, db < 0], [df['bv1'], 0], default=dvb)
        delta_va = np.select([ds > 0, ds < 0], [0, df['sv1']], default=dvs)
        voi = delta_vb - delta_va
        res['alpha_voi'] = voi
        # Cross
        if 'ctx_mid_price' in df.columns:
            res['alpha_cross_rs'] = df['log_ret'] - df['ctx_log_ret']
            spread = np.log(df['mid_price']) - np.log(df['ctx_mid_price'])
            res['alpha_cross_arb_z'] = (spread - spread.rolling(100).mean()) / (spread.rolling(100).std() + 1e-6)
        # Scenario
        if 'fut_price' in df.columns:
            fut_ret = df['fut_price'].pct_change().fillna(0)
            res['alpha_fut_lead'] = fut_ret - df['log_ret']
        return res.fillna(0)

class DeepModelManager(FactorGenerator):
    def __init__(self, name, model_cls, input_cols, lookback, n_factors=128, is_cls=False):
        self.name = name
        self.model_cls = model_cls
        self.input_cols = input_cols
        self.lookback = lookback
        self.n_factors = n_factors
        self.is_cls = is_cls
        self.trained_model = None
        self.trained_scaler = None

    def _prepare(self, df, fit=False, scaler=None):
        raw = df[self.input_cols].values
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        if fit:
            scaler = StandardScaler()
            data = scaler.fit_transform(raw)
        else:
            data = scaler.transform(raw) if scaler else raw
            
        X_list = []
        for i in range(CONFIG['MAX_LOOKBACK'], len(data)):
            X_list.append(data[i-self.lookback : i])
        X = np.array(X_list)
        
        y = None
        if fit:
            if self.is_cls:
                fut = df['mid_price'].shift(-CONFIG['HORIZON']).values
                curr = df['mid_price'].values
                ret = (fut - curr) / (curr + 1e-6)
                labels = np.zeros(len(data))
                labels[ret > 0.001] = 1; labels[ret < -0.001] = 2
                y = labels[CONFIG['MAX_LOOKBACK']:]
            else:
                vol = df['log_ret'].rolling(20).std().shift(-20).values * 1000
                y = np.nan_to_num(vol[CONFIG['MAX_LOOKBACK']:], nan=0)
        return torch.FloatTensor(X), y, scaler

    def train(self, df):
        print(f"🔄 训练模型: {self.name}")
        train_df = df.iloc[:int(len(df)*0.8)]
        X, y, scaler = self._prepare(train_df, fit=True)
        X, y = X[:-CONFIG['HORIZON']], y[:-CONFIG['HORIZON']]
        
        ds = TensorDataset(X, torch.LongTensor(y) if self.is_cls else torch.FloatTensor(y))
        dl = DataLoader(ds, batch_size=64, shuffle=True)
        
        model = self.model_cls(len(self.input_cols), n_factors=self.n_factors).to(CONFIG['DEVICE']) \
                if not self.is_cls else self.model_cls(len(self.input_cols)).to(CONFIG['DEVICE'])
        
        lr = 5e-5 if self.is_cls else 1e-4
        opt = optim.AdamW(model.parameters(), lr=lr)
        
        if self.is_cls:
            weights = torch.FloatTensor([0.2, 1.0, 1.0]).to(CONFIG['DEVICE'])
            loss_fn = nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fn = nn.MSELoss()
        
        model.train()
        for epoch in range(CONFIG['TRAIN_EPOCHS']):
            for bx, by in dl:
                bx, by = bx.to(CONFIG['DEVICE']), by.to(CONFIG['DEVICE'])
                opt.zero_grad()
                out = model(bx)
                loss = loss_fn(out if self.is_cls else out[0].squeeze(), by)
                loss.backward()
                opt.step()
        
        self.trained_model = model
        self.trained_scaler = scaler
        return model, scaler

    def process(self, df):
        model, scaler = self.train(df)
        model.eval()
        X, _, _ = self._prepare(df, fit=False, scaler=scaler)
        X = X.to(CONFIG['DEVICE'])
        outs = []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                batch = X[i:i+256]
                res = model(batch)
                if self.is_cls:
                    prob = torch.softmax(res, dim=1)
                    outs.append((prob[:,1]-prob[:,2]).cpu().numpy())
                else:
                    outs.append(res[1].cpu().numpy())
        res_df = pd.DataFrame(index=df.index[CONFIG['MAX_LOOKBACK']:])
        vals = np.concatenate(outs)
        if self.is_cls: res_df[f'alpha_{self.name}_score'] = vals
        else:
            for i in range(vals.shape[1]): res_df[f'alpha_{self.name}_{i:03d}'] = vals[:, i]
        return res_df

# ==============================================================================
# 4. 主程序 (生成 Artifact)
# ==============================================================================
def main():
    df = DataLoaderService.get_merged_data()
    if df is None: return

    excludes = ['tx_server_time', 'datetime']
    feats = [c for c in df.columns if c not in excludes and np.issubdtype(df[c].dtype, np.number)]
    print(f"📊 输入特征维度: {len(feats)}")

    # 1. 实例化生成器
    manual_gen = ManualFactorGenerator()
    dir_mgr = DeepModelManager("dl_dir", DeepLOB_Ultimate, feats, lookback=120, is_cls=True)
    miner_mgr = DeepModelManager("miner", HybridMinerNet, feats, lookback=60, n_factors=128, is_cls=False)

    # 2. 执行计算
    gens = [manual_gen, dir_mgr, miner_mgr]
    results = []
    for g in gens:
        try: results.append(g.process(df))
        except Exception as e: print(f"Error {g}: {e}")
            
    if not results: return
    
    # 3. 筛选有效因子
    raw_df = pd.concat(results, axis=1).dropna()
    target = df['log_ret'].shift(-20).reindex(raw_df.index).fillna(0)
    
    ic_map = {}
    for c in raw_df.columns:
        if raw_df[c].std() == 0: continue
        corr = spearmanr(raw_df[c].values, target.values)[0]
        if not np.isnan(corr): ic_map[c] = abs(corr)
    
    # 选出 Top 135 (或150)
    selected_factors = sorted(ic_map.keys(), key=lambda x: ic_map[x], reverse=True)[:135]
    final_df = raw_df[selected_factors]
    
    print(f"✅ 筛选出 Top {len(selected_factors)} 因子，正在打包...")

    # ==========================================================================
    # 4. 关键：生成策略 Artifact (交易模型直接使用包)
    # ==========================================================================
    strategy_artifact = {
        'meta': {
            'description': 'Hybrid Factor Strategy Artifact',
            'input_feature_count': len(feats),
            'output_factor_count': len(selected_factors),
            'lookback': CONFIG['MAX_LOOKBACK']
        },
        'features': {
            'input_names': feats,           # 必须严格按此顺序构造输入
            'output_names': selected_factors # 最终因子的名称列表
        },
        'models': {
            # DeepLOB (方向预测)
            'dl_state_dict': dir_mgr.trained_model.state_dict(),
            'dl_scaler': dir_mgr.trained_scaler,
            # Miner (因子挖掘)
            'miner_state_dict': miner_mgr.trained_model.state_dict(),
            'miner_scaler': miner_mgr.trained_scaler,
        }
    }
    
    torch.save(strategy_artifact, CONFIG['ARTIFACT_NAME'])
    final_df.to_csv("factor_lib_final.csv")
    
    print(f"\n📦 策略打包完成: {CONFIG['ARTIFACT_NAME']}")
    print(f"   -> 包含模型权重: ✅")
    print(f"   -> 包含标准化参数: ✅")
    print(f"   -> 包含特征映射表: ✅")
    print("现在，你可以将此文件直接加载到交易模型中。")

if __name__ == "__main__":
    main()