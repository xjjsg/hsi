import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib
import math

# 检测 GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 计算设备: {DEVICE}")

# ==============================================================================
# 1. 专业级数据处理器 (针对 34 列数据定制)
# ==============================================================================
class DataProcessorPro:
    def __init__(self, filepath, lookback=60, horizon=20):
        self.filepath = filepath
        self.lookback = lookback  # 输入过去 3分钟 (60 * 3s)
        self.horizon = horizon    # 预测未来 1分钟
        self.scaler = StandardScaler()

    def load_and_process(self):
        print(f"⚡ 正在读取全量数据: {self.filepath}")
        try:
            # 1. 读取数据
            raw = pd.read_csv(self.filepath)
            
            # 2. 时间索引处理 (关键)
            # 使用 tx_local_time (毫秒时间戳) 最准确，tx_server_time 是字符串，不好处理
            raw['datetime'] = pd.to_datetime(raw['tx_local_time'], unit='ms')
            
            # 转换为北京时间 (如果机器是UTC)
            if raw['datetime'].dt.tz is None:
                raw['datetime'] = raw['datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
            
            df = raw.set_index('datetime').sort_index()

            # 3. 数据重采样 (Resampling) - 必须覆盖所有关键字段
            # 我们将数据降频到 3秒 一个点，以减少噪音
            agg_rules = {
                # 价格与基础
                'price': 'last', 'tick_vol': 'sum', 'tick_amt': 'sum', 'tick_vwap': 'mean',
                'premium_rate': 'last', 'iopv': 'last', 'index_price': 'last', 
                'fx_rate': 'last', 'sentiment': 'last', 'interval_s': 'sum',
                
                # 五档盘口 (L1-L5)
                'bp1':'last', 'bv1':'last', 'sp1':'last', 'sv1':'last',
                'bp2':'last', 'bv2':'last', 'sp2':'last', 'sv2':'last',
                'bp3':'last', 'bv3':'last', 'sp3':'last', 'sv3':'last',
                'bp4':'last', 'bv4':'last', 'sp4':'last', 'sv4':'last',
                'bp5':'last', 'bv5':'last', 'sp5':'last', 'sv5':'last',
            }
            
            # 兼容期货字段 (如果有)
            if 'fut_price' in df.columns:
                agg_rules.update({
                    'fut_price': 'last', 'fut_mid': 'last', 
                    'fut_imb': 'mean', 'fut_delta_vol': 'sum', 'fut_pct': 'last'
                })

            # 执行重采样
            df = df.resample('3s').agg(agg_rules).ffill().dropna()

            # 4. === 深度特征工程 (Hardcore Feature Engineering) ===
            print("正在构建高阶特征...")

            # --- A. 基础微观特征 ---
            df['mid_price'] = (df['bp1'] + df['sp1']) / 2
            df['log_ret'] = np.log(df['mid_price'] / df['mid_price'].shift(1)).fillna(0)
            
            # --- B. Smart Money 痕迹 (VWAP Bias) ---
            # 如果成交均价 > 中间价，说明买方在主动向上吃单
            df['vwap_bias'] = (df['tick_vwap'] - df['mid_price']) / df['mid_price'] * 10000

            # --- C. 深度失衡 (Weighted Depth Imbalance) ---
            # 越靠近盘口的挂单，权重越大
            weights = [1.0, 0.8, 0.6, 0.4, 0.2]
            sum_bid = sum(df[f'bv{i}'] * w for i, w in zip(range(1, 6), weights))
            sum_ask = sum(df[f'sv{i}'] * w for i, w in zip(range(1, 6), weights))
            df['depth_imb'] = (sum_bid - sum_ask) / (sum_bid + sum_ask + 1e-6)

            # --- D. 盘口斜率 (Order Book Slope) ---
            # 判断 L1 到 L5 的价格分布是否陡峭
            # 斜率越小，说明挂单越密集，支撑/压力越强
            df['bid_slope'] = (df['bp1'] - df['bp5']) / 5
            df['ask_slope'] = (df['sp5'] - df['sp1']) / 5
            
            # --- E. 广义流动性 (Total Liquidity) ---
            df['total_depth'] = np.log(sum_bid + sum_ask + 1)

            # --- F. 期现联动 (Futures Basis) ---
            if 'fut_price' in df.columns:
                # 基差率
                df['basis_rate'] = (df['fut_price'] - df['price']) / df['price']
                # 期货买卖压力
                df['fut_pressure'] = df['fut_imb']

            # --- G. 情绪加速 (Sentiment Momentum) ---
            df['sent_acc'] = df['sentiment'].diff().fillna(0)

            # 5. 剔除无用列，保留纯数值特征
            # 我们不需要 tx_server_time 等字符串了
            # 将所有计算好的特征放入 feature_cols
            drop_cols = ['tx_server_time', 'tx_local_time', 'bd_server_time', 'bd_local_time']
            # 只保留数值类型的列
            numeric_df = df.select_dtypes(include=[np.number])
            
            # 6. 构建预测目标 (Target): 未来波动率
            # 预测未来 Horizon 内的对数收益率标准差
            indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=self.horizon)
            target_vol = df['log_ret'].rolling(window=indexer).std() * 10000 # 放大为 bp
            
            # 最终清洗
            final_df = numeric_df.copy()
            final_df['target'] = target_vol
            final_df = final_df.replace([np.inf, -np.inf], 0).dropna()
            
            self.feature_cols = [c for c in final_df.columns if c != 'target']
            print(f"✅ 特征工程完成，输入维度: {len(self.feature_cols)} (含L5深度+期货数据)")
            
            return final_df, self.feature_cols
            
        except Exception as e:
            print(f"数据处理出错: {e}")
            return None, []

    def get_tensors(self, df, feature_cols, fit_scaler=False):
        data = df[feature_cols].values
        target = df['target'].values
        
        if fit_scaler:
            data = self.scaler.fit_transform(data)
        else:
            data = self.scaler.transform(data)
            
        X, y = [], []
        # 滑动窗口切片
        for i in range(self.lookback, len(data)):
            X.append(data[i-self.lookback : i])
            y.append(target[i])
            
        return np.array(X), np.array(y)

# ==============================================================================
# 2. 混合挖掘模型 (Hybrid Miner Architecture)
# ==============================================================================
class AlphaLayer(nn.Module):
    """ 模拟 Quant 手工挖掘: 自动计算滚动均值、波动率 """
    def __init__(self, input_dim, window=20):
        super(AlphaLayer, self).__init__()
        self.window = window
        self.pool = nn.AvgPool1d(kernel_size=window, stride=1, padding=0)
    
    def forward(self, x):
        x = x.permute(0, 2, 1) # (Batch, Feat, Seq)
        # Padding
        pad = torch.zeros(x.shape[0], x.shape[1], self.window-1).to(x.device)
        x_pad = torch.cat([pad, x], dim=2)
        
        mean = self.pool(x_pad)
        x2_pad = torch.cat([pad, x**2], dim=2)
        mean_sq = self.pool(x2_pad)
        std = torch.sqrt(torch.clamp(mean_sq - mean**2, min=1e-6))
        
        # 拼接: 原始 + 均值 + 波动率
        out = torch.cat([x, mean, std], dim=1)
        return out.permute(0, 2, 1)

class TemporalBlock(nn.Module):
    """ TCN Block: 捕捉局部突变 """
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        self.act1 = nn.GELU()
        self.do1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        self.act2 = nn.GELU()
        self.do2 = nn.Dropout(dropout)
        self.chomp = padding
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

    def forward(self, x):
        out = self.conv1(x)[:, :, :-self.chomp]
        out = self.act1(self.do1(out))
        out = self.conv2(out)[:, :, :-self.chomp]
        out = self.act2(self.do2(out))
        res = x if self.downsample is None else self.downsample(x)
        return out + res

class HybridMinerNet(nn.Module):
    def __init__(self, input_dim, d_model=128, n_factors=128):
        super(HybridMinerNet, self).__init__()
        
        # 1. 特征裂变 (Input -> 3x Input)
        self.alpha_layer = AlphaLayer(input_dim, window=20)
        
        # 2. 局部感知 (TCN)
        # 将裂变后的特征压缩到 d_model
        self.tcn = TemporalBlock(input_dim*3, d_model, kernel_size=3, dilation=1)
        
        # 3. 全局注意力 (Transformer)
        self.pos_encoder = self._make_pos_encoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=256, dropout=0.1)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=3)
        
        # 4. 因子生成头 (输出 128 个因子)
        self.factor_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, n_factors),
            nn.Tanh() # 归一化到 [-1, 1]
        )
        
        # 5. 辅助预测 (Loss来源)
        self.predictor = nn.Linear(n_factors, 1)

    def _make_pos_encoding(self, d_model, max_len=5000):
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return nn.Parameter(pe.unsqueeze(0).transpose(0, 1), requires_grad=False)

    def forward(self, x):
        # x: (Batch, Seq, Feat)
        x = self.alpha_layer(x)       # -> (Batch, Seq, Feat*3)
        
        x = x.permute(0, 2, 1)        # -> (Batch, Channel, Seq) for TCN
        x = self.tcn(x)               # -> (Batch, d_model, Seq)
        
        x = x.permute(2, 0, 1)        # -> (Seq, Batch, d_model) for Transformer
        x = x + self.pos_encoder[:x.size(0), :]
        x = self.transformer(x)
        
        last_step = x[-1, :, :]       # (Batch, d_model)
        factors = self.factor_head(last_step) # (Batch, 128)
        pred = self.predictor(factors)
        
        return pred, factors

# ==============================================================================
# 3. 训练与提取流程
# ==============================================================================
def run_mining():
    FILE = 'sz159920.csv' # 确保你的CSV在这个路径
    
    # 1. 加载处理
    proc = DataProcessorPro(FILE)
    df, feat_cols = proc.load_and_process()
    
    if df is None: return

    # 划分数据集
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]
    
    X_train, y_train = proc.get_tensors(train_df, feat_cols, fit_scaler=True)
    X_test, y_test = proc.get_tensors(test_df, feat_cols, fit_scaler=False)
    
    # 保存 Scaler (用于实时实盘)
    joblib.dump(proc.scaler, 'miner_scaler.pkl')
    
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=128, shuffle=True
    )
    
    # 2. 初始化模型
    model = HybridMinerNet(input_dim=len(feat_cols), d_model=128, n_factors=128).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # 3. 训练
    print("\n🚀 开始挖掘 (AlphaNet + TCN + Transformer)...")
    model.train()
    for epoch in range(50):
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred, _ = model(X)
            loss = criterion(pred.squeeze(), y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss {total_loss/len(train_loader):.6f}")
        
    torch.save(model.state_dict(), 'miner_model_128.pth')
    print("模型已保存 -> miner_model_128.pth")
    
    # 4. 导出因子
    print("\n正在导出 128 维合成因子...")
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
        _, factors = model(X_test_tensor)
        factors_np = factors.cpu().numpy()
        
        # 构造 DataFrame
        cols = [f'Latent_{i:03d}' for i in range(128)]
        # 注意索引对齐 (lookback=60)
        valid_idx = test_df.index[60:]
        factor_df = pd.DataFrame(factors_np, columns=cols, index=valid_idx)
        factor_df['target_vol'] = y_test
        
        # 简单验证
        ic_scores = []
        for c in cols:
            ic = factor_df[c].corr(factor_df['target_vol'])
            ic_scores.append((c, abs(ic)))
        ic_scores.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Top 3 因子 IC: {ic_scores[:3]}")
        factor_df.to_csv("mined_128_factors.csv")
        print("✅ 因子表已保存: mined_128_factors.csv")

if __name__ == "__main__":
    run_mining()