import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib
import math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 计算设备: {DEVICE}")

# ==============================================================================
# 1. 模块 A: AlphaNet 风格的特征扩展层
# 模拟 Quant 的手工算子: TS_MEAN, TS_STD, TS_CORR
# ==============================================================================
class AlphaLayer(nn.Module):
    def __init__(self, input_dim, window=10):
        super(AlphaLayer, self).__init__()
        self.window = window
        # 使用 1D 卷积模拟滚动窗口操作 (固定权重不更新，或者可学习)
        # 这里为了效率，我们用 AvgPool1d 模拟 TS_MEAN
        self.pool = nn.AvgPool1d(kernel_size=window, stride=1, padding=0)
        
    def forward(self, x):
        # x shape: (Batch, Seq, Feat) -> Permute to (Batch, Feat, Seq) for Pool
        x = x.permute(0, 2, 1)
        
        # 1. 滚动均值 (Ts_Mean)
        # padding 确保输出长度一致 (Causal Padding)
        x_padded = torch.cat([torch.zeros(x.shape[0], x.shape[1], self.window-1).to(x.device), x], dim=2)
        ts_mean = self.pool(x_padded)
        
        # 2. 滚动标准差 (Ts_Std) approx: E[x^2] - (E[x])^2
        x_sq_padded = torch.cat([torch.zeros(x.shape[0], x.shape[1], self.window-1).to(x.device), x**2], dim=2)
        ts_sq_mean = self.pool(x_sq_padded)
        ts_var = ts_sq_mean - ts_mean**2
        ts_std = torch.sqrt(torch.clamp(ts_var, min=1e-6))
        
        # 3. 拼接: 原始特征 + 均值 + 标准差
        # Output shape: (Batch, Feat * 3, Seq)
        out = torch.cat([x, ts_mean, ts_std], dim=1)
        return out.permute(0, 2, 1) # Back to (Batch, Seq, Feat*3)

# ==============================================================================
# 2. 模块 B: TCN (时间卷积网络) - 捕捉局部突变
# 使用膨胀卷积 (Dilated Conv) 扩大感受野
# ==============================================================================
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # 确保因果性 (Causal): Padding 在左侧
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding) # 裁剪右侧多余的 Padding
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        # 残差连接 (ResNet)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

# ==============================================================================
# 3. 模块 C: 混合模型主架构 (Hybrid Net)
# 流程: Input -> AlphaLayer -> TCN -> Transformer -> Latent Factors
# ==============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x): return x + self.pe[:x.size(0), :]

class HybridMinerNet(nn.Module):
    def __init__(self, input_dim, d_model=64):
        super(HybridMinerNet, self).__init__()
        
        # 1. AlphaNet 层: 特征扩充 (Input: F -> Output: 3*F)
        self.alpha_layer = AlphaLayer(input_dim, window=10)
        expanded_dim = input_dim * 3
        
        # 2. TCN 层: 局部特征提取
        # 将扩充后的特征压缩映射到 d_model 维度
        self.tcn = TemporalBlock(expanded_dim, d_model, kernel_size=3, stride=1, dilation=1)
        
        # 3. Transformer 层: 全局上下文关联
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=128, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 4. 挖掘头 (Bottleneck): 输出 64 个合成因子
        self.factor_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.Tanh()
        )
        
        # 5. 预测头
        self.predictor = nn.Linear(64, 1)

    def forward(self, x):
        # x: (Batch, Seq, Feat)
        
        # --- Step 1: Alpha Expansion ---
        x_alpha = self.alpha_layer(x) # (Batch, Seq, 3*Feat)
        
        # --- Step 2: TCN Processing ---
        # TCN expects (Batch, Channel, Seq)
        x_tcn_in = x_alpha.permute(0, 2, 1) 
        x_tcn_out = self.tcn(x_tcn_in) # (Batch, d_model, Seq)
        
        # --- Step 3: Transformer Processing ---
        # Transformer expects (Seq, Batch, d_model)
        x_trans_in = x_tcn_out.permute(2, 0, 1) 
        x_trans_in = self.pos_encoder(x_trans_in)
        x_trans_out = self.transformer(x_trans_in) # (Seq, Batch, d_model)
        
        # --- Step 4: Latent Factor Extraction ---
        # 取最后一个时间步，代表综合了历史信息后的当前状态
        last_step = x_trans_out[-1, :, :] # (Batch, d_model)
        
        factors = self.factor_head(last_step) # (Batch, 64)
        prediction = self.predictor(factors)   # (Batch, 1)
        
        return prediction, factors

# ==============================================================================
# 4. 数据处理与训练引擎 (标准流程)
# ==============================================================================
class DataProcessor:
    def __init__(self, filepath, lookback=30, horizon=10):
        self.filepath = filepath
        self.lookback = lookback
        self.horizon = horizon
        self.scaler = StandardScaler()
        
    def load_and_process(self):
        print(f"⚡ 加载数据: {self.filepath}")
        raw = pd.read_csv(self.filepath)
        if 'tx_local_time' in raw.columns:
            raw['datetime'] = pd.to_datetime(raw['tx_local_time'], unit='ms')
        
        df = raw.set_index('datetime').sort_index() if 'datetime' in raw.columns else raw
        
        # 3s 重采样
        agg = {
            'price': 'last', 'tick_vol': 'sum', 
            'bp1': 'last', 'sp1': 'last', 'bv1': 'last', 'sv1': 'last',
            'premium_rate': 'last', 'sentiment': 'last'
        }
        if 'fut_price' in df.columns: agg.update({'fut_price': 'last', 'fut_imb': 'mean'})
        df = df.resample('3s').agg(agg).ffill().dropna()
        
        # --- 特征工程 ---
        df['mid'] = (df['bp1'] + df['sp1']) / 2
        df['log_ret'] = np.log(df['mid'] / df['mid'].shift(1)).fillna(0)
        df['spread'] = (df['sp1'] - df['bp1']) / df['mid']
        df['imb'] = (df['bv1'] - df['sv1']) / (df['bv1'] + df['sv1'] + 1e-6)
        
        if 'fut_price' in df.columns:
            df['basis_chg'] = df['fut_price'].pct_change() - df['price'].pct_change()
            df['fut_imb'] = df['fut_imb']
        
        df['sent_diff'] = df['sentiment'].diff()
        df = df.fillna(0)
        
        # 目标: 预测未来波动率 (Standard Deviation of returns)
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=self.horizon)
        df['target'] = df['log_ret'].rolling(window=indexer).std() * 10000 
        
        df = df.dropna()
        feature_cols = ['log_ret', 'spread', 'imb', 'tick_vol', 'premium_rate', 'sentiment', 'sent_diff']
        if 'fut_price' in df.columns: feature_cols.extend(['basis_chg', 'fut_imb'])
        
        self.feature_cols = feature_cols
        return df, feature_cols

    def get_tensors(self, df, feature_cols, fit_scaler=False):
        data = df[feature_cols].values
        target = df['target'].values
        if fit_scaler: data = self.scaler.fit_transform(data)
        else: data = self.scaler.transform(data)
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i-self.lookback : i])
            y.append(target[i])
        return np.array(X), np.array(y)

def run_hybrid_mining():
    FILE = 'sz159920.csv'
    LOOKBACK = 30
    HORIZON = 10
    
    # 1. 数据准备
    proc = DataProcessor(FILE, LOOKBACK, HORIZON)
    df, feat_cols = proc.load_and_process()
    print(f"基础特征维度: {len(feat_cols)} (经 AlphaLayer 扩展后将变大)")
    
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]
    
    X_train, y_train = proc.get_tensors(train_df, feat_cols, fit_scaler=True)
    X_test, y_test = proc.get_tensors(test_df, feat_cols, fit_scaler=False)
    
    joblib.dump(proc.scaler, 'hybrid_scaler.pkl')
    
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)), batch_size=64, shuffle=True)
    
    # 2. 初始化混合模型
    model = HybridMinerNet(input_dim=len(feat_cols)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    
    # 3. 训练
    print("\n🚀 开始训练混合模型 (AlphaNet + TCN + Transformer)...")
    model.train()
    for epoch in range(10):
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred, _ = model(X)
            loss = criterion(pred.squeeze(), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f}")
        
    torch.save(model.state_dict(), 'hybrid_miner_model.pth')
    
    # 4. 因子挖掘演示
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
        _, factors = model(X_test_tensor)
        factors_np = factors.cpu().numpy()
        
        # 简单评估 Top 1 因子
        target_np = y_test
        corrs = [np.corrcoef(factors_np[:, i], target_np)[0,1] for i in range(64)]
        best_idx = np.argmax(np.abs(corrs))
        print(f"\n🏆 混合模型挖掘最强因子 (Index {best_idx}): IC = {corrs[best_idx]:.4f}")

if __name__ == "__main__":
    run_hybrid_mining()