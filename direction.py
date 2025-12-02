import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib
import math
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 计算设备: {DEVICE} (Ultimate Mode)")

# ==============================================================================
# 1. 终极数据处理器 (支持双品种融合)
# ==============================================================================
class UltimateDataProcessor:
    def __init__(self, target_file, context_file, lookback=120, horizon=20):
        self.target_file = target_file
        self.context_file = context_file
        self.lookback = lookback  # [优化1] 120 tick = 6分钟
        self.horizon = horizon
        self.scaler = StandardScaler()

    def _load_single(self, filepath):
        """读取单个文件并重采样"""
        if not os.path.exists(filepath):
            return None
        try:
            raw = pd.read_csv(filepath)
            if 'tx_local_time' in raw.columns:
                raw['datetime'] = pd.to_datetime(raw['tx_local_time'], unit='ms')
                if raw['datetime'].dt.tz is None:
                    raw['datetime'] = raw['datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
                df = raw.set_index('datetime').sort_index()
            else:
                df = raw
                
            # 3s 重采样
            agg_rules = {
                'price': 'last', 'tick_vol': 'sum', 'tick_vwap': 'mean',
                'bp1':'last', 'bv1':'last', 'sp1':'last', 'sv1':'last',
                'premium_rate': 'last', 'sentiment': 'last'
            }
            # 尝试添加 L2-L5
            for i in range(2, 6):
                if f'bp{i}' in df.columns:
                    agg_rules[f'bp{i}'] = 'last'; agg_rules[f'bv{i}'] = 'last'
                    agg_rules[f'sp{i}'] = 'last'; agg_rules[f'sv{i}'] = 'last'
            
            if 'fut_price' in df.columns: agg_rules['fut_price'] = 'last'

            # 聚合
            df = df.resample('3s').agg(agg_rules).ffill().dropna()
            
            # 基础特征
            df['mid_price'] = (df['bp1'] + df['sp1']) / 2
            df['log_ret'] = np.log(df['mid_price'] / df['mid_price'].shift(1)).fillna(0)
            df['l1_imb'] = (df['bv1'] - df['sv1']) / (df['bv1'] + df['sv1'] + 1e-6)
            df['spread'] = (df['sp1'] - df['bp1']) / df['mid_price']
            
            return df
        except Exception as e:
            print(f"读取 {filepath} 失败: {e}")
            return None

    def load_and_process(self):
        print("⚡ 正在进行跨品种数据融合...")
        
        # 1. 读取主标的 (159920)
        df_target = self._load_single(self.target_file)
        if df_target is None: return None, []
        
        # 2. 读取参考标的 (513130) - [优化4]
        df_ctx = self._load_single(self.context_file)
        
        # 3. 数据融合 (Merge)
        if df_ctx is not None:
            # 重命名参考标的列，防止冲突
            df_ctx = df_ctx.add_prefix('ctx_')
            # 按时间索引合并 (Inner Join)
            df_merged = df_target.join(df_ctx, how='inner')
            print(f"双品种合并成功: {len(df_merged)} 样本")
            
            # === [优化4] 构造跨品种特征 (Relative Strength) ===
            # 相对强弱: 恒指涨幅 - 恒科涨幅
            df_merged['rel_str'] = df_merged['log_ret'] - df_merged['ctx_log_ret']
            # 相对深度: 恒指深度 / 恒科深度 (标准化后)
            # 这里简单用买一量对比
            df_merged['rel_depth'] = np.log(df_merged['bv1'] + 1) - np.log(df_merged['ctx_bv1'] + 1)
            # 情绪共振: 恒指情绪 + 恒科情绪
            if 'ctx_sentiment' in df_merged.columns:
                df_merged['total_sentiment'] = df_merged['sentiment'] + df_merged['ctx_sentiment']
        else:
            print("⚠️ 未找到参考标的文件，降级为单品种模式。")
            df_merged = df_target

        # 4. 清洗
        df_merged = df_merged.replace([np.inf, -np.inf], 0).fillna(0)

        # 5. 生成标签 (三势垒法)
        # 依然针对主标的 (mid_price) 生成标签
        labels = self.apply_triple_barrier(df_merged['mid_price'], horizon=self.horizon)
        df_merged['target'] = labels
        
        # 去除最后无法计算标签的行
        df_merged = df_merged.iloc[:-self.horizon]

        # 6. 特征选择
        drop_cols = ['tx_server_time', 'target']
        feat_cols = [c for c in df_merged.columns if c not in drop_cols and c in df_merged.select_dtypes(include=np.number).columns]
        
        self.feature_cols = feat_cols
        return df_merged, feat_cols

    def apply_triple_barrier(self, prices, horizon=20, min_ret=0.002):
        # 简化的三势垒: 1(涨), 2(跌), 0(震荡)
        out = pd.Series(0, index=prices.index)
        vals = prices.values
        for i in range(len(vals) - horizon):
            curr = vals[i]
            fut = vals[i+1 : i+horizon+1]
            if np.any(fut >= curr + min_ret): out.iloc[i] = 1
            elif np.any(fut <= curr - min_ret): out.iloc[i] = 2
        return out

    def get_tensors(self, df, feature_cols, fit_scaler=False):
        data = df[feature_cols].values
        target = df['target'].values
        
        if fit_scaler: data = self.scaler.fit_transform(data)
        else: data = self.scaler.transform(data)
            
        X, y = [], []
        # Lookback = 120
        for i in range(self.lookback, len(data)):
            X.append(data[i-self.lookback : i])
            y.append(target[i])
        return np.array(X), np.array(y)

# ==============================================================================
# 2. 终极模型架构 (Deep Capacity + Inception + SE)
# ==============================================================================
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()
        # 四分支多尺度提取
        self.b1 = nn.Conv1d(in_channels, out_channels//4, 1)
        self.b2 = nn.Sequential(nn.Conv1d(in_channels, out_channels//4, 1), nn.Conv1d(out_channels//4, out_channels//4, 3, padding=1))
        self.b3 = nn.Sequential(nn.Conv1d(in_channels, out_channels//4, 1), nn.Conv1d(out_channels//4, out_channels//4, 5, padding=2))
        self.b4 = nn.Sequential(nn.MaxPool1d(3, 1, 1), nn.Conv1d(in_channels, out_channels//4, 1))
    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)

class DeepLOB_Ultimate(nn.Module):
    def __init__(self, input_dim, d_model=256, nhead=16, num_layers=6, num_classes=3):
        super(DeepLOB_Ultimate, self).__init__()
        
        # --- [优化2] 模型扩容 ---
        # d_model 128 -> 256
        
        # 1. Stem
        self.stem = nn.Sequential(
            nn.Conv1d(input_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU()
        )
        # 2. Inception + SE (提取特征)
        self.inception = InceptionBlock(64, d_model)
        self.se = SEBlock(d_model)
        
        # 3. Deep Transformer (深层逻辑推理)
        # num_layers 4 -> 6, nhead 8 -> 16
        self.pos_encoder = self._make_pos_encoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=512, dropout=0.1)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        
        # 4. Head
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def _make_pos_encoding(self, d_model, max_len=5000):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        return nn.Parameter(pe.unsqueeze(0).transpose(0, 1), requires_grad=False)

    def forward(self, x):
        # x: (Batch, Seq, Feat) -> (Batch, Feat, Seq)
        x = x.permute(0, 2, 1)
        x = self.stem(x)
        x = self.inception(x)
        x = self.se(x)
        
        # Transformer: (Seq, Batch, Feat)
        x = x.permute(2, 0, 1)
        x = x + self.pos_encoder[:x.size(0), :]
        x = self.transformer(x)
        
        x = x.mean(dim=0)
        return self.fc(x)

# ==============================================================================
# 3. 训练流程
# ==============================================================================
def run_ultimate_training():
    # 输入文件
    TARGET_FILE = 'sz159920.csv'
    CONTEXT_FILE = 'sh513130.csv' # [优化4] 必须存在
    
    # 1. 加载
    proc = UltimateDataProcessor(TARGET_FILE, CONTEXT_FILE, lookback=120, horizon=20)
    df, feat_cols = proc.load_and_process()
    
    if df is None or len(df) < 200: 
        print("数据不足，无法训练。")
        return

    print(f"最终特征维度: {len(feat_cols)}")
    print(f"回看窗口: 120 (6分钟)")
    
    # 划分
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]
    
    X_train, y_train = proc.get_tensors(train_df, feat_cols, fit_scaler=True)
    X_test, y_test = proc.get_tensors(test_df, feat_cols, fit_scaler=False)
    
    joblib.dump(proc.scaler, 'ultimate_scaler.pkl')
    
    # DataLoader
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
        batch_size=64, shuffle=True
    )
    
    # 2. 初始化模型 [优化2]
    # d_model=256, nhead=16, num_layers=6
    model = DeepLOB_Ultimate(input_dim=len(feat_cols), d_model=256, nhead=16, num_layers=6).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4) # 深层模型需要更小的学习率
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.2, 1.0, 1.0]).to(DEVICE))
    
    # 3. 训练
    print("\n🚀 开始训练 Ultimate Model...")
    model.train()
    for epoch in range(20): # 深层模型多训练几轮
        total_loss = 0
        correct = 0
        total = 0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
        print(f"Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f} | Acc {correct/total:.2%}")
        
    torch.save(model.state_dict(), 'ultimate_model.pth')
    print("模型已保存 -> ultimate_model.pth")
    
    # 4. 验证与输出
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
        logits = model(X_test_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        
        # 构造 Score (-1, 1)
        scores = probs[:, 1] - probs[:, 2]
        
        res_df = pd.DataFrame({
            'score': scores,
            'target': y_test
        }, index=test_df.index[120:])
        
        res_df[['score']].to_csv("ultimate_factor.csv")
        
        # 验证胜率
        trades = res_df[np.abs(res_df['score']) > 0.5]
        wins = (np.sign(trades['score']) == np.where(trades['target']==2, -1, trades['target'])).sum()
        win_rate = wins / len(trades) if len(trades)>0 else 0
        
        print(f"\n🏆 Ultimate 模型战报:")
        print(f"强信号数量: {len(trades)}")
        print(f"方向胜率: {win_rate:.2%}")

if __name__ == "__main__":
    run_ultimate_training()