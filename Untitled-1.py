import os
import glob
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings('ignore')

# ==========================================
# 1. 全局配置 (Configuration)
# ==========================================
CONFIG = {
    # --- 路径 ---
    'DATA_DIR': './data',          # 数据根目录
    'MAIN_SYMBOL': 'sz159920',     # 交易标的
    'AUX_SYMBOL': 'sh513130',      # 辅助标的
    
    # --- 因子与数据 ---
    'RESAMPLE_FREQ': '3S',         # 3秒重采样
    'PREDICT_HORIZON': 60,         # 预测未来 60个周期 (180秒)
    'COST_THRESHOLD': 0.002,       # 利润门槛 (20bps)
    'LOB_DEPTH': 5,                # 盘口深度
    
    # --- 训练参数 ---
    'BATCH_SIZE': 512,
    'EPOCHS': 30,
    'LR': 1e-4,
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'TRAIN_SPLIT': 0.8,            # 前80%日期训练，后20%验证
    'LOOKBACK': 60,                # 每一个样本回看 60 个时间步 (180秒)
}

# ==========================================
# 2. 数据处理与因子熔炉 (Alpha Forge)
# ==========================================
class AlphaForge:
    def __init__(self, cfg):
        self.cfg = cfg
        # 盘口加权权重 (Level 1 -> Level 5)
        self.weights = np.array([1.0, 0.8, 0.6, 0.4, 0.2])

    def load_and_process(self):
        """主流程：加载所有文件并生成全量数据"""
        print(f"🚀 [AlphaForge] 启动... 扫描目录: {self.cfg['DATA_DIR']}")
        
        pairs = self._match_files()
        all_dfs = []
        
        for date, main_f, aux_f in pairs:
            try:
                # 1. 加载 & 对齐
                df = self._load_pair(main_f, aux_f, date)
                if df is None or len(df) < 200: continue
                
                # 2. 计算因子
                df = self._calc_factors(df)
                
                # 3. 生成标签
                df = self._make_labels(df)
                
                all_dfs.append(df.dropna())
                print(f"  -> {date}: 样本数 {len(df)} | Buy信号 {(df['label']==1).sum()}")
            except Exception as e:
                print(f"  -> {date} 处理出错: {e}")
                
        if not all_dfs:
            raise ValueError("未生成任何有效数据！")
            
        full_df = pd.concat(all_dfs)
        return full_df.sort_index()

    def _match_files(self):
        """文件配对"""
        m_path = os.path.join(self.cfg['DATA_DIR'], self.cfg['MAIN_SYMBOL'], f"*-*.csv")
        a_path = os.path.join(self.cfg['DATA_DIR'], self.cfg['AUX_SYMBOL'], f"*-*.csv")
        m_files = {self._get_date(f): f for f in glob.glob(m_path)}
        a_files = {self._get_date(f): f for f in glob.glob(a_path)}
        common = sorted(list(set(m_files.keys()) & set(a_files.keys())))
        return [(d, m_files[d], a_files[d]) for d in common]

    def _get_date(self, path):
        # 假设文件名: sz159920-2025-12-05.csv
        return os.path.basename(path).split('.')[0].split('-')[-1] # 取最后一段作为日期，或者根据实际情况调整

    def _load_pair(self, m_path, a_path, date_str):
        """读取双流数据并内连接"""
        def read_one(path):
            d = pd.read_csv(path)
            # 兼容多种日期格式，这里假设文件名已包含日期，或者通过参数传入
            # 为了稳健，直接拼合
            base_date = os.path.basename(path).split('-')[1:] # 假设 sz159920-2025-12-05
            date_part = "-".join(base_date).replace('.csv','')
            
            d['datetime'] = pd.to_datetime(date_part + ' ' + d['tx_server_time'])
            d = d.set_index('datetime').sort_index()
            # 快照去重
            return d.groupby(level=0).last()

        df_m = read_one(m_path)
        df_a = read_one(a_path)
        
        # 定义聚合规则
        agg_dict = {
            'price': 'last', 'tick_vol': 'sum',
            'bp1': 'last', 'sp1': 'last', # Level 1
            'bp2': 'last', 'sp2': 'last',
            'bp3': 'last', 'sp3': 'last',
            'bp4': 'last', 'sp4': 'last',
            'bp5': 'last', 'sp5': 'last',
            'bv1': 'last', 'sv1': 'last',
            'bv2': 'last', 'sv2': 'last',
            'bv3': 'last', 'sv3': 'last',
            'bv4': 'last', 'sv4': 'last',
            'bv5': 'last', 'sv5': 'last',
        }
        # 检查可选列
        for c in ['index_price', 'fut_price', 'fut_imb']:
            if c in df_m.columns: agg_dict[c] = 'last'
            
        # 重采样
        df_m_res = df_m.resample(self.cfg['RESAMPLE_FREQ']).agg(agg_dict)
        df_a_res = df_a.resample(self.cfg['RESAMPLE_FREQ']).agg({'price': 'last', 'tick_vol': 'sum'})
        df_a_res.columns = ['peer_price', 'peer_vol']
        
        # 内连接对齐
        return df_m_res.join(df_a_res, how='inner')

    def _calc_factors(self, df):
        """计算混合因子"""
        # --- 1. Meta Factors (时间/状态) ---
        seconds = df.index.hour * 3600 + df.index.minute * 60 + df.index.second
        df['meta_time_norm'] = (seconds - 34200) / 14400 # 简单归一化
        
        # --- 2. Micro Factors (微观盘口) ---
        mid = (df['bp1'] + df['sp1']) / 2
        # 加权压力
        wb = sum(df[f'bv{i}'] * self.weights[i-1] for i in range(1,6))
        wa = sum(df[f'sv{i}'] * self.weights[i-1] for i in range(1,6))
        df['feat_micro_pressure'] = (wb - wa) / (wb + wa + 1e-8)
        # OFI
        price_d = df['price'].diff()
        ofi = np.where(price_d>0, df['tick_vol'], np.where(price_d<0, -df['tick_vol'], 0))
        df['feat_micro_ofi'] = pd.Series(ofi, index=df.index).rolling(3).sum()
        
        # --- 3. Oracle Factors (上帝视角) ---
        if 'index_price' in df.columns:
            # 基差 (利用滞后)
            df['feat_oracle_basis'] = (df['index_price'] - mid) / mid
            # 动量
            df['feat_oracle_idx_mom'] = df['index_price'].pct_change(2) # 6s change
        
        if 'fut_price' in df.columns:
            df['feat_oracle_fut_lead'] = df['fut_price'].pct_change()
            
        # --- 4. Peer Factors (共振) ---
        df['feat_peer_diff'] = df['price'].pct_change() - df['peer_price'].pct_change()
        
        return df

    def _make_labels(self, df):
        """三重屏障打标"""
        mid = (df['bp1'] + df['sp1']) / 2
        # 未来 Horizon 收益率
        fwd_ret = mid.shift(-self.cfg['PREDICT_HORIZON']) / mid - 1
        
        labels = np.zeros(len(df))
        labels[fwd_ret > self.cfg['COST_THRESHOLD']] = 1   # Buy
        labels[fwd_ret < -self.cfg['COST_THRESHOLD']] = 2  # Sell
        
        df['label'] = labels
        return df

# ==========================================
# 3. 混合深度模型 (Hybrid DeepLOB)
# ==========================================
class HybridDeepLOB(nn.Module):
    def __init__(self, num_expert_feats):
        super(HybridDeepLOB, self).__init__()
        
        # A. 视觉流 (CNN处理LOB)
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 16, (1, 2), stride=(1, 2)), nn.LeakyReLU(), nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, (4, 1)), nn.LeakyReLU(), nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, (4, 1)), nn.LeakyReLU(), nn.BatchNorm2d(16),
        )
        
        # B. 逻辑流 (MLP处理手工因子)
        self.expert_net = nn.Sequential(
            nn.Linear(num_expert_feats, 32),
            nn.LeakyReLU(),
            nn.BatchNorm1d(32)
        )
        
        # C. 融合与时序 (LSTM)
        # CNN output approx 16 channels, need to flatten? 
        # DeepLOB standard output is (Batch, Time, Features)
        # 简化处理：假设CNN最后输出维度为 16
        self.lstm = nn.LSTM(input_size=16+32, hidden_size=64, batch_first=True)
        self.classifier = nn.Linear(64, 3) # 3 Classes

    def forward(self, x_lob, x_exp):
        # x_lob: (N, T, 20) -> (N, 1, T, 20)
        x_lob = x_lob.unsqueeze(1)
        
        # CNN Forward
        # 注意：这里简化了 DeepLOB 的 Inception 结构，用标准 Conv 演示原理
        # 实际 output 需要 reshape 成 (N, T, 16)
        # 为了演示，我们假设经过卷积层后，特征维被压缩，保留时间维
        # 在真实实现中需要仔细调整 Padding 以保持 Time 维度不变
        
        # Placeholder logic for dimension matching (In real code, calculate padding)
        # 这里使用 AdaptivePool 强行对齐时间维度 (T)，保证拼接
        feat_cnn = self.conv_net(x_lob) 
        # (N, 16, T', 1) -> (N, T', 16)
        feat_cnn = feat_cnn.permute(0, 2, 1, 3).squeeze(-1)
        
        # 强制对齐时间维度 (可能会有少量损失)
        target_len = x_exp.shape[1]
        feat_cnn = torch.nn.functional.adaptive_avg_pool1d(feat_cnn.permute(0,2,1), target_len).permute(0,2,1)
        
        # Expert Forward
        # Shared weights across time
        B, T, F = x_exp.shape
        feat_exp = self.expert_net(x_exp.reshape(-1, F)).reshape(B, T, -1)
        
        # Fusion
        combined = torch.cat([feat_cnn, feat_exp], dim=2)
        
        # LSTM
        out, _ = self.lstm(combined)
        # Take last step
        return self.classifier(out[:, -1, :])

# ==========================================
# 4. 数据集与训练器 (Dataset & Trainer)
# ==========================================
class ETFDataset(Dataset):
    def __init__(self, df, lookback, scaler=None):
        self.lookback = lookback
        
        # 提取特征列
        self.lob_cols = [f'{s}{i}' for i in range(1,6) for s in ['bp','sp']] + \
                        [f'{s}{i}' for i in range(1,6) for s in ['bv','sv']]
        self.exp_cols = [c for c in df.columns if c.startswith('feat_') or c.startswith('meta_')]
        
        # 数据预处理
        # 1. LOB 归一化 (Log Vol, Relative Price)
        mid = (df['bp1'] + df['sp1']) / 2
        lob_data = df[self.lob_cols].copy()
        for c in lob_data.columns:
            if 'b' in c and 'p' in c: lob_data[c] = (lob_data[c] - mid)/mid*10000
            if 'v' in c: lob_data[c] = np.log1p(lob_data[c])
        self.X_lob = lob_data.values.astype(np.float32)
        
        # 2. Expert 归一化 (StandardScaler)
        exp_data = df[self.exp_cols].values
        if scaler is None:
            self.scaler = StandardScaler()
            self.X_exp = self.scaler.fit_transform(exp_data).astype(np.float32)
        else:
            self.scaler = scaler
            self.X_exp = self.scaler.transform(exp_data).astype(np.float32)
            
        self.Y = df['label'].values.astype(np.int64)
        
    def __len__(self):
        return len(self.Y) - self.lookback

    def __getitem__(self, idx):
        # Time Window: [i : i+lookback]
        # Label: i+lookback-1 (prediction for next horizon)
        s, e = idx, idx + self.lookback
        return self.X_lob[s:e], self.X_exp[s:e], self.Y[e-1]

def train_model(train_df, val_df, cfg):
    print("\n🧠 [Trainer] 开始构建数据集与模型...")
    
    # 1. 构建 Dataset
    ds_train = ETFDataset(train_df, cfg['LOOKBACK'])
    ds_val = ETFDataset(val_df, cfg['LOOKBACK'], scaler=ds_train.scaler)
    
    dl_train = DataLoader(ds_train, batch_size=cfg['BATCH_SIZE'], shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=cfg['BATCH_SIZE'], shuffle=False)
    
    # 2. 计算 Class Weights (解决样本不平衡)
    labels = train_df['label'].values
    counts = np.bincount(labels.astype(int))
    # 权重 = 总数 / (类别数 * 频次)
    weights = torch.tensor([sum(counts)/c for c in counts], dtype=torch.float32).to(cfg['DEVICE'])
    print(f"  -> 类别分布: {counts}")
    print(f"  -> 自动权重: {weights.cpu().numpy()}")
    
    # 3. 模型与优化器
    model = HybridDeepLOB(num_expert_feats=len(ds_train.exp_cols)).to(cfg['DEVICE'])
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=cfg['LR'])
    
    # 4. 训练循环
    best_f1 = 0
    
    for epoch in range(cfg['EPOCHS']):
        model.train()
        train_loss = 0
        for x_lob, x_exp, y in dl_train:
            x_lob, x_exp, y = x_lob.to(cfg['DEVICE']), x_exp.to(cfg['DEVICE']), y.to(cfg['DEVICE'])
            
            optimizer.zero_grad()
            pred = model(x_lob, x_exp)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # 验证
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x_lob, x_exp, y in dl_val:
                x_lob, x_exp, y = x_lob.to(cfg['DEVICE']), x_exp.to(cfg['DEVICE']), y.to(cfg['DEVICE'])
                pred = model(x_lob, x_exp)
                all_preds.extend(pred.argmax(1).cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        # 评估报告
        report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        macro_f1 = report['macro avg']['f1-score']
        buy_precision = report['1']['precision']
        
        print(f"Epoch {epoch+1}/{cfg['EPOCHS']} | Loss: {train_loss/len(dl_train):.4f} | "
              f"Val F1: {macro_f1:.4f} | Buy Precision: {buy_precision:.4f}")
        
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), 'best_model.pth')
            
    print("✅ 训练完成。最佳模型已保存。")

# ==========================================
# 5. 主程序 (Main Execution)
# ==========================================
if __name__ == "__main__":
    # 1. 熔炼数据
    forge = AlphaForge(CONFIG)
    try:
        full_df = forge.load_and_process()
    except Exception as e:
        print(f"数据加载失败: {e}")
        exit()
        
    # 2. 切分训练/验证集 (按时间切分，严禁 Shuffle)
    split_idx = int(len(full_df) * CONFIG['TRAIN_SPLIT'])
    train_df = full_df.iloc[:split_idx]
    val_df = full_df.iloc[split_idx:]
    
    print(f"\n📊 数据切分: Train={len(train_df)}, Val={len(val_df)}")
    
    # 3. 训练模型
    train_model(train_df, val_df, CONFIG)