import os
import re
import glob
import warnings
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")

# ==========================================
# 1) 全局配置
# ==========================================
CONFIG = {
    # --- 路径配置 ---
    "DATA_DIR": "./data",
    "MAIN_SYMBOL": "sz159920",
    "AUX_SYMBOL": "sh513130",

    # --- 采样/标签 ---
    "RESAMPLE_FREQ": "3S",         # 3秒重采样
    "PREDICT_HORIZON": 60,         # 未来窗口(条) -> 60*3s=180s
    "LOOKBACK": 60,                # 输入回看窗口(条)

    # --- 打标与成本 ---
    "TRADE_COST": 0.0001,          # 单边费率(可调)
    "COST_THRESHOLD": 0.0004,      # 净收益阈值（已考虑点差与成本后仍要达标）

    # --- 外部数据门控（毫秒） ---
    "IDX_DELAY_CUTOFF_MS": 3000,
    "FUT_DELAY_CUTOFF_MS": 3000,

    # --- 回测/执行 ---
    "INITIAL_CAP": 200000,
    "CONF_OPEN": 0.60,             # 迟滞：开仓阈值
    "CONF_CLOSE": 0.45,            # 迟滞：平仓阈值（SigLost）
    "MAX_POSITION": 0.90,
    "STOP_LOSS_PCT": 0.008,
    "MIN_HOLD_BARS": 10,           # 最短持仓(条) -> 10*3s=30s
    "EXEC_DELAY_BARS": 1,          # 信号 -> 成交 延迟(条)
    "LIQ_PARTICIPATION": 0.10,     # 单次吃掉卖一/买一深度的比例上限
    "MIN_TRADE_AMT": 1000,
    "LOT_SIZE": 100,

    # --- 切分（按天） ---
    "VAL_DAYS": 1,
    "TEST_DAYS": 1,

    # --- 时区（用于 tx_local_time 转换到交易所本地时间） ---
    "TIMEZONE": "Asia/Shanghai",

    # --- 训练参数 ---
    "BATCH_SIZE": 512,
    "EPOCHS": 100,
    "LR": 1e-4,
    "WEIGHT_DECAY": 1e-4,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "PATIENCE": 200,
    "WARMUP_EPOCHS": 10,
}

# ==========================================
# 2) 数据工厂：AlphaForge
# ==========================================
class AlphaForge:
    """
    目标：
    - 保留原 HybridDeepLOB 模型架构
    - 重点修复：时间轴、数据对齐、门控、标签与回测一致性
    - 增加适配 3~5min 频率的“状态/结构”因子（纯后视滚动，不引入未来）
    """
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.weights = np.array([1.0, 0.8, 0.6, 0.4, 0.2], dtype=np.float32)

    def load_and_split(self):
        print(f"🚀 [AlphaForge] 扫描: {self.cfg['DATA_DIR']}")
        pairs = self._match_files()
        pairs.sort(key=lambda x: x[0])
        if len(pairs) < (1 + self.cfg["VAL_DAYS"] + self.cfg["TEST_DAYS"]):
            raise ValueError("数据天数不足以切分 train/val/test")

        n_val = self.cfg["VAL_DAYS"]
        n_test = self.cfg["TEST_DAYS"]
        train_pairs = pairs[: -(n_val + n_test)]
        val_pairs = pairs[-(n_val + n_test): -n_test]
        test_pairs = pairs[-n_test:]

        print(f"训练集: {train_pairs[0][0]} ~ {train_pairs[-1][0]}")
        print(f"验证集: {val_pairs[0][0]} ~ {val_pairs[-1][0]}")
        print(f"测试集: {test_pairs[0][0]} ~ {test_pairs[-1][0]}")

        train_df = self._process_batch(train_pairs)
        val_df = self._process_batch(val_pairs)
        test_df = self._process_batch(test_pairs)
        return train_df, val_df, test_df

    def _process_batch(self, pairs: List[Tuple[str, str, str]]) -> pd.DataFrame:
        dfs = []
        for date, mf, af in pairs:
            try:
                df = self._load_pair(mf, af, date)
                if df is None or len(df) < 300:
                    continue

                df = self._calc_factors(df)
                df = self._make_labels(df)

                # 只对“绝对必要列”做 dropna，避免外部列缺失导致选择偏差
                required = ["mid", "bp1", "sp1", "label"]
                df = df.replace([np.inf, -np.inf], np.nan)
                df = df.dropna(subset=[c for c in required if c in df.columns])

                dfs.append(df)
            except Exception as e:
                print(f"⚠️ 跳过 {date}: {e}")

        return pd.concat(dfs).sort_index() if dfs else pd.DataFrame()

    def _match_files(self):
        m_pattern = os.path.join(self.cfg["DATA_DIR"], "**", f"{self.cfg['MAIN_SYMBOL']}*.csv")
        a_pattern = os.path.join(self.cfg["DATA_DIR"], "**", f"{self.cfg['AUX_SYMBOL']}*.csv")
        m_files = glob.glob(m_pattern, recursive=True)
        a_files = glob.glob(a_pattern, recursive=True)

        def get_date(p: str) -> str:
            match = re.search(r"(\d{4}-\d{2}-\d{2})", p)
            if match:
                return match.group(1)
            return ""

        m_map = {get_date(p): p for p in m_files}
        a_map = {get_date(p): p for p in a_files}
        common = sorted(list(set(m_map.keys()) & set(a_map.keys())))
        return [(d, m_map[d], a_map[d]) for d in common]

    # -----------------------------
    # 读取 & 对齐
    # -----------------------------
    def _load_pair(self, m_path: str, a_path: str, date_str: str) -> Optional[pd.DataFrame]:
        def _read(p: str) -> pd.DataFrame:
            d = pd.read_csv(p)

            # 1) 用 tx_local_time 作为唯一主时间轴（如果不存在则回退到 tx_server_time）
            if "tx_local_time" in d.columns and d["tx_local_time"].notna().any():
                dt_utc = pd.to_datetime(d["tx_local_time"], unit="ms", utc=True, errors="coerce")
                tz = self.cfg.get("TIMEZONE", "Asia/Shanghai")
                d["datetime"] = dt_utc.dt.tz_convert(tz).dt.tz_localize(None)
            else:
                d["datetime"] = pd.to_datetime(date_str + " " + d["tx_server_time"], errors="coerce")

            # 2) 强制数值类型（空串/脏字符 -> NaN）
            numeric_like = set([
                "price","iopv","premium_rate","index_price","fx_rate","sentiment",
                "tick_vol","tick_amt","tick_vwap","interval_s",
                "idx_delay_ms","fut_delay_ms","data_flags",
                "fut_price","fut_mid","fut_imb","fut_delta_vol","fut_pct",
                "fut_local_time","fut_tick_time",
            ])
            # LOB
            for s in ["bp","bv","sp","sv"]:
                for i in range(1,6):
                    numeric_like.add(f"{s}{i}")

            cols = [c for c in d.columns if c in numeric_like]
            if cols:
                d[cols] = d[cols].apply(pd.to_numeric, errors="coerce")

            d = d.sort_values("datetime")
            d = d.drop_duplicates(subset="datetime", keep="last")
            d = d.set_index("datetime").sort_index()
            return d

        df_m, df_a = _read(m_path), _read(a_path)

        # --- 主标的聚合：必须包含 tick_amt，重算 bin_vwap ---
        agg = {
            "price": "last",
            "tick_vol": "sum",
            "tick_amt": "sum",
            "bp1": "last", "sp1": "last", "bp2": "last", "sp2": "last",
            "bp3": "last", "sp3": "last", "bp4": "last", "sp4": "last",
            "bp5": "last", "sp5": "last",
            "bv1": "last", "sv1": "last", "bv2": "last", "sv2": "last",
            "bv3": "last", "sv3": "last", "bv4": "last", "sv4": "last",
            "bv5": "last", "sv5": "last",
            # 外部列（存在则加进去）
            "index_price": "last",
            "premium_rate": "last",
            "sentiment": "last",
            "fx_rate": "last",
            "tick_vwap": "last",
            "interval_s": "last",
            "idx_delay_ms": "max",
            "fut_delay_ms": "max",
            "data_flags": "max",
            "fut_price": "last",
            "fut_mid": "last",
            "fut_imb": "last",
            "fut_delta_vol": "last",
            "fut_pct": "last",
        }
        # 只保留 df_m 里真正存在的列
        agg = {k:v for k,v in agg.items() if k in df_m.columns}

        df_m = df_m.resample(self.cfg["RESAMPLE_FREQ"]).agg(agg)

        # tick_vol/tick_amt 防御性修正
        if "tick_vol" in df_m.columns:
            df_m["tick_vol"] = df_m["tick_vol"].clip(lower=0)
        if "tick_amt" in df_m.columns:
            df_m["tick_amt"] = df_m["tick_amt"].clip(lower=0)

        # 重算 bin_vwap（更符合 3S 聚合统计意义）
        if "tick_vol" in df_m.columns and "tick_amt" in df_m.columns:
            denom = df_m["tick_vol"].replace(0, np.nan)
            df_m["tick_vwap_bin"] = (df_m["tick_amt"] / denom).fillna(df_m.get("price"))
        else:
            df_m["tick_vwap_bin"] = df_m.get("tick_vwap", df_m.get("price"))

        # --- 辅标的聚合：更宽容（left + ffill） ---
        agg_a = {"price": "last", "tick_vol": "sum"}
        agg_a = {k:v for k,v in agg_a.items() if k in df_a.columns}
        df_a = df_a.resample(self.cfg["RESAMPLE_FREQ"]).agg(agg_a)
        df_a = df_a.rename(columns={"price":"peer_price", "tick_vol":"peer_vol"})

        df = df_m.join(df_a, how="left")
        # 只用过去值填充，避免未来
        if "peer_price" in df.columns:
            df["peer_price"] = df["peer_price"].ffill()
        if "peer_vol" in df.columns:
            df["peer_vol"] = df["peer_vol"].fillna(0)

        df = df.dropna(subset=["price", "bp1", "sp1"])
        return df

    # -----------------------------
    # 因子
    # -----------------------------
    def _calc_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        # ---------- 基础 ----------
        df["mid"] = (df["bp1"] + df["sp1"]) / 2.0
        df["spread"] = (df["sp1"] - df["bp1"]).clip(lower=0)

        # Meta time: 映射到 [0,1]
        # 简化：按 09:30-15:00(含午休) 线性映射，作为“日内位置”特征
        t = df.index
        seconds = t.hour*3600 + t.minute*60 + t.second
        start = 9*3600 + 30*60
        end = 15*3600
        df["meta_time"] = np.clip((seconds - start) / (end - start), 0, 1)

        # ---------- 微观结构：原 micro_pressure ----------
        wb = (df[[f"bv{i}" for i in range(1,6)]].values * self.weights).sum(axis=1)
        wa = (df[[f"sv{i}" for i in range(1,6)]].values * self.weights).sum(axis=1)
        denom = (wb + wa)
        df["feat_micro_pressure"] = np.where(denom == 0, 0.0, (wb - wa) / denom)

        # ---------- 新增：3~5min 频率更稳的因子 ----------
        # 1) 点差成本占比
        df["feat_spread_pct"] = (df["spread"] / df["mid"]).replace([np.inf, -np.inf], np.nan)

        # 2) 一档深度与深度不平衡
        depth1 = (df["bv1"] + df["sv1"]).replace(0, np.nan)
        df["feat_depth1_log"] = np.log1p((df["bv1"] + df["sv1"]).clip(lower=0))
        df["feat_depth_imb1"] = ((df["bv1"] - df["sv1"]) / depth1).fillna(0.0)

        # 3) TFI：成交流失衡（用 tick_vwap_bin 相对 mid 判断主动方向）
        tv = df["tick_vwap_bin"].fillna(df["mid"])
        vol = df.get("tick_vol", pd.Series(0, index=df.index)).fillna(0).clip(lower=0)
        df["feat_tfi"] = np.sign(tv - df["mid"]).fillna(0.0) * np.log1p(vol)

        # 4) OFI（简化 Cont-OFI，适配 3s 快照）
        bp1, bv1 = df["bp1"], df["bv1"]
        sp1, sv1 = df["sp1"], df["sv1"]
        bp1_prev, bv1_prev = bp1.shift(1), bv1.shift(1)
        sp1_prev, sv1_prev = sp1.shift(1), sv1.shift(1)

        ofi_bid = np.where(bp1 > bp1_prev, bv1,
                    np.where(bp1 == bp1_prev, bv1 - bv1_prev, -bv1_prev))
        ofi_ask = np.where(sp1 < sp1_prev, sv1,
                    np.where(sp1 == sp1_prev, sv1 - sv1_prev, -sv1_prev))
        ofi_raw = np.nan_to_num(ofi_bid) - np.nan_to_num(ofi_ask)
        df["feat_ofi1"] = np.sign(ofi_raw) * np.log1p(np.abs(ofi_raw))

        # 5) LOB skew：价差形态偏度
        bid_span = (df["bp1"] - df["bp5"]).clip(lower=0)
        ask_span = (df["sp5"] - df["sp1"]).clip(lower=0)
        denom2 = (bid_span + ask_span).replace(0, np.nan)
        df["feat_lob_skew"] = ((bid_span - ask_span) / denom2).fillna(0.0)

        # 6) Book slope：远端挂单“陡峭程度”
        df["feat_ask_slope"] = ((df["sp5"] - df["sp1"]) / df["mid"]).replace([np.inf, -np.inf], np.nan)
        df["feat_bid_slope"] = ((df["bp1"] - df["bp5"]) / df["mid"]).replace([np.inf, -np.inf], np.nan)

        # ---------- 原核心 Alpha 因子源列 ----------
        # premium: 优先用采集 premium_rate，否则用 index-mid 近似
        if "premium_rate" in df.columns and df["premium_rate"].notna().any():
            df["feat_premium_rate"] = df["premium_rate"]
        else:
            idx = df.get("index_price")
            if idx is not None:
                df["feat_premium_rate"] = (idx - df["mid"]) / df["mid"]
            else:
                df["feat_premium_rate"] = 0.0

        if "sentiment" in df.columns:
            df["feat_sentiment"] = df["sentiment"]
        else:
            df["feat_sentiment"] = 0.0

        if "fut_imb" in df.columns:
            df["feat_fut_imb"] = df["fut_imb"]
        else:
            df["feat_fut_imb"] = 0.0

        # Flow force（用 bin_vwap + vol）
        df["feat_flow_force"] = (tv - df["mid"]) * np.log1p(vol)

        # ---------- 3~5min 状态因子：纯后视 rolling ----------
        # window sizes based on RESAMPLE_FREQ
        # 5min: 300s
        freq = pd.to_timedelta(self.cfg["RESAMPLE_FREQ"])
        w5 = int(pd.Timedelta("5min") / freq)
        w3 = int(pd.Timedelta("3min") / freq)
        w15 = int(pd.Timedelta("15min") / freq)

        mid = df["mid"]
        logret = np.log(mid).diff()

        # 5min 方向/动量
        df["feat_ret_5m"] = mid.pct_change(w5)
        # 5min RV
        df["feat_rv_5m"] = np.sqrt((logret**2).rolling(w5).sum())
        # 5min 价格效率比
        hi5 = mid.rolling(w5).max()
        lo5 = mid.rolling(w5).min()
        denom3 = (hi5 - lo5).replace(0, np.nan)
        df["feat_eff_5m"] = (mid - mid.shift(w5)).abs() / denom3

        # 5min 平均点差/深度/流动性状态
        df["feat_spread_mean_5m"] = df["spread"].rolling(w5).mean()
        df["feat_depth1_mean_5m"] = (df["bv1"] + df["sv1"]).rolling(w5).mean()

        # 5min 订单流聚合
        df["feat_ofi_5m"] = df["feat_ofi1"].rolling(w5).sum()
        df["feat_tfi_5m"] = df["feat_tfi"].rolling(w5).sum()

        # 量能 z-score（用更长窗口做均值方差）
        bar_vol_5m = vol.rolling(w5).sum()
        mu = bar_vol_5m.rolling(w15).mean()
        sd = bar_vol_5m.rolling(w15).std().replace(0, np.nan)
        df["feat_volz_5m"] = ((bar_vol_5m - mu) / sd).fillna(0.0)

        # ---------- Peer 残差动量（滚动 beta） ----------
        if "peer_price" in df.columns and df["peer_price"].notna().any():
            peer_mid = df["peer_price"]
            r = mid.pct_change()
            rp = peer_mid.pct_change()
            # beta: cov(r, rp)/var(rp)
            cov = (r * rp).rolling(w15).mean() - r.rolling(w15).mean() * rp.rolling(w15).mean()
            var = (rp**2).rolling(w15).mean() - (rp.rolling(w15).mean()**2)
            beta = (cov / var.replace(0, np.nan)).fillna(1.0)
            df["feat_peer_resid"] = (r - beta * rp).fillna(0.0)
        else:
            df["feat_peer_resid"] = 0.0

        # ---------- Oracle（外部因子） + 门控 ----------
        # 指数动量、期货领先等：只有在 delay 不坏时才“开放”
        # 兼容旧数据：没有 delay 字段时默认可用
        idx_delay = df.get("idx_delay_ms", pd.Series(np.nan, index=df.index))
        fut_delay = df.get("fut_delay_ms", pd.Series(np.nan, index=df.index))
        flags = df.get("data_flags", pd.Series(0, index=df.index)).fillna(0)

        df["feat_idx_staleness"] = np.log1p(idx_delay.fillna(999999))
        df["feat_fut_staleness"] = np.log1p(fut_delay.fillna(999999))

        bad_idx = (idx_delay > self.cfg["IDX_DELAY_CUTOFF_MS"]) | (flags > 0)
        bad_fut = (fut_delay > self.cfg["FUT_DELAY_CUTOFF_MS"]) | (flags > 0)

        if "index_price" in df.columns:
            df["feat_oracle_idx_mom"] = df["index_price"].pct_change(2)
            df["feat_oracle_basis"] = (df["index_price"] - df["mid"]) / df["mid"]
            df.loc[bad_idx.fillna(False), ["feat_oracle_idx_mom", "feat_oracle_basis"]] = np.nan
        else:
            df["feat_oracle_idx_mom"] = 0.0
            df["feat_oracle_basis"] = 0.0

        if "fut_price" in df.columns:
            # 期货“领先”简化：期货短动量 - 现货短动量
            fut_mom = df["fut_price"].pct_change(2)
            spot_mom = df["mid"].pct_change(2)
            df["feat_oracle_fut_lead"] = (fut_mom - spot_mom)
            df.loc[bad_fut.fillna(False), ["feat_oracle_fut_lead"]] = np.nan
        else:
            df["feat_oracle_fut_lead"] = 0.0

        # 清理极端值
        for c in [c for c in df.columns if c.startswith("feat_")]:
            df[c] = df[c].replace([np.inf, -np.inf], np.nan)

        return df

    # -----------------------------
    # 标签：与回测成交口径一致（Ask 买 / Bid 卖）
    # -----------------------------
    def _make_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        horizon = int(self.cfg["PREDICT_HORIZON"])
        thr = float(self.cfg["COST_THRESHOLD"])
        cost = float(self.cfg["TRADE_COST"])

        ask = df["sp1"]
        bid = df["bp1"]

        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=horizon)

        # 未来最高 Bid：我现在按 Ask 买，未来能否按 Bid 卖出赚钱？
        future_max_bid = bid.rolling(window=indexer).max()
        ret_buy = (future_max_bid / ask) - 1.0 - 2.0 * cost

        # 未来最低 Bid：如果未来会出现明显回撤，作为“卖出/避险”标签
        future_min_bid = bid.rolling(window=indexer).min()
        ret_drawdown = (future_min_bid / bid) - 1.0

        label = np.zeros(len(df), dtype=np.int64)
        label[ret_buy > thr] = 1
        label[ret_drawdown < -thr] = 2

        # 同时给回测/分析保留一个未来点收益（不用于打标决策）
        future_mid = df["mid"].shift(-horizon)
        df["real_future_ret"] = (future_mid / df["mid"]) - 1.0

        df["label"] = label
        return df


# ==========================================
# 3) 模型：HybridDeepLOB（保持原架构）
# ==========================================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

class InceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=1, padding=0)
        self.conv3 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm1d(out_ch * 3)
    def forward(self, x):
        y1 = F.relu(self.conv1(x))
        y3 = F.relu(self.conv3(x))
        y5 = F.relu(self.conv5(x))
        y = torch.cat([y1, y3, y5], dim=1)
        return F.relu(self.bn(y))

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    def forward(self, x):
        # x: (B, T, H)
        scores = self.v(torch.tanh(self.W(x))).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (x * weights).sum(dim=1)

class HybridDeepLOB(nn.Module):
    def __init__(self, num_exp_features, num_classes=3):
        super().__init__()
        # CNN branch (DeepLOB-like)
        self.conv1 = nn.Conv1d(20, 32, kernel_size=3, padding=1)
        self.se1 = SEBlock(32)
        self.inception1 = InceptionBlock(32, 16)
        self.inception2 = InceptionBlock(48, 16)

        # Expert branch
        self.expert = nn.Sequential(
            nn.Linear(num_exp_features, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Fusion + LSTM + Attention
        self.lstm = nn.LSTM(input_size=48 + 32, hidden_size=64, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.attention = TemporalAttention(hidden_dim=128)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x_lob, x_exp):
        # x_lob: (B, T, 20) -> (B, 20, T)
        x = x_lob.permute(0, 2, 1)
        feat = F.relu(self.conv1(x))
        feat = self.se1(feat)
        feat = self.inception1(feat)
        feat = self.inception2(feat)
        feat = feat.permute(0, 2, 1)  # (B, T, C)

        # expert on each timestep
        B, T, Fexp = x_exp.shape
        exp = self.expert(x_exp.reshape(-1, Fexp)).reshape(B, T, -1)

        combined = torch.cat([feat, exp], dim=2)
        lstm_out, _ = self.lstm(combined)
        context = self.attention(lstm_out)
        return self.fc(context)


# ==========================================
# 4) 数据集（修复：只在 train 上 fit scaler + 统一填充策略）
# ==========================================
class ETFDataset(Dataset):
    def __init__(self, df: pd.DataFrame, lookback: int,
                 scaler: Optional[StandardScaler]=None,
                 imputer: Optional[Dict[str, float]]=None):
        self.lookback = int(lookback)

        lob_cols = [f"{s}{i}" for i in range(1,6) for s in ["bp","sp"]] + \
                   [f"{s}{i}" for i in range(1,6) for s in ["bv","sv"]]
        exp_cols = [c for c in df.columns if c.startswith("feat_") or c.startswith("meta_")]

        # --- LOB tensor ---
        mid = df["mid"].values.reshape(-1, 1)
        safe_mid = np.where(mid == 0, 1.0, mid)
        lob_data = df[lob_cols].values.astype(np.float32)

        # price levels -> relative bps
        lob_data[:, :10] = (lob_data[:, :10] - mid) / safe_mid * 10000.0
        # sizes -> log1p
        lob_data[:, 10:] = np.log1p(np.clip(lob_data[:, 10:], a_min=0, a_max=None))

        self.X_lob = np.nan_to_num(lob_data).astype(np.float32)

        # --- Expert features ---
        exp_df = df[exp_cols].copy()

        # 缺失指示（让模型知道“不可用/被门控”）
        miss_flags = exp_df.isna().astype(np.float32)
        miss_flags.columns = [c + "_isna" for c in miss_flags.columns]

        # impute：只用 train 统计的中位数
        if imputer is None:
            self.imputer = {c: float(exp_df[c].median(skipna=True)) if exp_df[c].notna().any() else 0.0 for c in exp_cols}
        else:
            self.imputer = imputer

        for c in exp_cols:
            exp_df[c] = exp_df[c].fillna(self.imputer.get(c, 0.0))

        exp_full = pd.concat([exp_df, miss_flags], axis=1)
        self.exp_feature_names = exp_full.columns.tolist()
        exp_data = exp_full.values.astype(np.float32)

        if scaler is None:
            self.scaler = StandardScaler()
            self.X_exp = self.scaler.fit_transform(exp_data).astype(np.float32)
        else:
            self.scaler = scaler
            self.X_exp = self.scaler.transform(exp_data).astype(np.float32)

        self.Y = df["label"].values.astype(np.int64)
        self.raw_ret = df.get("real_future_ret", pd.Series(np.nan, index=df.index)).values.astype(np.float32)

    def __len__(self):
        # 让最后一个标签也可用
        return max(0, len(self.Y) - self.lookback + 1)

    def __getitem__(self, i):
        s = i
        e = i + self.lookback
        return (
            torch.from_numpy(self.X_lob[s:e]),
            torch.from_numpy(self.X_exp[s:e]),
            torch.tensor(self.Y[e-1], dtype=torch.long),
            torch.tensor(self.raw_ret[e-1], dtype=torch.float32),
        )

# ==========================================
# 5) 回测引擎（迟滞 + 深度约束 + 最小持仓 + 延迟成交）
# ==========================================
@torch.no_grad()
def backtest_evaluate(model: nn.Module, dataloader: DataLoader, cfg: Dict, raw_df: Optional[pd.DataFrame]=None) -> float:
    model.eval()
    device = cfg["DEVICE"]

    all_probs = []
    for x_lob, x_exp, _, _ in dataloader:
        x_lob = x_lob.to(device)
        x_exp = x_exp.to(device)
        logits = model(x_lob, x_exp)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)

    if not all_probs or raw_df is None:
        return 0.0

    probs_stream = np.concatenate(all_probs, axis=0)
    # 预测对齐：Dataset 的第一个预测对应 raw_df 的 index = lookback-1
    lookback = cfg["LOOKBACK"]
    sim_df = raw_df.iloc[lookback-1: lookback-1 + len(probs_stream)].copy()
    if len(sim_df) != len(probs_stream):
        # 兜底：尾部对齐
        sim_df = raw_df.tail(len(probs_stream)).copy()

    ask = sim_df["sp1"].values
    bid = sim_df["bp1"].values
    ask_v = sim_df.get("sv1", pd.Series(np.inf, index=sim_df.index)).values
    bid_v = sim_df.get("bv1", pd.Series(np.inf, index=sim_df.index)).values
    times = sim_df.index

    initial_cap = cfg["INITIAL_CAP"]
    cash = float(initial_cap)
    shares = 0.0
    is_holding = False
    entry_price = 0.0
    entry_idx = -1

    cost_rate = cfg["TRADE_COST"]
    open_th = cfg["CONF_OPEN"]
    close_th = cfg["CONF_CLOSE"]
    max_pos = cfg["MAX_POSITION"]
    stop_loss = cfg["STOP_LOSS_PCT"]
    min_hold = cfg["MIN_HOLD_BARS"]
    delay = int(cfg["EXEC_DELAY_BARS"])
    lot = int(cfg["LOT_SIZE"])
    min_amt = float(cfg["MIN_TRADE_AMT"])
    part = float(cfg["LIQ_PARTICIPATION"])

    def is_eod(ts):
        return (ts.hour == 14 and ts.minute >= 55) or (ts.hour >= 15)

    # 日志
    log_path = "backtest_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("[Backtest] start\n")

        for t in range(len(probs_stream) - delay):
            ts = times[t]
            p_hold, p_buy, p_sell = probs_stream[t]

            # 延迟成交用 t+delay 的盘口
            ex_ask = ask[t + delay]
            ex_bid = bid[t + delay]
            ex_ask_v = ask_v[t + delay] if np.isfinite(ask_v[t + delay]) else np.inf
            ex_bid_v = bid_v[t + delay] if np.isfinite(bid_v[t + delay]) else np.inf

            # EOD 强制清仓
            if is_eod(ts):
                if is_holding:
                    revenue = shares * ex_bid * (1 - cost_rate)
                    cash += revenue
                    pnl = revenue - shares * entry_price * (1 + cost_rate)
                    f.write(f"[{ts}] SELL(EOD) @ {ex_bid:.4f} shares={shares:.0f} pnl={pnl:+.2f}\n")
                    shares = 0.0
                    is_holding = False
                continue

            # 决策（迟滞）：只对 Buy 做多，Sell 作为退出/风控信号
            want_buy = (p_buy > p_hold) and (p_buy > p_sell) and (p_buy >= open_th)
            want_exit = (p_sell > p_buy and p_sell >= open_th) or (p_buy <= close_th)

            if not is_holding:
                if want_buy:
                    # 动态仓位
                    confidence = float(p_buy)
                    pos = max_pos * (confidence - open_th) / (1.0 - open_th)
                    pos = float(np.clip(pos, 0.0, max_pos))

                    budget = cash * pos
                    if budget < min_amt:
                        continue

                    # 流动性约束：最多吃掉卖一深度的一部分
                    max_shares_liq = math.floor((ex_ask_v * part) / lot) * lot if np.isfinite(ex_ask_v) else 10**12
                    # 资金约束
                    max_shares_cash = math.floor((budget / (ex_ask * (1 + cost_rate))) / lot) * lot
                    buy_shares = max(0, min(max_shares_liq, max_shares_cash))

                    if buy_shares >= lot:
                        cost = buy_shares * ex_ask * (1 + cost_rate)
                        cash -= cost
                        shares = float(buy_shares)
                        entry_price = float(ex_ask)
                        entry_idx = t
                        is_holding = True
                        f.write(f"[{ts}] BUY @ {ex_ask:.4f} shares={shares:.0f} p_buy={p_buy:.3f}\n")

            else:
                # 最短持仓
                if (t - entry_idx) < min_hold:
                    continue

                # 止损（按 Bid 估值）
                pnl_pct = (ex_bid - entry_price) / entry_price
                if pnl_pct <= -stop_loss:
                    revenue = shares * ex_bid * (1 - cost_rate)
                    cash += revenue
                    pnl = revenue - shares * entry_price * (1 + cost_rate)
                    f.write(f"[{ts}] SELL(StopLoss) @ {ex_bid:.4f} pnl={pnl:+.2f} ({pnl_pct:.2%})\n")
                    shares = 0.0
                    is_holding = False
                    continue

                if want_exit:
                    revenue = shares * ex_bid * (1 - cost_rate)
                    cash += revenue
                    pnl = revenue - shares * entry_price * (1 + cost_rate)
                    f.write(f"[{ts}] SELL(Exit) @ {ex_bid:.4f} pnl={pnl:+.2f} p_buy={p_buy:.3f} p_sell={p_sell:.3f}\n")
                    shares = 0.0
                    is_holding = False
                    continue

        # 结算
        nav = cash
        if is_holding:
            nav += shares * bid[-1] * (1 - cost_rate)

        f.write(f"[Backtest] final_nav={nav:.2f} ret={(nav/initial_cap-1):.4%}\n")

    return (nav / initial_cap) - 1.0


# ==========================================
# 6) 训练
# ==========================================
def train_system(cfg: Dict = CONFIG):
    forge = AlphaForge(cfg)
    train_df, val_df, test_df = forge.load_and_split()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("train/val/test 数据为空，请检查数据目录与文件命名")

    ds_train = ETFDataset(train_df, cfg["LOOKBACK"], scaler=None, imputer=None)
    ds_val = ETFDataset(val_df, cfg["LOOKBACK"], scaler=ds_train.scaler, imputer=ds_train.imputer)
    ds_test = ETFDataset(test_df, cfg["LOOKBACK"], scaler=ds_train.scaler, imputer=ds_train.imputer)

    dl_train = DataLoader(ds_train, batch_size=cfg["BATCH_SIZE"], shuffle=True, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=cfg["BATCH_SIZE"], shuffle=False)
    dl_test = DataLoader(ds_test, batch_size=cfg["BATCH_SIZE"], shuffle=False)

    device = cfg["DEVICE"]
    model = HybridDeepLOB(num_exp_features=ds_train.X_exp.shape[1], num_classes=3).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg["LR"], weight_decay=cfg["WEIGHT_DECAY"])
    criterion = nn.CrossEntropyLoss()

    best_val = -1e9
    patience = 0

    for epoch in range(cfg["EPOCHS"]):
        model.train()
        losses = []
        for x_lob, x_exp, y, _ in dl_train:
            x_lob = x_lob.to(device)
            x_exp = x_exp.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x_lob, x_exp)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # --- 验证：用更严格的回测做选模 ---
        val_ret = backtest_evaluate(model, dl_val, cfg, raw_df=val_df)
        print(f"Epoch {epoch+1:03d} | loss={np.mean(losses):.4f} | val_ret={val_ret:.4%}")

        if epoch < cfg["WARMUP_EPOCHS"]:
            continue

        if val_ret > best_val:
            best_val = val_ret
            patience = 0
            torch.save(model.state_dict(), "alpha_model_hybriddeeplob.pth")
            print(f"✅ 保存最好模型: val_ret={best_val:.4%}")
        else:
            patience += 1
            if patience >= cfg["PATIENCE"]:
                print("⏹️ 早停触发")
                break

    # --- 测试评估 ---
    model.load_state_dict(torch.load("alpha_model_hybriddeeplob.pth", map_location=device))
    test_ret = backtest_evaluate(model, dl_test, cfg, raw_df=test_df)
    print(f"\n[Test] ret={test_ret:.4%}")

    # 额外：分类报告（仅供诊断，不做选模依据）
    model.eval()
    ys, yp = [], []
    with torch.no_grad():
        for x_lob, x_exp, y, _ in dl_test:
            x_lob = x_lob.to(device); x_exp = x_exp.to(device)
            logits = model(x_lob, x_exp)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            ys.append(y.numpy()); yp.append(pred)
    ys = np.concatenate(ys); yp = np.concatenate(yp)
    print(classification_report(ys, yp, digits=4))

    return model


if __name__ == "__main__":
    train_system(CONFIG)
