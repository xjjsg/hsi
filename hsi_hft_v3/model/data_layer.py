#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HSI HFT V3 - Data Layer (Consolidated Module)

整合模块：
- core/data_contract.py - 数据契约定义
- data/bar_builder.py - K线构建器
- data/aligner.py - 双流对齐器
- data/loader.py - 数据加载器

功能：
1. 数据契约：Bar (K线) 和 AlignedSample (对齐样本)
2. K线构建：从Tick数据聚合为3秒K线
3. 双流对齐：Target和Aux流的因果对齐
4. 数据加载：批量加载和预处理数据
"""

import os
import glob
import re
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

# 导入配置（从config）
from hsi_hft_v3.config import (
    TARGET_SYMBOL,
    AUX_SYMBOL,
    BAR_SIZE_S,
    ALLOWLIST_FIELDS,
)


# ==========================================
# 1. 数据契约 (Data Contract)
# ==========================================


@dataclass
class Bar:
    """标准 3秒 K线结构"""

    ts_ms: int  # 桶结束时间
    symbol: str

    # 市场数据
    mid: float
    vwap: float
    volume: int
    amount: float

    # LOB (桶结束时的快照)
    bids: List[tuple] = field(default_factory=list)  # [(价格, 数量), ...] 前5档
    asks: List[tuple] = field(default_factory=list)

    # 外部 / 衍生数据
    sentiment: float = 0.0
    premium_rate: float = 0.0
    index_price: float = 0.0  # V5 规范新增
    fx_rate: float = 0.0  # V5 规范新增
    iopv: float = 0.0  # V5 规范新增

    # 期货数据 (Target 独有, 可选)
    fut_price: Optional[float] = None
    fut_imb: Optional[float] = None

    def sanity_check(self) -> bool:
        """基础数据完整性检查"""
        # 1. 价格逻辑
        if self.mid <= 0 or not np.isfinite(self.mid):
            return False

        # 2. LOB 逻辑
        if len(self.bids) > 0 and len(self.asks) > 0:
            best_bid = self.bids[0][0]
            best_ask = self.asks[0][0]
            if best_bid > best_ask:  # 交叉盘
                return False

        # 3. 成交量逻辑
        if self.volume < 0 or self.amount < 0:
            return False

        return True


@dataclass
class AlignedSample:
    """用于特征的双流对齐输入"""

    ts_ms: int
    target: Bar
    aux: Optional[Bar]

    # Masks
    aux_available: bool  # 1 if aux exists and lag <= max_lag
    aux_lag_ms: Optional[int]
    has_fut: bool

    def to_whitebox_input(self) -> Dict:
        """转换为 WhiteBoxFactory 的字典结构"""
        return {
            "target": self.target,
            "aux": self.aux,
            "masks": {
                "aux_available": 1.0 if self.aux_available else 0.0,
                "has_fut": 1.0 if self.has_fut else 0.0,
            },
        }


# ==========================================
# 2. K线构建器 (Bar Builder)
# ==========================================


class BarBuilder:
    """将原始Tick数据转换为3秒K线"""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bucket_ms = BAR_SIZE_S * 1000

    def process_dataframe(self, df: pd.DataFrame) -> List[Bar]:
        """将原始 DataFrame 转换为 List[Bar]"""
        # 1. 过滤字段
        valid_cols = [c for c in ALLOWLIST_FIELDS if c in df.columns]
        df = df[valid_cols].copy()

        # 2. 添加时间桶
        df["ts_bucket"] = (df["tx_local_time"] // self.bucket_ms) * self.bucket_ms

        # 3. 慢速变量前向填充 (V5 规范要求)
        # 慢速变量: iopv, index_price, fx_rate, sentiment, premium_rate
        slow_cols = ["iopv", "index_price", "fx_rate", "sentiment", "premium_rate"]
        for c in slow_cols:
            if c in df.columns:
                df[c] = df[c].ffill()
                # 可选: 缺失掩码? Spec 说 Bar 中有 'missing_mask'?
                # 先简单处理: ffill 确保 Tick 之间的 Bar 不为零。

        bars = []
        for ts, group in df.groupby("ts_bucket"):
            bar = self._aggregate_group(ts, group)
            if bar.sanity_check():
                bars.append(bar)
            # else: log dropped bar

        return bars

    def _aggregate_group(self, ts: int, group: pd.DataFrame) -> Bar:
        """在时间桶内聚合 Tick 数据以创建 Bar"""
        last_row = group.iloc[-1]

        # LOB 快照
        bids = []
        asks = []
        for i in range(1, 6):
            if f"bp{i}" in last_row and f"bv{i}" in last_row:
                bids.append((float(last_row[f"bp{i}"]), float(last_row[f"bv{i}"])))
            if f"sp{i}" in last_row and f"sv{i}" in last_row:
                asks.append((float(last_row[f"sp{i}"]), float(last_row[f"sv{i}"])))

        # 成交量聚合
        vol = int(group["tick_vol"].sum())
        amt = float(group["tick_amt"].sum())
        vwap = amt / vol if vol > 0 else last_row.get("price", 0.0)

        # 期货 (可选)
        fut_price = last_row.get("fut_price", None)
        fut_imb = last_row.get("fut_imb", None)
        # Handle nan
        if pd.isna(fut_price):
            fut_price = None
        if pd.isna(fut_imb):
            fut_imb = None

        # 🔧 修复：使用.get()从DataFrame读取fx_rate等字段
        return Bar(
            ts_ms=ts,
            symbol=self.symbol,
            mid=float((bids[0][0] + asks[0][0]) / 2.0) if bids and asks else 0.0,
            vwap=float(vwap) if not pd.isna(vwap) else 0.0,
            volume=vol,
            amount=amt,
            bids=bids,
            asks=asks,
            sentiment=float(last_row.get("sentiment", 0.0)),
            premium_rate=float(last_row.get("premium_rate", 0.0)),
            index_price=float(last_row.get("index_price", 0.0)),  # 🔧 修复
            fx_rate=float(last_row.get("fx_rate", 0.0)),  # 🔧 修复
            iopv=float(last_row.get("iopv", 0.0)),  # 🔧 修复
            fut_price=fut_price,
            fut_imb=fut_imb,
        )


# ==========================================
# 3. 双流对齐器 (Dual Stream Aligner)
# ==========================================


class DualStreamAligner:
    """基于 Asof 逻辑的严格因果对齐"""

    def __init__(self, max_lag_ms: int = 30000):

        from hsi_hft_v3.config import DataConfig

        data_cfg = DataConfig()
        self.max_lag_ms = max_lag_ms if max_lag_ms != 30000 else data_cfg.max_lag_ms

    def align(self, target_bars: List[Bar], aux_bars: List[Bar]) -> List[AlignedSample]:
        """
        对齐 Target 和 Aux 两个数据流，并应用因果约束

        对于每个 Target Bar，找到在其之前到达的最近的 Aux Bar，
        前提是滞后时间不超过 max_lag_ms。
        """
        # 确保严格的时间排序
        target_bars.sort(key=lambda x: x.ts_ms)
        aux_bars.sort(key=lambda x: x.ts_ms)

        aligned_samples = []

        # 双指针逻辑
        aux_idx = 0
        n_aux = len(aux_bars)
        last_valid_aux = None

        for t_bar in target_bars:
            # 推进 Aux 指针以找到 t_bar.ts_ms 之前或同时的快照
            # 我们需要满足 aux.ts_ms <= t_bar.ts_ms 的最后一个 aux

            while aux_idx < n_aux and aux_bars[aux_idx].ts_ms <= t_bar.ts_ms:
                last_valid_aux = aux_bars[aux_idx]
                aux_idx += 1

            # Determine logic
            aux_val = None
            aux_available = False
            aux_lag = None

            if last_valid_aux is not None:
                lag = t_bar.ts_ms - last_valid_aux.ts_ms
                if lag <= self.max_lag_ms:
                    aux_val = last_valid_aux
                    aux_available = True
                    aux_lag = lag
                else:
                    # Stale aux data
                    aux_available = False
                    aux_lag = lag  # Keep lag for debugging, but not available

            # Futures check
            has_fut = t_bar.fut_price is not None and t_bar.fut_imb is not None

            sample = AlignedSample(
                ts_ms=t_bar.ts_ms,
                target=t_bar,
                aux=aux_val,
                aux_available=aux_available,
                aux_lag_ms=aux_lag,
                has_fut=has_fut,
            )
            aligned_samples.append(sample)

        return aligned_samples


# ==========================================
# 4. 数据加载器 (Data Loader)
# ==========================================


class V5DataLoader:
    """V5 架构的高级数据加载器"""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.target_symbol = TARGET_SYMBOL
        self.aux_symbol = AUX_SYMBOL

    def load_date_range(
        self,
        start_date: str = None,
        end_date: str = None,
        exclude_dates: List[str] = None,
    ) -> Dict[str, List[AlignedSample]]:
        """
        加载指定日期范围的数据。
        Returns: Dict {date_str: List[AlignedSample]}
        """
        pairs = self._match_files()
        results = {}

        # 按日期过滤
        filtered_pairs = []
        for date, tgt_path, aux_path in pairs:
            if start_date and date < start_date:
                continue
            if end_date and date > end_date:
                continue
            if exclude_dates and date in exclude_dates:
                continue
            filtered_pairs.append((date, tgt_path, aux_path))

        print(f"[Loader] Found {len(filtered_pairs)} valid days in {self.data_dir}")

        # Process each day
        bb_tgt = BarBuilder(self.target_symbol)
        bb_aux = BarBuilder(self.aux_symbol)
        aligner = DualStreamAligner()

        for date, tgt_path, aux_path in filtered_pairs:
            print(f"Loading {date}...")
            try:
                # Read CSVs
                df_tgt = pd.read_csv(tgt_path)
                df_aux = pd.read_csv(aux_path) if aux_path else pd.DataFrame()

                # Build Bars
                bars_tgt = bb_tgt.process_dataframe(df_tgt)
                bars_aux = bb_aux.process_dataframe(df_aux) if not df_aux.empty else []

                # Align
                samples = aligner.align(bars_tgt, bars_aux)

                if samples:
                    results[date] = samples
                    print(f"  -> {len(samples)} samples")
                else:
                    print("  -> No samples produced")

            except Exception as e:
                print(f"  -> Error loading {date}: {e}")

        return results

    def _match_files(self) -> List[Tuple[str, str, str]]:
        """按日期匹配文件"""
        # 假设结构: data_dir/sz159920/*.csv
        tgt_pattern = os.path.join(self.data_dir, self.target_symbol, "*.csv")
        # 假设结构: data_dir/sh513130/*.csv
        aux_pattern = os.path.join(self.data_dir, self.aux_symbol, "*.csv")

        tgt_files = glob.glob(tgt_pattern)
        aux_files = glob.glob(aux_pattern)

        print(
            f"[Loader] 扫描到: {len(tgt_files)} 个目标文件, {len(aux_files)} 个辅助文件"
        )

        def get_date(path):
            # Expecting *-YYYY-MM-DD.csv
            m = re.search(r"(\d{4}-\d{2}-\d{2})", path)
            return m.group(1) if m else None

        tgt_map = {get_date(f): f for f in tgt_files if get_date(f)}
        aux_map = {get_date(f): f for f in aux_files if get_date(f)}

        common_dates = sorted(tgt_map.keys())

        pairs = []
        for d in common_dates:
            pairs.append(
                (d, tgt_map[d], aux_map.get(d))
            )  # Aux is optional but preferred

        return pairs
