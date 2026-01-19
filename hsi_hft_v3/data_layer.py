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

# 导入配置（从trading_layer）
TARGET_SYMBOL = "sz159920"
AUX_SYMBOL = "sh513130"
BAR_SIZE_S = 3

# ALLOWLIST_FIELDS - 定义允许的字段
ALLOWLIST_FIELDS = [
    "tx_local_time",
    "symbol",
    "price",
    "tick_vol",
    "tick_amt",
    "bp1",
    "bv1",
    "bp2",
    "bv2",
    "bp3",
    "bv3",
    "bp4",
    "bv4",
    "bp5",
    "bv5",
    "sp1",
    "sv1",
    "sp2",
    "sv2",
    "sp3",
    "sv3",
    "sp4",
    "sv4",
    "sp5",
    "sv5",
    "sentiment",
    "premium_rate",
    "index_price",
    "fx_rate",
    "iopv",
    "fut_price",
    "fut_imb",
]


# ==========================================
# 1. 数据契约 (Data Contract)
# ==========================================


@dataclass
class Bar:
    """Standard 3s Bar Structure"""

    ts_ms: int  # Bucket end time
    symbol: str

    # Market Data
    mid: float
    vwap: float
    volume: int
    amount: float

    # LOB (Snapshot at bucket end)
    bids: List[tuple] = field(default_factory=list)  # [(price, vol), ...] for 5 levels
    asks: List[tuple] = field(default_factory=list)

    # External / Derived
    sentiment: float = 0.0
    premium_rate: float = 0.0
    index_price: float = 0.0  # Added for V5 Spec
    fx_rate: float = 0.0  # Added for V5 Spec
    iopv: float = 0.0  # Added for V5 Spec

    # Futures (Target Only, Optional)
    fut_price: Optional[float] = None
    fut_imb: Optional[float] = None

    def sanity_check(self) -> bool:
        """Basic data integrity checks"""
        # 1. Price Logic
        if self.mid <= 0 or not np.isfinite(self.mid):
            return False

        # 2. LOB Logic
        if len(self.bids) > 0 and len(self.asks) > 0:
            best_bid = self.bids[0][0]
            best_ask = self.asks[0][0]
            if best_bid > best_ask:  # Crossed book
                return False

        # 3. Volume Logic
        if self.volume < 0 or self.amount < 0:
            return False

        return True


@dataclass
class AlignedSample:
    """Dual-Stream Aligned Input for Features"""

    ts_ms: int
    target: Bar
    aux: Optional[Bar]

    # Masks
    aux_available: bool  # 1 if aux exists and lag <= max_lag
    aux_lag_ms: Optional[int]
    has_fut: bool

    def to_whitebox_input(self) -> Dict:
        """Convert to dict structure for WhiteBoxFactory"""
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
    """Convert raw tick data to 3-second bars"""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bucket_ms = BAR_SIZE_S * 1000

    def process_dataframe(self, df: pd.DataFrame) -> List[Bar]:
        """Convert Raw DataFrame to List[Bar]"""
        # 1. Filter Fields
        valid_cols = [c for c in ALLOWLIST_FIELDS if c in df.columns]
        df = df[valid_cols].copy()

        # 2. Add bucket time
        df["ts_bucket"] = (df["tx_local_time"] // self.bucket_ms) * self.bucket_ms

        # 3. Slow Variable Forward Fill (V5 Spec Req)
        # Slow vars: iopv, index_price, fx_rate, sentiment, premium_rate
        slow_cols = ["iopv", "index_price", "fx_rate", "sentiment", "premium_rate"]
        for c in slow_cols:
            if c in df.columns:
                df[c] = df[c].ffill()
                # Optional: missing mask? Spec says 'missing_mask' in Bar?
                # Start simple: ffill ensures we don't have zeros in bars between ticks.

        bars = []
        for ts, group in df.groupby("ts_bucket"):
            bar = self._aggregate_group(ts, group)
            if bar.sanity_check():
                bars.append(bar)
            # else: log dropped bar

        return bars

    def _aggregate_group(self, ts: int, group: pd.DataFrame) -> Bar:
        """Aggregate tick data within a bucket to create a Bar"""
        last_row = group.iloc[-1]

        # LOB Snapshot
        bids = []
        asks = []
        for i in range(1, 6):
            if f"bp{i}" in last_row and f"bv{i}" in last_row:
                bids.append((float(last_row[f"bp{i}"]), float(last_row[f"bv{i}"])))
            if f"sp{i}" in last_row and f"sv{i}" in last_row:
                asks.append((float(last_row[f"sp{i}"]), float(last_row[f"sv{i}"])))

        # Volume Aggregation
        vol = int(group["tick_vol"].sum())
        amt = float(group["tick_amt"].sum())
        vwap = amt / vol if vol > 0 else last_row.get("price", 0.0)

        # Futures (Optional)
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
    """Strict Causal Alignment using Asof Logic"""

    def __init__(self, max_lag_ms: int = 30000):
        self.max_lag_ms = max_lag_ms

    def align(self, target_bars: List[Bar], aux_bars: List[Bar]) -> List[AlignedSample]:
        """
        Align target and auxiliary bar streams with causal constraints

        For each target bar, find the most recent aux bar that arrived before it,
        subject to max_lag_ms constraint.
        """
        # Ensure strict sorted order
        target_bars.sort(key=lambda x: x.ts_ms)
        aux_bars.sort(key=lambda x: x.ts_ms)

        aligned_samples = []

        # Dual Pointer Logic
        aux_idx = 0
        n_aux = len(aux_bars)
        last_valid_aux = None

        for t_bar in target_bars:
            # Advance aux pointer to find the snapshot right before or at t_bar.ts_ms
            # We want last aux where aux.ts_ms <= t_bar.ts_ms

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
    """High-level data loader for V5 architecture"""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.target_symbol = TARGET_SYMBOL
        self.aux_symbol = AUX_SYMBOL

    def load_date_range(
        self, start_date: str = None, end_date: str = None
    ) -> Dict[str, List[AlignedSample]]:
        """
        Load data for a range of dates.
        Returns: Dict {date_str: List[AlignedSample]}
        """
        pairs = self._match_files()
        results = {}

        # Filter by date
        filtered_pairs = []
        for date, tgt_path, aux_path in pairs:
            if start_date and date < start_date:
                continue
            if end_date and date > end_date:
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
        """Find paired files by date"""
        # Assume structure: data_dir/sz159920/*.csv
        tgt_pattern = os.path.join(self.data_dir, self.target_symbol, "*.csv")
        # Assume structure: data_dir/sh513130/*.csv
        aux_pattern = os.path.join(self.data_dir, self.aux_symbol, "*.csv")

        tgt_files = glob.glob(tgt_pattern)
        aux_files = glob.glob(aux_pattern)

        print(
            f"[Loader] Scanning: {len(tgt_files)} target files, {len(aux_files)} aux files"
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
