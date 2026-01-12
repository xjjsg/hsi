import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from hsi_hft_v3.core.config import BAR_SIZE_S, ALLOWLIST_FIELDS
from hsi_hft_v3.core.data_contract import Bar

class BarBuilder:
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
        if pd.isna(fut_price): fut_price = None
        if pd.isna(fut_imb): fut_imb = None
        
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
            index_price=float(last_row.get("index_price", 0.0)),
            fx_rate=float(last_row.get("fx_rate", 0.0)),
            iopv=float(last_row.get("iopv", 0.0)),
            fut_price=fut_price,
            fut_imb=fut_imb
        )
