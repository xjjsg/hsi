"""
CSVWriter - CSV 写入模块
负责将聚合后的数据写入文件
"""
import os
import aiofiles
from datetime import datetime
from typing import Dict

from .aggregator import AggregatedRow


# CSV 表头
CSV_HEADER_BASE = [
    "symbol",
    "tx_server_time", "tx_local_time",
    "index_price", "fx_rate", "sentiment",
    "price", "iopv", "premium_rate",
    "tick_vol", "tick_amt", "tick_vwap",
    "bp1", "bv1", "bp2", "bv2", "bp3", "bv3", "bp4", "bv4", "bp5", "bv5",
    "sp1", "sv1", "sp2", "sv2", "sp3", "sv3", "sp4", "sv4", "sp5", "sv5",
    "idx_delay_ms", "fut_delay_ms", "data_flags"
]

CSV_HEADER_FUTURES = [
    "fut_price", "fut_mid", "fut_imb", "fut_delta_vol", "fut_pct"
]


class CSVWriter:
    """
    CSV 写入器
    
    特性：
    1. 按 symbol 分目录存储
    2. 按日期分文件
    3. 自动创建表头
    4. 异步写入
    """
    
    def __init__(self, base_dir: str = "./data"):
        self.base_dir = base_dir
        self.file_handles: Dict[str, str] = {}  # symbol -> current file path
        
        # ETF 配置
        self.etf_config = {
            "sz159920": {"has_futures": True},
            "sh513130": {"has_futures": False}
        }
    
    def _get_file_path(self, symbol: str) -> str:
        """获取当前日期的文件路径"""
        today = datetime.now().strftime("%Y-%m-%d")
        dir_path = os.path.join(self.base_dir, symbol)
        os.makedirs(dir_path, exist_ok=True)
        return os.path.join(dir_path, f"{symbol}-{today}.csv")
    
    def _get_header(self, symbol: str) -> list:
        """获取 symbol 对应的表头"""
        config = self.etf_config.get(symbol, {})
        if config.get("has_futures", False):
            return CSV_HEADER_BASE + CSV_HEADER_FUTURES
        return CSV_HEADER_BASE
    
    async def _ensure_file(self, symbol: str) -> str:
        """确保文件存在并有表头"""
        file_path = self._get_file_path(symbol)
        
        # 检查是否需要创建新文件
        if not os.path.exists(file_path):
            header = self._get_header(symbol)
            async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                await f.write(",".join(header) + "\n")
            print(f"[Writer] 创建新文件: {file_path}")
        
        return file_path
    
    async def write(self, row: AggregatedRow):
        """写入一行数据"""
        symbol = row.symbol
        has_futures = self.etf_config.get(symbol, {}).get("has_futures", False)
        
        file_path = await self._ensure_file(symbol)
        csv_row = row.to_csv_row(has_futures=has_futures)
        
        async with aiofiles.open(file_path, "a", encoding="utf-8") as f:
            await f.write(",".join(map(str, csv_row)) + "\n")
