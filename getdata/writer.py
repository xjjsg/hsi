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
    "tx_server_time",
    "tx_local_time",
    "index_price",
    "fx_rate",
    "sentiment",
    "price",
    "iopv",
    "premium_rate",
    "tick_vol",
    "tick_amt",
    "tick_vwap",
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
    "idx_delay_ms",
    "fut_delay_ms",
    "data_flags",
]

CSV_HEADER_FUTURES = ["fut_price", "fut_mid", "fut_imb", "fut_delta_vol", "fut_pct"]


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
            "sh513130": {"has_futures": False},
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

    async def validate_daily_data(self, symbol: str) -> dict:
        """
        数据质量检查（日终调用）

        检查项：
        1. 时间戳唯一值数量（防止2025-12-22类似问题）
        2. 数据行数合理性
        3. 关键字段缺失率

        Returns:
            dict: 检查结果 {'valid': bool, 'warnings': list, 'stats': dict}
        """
        import pandas as pd

        file_path = self._get_file_path(symbol)
        if not os.path.exists(file_path):
            return {"valid": False, "warnings": ["文件不存在"], "stats": {}}

        try:
            df = pd.read_csv(file_path)
            warnings = []
            stats = {
                "total_rows": len(df),
                "tx_local_time_unique": (
                    df["tx_local_time"].nunique()
                    if "tx_local_time" in df.columns
                    else 0
                ),
            }

            # 🔧 检查1：时间戳唯一性异常
            if "tx_local_time" in df.columns:
                unique_ratio = stats["tx_local_time_unique"] / max(len(df), 1)

                if unique_ratio < 0.01:  # 唯一值<1%说明时间戳损坏
                    warnings.append(
                        f"⚠️ CRITICAL: 时间戳异常！"
                        f"总行数{len(df)}，但唯一时间戳仅{stats['tx_local_time_unique']}个 "
                        f"({unique_ratio*100:.2f}%)"
                    )
                elif unique_ratio < 0.5:  # 50%以下也不正常
                    warnings.append(
                        f"⚠️ WARNING: 时间戳重复率过高 "
                        f"({unique_ratio*100:.1f}% 唯一)"
                    )

            # 🔧 检查2：数据量异常
            if len(df) < 100:
                warnings.append(f"⚠️ WARNING: 数据量过少（{len(df)}行）")
            elif len(df) < 500:
                warnings.append(f"⚠️ INFO: 数据量偏少（{len(df)}行），可能是半天交易")

            # 🔧 检查3：关键字段缺失
            critical_fields = ["tx_local_time", "price", "bp1", "sp1"]
            for field in critical_fields:
                if field in df.columns:
                    null_ratio = df[field].isna().sum() / len(df)
                    if null_ratio > 0.5:
                        warnings.append(
                            f"⚠️ WARNING: 字段{field}缺失率{null_ratio*100:.1f}%"
                        )

            # 返回结果
            is_valid = len([w for w in warnings if "CRITICAL" in w]) == 0

            return {"valid": is_valid, "warnings": warnings, "stats": stats}

        except Exception as e:
            return {
                "valid": False,
                "warnings": [f"数据检查失败: {str(e)}"],
                "stats": {},
            }

    async def run_daily_validation(self):
        """运行所有symbol的日终验证"""
        print("\n" + "=" * 60)
        print("数据质量检查报告")
        print("=" * 60)

        all_valid = True
        for symbol in self.etf_config.keys():
            result = await self.validate_daily_data(symbol)

            print(f"\n【{symbol}】")
            print(f"  总行数: {result['stats'].get('total_rows', 0)}")
            print(f"  时间戳唯一值: {result['stats'].get('tx_local_time_unique', 0)}")

            if result["warnings"]:
                for w in result["warnings"]:
                    print(f"  {w}")
            else:
                print("  ✅ 数据质量良好")

            if not result["valid"]:
                all_valid = False
                print(f"  ❌ {symbol} 数据质量不合格，建议删除")

        print("\n" + "=" * 60)
        if all_valid:
            print("✅ 所有数据验证通过")
        else:
            print("❌ 发现数据质量问题，请检查上述警告")
        print("=" * 60 + "\n")

        return all_valid
