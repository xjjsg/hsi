"""
历史数据清洗脚本
用于清洗 getdata V1 产生的冗余数据

清洗规则：
1. 按 tx_server_time 去重（保留每个时间点的最后一条）
2. 过滤 tick_vol == 0 且 data_flags == 0 的无效行
3. 过滤 data_flags == 1 的 Reset 行（可选）
4. 过滤 idx_delay_ms > 阈值的行
5. 重新计算 tick_vol（基于清洗后的数据）
"""
import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import argparse


def load_csv(file_path: str) -> pd.DataFrame:
    """加载 CSV 文件"""
    try:
        df = pd.read_csv(file_path)
        print(f"  加载: {file_path} ({len(df)} 行)")
        return df
    except Exception as e:
        print(f"  加载失败: {file_path} - {e}")
        return pd.DataFrame()


def clean_dataframe(
    df: pd.DataFrame,
    remove_zero_vol: bool = True,
    remove_reset: bool = False,
    max_delay_ms: int = 5000,
    recalc_volume: bool = True
) -> pd.DataFrame:
    """
    清洗 DataFrame
    
    参数：
        df: 原始数据
        remove_zero_vol: 是否移除 tick_vol=0 的行
        remove_reset: 是否移除 data_flags=1 的行
        max_delay_ms: 最大允许延迟（毫秒）
        recalc_volume: 是否重新计算 tick_vol
    """
    original_len = len(df)
    
    if df.empty:
        return df
    
    # 确保必要的列存在
    required_cols = ["tx_server_time"]
    for col in required_cols:
        if col not in df.columns:
            print(f"  警告: 缺少必要列 {col}")
            return df
    
    # Step 1: 按 tx_server_time 去重
    df = df.drop_duplicates(subset=["tx_server_time"], keep="last")
    dedup_len = len(df)
    print(f"  去重: {original_len} -> {dedup_len} ({original_len - dedup_len} 重复)")
    
    # Step 2: 移除 tick_vol=0 且 data_flags=0 的行
    if remove_zero_vol and "tick_vol" in df.columns and "data_flags" in df.columns:
        mask = ~((df["tick_vol"] == 0) & (df["data_flags"] == 0))
        df = df[mask]
        print(f"  移除空成交: {dedup_len} -> {len(df)} ({dedup_len - len(df)} 空行)")
    
    # Step 3: 移除 Reset 行（可选）
    if remove_reset and "data_flags" in df.columns:
        before_len = len(df)
        df = df[df["data_flags"] != 1]
        print(f"  移除Reset: {before_len} -> {len(df)} ({before_len - len(df)} Reset)")
    
    # Step 4: 过滤高延迟行
    if "idx_delay_ms" in df.columns:
        before_len = len(df)
        df = df[df["idx_delay_ms"] <= max_delay_ms]
        print(f"  过滤高延迟: {before_len} -> {len(df)} ({before_len - len(df)} 高延迟)")
    
    # Step 5: 过滤负延迟（时钟异常）
    if "idx_delay_ms" in df.columns:
        before_len = len(df)
        df = df[df["idx_delay_ms"] >= 0]
        print(f"  过滤负延迟: {before_len} -> {len(df)} ({before_len - len(df)} 负延迟)")
    
    # Step 6: 重新计算 tick_vol（可选）
    if recalc_volume and "tick_vol" in df.columns:
        # 需要找到累计成交量列来重算
        # 这里假设原始数据没有累计成交量，跳过
        pass
    
    final_len = len(df)
    reduction = (1 - final_len / original_len) * 100 if original_len > 0 else 0
    print(f"  最终: {final_len} 行 (减少 {reduction:.1f}%)")
    
    return df


def process_directory(
    data_dir: str,
    output_dir: str,
    symbol: str,
    **clean_kwargs
):
    """处理一个目录下的所有 CSV 文件"""
    pattern = os.path.join(data_dir, symbol, f"{symbol}-*.csv")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"未找到匹配文件: {pattern}")
        return
    
    print(f"\n{'='*60}")
    print(f"处理 {symbol}: 发现 {len(files)} 个文件")
    print(f"{'='*60}")
    
    # 创建输出目录
    out_symbol_dir = os.path.join(output_dir, symbol)
    os.makedirs(out_symbol_dir, exist_ok=True)
    
    total_original = 0
    total_cleaned = 0
    
    for file_path in files:
        print(f"\n处理文件: {os.path.basename(file_path)}")
        
        df = load_csv(file_path)
        if df.empty:
            continue
        
        total_original += len(df)
        
        # 清洗
        df_clean = clean_dataframe(df, **clean_kwargs)
        
        total_cleaned += len(df_clean)
        
        # 保存
        out_path = os.path.join(out_symbol_dir, os.path.basename(file_path))
        df_clean.to_csv(out_path, index=False)
        print(f"  保存: {out_path}")
    
    # 汇总
    print(f"\n{'='*60}")
    print(f"{symbol} 清洗完成:")
    print(f"  原始总行数: {total_original:,}")
    print(f"  清洗后行数: {total_cleaned:,}")
    print(f"  减少比例: {(1 - total_cleaned/total_original)*100:.1f}%")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="清洗 GetData V1 历史数据")
    parser.add_argument("--data-dir", default="./data", help="数据目录")
    parser.add_argument("--output-dir", default="./data_cleaned", help="输出目录")
    parser.add_argument("--symbols", nargs="+", default=["sz159920", "sh513130"], help="要处理的 symbol")
    parser.add_argument("--remove-reset", action="store_true", help="移除 data_flags=1 的行")
    parser.add_argument("--max-delay", type=int, default=5000, help="最大允许延迟(ms)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  GetData V1 历史数据清洗工具")
    print("=" * 60)
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"处理 Symbol: {args.symbols}")
    print(f"移除 Reset: {args.remove_reset}")
    print(f"最大延迟: {args.max_delay}ms")
    
    for symbol in args.symbols:
        process_directory(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            symbol=symbol,
            remove_zero_vol=True,
            remove_reset=args.remove_reset,
            max_delay_ms=args.max_delay,
            recalc_volume=False
        )
    
    print("\n清洗完成!")


if __name__ == "__main__":
    main()
