import pandas as pd
import numpy as np
import os
from datetime import datetime


def analyze_v1(file_path):
    print(f"分析 V1: {file_path}")
    df = pd.read_csv(file_path)

    total_rows = len(df)
    unique_times = df["tx_server_time"].nunique()

    # 冗余度
    redundancy_ratio = total_rows / unique_times if unique_times > 0 else 0

    # 零成交量 Tick (心跳/噪声)
    zero_vol_rows = len(df[df["tick_vol"] == 0])

    # 数据标志 (如果存在, V1中通常为0)
    reset_rows = len(df[df["data_flags"] == 1]) if "data_flags" in df.columns else 0

    # 指数价格缺失
    missing_index = df["index_price"].isna().sum()

    print(f"  总行数: {total_rows}")
    print(f"  唯一服务端时间戳: {unique_times}")
    print(f"  冗余比率: {redundancy_ratio:.2f}x")
    print(f"  零成交量行数: {zero_vol_rows} ({zero_vol_rows/total_rows:.1%})")
    print(f"  重置行数 (Flag=1): {reset_rows}")
    print(f"  缺失指数价格: {missing_index}")

    return df


def analyze_v2(file_path):
    print(f"\n分析 V2: {file_path}")
    df = pd.read_csv(file_path)

    total_rows = len(df)
    unique_times = df["tx_server_time"].nunique()

    # 冗余度 (应接近 1.0)
    redundancy_ratio = total_rows / unique_times if unique_times > 0 else 0

    # 零成交量 Tick
    zero_vol_rows = len(df[df["tick_vol"] == 0])

    # 数据标志
    reset_rows = len(df[df["data_flags"] == 1])

    # 指数价格缺失
    missing_index = df["index_price"].isna().sum()

    print(f"  总行数: {total_rows}")
    print(f"  唯一服务端时间戳: {unique_times}")
    print(f"  冗余比率: {redundancy_ratio:.2f}x")
    print(f"  零成交量行数: {zero_vol_rows} ({zero_vol_rows/total_rows:.1%})")
    print(f"  重置行数 (Flag=1): {reset_rows}")
    print(f"  缺失指数价格: {missing_index}")

    return df


def analyze_cleaned_v1(file_path):
    print(f"\n分析已清洗的 V1: {file_path}")
    df = pd.read_csv(file_path)

    total_rows = len(df)
    unique_times = df["tx_server_time"].nunique()

    # 冗余度
    redundancy_ratio = total_rows / unique_times if unique_times > 0 else 0

    # 零成交量 Tick
    zero_vol_rows = len(df[df["tick_vol"] == 0])

    print(f"  总行数: {total_rows}")
    print(f"  唯一服务端时间戳: {unique_times}")
    print(f"  冗余比率: {redundancy_ratio:.2f}x")
    print(f"  零成交量行数: {zero_vol_rows} ({zero_vol_rows/total_rows:.1%} if > 0)")

    return df


if __name__ == "__main__":
    v1_path = "/Users/xjjsg/Desktop/hsi/data/sz159920/sz159920-2025-11-26.csv"
    v1_cleaned_path = (
        "/Users/xjjsg/Desktop/hsi/cleaned_data/sz159920-2025-11-26-cleaned.csv"
    )
    v2_path = "/Users/xjjsg/Desktop/hsi/data/sz159920/sz159920-2026-01-08.csv"

    analyze_v1(v1_path)
    if os.path.exists(v1_cleaned_path):
        analyze_cleaned_v1(v1_cleaned_path)
    else:
        print(f"Cleaned file not found: {v1_cleaned_path}")
    analyze_v2(v2_path)
