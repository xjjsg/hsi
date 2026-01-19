

import pandas as pd
import numpy as np
import os
from datetime import datetime

def analyze_v1(file_path):
    print(f"Analyzing V1: {file_path}")
    df = pd.read_csv(file_path)
    
    total_rows = len(df)
    unique_times = df['tx_server_time'].nunique()
    
    # Redundancy
    redundancy_ratio = total_rows / unique_times if unique_times > 0 else 0
    
    # Zero volume ticks (heartbeats/noise)
    zero_vol_rows = len(df[df['tick_vol'] == 0])
    
    # Data flags (if present, usually 0 in V1)
    reset_rows = len(df[df['data_flags'] == 1]) if 'data_flags' in df.columns else 0
    
    # Index price missing
    missing_index = df['index_price'].isna().sum()
    
    print(f"  Total Rows: {total_rows}")
    print(f"  Unique Server Times: {unique_times}")
    print(f"  Redundancy Ratio: {redundancy_ratio:.2f}x")
    print(f"  Zero Volume Rows: {zero_vol_rows} ({zero_vol_rows/total_rows:.1%})")
    print(f"  Reset Rows (Flag=1): {reset_rows}")
    print(f"  Missing Index Price: {missing_index}")
    
    return df

def analyze_v2(file_path):
    print(f"\nAnalyzing V2: {file_path}")
    df = pd.read_csv(file_path)
    
    total_rows = len(df)
    unique_times = df['tx_server_time'].nunique()
    
    # Redundancy (should be close to 1.0)
    redundancy_ratio = total_rows / unique_times if unique_times > 0 else 0
    
    # Zero volume ticks
    zero_vol_rows = len(df[df['tick_vol'] == 0])
    
    # Data flags
    reset_rows = len(df[df['data_flags'] == 1])
    
    # Index price missing
    missing_index = df['index_price'].isna().sum()
    
    print(f"  Total Rows: {total_rows}")
    print(f"  Unique Server Times: {unique_times}")
    print(f"  Redundancy Ratio: {redundancy_ratio:.2f}x")
    print(f"  Zero Volume Rows: {zero_vol_rows} ({zero_vol_rows/total_rows:.1%})")
    print(f"  Reset Rows (Flag=1): {reset_rows}")
    print(f"  Missing Index Price: {missing_index}")
    
    return df


def analyze_cleaned_v1(file_path):
    print(f"\nAnalyzing Cleaned V1: {file_path}")
    df = pd.read_csv(file_path)
    
    total_rows = len(df)
    unique_times = df['tx_server_time'].nunique()
    
    # Redundancy
    redundancy_ratio = total_rows / unique_times if unique_times > 0 else 0
    
    # Zero volume ticks
    zero_vol_rows = len(df[df['tick_vol'] == 0])
    
    print(f"  Total Rows: {total_rows}")
    print(f"  Unique Server Times: {unique_times}")
    print(f"  Redundancy Ratio: {redundancy_ratio:.2f}x")
    print(f"  Zero Volume Rows: {zero_vol_rows} ({zero_vol_rows/total_rows:.1%} if > 0)")
    
    return df

if __name__ == "__main__":
    v1_path = "/Users/xjjsg/Desktop/hsi/data/sz159920/sz159920-2025-11-26.csv"
    v1_cleaned_path = "/Users/xjjsg/Desktop/hsi/cleaned_data/sz159920-2025-11-26-cleaned.csv"
    v2_path = "/Users/xjjsg/Desktop/hsi/data/sz159920/sz159920-2026-01-08.csv"
    
    analyze_v1(v1_path)
    if os.path.exists(v1_cleaned_path):
        analyze_cleaned_v1(v1_cleaned_path)
    else:
        print(f"Cleaned file not found: {v1_cleaned_path}")
    analyze_v2(v2_path)
