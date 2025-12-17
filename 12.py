import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime

# 配置：你的数据目录
DATA_DIR = "./data"

def repair_csv(file_path):
    print(f"正在处理: {file_path} ...", end=" ")
    
    try:
        # 1. 读取原始数据 (所有列都先读作 object/float 混合，防止报错)
        df = pd.read_csv(file_path, low_memory=False)
        
        # --- 修复 A: 负成交量 (Data Corruption) ---
        if 'tick_vol' in df.columns:
            # 强制转数字
            df['tick_vol'] = pd.to_numeric(df['tick_vol'], errors='coerce').fillna(0)
            
            mask_neg = df['tick_vol'] < 0
            neg_count = mask_neg.sum()
            
            if neg_count > 0:
                # 负值归零
                df.loc[mask_neg, 'tick_vol'] = 0
                # 如果有 tick_amt 也顺便处理
                if 'tick_amt' in df.columns:
                    df.loc[mask_neg, 'tick_amt'] = 0
                
                print(f"[发现 {neg_count} 行负成交量 -> 已归零]", end=" ")
            
        # --- 修复 B: Index Price = 0 (Data Corruption) ---
        if 'index_price' in df.columns:
            df['index_price'] = pd.to_numeric(df['index_price'], errors='coerce')
            
            mask_zero = (df['index_price'] == 0) | (df['index_price'].isna())
            zero_count = mask_zero.sum()
            
            if zero_count > 0:
                # 0 转 NaN
                df.loc[df['index_price'] == 0, 'index_price'] = np.nan
                # 前向填充
                df['index_price'] = df['index_price'].ffill()
                print(f"[修复 {zero_count} 行指数缺失]", end=" ")

        # --- 修复 C: 补全延迟字段 (Latency & Time Alignment) ---
        if 'tx_local_time' in df.columns:
            tx_local = pd.to_numeric(df['tx_local_time'], errors='coerce')
            
            # 1. 计算 idx_delay_ms
            if 'bd_local_time' in df.columns and 'idx_delay_ms' not in df.columns:
                bd_local = pd.to_numeric(df['bd_local_time'], errors='coerce')
                # 延迟 = 当前抓取时间 - 百度抓取时间
                df['idx_delay_ms'] = tx_local - bd_local
                # 处理未初始化的行 (bd_local=0)
                df.loc[bd_local == 0, 'idx_delay_ms'] = 99999
            
            # 2. 计算 fut_delay_ms (如果能算的话)
            if 'fut_local_time' in df.columns and 'fut_delay_ms' not in df.columns:
                # 尝试转换并计算，如果失败则跳过
                pass 
                
        # --- 修复 D: 增加 data_flags ---
        if 'data_flags' not in df.columns:
            df['data_flags'] = 0

        # --- 保存 (直接覆盖原文件) ---
        # 仅保留有效列 (去掉可能存在的 Unnamed)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # 覆盖写入
        df.to_csv(file_path, index=False)
        print("-> 完成 (已覆盖)")
        
    except Exception as e:
        print(f"\n[错误] 处理 {file_path} 失败: {e}")

def main():
    print(f"开始扫描并修复数据 (直接覆盖模式): {DATA_DIR}")
    print("⚠️  警告: 请确保你已经备份了 data 文件夹！")
    
    # 递归查找所有 csv
    csv_files = glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True)
    
    count = 0
    for f in csv_files:
        # 跳过错误日志文件
        if "error.txt" in f:
            continue
            
        repair_csv(f)
        count += 1
        
    print(f"\n全部完成! 共处理 {count} 个文件。")

if __name__ == "__main__":
    main()