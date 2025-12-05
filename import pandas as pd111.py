import pandas as pd
import numpy as np
import glob
import os

# === 配置区域 ===
# 建议分析最近几天的文件
FILE_PATTERN = "./data/sz159920/sz159920-*.csv"
BAR_INTERVAL_SEC = 3 # 数据的重采样频率 (3秒)

def run_analysis():
    files = glob.glob(FILE_PATTERN)
    if not files:
        print("未找到数据文件，请检查路径。")
        return

    print(f"正在分析 {len(files)} 个文件...")
    
    # 我们关注的时间窗口 (分钟)
    horizons_minutes = [1, 3, 5, 10, 20]
    
    combined_data = {m: [] for m in horizons_minutes}
    
    for f in files:
        try:
            # 读取数据
            df = pd.read_csv(f)
            if 'tx_local_time' not in df.columns: continue
            
            # 基础清洗与重采样
            df['datetime'] = pd.to_datetime(df['tx_local_time'], unit='ms')
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)
            
            # 重采样为 3s (与策略保持一致)
            df_resampled = df.resample(f'{BAR_INTERVAL_SEC}s').last().ffill().dropna()
            
            # 计算中间价
            mid_price = (df_resampled['bp1'] + df_resampled['sp1']) / 2
            prices = mid_price.values
            
            # 对每个时间窗口计算最大波动
            for minutes in horizons_minutes:
                horizon_bars = int(minutes * 60 / BAR_INTERVAL_SEC)
                if len(prices) <= horizon_bars: continue
                
                # 使用 pandas 的 rolling window 计算未来 N 个 bar 的最大/最小值
                # 注意：这里使用 FixedForwardWindowIndexer 来实现“向前看”
                indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=horizon_bars)
                rolling_max = pd.Series(prices).rolling(window=indexer).max()
                rolling_min = pd.Series(prices).rolling(window=indexer).min()
                
                # 计算相对于当前价格的最大涨幅和最大跌幅 (绝对值)
                curr_prices = pd.Series(prices)
                upside = (rolling_max - curr_prices) / curr_prices
                downside = (curr_prices - rolling_min) / curr_prices
                
                # 取两者的最大值作为该时刻的“波动幅度”
                max_dev = np.maximum(upside, downside).dropna()
                
                combined_data[minutes].extend(max_dev.values)
                
        except Exception as e:
            print(f"处理文件 {f} 出错: {e}")

    # === 打印分析报告 ===
    print("\n" + "="*60)
    print(f"📊 波动率阈值分析报告 (基于 {len(files)} 天数据)")
    print("="*60)
    print(f"{'预测窗口':<10} | {'50%位(中位数)':<12} | {'80%位':<12} | {'90%位':<12} | {'95%位':<12} | {'99%位':<12}")
    print("-" * 80)

    for m in horizons_minutes:
        data = np.array(combined_data[m])
        if len(data) == 0: continue
        
        # 计算分位数
        p50 = np.percentile(data, 50)
        p80 = np.percentile(data, 80)
        p90 = np.percentile(data, 90)
        p95 = np.percentile(data, 95)
        p99 = np.percentile(data, 99)
        
        print(f"{m} 分钟{'':<4} | {p50:.6f}{'':<4} | {p80:.6f}{'':<4} | {p90:.6f}{'':<4} | {p95:.6f}{'':<4} | {p99:.6f}")

if __name__ == "__main__":
    run_analysis()