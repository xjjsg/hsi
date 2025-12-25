import os
import glob
import re
import pandas as pd
import numpy as np
import warnings
from datetime import time

# 屏蔽 Pandas 的链式赋值警告
warnings.filterwarnings("ignore")

# ==========================================
# 1. 核心配置 (Config)
# ==========================================
CONFIG = {
    # 数据路径 (递归搜索)
    "DATA_DIR": "./xjjsg",  
    "SYMBOL": "sz159920",
    
    # 数据处理 (完全复刻 modelbuild.py)
    "TIMEZONE": "Asia/Shanghai",
    "RESAMPLE_FREQ": "3S",
    
    # 实盘参数
    "CAPITAL": 100000.0,    # 10万本金
    "COMMISSION": 0.0001,   # 万1
    
    # 要测试的时间窗口 (分钟)
    "HORIZONS": [1, 3, 5, 10, 15, 30]
}

# ==========================================
# 2. 正宗数据加载器 (AlphaForge Logic)
# ==========================================
def load_data_strict():
    """
    复刻 modelbuild.py 的数据清洗流程：
    Glob递归 -> 正则匹配 -> 时区转换 -> 3S重采样 -> 数值清洗
    """
    print(f"🚀 [Loader] 启动严格模式，扫描路径: {CONFIG['DATA_DIR']}")
    
    # 1. 递归查找文件
    pattern = os.path.join(CONFIG['DATA_DIR'], "**", f"{CONFIG['SYMBOL']}*.csv")
    files = sorted(glob.glob(pattern, recursive=True))
    
    if not files:
        # 尝试备用路径逻辑 (适配不同的解压结构)
        pattern = os.path.join(".", "**", f"{CONFIG['SYMBOL']}*.csv")
        files = sorted(glob.glob(pattern, recursive=True))
    
    print(f"🔎 [Loader] 发现 {len(files)} 个源文件")
    
    df_list = []
    
    for f in files:
        try:
            # 只读需要的列
            df = pd.read_csv(f, usecols=["tx_local_time", "bp1", "sp1"])
            
            # --- 时间清洗 (核心) ---
            if "tx_local_time" not in df.columns: continue
            
            # 毫秒时间戳 -> UTC -> 上海时间 -> 去时区
            dt_utc = pd.to_datetime(df["tx_local_time"], unit="ms", utc=True, errors="coerce")
            df["datetime"] = dt_utc.dt.tz_convert(CONFIG["TIMEZONE"]).dt.tz_localize(None)
            
            # --- 数值清洗 ---
            for c in ["bp1", "sp1"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            
            # 排序与去重
            df = df.sort_values("datetime")
            df = df.drop_duplicates(subset="datetime", keep="last")
            df = df.set_index("datetime").sort_index()
            
            # --- 3S 重采样 (核心) ---
            # 模拟模型视角的“快照”
            df_res = df.resample(CONFIG["RESAMPLE_FREQ"]).last().dropna()
            
            # 过滤 0 值和非交易时间 (只保留连续竞价)
            df_res = df_res[(df_res["bp1"] > 0) & (df_res["sp1"] > 0)]
            
            t = df_res.index.time
            # 简单过滤：9:30-11:30, 13:00-14:57 (去掉尾盘竞价)
            mask = ((t >= time(9, 30)) & (t <= time(11, 30))) | \
                   ((t >= time(13, 0)) & (t <= time(14, 57)))
            df_res = df_res[mask]
            
            df_list.append(df_res)
            
        except Exception as e:
            # print(f"⚠️ 跳过坏文件 {f}: {e}")
            continue
            
    if not df_list:
        raise ValueError("❌ 未加载到有效数据，请检查路径！")
        
    full_df = pd.concat(df_list).sort_index()
    print(f"✅ [Loader] 数据加载完毕: {len(full_df)} 条 3S 快照")
    return full_df

# ==========================================
# 3. 10w 实盘生存模拟 (Simulation)
# ==========================================
def run_simulation(df):
    print("\n" + "="*60)
    print(f"💰 实盘极限Tick挑战 (本金: {int(CONFIG['CAPITAL'])} | 成本: 万{int(CONFIG['COMMISSION']*10000)})")
    print(f"🎯 逻辑: 对手价(Ask)买入 -> 等待(Bid)覆盖成本 -> 只要能微利就跑")
    print("="*60)
    
    # --- 向量化计算进场成本 ---
    ask_price = df["sp1"]
    
    # 向下取整到 100 股
    shares = (CONFIG["CAPITAL"] // (ask_price * 100)) * 100
    
    # 过滤钱不够买一手的情况
    valid_mask = shares > 0
    if valid_mask.sum() == 0:
        print("❌ 资金不足以买入一手，模拟结束")
        return

    # 实际买入金额与费用
    entry_amt = shares * ask_price
    entry_fee = entry_amt * CONFIG["COMMISSION"]
    total_cost = entry_amt + entry_fee
    
    # --- 计算保本卖出价 (Break-even Bid) ---
    # 公式: Revenue * (1 - comm) > Total Cost
    # Revenue > Total Cost / (1 - comm)
    # Bid Price > (Total Cost / (1 - comm)) / shares
    min_revenue_needed = total_cost / (1 - CONFIG["COMMISSION"])
    break_even_bid = min_revenue_needed / shares
    
    # 保存结果容器
    results = {}
    
    print(f"{'持有时间':<10} | {'胜率 (能活着出来)':<20} | {'评价'}")
    print("-" * 60)
    
    best_horizon = None
    best_win_rate = -1
    
    for minutes in CONFIG["HORIZONS"]:
        # 将分钟转为 Bar 数 (3秒一个Bar)
        bars = int(minutes * 60 / 3)
        
        # 获取未来 N 分钟内的 "最高买一价" (Max Future Bid)
        # 使用 FixedForwardWindow 进行向量化 Look-ahead
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=bars)
        future_max_bid = df["bp1"].rolling(window=indexer).max()
        
        # 判定: 未来最高Bid 是否 > 保本价
        is_win = (future_max_bid > break_even_bid) & valid_mask
        
        win_rate = is_win.mean()
        results[minutes] = is_win
        
        # 评价体系
        if win_rate < 0.20: verdict = "💀 必死无疑"
        elif win_rate < 0.30: verdict = "⚠️ 高风险"
        elif win_rate < 0.45: verdict = "🎲 勉强博弈"
        else: verdict = "✅ 推荐区间"
        
        print(f"{minutes:<3} 分钟{'':<5} | {win_rate:<22.2%} | {verdict}")
        
        if win_rate > best_win_rate:
            best_win_rate = win_rate
            best_horizon = minutes

    # ==========================================
    # 4. 黄金时段热力图 (Heatmap)
    # ==========================================
    if best_horizon:
        print("\n" + "="*60)
        print(f"⏰ 日内最佳时机分析 (基于 {best_horizon} 分钟持仓)")
        print("="*60)
        
        # 将最佳周期的胜负结果并入 DataFrame
        df["is_win"] = results[best_horizon]
        
        # 按 15分钟 分桶
        df["time_bucket"] = df.index.hour * 100 + (df.index.minute // 15) * 15
        
        stats = df.groupby("time_bucket")["is_win"].mean()
        counts = df.groupby("time_bucket")["is_win"].count()
        
        print(f"{'时段':<10} | {'胜率':<10} | {'样本量':<8} | {'热度'}")
        print("-" * 60)
        
        for t in stats.index:
            rate = stats[t]
            n = counts[t]
            if n < 50: continue # 忽略样本太少的时段
            
            # 格式化时间
            h_str = f"{t//100:02d}:{t%100:02d}"
            
            # 可视化条
            bar_len = int(rate * 25)
            bar = "█" * bar_len
            
            # 标记高光时刻
            highlight = "🔥 BEST" if rate == stats.max() else ""
            if rate > 0.30 and not highlight: highlight = "✨"
            
            print(f"{h_str:<10} | {rate:<10.2%} | {n:<8} | {bar} {highlight}")

if __name__ == "__main__":
    try:
        # 1. 加载数据
        data = load_data_strict()
        # 2. 运行模拟
        run_simulation(data)
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        import traceback
        traceback.print_exc()