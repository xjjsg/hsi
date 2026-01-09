"""
理想模型交易模拟器 (Ideal Model Simulator)

场景假设:
- 模型拥有上帝视角 (Oracle)，能完美预测未来 H* (2分钟) 内的走势。
- 交易约束: Taker-Taker 模式 (Ask买, Bid卖), 双边成本。

模拟目标:
1. 这种"完美模型"在历史数据上到底能赚多少钱？(理论上限)
2. 它的交易频率是多少？(是一直在做，还是偶尔出手？)
3. 它的平均持仓时间是多少？(虽然预测2分钟，但实际多久止盈？)
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
from datetime import time as dt_time

warnings.filterwarnings("ignore")

# ==========================================
# 1. 核心配置
# ==========================================
CONFIG = {
    "DATA_DIR": "./data",
    "SYMBOL": "sz159920",
    "TIMEZONE": "Asia/Shanghai",
    
    # 理想预测窗口 H* (基于之前的分析)
    "H_STAR_SECONDS": 120,    # 2分钟
    "H_STAR_BARS": 40,        # 40个3秒bar
    
    # 交易参数
    "INITIAL_CAP": 200000,
    "COST_RATE": 0.0001,      # 万1
    "MIN_PROFIT_THRESHOLD": 0.0000, # 只要能覆盖成本并哪怕赚0.00001都做
    
    # 模拟限制
    "COOLDOWN_BARS": 0,       # 理想模型假设并发能力强，或者设为1表示刚平仓才能开
}

# ==========================================
# 2. 数据加载 (复用)
# ==========================================
def load_data(data_dir: str = None, symbol: str = None) -> pd.DataFrame:
    print(f"🚀 [Loader] 加载数据...")
    data_dir = data_dir or CONFIG["DATA_DIR"]
    symbol = symbol or CONFIG["SYMBOL"]
    pattern = os.path.join(data_dir, "**", f"{symbol}*.csv")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        pattern = os.path.join(".", "**", f"{symbol}*.csv")
        files = sorted(glob.glob(pattern, recursive=True))
    
    df_list = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if "tx_local_time" not in df.columns: continue
            dt_utc = pd.to_datetime(df["tx_local_time"], unit="ms", utc=True, errors="coerce")
            df["timestamp"] = dt_utc.dt.tz_convert(CONFIG["TIMEZONE"]).dt.tz_localize(None)
            df["bid1"] = pd.to_numeric(df.get("bp1"), errors="coerce")
            df["ask1"] = pd.to_numeric(df.get("sp1"), errors="coerce")
            df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").set_index("timestamp").sort_index()
            df_res = df[["bid1", "ask1"]].resample("3S").last().dropna()
            df_res = df_res[(df_res["bid1"] > 0) & (df_res["ask1"] > 0)]
            t = df_res.index.time
            mask = ((t >= dt_time(9, 30)) & (t <= dt_time(11, 30))) | \
                   ((t >= dt_time(13, 0)) & (t <= dt_time(14, 57)))
            df_list.append(df_res[mask])
        except: continue
    
    if not df_list: raise ValueError("无数据")
    return pd.concat(df_list).sort_index().reset_index()

# ==========================================
# 3. 理想模型模拟核心
# ==========================================

# ==========================================
# 3. 理想模型模拟核心 (增强版)
# ==========================================
def run_ideal_simulation(df: pd.DataFrame):
    print("\n" + "="*80)
    print(f"🤖 理想模型交易模拟 - 交易日志与合并分析 (H* = {CONFIG['H_STAR_SECONDS']}秒)")
    print("="*80)
    
    n = len(df)
    times = df["timestamp"].values
    bid = df["bid1"].values
    ask = df["ask1"].values
    
    # 模拟状态
    cash = CONFIG["INITIAL_CAP"]
    trade_log = []
    
    # 向前扫描窗口 (bars)
    horizon = CONFIG["H_STAR_BARS"]
    cost_rate = CONFIG["COST_RATE"]
    
    print("⏳ 正在回放并详细记录...")
    
    i = 0
    while i < n - 1:
        # 当前只能看到 Ask
        curr_ask = ask[i]
        curr_time = times[i]
        
        # 快速定位潜在的结束索引 (优化性能)
        future_bids = []
        future_times = []
        
        # 往后搜索直到超过时间窗口
        j = i + 1
        while j < n:
            dt = (times[j] - curr_time).astype('timedelta64[s]').astype(int)
            if dt > CONFIG["H_STAR_SECONDS"]:
                break
            if bid[j] > 0:
                future_bids.append(bid[j])
                future_times.append(times[j])
            j += 1
            
        if not future_bids:
            i += 1
            continue
            
        # 寻找这一波里的最高价 (最佳卖点)
        future_bids = np.array(future_bids)
        max_bid = np.max(future_bids)
        max_idx_rel = np.argmax(future_bids)
        exit_time = future_times[max_idx_rel]
        
        # 计算潜在收益
        gross_ret = (max_bid - curr_ask) / curr_ask
        net_ret = gross_ret - cost_rate * 2
        
        # 决策
        if net_ret > CONFIG["MIN_PROFIT_THRESHOLD"]:
            # 全仓梭哈 (模拟)
            can_buy_shares = (cash / (curr_ask * (1 + cost_rate))) // 100 * 100
            
            if can_buy_shares > 0:
                entry_cost = can_buy_shares * curr_ask * (1 + cost_rate)
                cash -= entry_cost
                
                exit_revenue = can_buy_shares * max_bid * (1 - cost_rate)
                cash += exit_revenue
                pnl = exit_revenue - entry_cost
                
                hold_time = (exit_time - curr_time).astype('timedelta64[s]').astype(int)
                
                # 记录详细交易日志
                trade_log.append({
                    "entry_time": curr_time,
                    "entry_price": curr_ask,
                    "exit_time": exit_time,
                    "exit_price": max_bid,
                    "hold_seconds": hold_time,
                    "quantity": can_buy_shares,
                    "profit_bps": (max_bid - curr_ask) / curr_ask * 10000, # 毛利
                    "cost_bps": cost_rate * 2 * 10000,
                    "net_profit_bps": net_ret * 10000,
                    "pnl_amount": pnl
                })
                
                # 跳过持仓期
                k = i + 1
                while k < n and times[k] <= exit_time:
                    k += 1
                i = k
                continue
        
        i += 1
        
    trades_df = pd.DataFrame(trade_log)
    if trades_df.empty:
        print("🤷 无交易记录")
        return

    # 保存原始交易记录
    trades_df.to_csv("trade_log.csv", index=False)
    print("💾 交易日志已保存至 trade_log.csv")

    # 执行分析
    analyze_trades(trades_df)


# ==========================================
# 4. 交易分析与合并潜力评估
# ==========================================
def analyze_trades(trades_df: pd.DataFrame):
    print("\n" + "="*80)
    print("📊 交易统计与合并潜力分析")
    print("="*80)

    # 1. 基础统计
    print("\n[1] 基础统计")
    print(f"总交易次数: {len(trades_df)}")
    print(f"平均净收益: {trades_df['net_profit_bps'].mean():.2f} bps")
    print(f"平均持仓时间: {trades_df['hold_seconds'].mean():.1f} 秒")
    print("\n盈利分布:")
    print(trades_df['net_profit_bps'].describe().to_string())

    # 2. 合并潜力分析
    # 逻辑: 如果前一笔交易的 exit_time 与 后一笔交易的 entry_time 很近 (例如 < 10秒)
    # 并且前一笔卖价 ~ 后一笔买价 (考虑点差)，则可能直接持有不卖，省去双边费用。
    
    print("\n[2] 合并潜力分析 (Chain Merging)")
    
    trades_df = trades_df.sort_values("entry_time").reset_index(drop=True)
    trades_df["prev_exit_time"] = trades_df["exit_time"].shift(1)
    trades_df["prev_exit_price"] = trades_df["exit_price"].shift(1)
    
    # 计算时间间隔 (秒)
    trades_df["gap_seconds"] = (trades_df["entry_time"] - trades_df["prev_exit_time"]).dt.total_seconds()
    
    # 假设如果 gap < 30秒，且 再次买入价 >= 前次卖出价 * (1 - cost_rate*2) 
    # (即: 再次买入成本 高于或接近 卖出到手价，说明卖飞了或者白交手续费了)
    # 这里我们简化逻辑: 只要时间足够短，就视为"连续机会"，统计如果合并能省多少钱
    
    # 可合并条件: 间隔小于某阈值 (比如 60秒，即 20 bars)
    MERGE_GAP_THRESHOLD = 60 
    
    potential_merges = trades_df[trades_df["gap_seconds"] < MERGE_GAP_THRESHOLD]
    
    num_merges = len(potential_merges)
    pct_merges = num_merges / len(trades_df)
    
    print(f"\n间隔 < {MERGE_GAP_THRESHOLD}秒 的连续交易: {num_merges} 笔 ({pct_merges:.1%})")
    
    if num_merges > 0:
        # 估算节省成本: 每合并且一次，省去一次卖出和一次买入的费用 (约 2 * cost_rate)
        # 简化计算: 每次合并节省 2 bps
        saved_costs_bps = 2.0 
        total_saved_bps = num_merges * saved_costs_bps
        
        print(f"潜在节省成本 (每笔 {saved_costs_bps} bps): {total_saved_bps:.1f} bps (总计)")
        print(f"这相当于将总收益提升了: {total_saved_bps / trades_df['net_profit_bps'].sum() * 100:.1%}")
        
        # 详细展示前 5 个合并案例
        print("\n前 5 个可合并案例示例:")
        print(potential_merges[["entry_time", "gap_seconds", "prev_exit_price", "entry_price"]].head(5).to_string())
    else:
        print("无明显的连续交易可合并。")

if __name__ == "__main__":
    try:
        data = load_data()
        run_ideal_simulation(data)
    except Exception as e:
        print(e)
        import traceback
        traceback.print_exc()
