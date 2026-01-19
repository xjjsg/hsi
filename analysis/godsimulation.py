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
    "H_STAR_SECONDS": 120,  # 2分钟
    "H_STAR_BARS": 40,  # 40个3秒bar
    # 交易参数
    "INITIAL_CAP": 200000,
    "COST_RATE": 0.0001,  # 万1
    "MIN_PROFIT_THRESHOLD": 0.0000,  # 只要能覆盖成本并哪怕赚0.00001都做
    # 模拟限制
    "COOLDOWN_BARS": 0,  # 理想模型假设并发能力强，或者设为1表示刚平仓才能开
    # 交易合并配置
    "MERGE_GAP_THRESHOLD": 60,  # 合并最大间隔 (秒)
    "MERGE_COST_THRESHOLD": 2e-4,  # 2bps (用于比较是否值得重新开仓)
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
            if "tx_local_time" not in df.columns:
                continue
            dt_utc = pd.to_datetime(
                df["tx_local_time"], unit="ms", utc=True, errors="coerce"
            )
            df["timestamp"] = dt_utc.dt.tz_convert(CONFIG["TIMEZONE"]).dt.tz_localize(
                None
            )
            df["bid1"] = pd.to_numeric(df.get("bp1"), errors="coerce")
            df["ask1"] = pd.to_numeric(df.get("sp1"), errors="coerce")
            df = (
                df.sort_values("timestamp")
                .drop_duplicates("timestamp", keep="last")
                .set_index("timestamp")
                .sort_index()
            )
            df_res = df[["bid1", "ask1"]].resample("3S").last().dropna()
            df_res = df_res[(df_res["bid1"] > 0) & (df_res["ask1"] > 0)]
            t = df_res.index.time
            mask = ((t >= dt_time(9, 30)) & (t <= dt_time(11, 30))) | (
                (t >= dt_time(13, 0)) & (t <= dt_time(14, 57))
            )
            df_list.append(df_res[mask])
        except:
            continue

    if not df_list:
        raise ValueError("无数据")
    return pd.concat(df_list).sort_index().reset_index()


# ==========================================
# 3. 理想模型模拟核心
# ==========================================


# ==========================================
# 3. 理想模型模拟核心 (增强版)
# ==========================================
def run_ideal_simulation(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print(
        f"🤖 理想模型交易模拟 - 交易日志与合并分析 (H* = {CONFIG['H_STAR_SECONDS']}秒)"
    )
    print("=" * 80)

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
            dt = (times[j] - curr_time).astype("timedelta64[s]").astype(int)
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

                hold_time = (exit_time - curr_time).astype("timedelta64[s]").astype(int)

                # 记录详细交易日志
                trade_log.append(
                    {
                        "entry_time": curr_time,
                        "entry_price": curr_ask,
                        "exit_time": exit_time,
                        "exit_price": max_bid,
                        "hold_seconds": hold_time,
                        "quantity": can_buy_shares,
                        "profit_bps": (max_bid - curr_ask) / curr_ask * 10000,  # 毛利
                        "cost_bps": cost_rate * 2 * 10000,
                        "net_profit_bps": net_ret * 10000,
                        "pnl_amount": pnl,
                    }
                )

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
    trades_df.to_csv("outputs/trade_log.csv", index=False)
    print("💾 交易日志已保存至 outputs/trade_log.csv")

    # 执行分析
    analyze_trades(trades_df)


# ==========================================
# 4. 交易合并逻辑 (Chain Merging)
# ==========================================
def merge_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    合并连续交易
    逻辑: 如果 (Buy_Next - Sell_Prev + 2*Cost) > 0，说明"做T"做反了或者空间不够覆盖成本，
    不如直接持有。
    """
    if trades_df.empty:
        return trades_df

    merged_list = []

    # 按入场时间排序
    df = trades_df.sort_values("entry_time").reset_index(drop=True)
    n = len(df)

    # 当前正在累积的交易
    current_trade = df.iloc[0].to_dict()

    merge_gap = CONFIG["MERGE_GAP_THRESHOLD"]
    cost = CONFIG["COST_RATE"]

    for i in range(1, n):
        next_trade = df.iloc[i]

        # 1. 检查时间间隔
        prev_exit_time = current_trade["exit_time"]
        next_entry_time = next_trade["entry_time"]
        gap_sec = (next_entry_time - prev_exit_time).total_seconds()

        # 2. 检查价格条件 (是否值得合并)
        # 如果 Entry_Next > Exit_Prev - 2*Cost
        # 意味着: 重新买回的成本 (Entry_Next + Cost) > 刚才卖出的到手价 (Exit_Prev - Cost)
        # 即: 卖早了/买贵了，不如一直拿着。

        entry_price_next = next_trade["entry_price"]
        exit_price_prev = current_trade["exit_price"]

        should_merge = False
        if gap_sec <= merge_gap:
            # PnL不等式检查
            # 维持持仓的收益 = Exit_Next - Entry_Current - 2*C
            # 拆开做的收益 = (Exit_Prev - Entry_Current - 2*C) + (Exit_Next - Entry_Next - 2*C)
            # 差额 (持有 - 拆开) = Exit_Next - Entry_Current - Exit_Prev + Entry_Current - Exit_Next + Entry_Next + 2*C
            #                 = Entry_Next - Exit_Prev + 2*C
            # 如果 差额 > 0，则持有更好。

            diff = (
                entry_price_next - exit_price_prev + (2 * cost * entry_price_next)
            )  # 近似计算
            # 注意: 严格来说 cost是按成交额算的，这里简化用价格近似

            if diff > 0:
                should_merge = True

        if should_merge:
            # 执行合并
            # 更新退出信息为最新的一笔
            current_trade["exit_time"] = next_trade["exit_time"]
            current_trade["exit_price"] = next_trade["exit_price"]

            # 重新计算收益
            old_entry_price = current_trade["entry_price"]
            new_exit_price = current_trade["exit_price"]

            # 更新持仓时间
            current_trade["hold_seconds"] = (
                current_trade["exit_time"] - current_trade["entry_time"]
            ).total_seconds()

            # 更新 PnL 相关字段
            gross_pnl = (new_exit_price - old_entry_price) / old_entry_price
            net_pnl = gross_pnl - cost * 2

            current_trade["profit_bps"] = gross_pnl * 10000
            current_trade["net_profit_bps"] = net_pnl * 10000

            # 更新金额 (假设 quantity 不变，或者简单累加？)
            # 理想模型每次全仓，所以quantity其实是随资金增长的。这里简化处理：
            # 仅更新比例收益，金额暂不重新模拟 (因为涉及到复利路径改变，如果要精确需要重跑回测循环)
            # 在 analyze 阶段我们主要关注 bps 提升。
            current_trade["pnl_amount"] = 0  # 标记为合并后金额需重算(或忽略)

            # 记录合并次数(可选)
            current_trade["merge_count"] = current_trade.get("merge_count", 0) + 1

        else:
            # 结束上一笔，开始新的一笔
            merged_list.append(current_trade)
            current_trade = next_trade.to_dict()
            current_trade["merge_count"] = 0

    # 最后一笔
    merged_list.append(current_trade)

    return pd.DataFrame(merged_list)


# ==========================================
# 5. 交易分析与合并潜力评估
# ==========================================
def analyze_trades(trades_df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("📊 交易统计与合并分析 (Chain Merging Optimized)")
    print("=" * 80)

    # 1. 基础统计 (原始)
    print("\n[1] 原始策略表现")
    n_orig = len(trades_df)
    avg_pnl_orig = trades_df["net_profit_bps"].mean()
    sum_pnl_orig = trades_df["net_profit_bps"].sum()
    print(f"交易次数: {n_orig}")
    print(f"平均净收益: {avg_pnl_orig:.2f} bps")
    print(f"总净收益: {sum_pnl_orig:.2f} bps")

    # 2. 执行合并
    print("\n[2] 执行交易合并...")
    merged_df = merge_trades(trades_df)

    # 3. 合并后统计
    print("\n[3] 合并策略表现")
    n_merged = len(merged_df)
    avg_pnl_merged = merged_df["net_profit_bps"].mean()
    sum_pnl_merged = merged_df["net_profit_bps"].sum()

    print(
        f"交易次数: {n_merged} (减少 {n_orig - n_merged} 笔, -{(n_orig - n_merged)/n_orig*100:.1f}%)"
    )
    print(f"平均净收益: {avg_pnl_merged:.2f} bps")
    print(f"总净收益: {sum_pnl_merged:.2f} bps")

    # 4. 提升分析
    delta_bps = sum_pnl_merged - sum_pnl_orig
    print(f"\n[4] 优化效果")
    print(f"总收益提升: +{delta_bps:.2f} bps")
    if sum_pnl_orig != 0:
        print(f"提升幅度: +{delta_bps / abs(sum_pnl_orig) * 100:.2f}%")

    # 保存
    merged_df.to_csv("outputs/trade_log_merged.csv", index=False)
    print("\n💾 合并后的交易日志已保存至 outputs/trade_log_merged.csv")


if __name__ == "__main__":
    try:
        data = load_data()
        run_ideal_simulation(data)
    except Exception as e:
        print(e)
        import traceback

        traceback.print_exc()
