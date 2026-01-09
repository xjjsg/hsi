"""
Barrier-hit 事件分析器

研究目标：
在真实交易约束下（只能做多、对手价成交、考虑交易成本），
识别"Barrier-hit 事件"在时间维度上的自然完成尺度，
确定模型理论上仍然可预测的最大时间窗口 H*。
"""

import os
import glob
import warnings
from datetime import time as dt_time
from typing import List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ==========================================
# 1. 配置
# ==========================================
CONFIG = {
    "DATA_DIR": "./data",
    "MAIN_SYMBOL": "sz159920",
    "TIMEZONE": "Asia/Shanghai",
    "RESAMPLE_FREQ": "3S",  # 每个 Bar = 3 秒
    
    # 交易成本
    "COST_RATE": 0.0001,  # 单边万1
    
    # H 候选列表 (秒)
    "H_LIST": [3, 6, 9, 12, 15, 18, 21, 24, 27, 30,  # 3-30秒
               45, 60, 90, 120, 180, 240, 300,        # 45秒-5分
               360, 420, 480, 540, 600,               # 6-10分
               720, 900, 1200, 1800, 3600],           # 12分-1小时
    
    # 最大向前扫描时间 (秒)
    "MAX_SCAN_SECONDS": 3600,  # 1小时
}


# ==========================================
# 2. 数据加载器
# ==========================================
def load_data(data_dir: str = None, symbol: str = None) -> pd.DataFrame:
    """加载并清洗数据"""
    data_dir = data_dir or CONFIG["DATA_DIR"]
    symbol = symbol or CONFIG["MAIN_SYMBOL"]
    
    print(f"🚀 [Loader] 扫描路径: {data_dir}, 标的: {symbol}")
    
    pattern = os.path.join(data_dir, "**", f"{symbol}*.csv")
    files = sorted(glob.glob(pattern, recursive=True))
    
    if not files:
        pattern = os.path.join(".", "**", f"{symbol}*.csv")
        files = sorted(glob.glob(pattern, recursive=True))
    
    print(f"🔎 [Loader] 发现 {len(files)} 个源文件")
    
    df_list = []
    
    for f in files:
        try:
            df = pd.read_csv(f)
            if "tx_local_time" not in df.columns:
                continue
            
            dt_utc = pd.to_datetime(df["tx_local_time"], unit="ms", utc=True, errors="coerce")
            df["timestamp"] = dt_utc.dt.tz_convert(CONFIG["TIMEZONE"]).dt.tz_localize(None)
            
            # 重命名为规范字段
            df["bid1"] = pd.to_numeric(df["bp1"], errors="coerce")
            df["ask1"] = pd.to_numeric(df["sp1"], errors="coerce")
            
            df = df.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="last")
            df = df.set_index("timestamp").sort_index()
            
            df_res = df[["bid1", "ask1"]].resample(CONFIG["RESAMPLE_FREQ"]).last()
            df_res = df_res.dropna()
            df_res = df_res[(df_res["bid1"] > 0) & (df_res["ask1"] > 0)]
            
            t = df_res.index.time
            mask = ((t >= dt_time(9, 30)) & (t <= dt_time(11, 30))) | \
                   ((t >= dt_time(13, 0)) & (t <= dt_time(14, 57)))
            df_res = df_res[mask]
            
            df_list.append(df_res)
        except Exception:
            continue
    
    if not df_list:
        raise ValueError("❌ 未加载到有效数据")
    
    full_df = pd.concat(df_list).sort_index()
    full_df = full_df.reset_index()
    print(f"✅ [Loader] 数据加载完毕: {len(full_df)} 条 Bar")
    return full_df


# ==========================================
# 3. Barrier-hit 核心统计计算
# ==========================================
def compute_barrier_stats(
    df: pd.DataFrame,
    cost_rate: float,
    H_list: List[int],
    max_scan_seconds: int,
    time_col: str = "timestamp",
) -> pd.DataFrame:
    """
    计算 Barrier-hit 事件统计
    
    Barrier-hit 定义:
    是否存在某个未来时间 τ > 0，使得
    bid1_{t+τ} - ask1_t >= θ_t
    其中 θ_t = ask1_t × 2 × cost_rate
    
    Args:
        df: 必须按时间排序，包含 bid1, ask1 列
        cost_rate: 单边交易成本率
        H_list: 候选 H 值列表 (秒)
        max_scan_seconds: 最大向前扫描时间
        time_col: 时间戳列名
    
    Returns:
        每个 H 的统计结果 DataFrame
    """
    print(f"\n⏳ 计算 Barrier-hit 统计 (cost_rate={cost_rate}, max_scan={max_scan_seconds}s)...")
    
    times = df[time_col].values
    bid = df["bid1"].values
    ask = df["ask1"].values
    
    n = len(df)
    
    # 首次命中时间 (秒)，未命中则为 inf
    tau_star = np.full(n, np.inf)
    # 最大不利变动 (MAE)
    mae = np.full(n, np.nan)
    # 命中时的收益
    hit_pnl = np.full(n, np.nan)
    
    print(f"   处理 {n} 个时间点...")
    
    # 进度显示
    progress_step = max(1, n // 20)
    
    for i in range(n):
        if i % progress_step == 0:
            print(f"   进度: {i/n*100:.0f}%", end="\r")
        
        entry = ask[i]
        # 有效止盈阈值: θ_t = ask1_t × 2 × cost_rate
        barrier = entry * (1 + 2 * cost_rate)
        
        worst = 0.0  # 最大不利变动 (从入场价计算)
        
        for j in range(i + 1, n):
            # 计算时间差 (秒)
            dt = (times[j] - times[i]).astype("timedelta64[s]").astype(int)
            
            if dt > max_scan_seconds:
                break
            
            # 当前盈亏 (未扣除成本)
            pnl = bid[j] - entry
            worst = min(worst, pnl)
            
            # 检查是否命中 barrier
            if bid[j] >= barrier:
                tau_star[i] = dt
                mae[i] = worst
                hit_pnl[i] = pnl
                break
        
        # 如果未命中，记录最大不利变动
        if np.isinf(tau_star[i]):
            mae[i] = worst
    
    print(f"   进度: 100%   ")
    
    # =============================================
    # 按 H 统计
    # =============================================
    results = []
    
    for H in H_list:
        # 在 H 时间内命中的 mask
        mask = tau_star <= H
        n_hits = mask.sum()
        
        if n_hits == 0:
            results.append({
                "H_seconds": H,
                "H_bars": H / 3,
                "H_minutes": H / 60,
                "hit_rate": 0.0,
                "mean_tau": np.nan,
                "median_tau": np.nan,
                "std_tau": np.nan,
                "mean_MAE": np.nan,
                "q05_MAE": np.nan,
                "mean_hit_pnl": np.nan,
                "num_samples": 0,
            })
            continue
        
        tau_hits = tau_star[mask]
        mae_hits = mae[mask]
        pnl_hits = hit_pnl[mask]
        
        results.append({
            "H_seconds": H,
            "H_bars": H / 3,
            "H_minutes": H / 60,
            "hit_rate": mask.mean(),
            "mean_tau": np.mean(tau_hits),
            "median_tau": np.median(tau_hits),
            "std_tau": np.std(tau_hits),
            "mean_MAE": np.mean(mae_hits),
            "q05_MAE": np.quantile(mae_hits, 0.05),
            "mean_hit_pnl": np.mean(pnl_hits),
            "num_samples": n_hits,
        })
    
    return pd.DataFrame(results)


# ==========================================
# 4. H* 判定与可视化
# ==========================================
def find_optimal_H(results_df: pd.DataFrame) -> dict:
    """
    找到最优 H*
    
    H* 应满足:
    - hit_rate ∈ [5%, 30%] (可调)
    - τ* 分布仍然集中 (std 不爆炸)
    - MAE 未显著恶化
    - 再增大 H，统计特征发生"质变"
    """
    
    # 筛选合理区间
    viable = results_df[
        (results_df["hit_rate"] >= 0.05) & 
        (results_df["hit_rate"] <= 0.50) &
        (results_df["num_samples"] > 100)
    ].copy()
    
    if viable.empty:
        viable = results_df[results_df["num_samples"] > 100].copy()
    
    if viable.empty:
        return None
    
    # 计算 τ* 的变异系数 (CV = std/mean)
    viable["tau_cv"] = viable["std_tau"] / viable["mean_tau"]
    
    # 计算各指标的相对变化率
    viable["hit_rate_change"] = viable["hit_rate"].diff() / viable["hit_rate"].shift(1)
    viable["tau_cv_change"] = viable["tau_cv"].diff() / viable["tau_cv"].shift(1)
    viable["mae_change"] = viable["mean_MAE"].diff().abs() / viable["mean_MAE"].shift(1).abs()
    
    # 综合评分: 寻找"质变"拐点前的最后一个稳定点
    # 质变信号: tau_cv 突然增大, mae 突然恶化
    
    # 简化判断: 找 hit_rate 在合理区间内，且 tau_cv 最小的点
    best_idx = viable["tau_cv"].idxmin()
    best = viable.loc[best_idx]
    
    return {
        "H_star_seconds": best["H_seconds"],
        "H_star_bars": best["H_bars"],
        "H_star_minutes": best["H_minutes"],
        "hit_rate": best["hit_rate"],
        "mean_tau": best["mean_tau"],
        "std_tau": best["std_tau"],
        "tau_cv": best["tau_cv"],
        "mean_MAE": best["mean_MAE"],
    }


def print_results(results_df: pd.DataFrame, optimal_H: dict):
    """打印分析结果"""
    
    print("\n" + "=" * 120)
    print("📊 Barrier-hit 事件统计 (按最大等待时间 H)")
    print("=" * 120)
    print(f"{'H(秒)':<8} | {'H(分)':<8} | {'命中率':<10} | {'E[τ*]秒':<10} | {'std[τ*]':<10} | "
          f"{'CV':<8} | {'E[MAE]':<12} | {'Q5%MAE':<12} | {'样本数':<10}")
    print("-" * 120)
    
    for _, row in results_df.iterrows():
        if row["num_samples"] == 0:
            continue
        
        cv = row["std_tau"] / row["mean_tau"] if row["mean_tau"] > 0 else np.nan
        
        highlight = ""
        if optimal_H and row["H_seconds"] == optimal_H["H_star_seconds"]:
            highlight = " ⭐ H*"
        
        print(f"{int(row['H_seconds']):<8} | {row['H_minutes']:<8.2f} | {row['hit_rate']:<10.2%} | "
              f"{row['mean_tau']:<10.1f} | {row['std_tau']:<10.1f} | {cv:<8.3f} | "
              f"{row['mean_MAE']*10000:<12.2f}bp | {row['q05_MAE']*10000:<12.2f}bp | "
              f"{int(row['num_samples']):<10}{highlight}")
    
    # 结构性崩坏分析
    print("\n" + "=" * 120)
    print("📈 结构性崩坏信号分析")
    print("=" * 120)
    
    if len(results_df) > 1:
        # 计算变化率
        results_df = results_df.copy()
        results_df["tau_cv"] = results_df["std_tau"] / results_df["mean_tau"]
        
        print(f"\n{'H(秒)':<10} | {'命中率趋势':<30} | {'τ*变异系数趋势':<30}")
        print("-" * 80)
        
        prev_hit_rate = None
        prev_cv = None
        
        for _, row in results_df.iterrows():
            if row["num_samples"] == 0:
                continue
            
            hit_bar_len = int(row["hit_rate"] * 30)
            hit_bar = "█" * hit_bar_len
            
            cv = row["std_tau"] / row["mean_tau"] if row["mean_tau"] > 0 else 0
            cv_bar_len = int(min(cv, 1.0) * 20)
            cv_bar = "▓" * cv_bar_len
            
            change_signal = ""
            if prev_hit_rate and row["hit_rate"] > 0.30 and prev_hit_rate < 0.30:
                change_signal = " ⚠️ 趋向随机"
            if prev_cv and cv > 0.5 and prev_cv < 0.5:
                change_signal = " ⚠️ τ*发散"
            
            prev_hit_rate = row["hit_rate"]
            prev_cv = cv
            
            print(f"{int(row['H_seconds']):<10} | {hit_bar:<30} | {cv_bar:<30}{change_signal}")
    
    # 最优 H* 建议
    print("\n" + "=" * 120)
    print("🎯 最优预测时间窗口 H* 建议")
    print("=" * 120)
    
    if optimal_H:
        print(f"\n推荐配置:")
        print(f"   PREDICT_HORIZON = {int(optimal_H['H_star_bars'])} bars")
        print(f"   ≈ {optimal_H['H_star_seconds']:.0f} 秒 = {optimal_H['H_star_minutes']:.2f} 分钟")
        print(f"\n统计特征:")
        print(f"   命中率: {optimal_H['hit_rate']:.2%}")
        print(f"   平均首次命中时间: {optimal_H['mean_tau']:.1f} 秒")
        print(f"   τ* 变异系数 (CV): {optimal_H['tau_cv']:.3f}")
        print(f"   平均最大不利变动: {optimal_H['mean_MAE']*10000:.2f} bps")
    else:
        print("\n⚠️ 未找到合适的 H*，请调整参数或检查数据")


# ==========================================
# 5. 主程序
# ==========================================
if __name__ == "__main__":
    try:
        # 1. 加载数据
        data = load_data()
        
        # 2. 计算 Barrier-hit 统计
        results = compute_barrier_stats(
            df=data,
            cost_rate=CONFIG["COST_RATE"],
            H_list=CONFIG["H_LIST"],
            max_scan_seconds=CONFIG["MAX_SCAN_SECONDS"],
        )
        
        # 3. 找到最优 H*
        optimal_H = find_optimal_H(results)
        
        # 4. 打印结果
        print_results(results, optimal_H)
        
        # 5. 保存结果
        results.to_csv("barrier_hit_stats.csv", index=False)
        print(f"\n💾 结果已保存到 barrier_hit_stats.csv")
        
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        import traceback
        traceback.print_exc()
