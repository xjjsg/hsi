"""
HSI HFT V3 - Regime分析脚本（两层状态设计）

目标：
1. 验证Micro Regime（illiquid/high_vol/normal）检测的稳定性
2. 验证Action Regime（trending/mean_reverting/neutral）的价格动力学证据
3. 计算日内分位数基线，替代硬阈值
4. 测试抖动控制（最小驻留期）的效果
5. 分体制统计白盒/黑盒贡献，校准alpha_by_regime

基于用户设计方案：
- 两层状态比单层五分类更清晰
- 规则驱动为主，学习型仅用于离线校准
- 可解释、可复现、可回测
"""

import sys
import os
import numpy as np
import pandas as pd
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from datetime import datetime, time as dt_time

# 添加路径
sys.path.append(os.getcwd())

from hsi_hft_v3.data_layer import V5DataLoader
from hsi_hft_v3.features.whitebox import WhiteBoxFeatureFactory


# ==========================================
# 1. 价格动力学指标计算
# ==========================================


class PriceDynamicsIndicators:
    """
    计算趋势和均值回复的证据指标

    趋势证据：
    - drift_to_vol_ratio: 绝对漂移与实现波动比值
    - directional_consistency: 方向一致性（正收益占比）

    均值回复证据：
    - lag1_autocorr: 收益的lag-1自相关
    - mean_reversion_strength: 回归强度（偏离度与回归速度）
    """

    def __init__(self, window=20):
        self.window = window
        self.returns_buffer = deque(maxlen=window)
        self.mid_buffer = deque(maxlen=window)

    def update(self, mid: float, prev_mid: float):
        """更新缓冲区"""
        if mid > 0 and prev_mid > 0:
            ret = np.log(mid / prev_mid)
        else:
            ret = 0.0

        self.returns_buffer.append(ret)
        self.mid_buffer.append(mid)

    def get_drift_to_vol_ratio(self) -> float:
        """
        漂移-波动比

        单边行情会显著抬高该比值
        震荡行情比值接近0
        """
        if len(self.returns_buffer) < 10:
            return 0.0

        rets = np.array(list(self.returns_buffer))

        # 绝对漂移（累积收益的绝对值）
        drift = abs(rets.sum())

        # 实现波动
        vol = rets.std() * np.sqrt(len(rets))

        if vol < 1e-9:
            return 0.0

        return drift / vol

    def get_directional_consistency(self) -> float:
        """
        方向一致性

        返回正收益占比 - 0.5（中心化）
        单边上涨：接近+0.5
        单边下跌：接近-0.5
        震荡：接近0
        """
        if len(self.returns_buffer) < 10:
            return 0.0

        rets = np.array(list(self.returns_buffer))
        pos_ratio = (rets > 0).sum() / len(rets)

        # 中心化：[-0.5, 0.5]
        return pos_ratio - 0.5

    def get_lag1_autocorr(self) -> float:
        """
        Lag-1自相关

        显著为负 → 均值回复证据
        显著为正 → 趋势持续证据
        """
        if len(self.returns_buffer) < 10:
            return 0.0

        rets = np.array(list(self.returns_buffer))

        if len(rets) < 2:
            return 0.0

        # 计算lag-1相关性
        corr = np.corrcoef(rets[:-1], rets[1:])[0, 1]

        return corr if not np.isnan(corr) else 0.0

    def get_mean_reversion_strength(self) -> float:
        """
        均值回复强度

        计算mid相对短均线的偏离与回归速度
        """
        if len(self.mid_buffer) < self.window:
            return 0.0

        mids = np.array(list(self.mid_buffer))
        ma = mids[:-5].mean() if len(mids) > 5 else mids.mean()

        # 偏离度
        deviation = (mids[-1] - ma) / (ma + 1e-9)

        # 回归速度（最近5个bar的趋势）
        if len(mids) >= 5:
            recent_trend = (mids[-1] - mids[-5]) / (mids[-5] + 1e-9)

            # 如果偏离向上但趋势向下（或反之），说明在回归
            reversion_signal = -deviation * recent_trend
            return reversion_signal

        return 0.0


# ==========================================
# 2. 日内分位数基线计算
# ==========================================


class IntradayQuantileBaseline:
    """
    日内分位数基线

    按5分钟桶维护VPIN、spread、depth的历史分位数
    避免硬阈值被日内季节性打穿
    """

    def __init__(self, bucket_minutes=5):
        self.bucket_minutes = bucket_minutes

        # {bucket_id: {'vpin': [], 'spread': [], 'depth': []}}
        self.historical_data = defaultdict(
            lambda: {"vpin": [], "spread": [], "depth": []}
        )

        # 计算好的分位数表
        self.quantile_table = {}

    def get_bucket_id(self, timestamp_ms: int) -> int:
        """获取时间桶ID"""
        dt = pd.Timestamp(timestamp_ms, unit="ms", tz="Asia/Shanghai")

        # 转为分钟数（从开盘算起，假设9:30开盘）
        minutes_since_open = (dt.hour - 9) * 60 + (dt.minute - 30)

        # 桶ID
        bucket_id = minutes_since_open // self.bucket_minutes

        return bucket_id

    def add_observation(
        self, timestamp_ms: int, vpin: float, spread: float, depth: int
    ):
        """添加观测值"""
        bucket_id = self.get_bucket_id(timestamp_ms)

        self.historical_data[bucket_id]["vpin"].append(vpin)
        self.historical_data[bucket_id]["spread"].append(spread)
        self.historical_data[bucket_id]["depth"].append(depth)

    def compute_quantiles(self, quantiles=[0.5, 0.9, 0.95]):
        """计算所有桶的分位数"""
        for bucket_id, data in self.historical_data.items():
            self.quantile_table[bucket_id] = {}

            for metric in ["vpin", "spread", "depth"]:
                if len(data[metric]) > 10:
                    self.quantile_table[bucket_id][metric] = {
                        f"p{int(q*100)}": np.percentile(data[metric], q * 100)
                        for q in quantiles
                    }
                else:
                    # 数据不足，用默认值
                    self.quantile_table[bucket_id][metric] = {
                        "p50": 0,
                        "p90": 0,
                        "p95": 0,
                    }

    def get_threshold(self, timestamp_ms: int, metric: str, percentile: str) -> float:
        """获取动态阈值"""
        bucket_id = self.get_bucket_id(timestamp_ms)

        if bucket_id in self.quantile_table:
            return self.quantile_table[bucket_id].get(metric, {}).get(percentile, 0)

        return 0.0


# ==========================================
# 3. 两层Regime检测器
# ==========================================


class TwoTierRegimeDetector:
    """
    两层状态检测器

    Layer 1 - Micro Regime (优先级high→low):
      1. illiquid (depth过低 OR spread过高)
      2. high_volatility (vpin过高 OR spread拉宽 + 实现波动抬升)
      3. normal

    Layer 2 - Action Regime (仅在micro≠illiquid时判断):
      1. trending (drift-vol ratio高 + 方向一致性强)
      2. mean_reverting (lag1自相关显著负 + 回归强度高)
      3. neutral

    输出: micro:action (例如 "high_vol:trending")
    """

    def __init__(self, baseline: IntradayQuantileBaseline, min_residence=10):
        self.baseline = baseline
        self.min_residence = min_residence  # 最小驻留期（bar数）

        # 状态历史
        self.current_micro = "normal"
        self.current_action = "neutral"
        self.residence_counter = 0
        self.dynamics = PriceDynamicsIndicators(window=20)

        # 阈值配置（进入/退出分离）
        # 🔧 调整后的阈值（基于HSI ETF流动性特征）
        self.thresholds = {
            "illiquid": {
                # 🔧 从0.5/0.7调整为0.2/0.4（更宽松，减少误判）
                "depth_enter": ("p50", 0.2),  # depth低于p50的20%
                "depth_exit": ("p50", 0.4),  # 恢复到p50的40%
                # 🔧 从p90调整为p95（只在极端价差时触发）
                "spread_enter": ("p95", 1.0),  # 超过p95
                "spread_exit": ("p95", 0.85),  # 低于p95的85%
            },
            "high_vol": {
                "vpin_enter": ("p90", 1.0),
                "vpin_exit": ("p90", 0.85),
                "spread_enter": ("p95", 1.0),  # 🔧 从p90改为p95
                "spread_exit": ("p95", 0.85),
            },
            "trending": {
                # 🔧 从1.5降低到0.8（更容易检测到趋势）
                "drift_vol_ratio": 0.8,
                "directional_consistency": 0.25,  # 🔧 从0.3降到0.25
            },
            "mean_reverting": {
                "autocorr_threshold": -0.25,  # 🔧 从-0.3放宽到-0.25
                "reversion_strength": 0.4,  # 🔧 从0.5降到0.4
            },
        }

    def detect(
        self, timestamp_ms: int, white_risk: Dict, mid: float, prev_mid: float
    ) -> Tuple[str, str, float]:
        """
        检测体制

        Returns:
            (micro_regime, action_regime, confidence)
        """
        # 更新价格动力学
        self.dynamics.update(mid, prev_mid)

        # 提取白盒指标
        vpin = white_risk.get("vpin_z", 0)
        spread = white_risk.get("spread_bps", 0)
        depth = white_risk.get("depth", 10000)

        # Layer 1: Micro Regime检测
        new_micro, micro_conf = self._detect_micro_regime(
            timestamp_ms, vpin, spread, depth
        )

        # 最小驻留期检查
        if new_micro != self.current_micro:
            if self.residence_counter >= self.min_residence:
                # 允许切换
                self.current_micro = new_micro
                self.residence_counter = 0
            # else: 保持当前状态，不切换

        self.residence_counter += 1

        # Layer 2: Action Regime检测（仅在非illiquid时）
        if self.current_micro == "illiquid":
            new_action = "neutral"  # illiquid时不交易，action无意义
            action_conf = 1.0
        else:
            new_action, action_conf = self._detect_action_regime()

        self.current_action = new_action

        # 综合置信度
        overall_conf = (micro_conf + action_conf) / 2

        return self.current_micro, self.current_action, overall_conf

    def _detect_micro_regime(
        self, timestamp_ms: int, vpin: float, spread: float, depth: int
    ) -> Tuple[str, float]:
        """检测Micro Regime"""

        # 获取动态阈值
        depth_p50 = self.baseline.get_threshold(timestamp_ms, "depth", "p50")
        spread_p90 = self.baseline.get_threshold(timestamp_ms, "spread", "p90")
        vpin_p90 = self.baseline.get_threshold(timestamp_ms, "vpin", "p90")

        # 优先级1: illiquid
        if self.current_micro == "illiquid":
            # 退出阈值（更宽松）
            depth_threshold = depth_p50 * self.thresholds["illiquid"]["depth_exit"][1]
            spread_threshold = (
                spread_p90 * self.thresholds["illiquid"]["spread_exit"][1]
            )

            if depth > depth_threshold and spread < spread_threshold:
                # 恢复正常
                pass
            else:
                return "illiquid", 1.0
        else:
            # 进入阈值（更严格）
            depth_threshold = depth_p50 * self.thresholds["illiquid"]["depth_enter"][1]
            spread_threshold = (
                spread_p90 * self.thresholds["illiquid"]["spread_enter"][1]
            )

            if depth < depth_threshold or spread > spread_threshold:
                return "illiquid", 0.9

        # 优先级2: high_volatility
        vpin_threshold = vpin_p90 * self.thresholds["high_vol"]["vpin_enter"][1]

        if abs(vpin) > vpin_threshold or spread > spread_p90:
            return "high_volatility", 0.8

        # 优先级3: normal
        return "normal", 0.7

    def _detect_action_regime(self) -> Tuple[str, float]:
        """检测Action Regime"""

        # 计算价格动力学指标
        drift_vol = self.dynamics.get_drift_to_vol_ratio()
        dir_cons = self.dynamics.get_directional_consistency()
        autocorr = self.dynamics.get_lag1_autocorr()
        mr_strength = self.dynamics.get_mean_reversion_strength()

        # 趋势证据
        is_trending = (
            drift_vol > self.thresholds["trending"]["drift_vol_ratio"]
            and abs(dir_cons) > self.thresholds["trending"]["directional_consistency"]
        )

        # 均值回复证据
        is_mean_reverting = (
            autocorr < self.thresholds["mean_reverting"]["autocorr_threshold"]
            or mr_strength > self.thresholds["mean_reverting"]["reversion_strength"]
        )

        if is_trending:
            conf = min(drift_vol / 2.0, 1.0)  # 置信度
            return "trending", conf
        elif is_mean_reverting:
            conf = min(abs(autocorr), 1.0)
            return "mean_reverting", conf
        else:
            return "neutral", 0.5


# ==========================================
# 4. 主分析流程
# ==========================================


def analyze_regime_feasibility(data_dir="./data", start_date=None, end_date=None):
    """
    Regime可行性分析

    流程：
    1. 加载历史数据
    2. 计算白盒指标
    3. 建立日内分位数基线
    4. 运行两层Regime检测
    5. 统计切换频率、驻留时间、分体制指标
    """

    print("=" * 60)
    print("HSI Regime分析 - 两层状态设计可行性验证")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1] 加载数据...")
    loader = V5DataLoader(data_dir)
    data_dict = loader.load_date_range(start_date=start_date, end_date=end_date)

    if not data_dict:
        print("❌ 无数据")
        return

    print(f"✅ 加载 {len(data_dict)} 天数据")

    # 2. 第一遍扫描：建立分位数基线
    print("\n[2] 建立日内分位数基线...")
    baseline = IntradayQuantileBaseline(bucket_minutes=5)
    wb_factory = WhiteBoxFeatureFactory()

    total_bars = 0
    for date, samples in data_dict.items():
        for s in samples:
            wb_out = wb_factory.compute(s)

            vpin = wb_out["white_derived"].get("tgt_VPIN_100_z_100", 0)
            spread = wb_out["white_target_raw"].get("tgt_spread_bps", 0)
            depth = (
                s.target.bids[0][1] + s.target.asks[0][1]
                if (s.target.bids and s.target.asks)
                else 0
            )

            baseline.add_observation(s.ts_ms, abs(vpin), spread, depth)
            total_bars += 1

    baseline.compute_quantiles()
    print(
        f"✅ 处理 {total_bars} bars，建立 {len(baseline.quantile_table)} 个时间桶基线"
    )

    # 3. 第二遍扫描：Regime检测
    print("\n[3] 运行两层Regime检测...")
    detector = TwoTierRegimeDetector(baseline, min_residence=10)

    regime_history = []
    switch_log = []
    prev_regime = None
    prev_mid = 0

    for date, samples in data_dict.items():
        for s in samples:
            wb_out = wb_factory.compute(s)

            # 准备white_risk
            white_risk = {
                "vpin_z": wb_out["white_derived"].get("tgt_VPIN_100_z_100", 0),
                "spread_bps": wb_out["white_target_raw"].get("tgt_spread_bps", 0),
                "depth": (
                    s.target.bids[0][1] + s.target.asks[0][1]
                    if (s.target.bids and s.target.asks)
                    else 0
                ),
            }

            mid = s.target.mid

            # 检测
            micro, action, conf = detector.detect(s.ts_ms, white_risk, mid, prev_mid)

            current_regime = f"{micro}:{action}"

            regime_history.append(
                {
                    "timestamp": s.ts_ms,
                    "date": date,
                    "micro": micro,
                    "action": action,
                    "regime": current_regime,
                    "confidence": conf,
                    "vpin": white_risk["vpin_z"],
                    "spread": white_risk["spread_bps"],
                    "depth": white_risk["depth"],
                }
            )

            # 记录切换
            if prev_regime and prev_regime != current_regime:
                switch_log.append(
                    {"timestamp": s.ts_ms, "from": prev_regime, "to": current_regime}
                )

            prev_regime = current_regime
            prev_mid = mid

    print(f"✅ 检测完成，共 {len(regime_history)} bars")

    # 4. 统计分析
    print("\n[4] 统计分析")
    print("-" * 60)

    df_regime = pd.DataFrame(regime_history)

    # 4.1 Regime分布
    print("\n【Regime分布】")
    regime_dist = df_regime["regime"].value_counts()
    print(regime_dist)
    print(f"\n占比：")
    print((regime_dist / len(df_regime) * 100).round(2))

    # 4.2 切换频率
    print(f"\n【切换频率】")
    print(f"总切换次数: {len(switch_log)}")
    print(f"平均切换间隔: {len(df_regime) / (len(switch_log)+1):.1f} bars")
    print(f"切换频率: {len(switch_log) / (len(df_regime)/1200):.2f} 次/小时")

    # 4.3 驻留时间
    print(f"\n【驻留时间统计】")
    residence_times = []
    current_regime = None
    residence_start = 0

    for i, row in df_regime.iterrows():
        if row["regime"] != current_regime:
            if current_regime:
                residence_times.append(
                    {"regime": current_regime, "duration": i - residence_start}
                )
            current_regime = row["regime"]
            residence_start = i

    df_residence = pd.DataFrame(residence_times)
    if len(df_residence) > 0:
        print(df_residence.groupby("regime")["duration"].describe())

    # 4.5 保存结果
    output_dir = "./analysis/regime_analysis"
    os.makedirs(output_dir, exist_ok=True)

    df_regime.to_csv(f"{output_dir}/regime_history.csv", index=False)
    pd.DataFrame(switch_log).to_csv(f"{output_dir}/regime_switches.csv", index=False)

    print(f"\n✅ 结果保存到 {output_dir}/")

    return df_regime, switch_log, baseline


if __name__ == "__main__":
    # 运行分析（加载所有可用数据）
    df_regime, switches, baseline = analyze_regime_feasibility(
        data_dir="./data",
        start_date=None,  # 从最早的数据开始
        end_date=None,  # 到最新的数据（移除限制）
    )

    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)
    print("\n下一步建议：")
    print("1. 检查 analysis/regime_analysis/regime_history.csv 查看时序")
    print("2. 调整min_residence和阈值，降低切换频率")
    print("3. 分体制统计白盒/黑盒表现，校准alpha")
    print("4. 将RegimeDetector集成到实时系统")
