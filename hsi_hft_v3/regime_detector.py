"""
RegimeDetector v1.1 - 工程化两层状态系统

核心改进：
1. 字段口径统一 + 健康度闸门
2. 分位数评分制（避免OR进入AND退出锁死）
3. Action层两阶段gating（确保可交易驻留长度）
4. Micro/Action分别的min_residence
5. 连续置信度驱动平滑

基于用户诊断方案v1.1
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from collections import deque, defaultdict


# ============================================
# 1. 字段口径统一与健康度闸门
# ============================================


class FeatureHealthMonitor:
    """
    特征健康度监控

    检查项：
    1. 非零比例（>50%）
    2. 标准差（避免常数）
    3. 极值分位数跨度（p95-p05 > threshold）
    """

    def __init__(self, window=100):
        self.window = window
        self.history = defaultdict(lambda: deque(maxlen=window))

    def update(self, feature_name: str, value: float):
        """更新特征历史"""
        self.history[feature_name].append(value)

    def is_healthy(self, feature_name: str) -> Tuple[bool, str]:
        """
        判断特征是否健康

        Returns:
            (is_healthy, reason)
        """
        if feature_name not in self.history:
            return False, "no_data"

        values = list(self.history[feature_name])
        if len(values) < 10:
            return False, "insufficient_samples"

        # 检查1：非零比例
        non_zero_ratio = sum(1 for v in values if abs(v) > 1e-9) / len(values)
        if non_zero_ratio < 0.05:  # Relaxed for ETF from 0.5 -> 0.05
            return False, f"low_non_zero_ratio_{non_zero_ratio:.2f}"

        # 检查2：标准差
        std = np.std(values)
        if std < 1e-6:
            return False, f"constant_std_{std:.2e}"

        # 检查3：极值跨度
        p95 = np.percentile(values, 95)
        p05 = np.percentile(values, 5)
        span = p95 - p05
        if span < 1e-6:
            return False, f"low_span_{span:.2e}"

        return True, "ok"


class CanonicalFeatureMapper:
    """
    字段口径统一映射器

    所有别名映射到canonical key
    """

    CANONICAL_KEYS = {
        "vpin": ["tgt_VPIN_100", "VPIN_100", "vpin_z", "VPIN"],
        "spread_bps": ["tgt_spread_bps", "spread_bps", "spread"],
        "depth": ["depth", "total_depth"],
    }

    @classmethod
    def get_canonical_value(cls, data: Dict, canonical_key: str) -> Optional[float]:
        """
        从data中获取canonical key对应的值

        尝试所有可能的别名，返回第一个找到的
        """
        aliases = cls.CANONICAL_KEYS.get(canonical_key, [canonical_key])

        for alias in aliases:
            if alias in data:
                val = data[alias]
                if val is not None and not np.isnan(val):
                    return float(val)

        return None


# ============================================
# 2. 增强的日内分位数基线
# ============================================


class IntradayQuantileBaseline:
    """
    日内分位数基线 v1.1

    改进：
    1. 更多分位数（p05, p10, p20, p50, p80, p90, p95, p99）
    2. Session分离（早盘/午盘）
    3. Out-of-session标记
    """

    # 交易时段定义（港股时间）
    SESSIONS = {
        "morning": ((9, 30), (12, 0)),  # 早盘
        "afternoon": ((13, 0), (16, 0)),  # 午盘
    }

    def __init__(self, bucket_minutes=5):
        self.bucket_minutes = bucket_minutes

        # {session: {bucket_id: {'metric': [values]}}}
        self.historical_data = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )

        # 计算好的分位数表
        self.quantile_table = {}

        # 分位数集合
        self.quantiles = [0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95, 0.99]

    def _get_session_and_bucket(self, timestamp_ms: int) -> Tuple[Optional[str], int]:
        """
        获取session和bucket_id

        Returns:
            (session_name, bucket_id) 或 (None, -1) if out of session
        """
        dt = pd.Timestamp(timestamp_ms, unit="ms", tz="Asia/Shanghai")
        hour, minute = dt.hour, dt.minute

        for session_name, ((start_h, start_m), (end_h, end_m)) in self.SESSIONS.items():
            # 检查是否在该session内
            time_minutes = hour * 60 + minute
            start_minutes = start_h * 60 + start_m
            end_minutes = end_h * 60 + end_m

            if start_minutes <= time_minutes < end_minutes:
                # 计算session内的bucket
                minutes_since_session_start = time_minutes - start_minutes
                bucket_id = minutes_since_session_start // self.bucket_minutes
                return session_name, bucket_id

        return None, -1  # Out of session

    def add_observation(
        self, timestamp_ms: int, vpin: float, spread: float, depth: float
    ):
        """添加观测值"""
        session, bucket_id = self._get_session_and_bucket(timestamp_ms)

        if session is None:
            return  # 跳过休市时间

        self.historical_data[session][bucket_id]["vpin"].append(vpin)
        self.historical_data[session][bucket_id]["spread"].append(spread)
        self.historical_data[session][bucket_id]["depth"].append(depth)

    def compute_quantiles(self):
        """计算所有session和bucket的分位数"""
        for session in self.historical_data:
            for bucket_id in self.historical_data[session]:
                key = f"{session}_{bucket_id}"
                self.quantile_table[key] = {}

                for metric in ["vpin", "spread", "depth"]:
                    data = self.historical_data[session][bucket_id][metric]

                    if len(data) > 10:
                        self.quantile_table[key][metric] = {
                            f"p{int(q*100):02d}": np.percentile(data, q * 100)
                            for q in self.quantiles
                        }
                    else:
                        # 数据不足，使用默认值
                        self.quantile_table[key][metric] = {
                            f"p{int(q*100):02d}": 0.0 for q in self.quantiles
                        }

    def get_threshold(self, timestamp_ms: int, metric: str, percentile: str) -> float:
        """获取动态阈值"""
        session, bucket_id = self._get_session_and_bucket(timestamp_ms)

        if session is None:
            return 0.0

        key = f"{session}_{bucket_id}"
        if key in self.quantile_table:
            return self.quantile_table[key].get(metric, {}).get(percentile, 0.0)

        return 0.0

    def get_rank(self, timestamp_ms: int, metric: str, value: float) -> float:
        """
        获取value在历史分布中的分位数位置（rank）

        Returns:
            0.0-1.0，表示value在该bucket历史分布中的位置
        """
        session, bucket_id = self._get_session_and_bucket(timestamp_ms)

        if session is None:
            return 0.5  # 默认中位数

        data = self.historical_data[session][bucket_id].get(metric, [])
        if len(data) < 10:
            return 0.5

        # 计算rank（小于等于value的比例）
        rank = sum(1 for v in data if v <= value) / len(data)
        return rank


# ============================================
# 3. 价格动力学指标（保持不变）
# ============================================


class PriceDynamicsIndicators:
    """价格动力学指标（已验证可用）"""

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
        """漂移-波动比"""
        if len(self.returns_buffer) < 10:
            return 0.0

        rets = np.array(list(self.returns_buffer))
        drift = abs(rets.sum())
        vol = rets.std() * np.sqrt(len(rets))

        if vol < 1e-9:
            return 0.0

        return drift / vol

    def get_directional_consistency(self) -> float:
        """方向一致性"""
        if len(self.returns_buffer) < 10:
            return 0.0

        rets = np.array(list(self.returns_buffer))
        pos_ratio = (rets > 0).sum() / len(rets)

        return pos_ratio - 0.5

    def get_lag1_autocorr(self) -> float:
        """Lag-1自相关"""
        if len(self.returns_buffer) < 10:
            return 0.0

        rets = np.array(list(self.returns_buffer))

        if len(rets) < 2:
            return 0.0

        corr = np.corrcoef(rets[:-1], rets[1:])[0, 1]

        return corr if not np.isnan(corr) else 0.0

    def get_mean_reversion_strength(self) -> float:
        """均值回复强度"""
        if len(self.mid_buffer) < self.window:
            return 0.0

        mids = np.array(list(self.mid_buffer))
        ma = mids[:-5].mean() if len(mids) > 5 else mids.mean()

        deviation = (mids[-1] - ma) / (ma + 1e-9)

        if len(mids) >= 5:
            recent_trend = (mids[-1] - mids[-5]) / (mids[-5] + 1e-9)
            reversion_signal = -deviation * recent_trend
            return reversion_signal

        return 0.0

    def get_realized_vol(self) -> float:
        """实现波动"""
        if len(self.returns_buffer) < 10:
            return 0.0

        rets = np.array(list(self.returns_buffer))
        return rets.std()


# ============================================
# 4. 两层Regime检测器 v1.1
# ============================================


class TwoTierRegimeDetector_v11:
    """
    两层Regime检测器 v1.1

    核心改进：
    1. ✅ 字段统一映射
    2. ✅ 健康度闸门
    3. ✅ 分位数评分制（避免锁死）
    4. ✅ 迟滞阈值（enter > exit）
    5. ✅ Action两阶段gating
    6. ✅ Micro/Action独立min_residence
    7. ✅ 连续置信度
    """

    def __init__(
        self,
        baseline: IntradayQuantileBaseline,
        min_residence_micro=10,
        min_residence_action=15,
    ):
        self.baseline = baseline
        self.min_residence_micro = 30  # Increased for stability (was 10)
        self.min_residence_action = 50  # Increased for stability (was 15)

        # 状态
        self.current_micro = "normal"
        self.current_action = "neutral"
        self.residence_counter_micro = 0
        self.residence_counter_action = 0

        # 价格动力学
        self.dynamics = PriceDynamicsIndicators(window=40)  # Slower window (was 20)

        # 健康度监控
        self.health_monitor = FeatureHealthMonitor(
            window=300
        )  # Longer window for sticky prices (was 100)

        # 字段映射器
        self.mapper = CanonicalFeatureMapper()

        # 评分历史（用于平滑）
        self.illiquid_score_buffer = deque(maxlen=10)  # Smoother (was 5)
        self.highvol_score_buffer = deque(maxlen=10)  # Smoother (was 5)

    def detect(
        self, timestamp_ms: int, white_risk: Dict, mid: float, prev_mid: float
    ) -> Tuple[str, str, float]:
        """
        主检测接口

        Returns:
            (micro_regime, action_regime, confidence)
        """
        # 更新动力学
        self.dynamics.update(mid, prev_mid)

        # 🔧 1. 字段统一映射
        vpin = self.mapper.get_canonical_value(white_risk, "vpin") or 0.0
        spread = self.mapper.get_canonical_value(white_risk, "spread_bps") or 0.0
        depth = self.mapper.get_canonical_value(white_risk, "depth") or 10000.0

        # 🔧 2. 更新健康度
        self.health_monitor.update("vpin", vpin)
        self.health_monitor.update("spread", spread)
        self.health_monitor.update("depth", depth)

        # 健康度检查
        vpin_healthy, _ = self.health_monitor.is_healthy("vpin")
        spread_healthy, _ = self.health_monitor.is_healthy("spread")
        depth_healthy, _ = self.health_monitor.is_healthy("depth")

        # 🔧 3. Micro层：分位数评分制
        new_micro, micro_conf = self._detect_micro_regime(
            timestamp_ms,
            vpin,
            spread,
            depth,
            vpin_healthy,
            spread_healthy,
            depth_healthy,
        )

        # Micro切换控制
        if new_micro != self.current_micro:
            if self.residence_counter_micro >= self.min_residence_micro:
                self.current_micro = new_micro
                self.residence_counter_micro = 0
                # Micro切换时重置Action
                self.current_action = "neutral"
                self.residence_counter_action = 0
        self.residence_counter_micro += 1

        # 🔧 4. Action层：两阶段gating
        if self.current_micro == "illiquid":
            # illiquid时Action无意义
            new_action = "neutral"
            action_conf = 0.0
        else:
            new_action, action_conf = self._detect_action_regime(timestamp_ms)

        # Action切换控制（仅在Micro稳定时允许）
        if new_action != self.current_action and self.residence_counter_micro >= 5:
            if self.residence_counter_action >= self.min_residence_action:
                self.current_action = new_action
                self.residence_counter_action = 0
        self.residence_counter_action += 1

        # 综合置信度
        overall_conf = micro_conf * (1.0 if self.current_micro != "illiquid" else 0.5)
        overall_conf *= max(0.5, action_conf) if action_conf > 0 else 0.7

        return self.current_micro, self.current_action, overall_conf

    def _detect_micro_regime(
        self,
        timestamp_ms: int,
        vpin: float,
        spread: float,
        depth: float,
        vpin_healthy: bool,
        spread_healthy: bool,
        depth_healthy: bool,
    ) -> Tuple[str, float]:
        """
        Micro层检测：分位数评分制

        核心改进：避免OR进入AND退出的锁死
        """
        # 获取rank（分位数位置）
        rank_spread = (
            self.baseline.get_rank(timestamp_ms, "spread", spread)
            if spread_healthy
            else 0.5
        )
        rank_depth = (
            self.baseline.get_rank(timestamp_ms, "depth", depth)
            if depth_healthy
            else 0.5
        )
        rank_vpin = (
            self.baseline.get_rank(timestamp_ms, "vpin", abs(vpin))
            if vpin_healthy
            else 0.5
        )

        # 计算illiquid_score
        # 价差异常扩大 + 深度异常塌陷
        illiquid_score = (
            max(0, rank_spread - 0.95) * 20 + max(0, 0.05 - rank_depth) * 20
        )
        self.illiquid_score_buffer.append(illiquid_score)
        illiquid_score_smooth = np.mean(self.illiquid_score_buffer)

        # 计算highvol_score
        # VPIN尾部 + 实现波动尾部
        realized_vol = self.dynamics.get_realized_vol()
        rank_vol = 0.99 if realized_vol > 0.001 else 0.5  # 简化：实际应该也用baseline

        highvol_score = max(0, rank_vpin - 0.90) * 10 + max(0, rank_vol - 0.90) * 10
        self.highvol_score_buffer.append(highvol_score)
        highvol_score_smooth = np.mean(self.highvol_score_buffer)

        # 🔧 迟滞阈值
        if self.current_micro == "illiquid":
            # 退出阈值更宽松
            if illiquid_score_smooth < 0.3:  # 退出阈值
                pass  # 允许退出到normal
            else:
                return "illiquid", 0.9

        if self.current_micro == "high_volatility":
            if highvol_score_smooth < 0.3:
                pass
            else:
                return "high_volatility", 0.85

        # 进入判断
        if illiquid_score_smooth > 0.5:  # 进入阈值更严格
            return "illiquid", 0.9

        if highvol_score_smooth > 0.5:
            return "high_volatility", 0.85

        return "normal", 0.7

    def _detect_action_regime(self, timestamp_ms: int) -> Tuple[str, float]:
        """
        Action层检测：两阶段gating

        改进：确保可交易驻留长度
        """
        # 🔧 Stage 1: 动力学信息gating
        realized_vol = self.dynamics.get_realized_vol()

        # 获取该bucket的低分位阈值
        vol_p10 = self.baseline.get_threshold(timestamp_ms, "spread", "p10")  # 近似

        if realized_vol < 1e-5:  # 极低波动，无动力学信息
            return "neutral", 0.3

        # 🔧 Stage 2: 证据竞争
        # trending证据
        drift_vol = self.dynamics.get_drift_to_vol_ratio()
        dir_cons = abs(self.dynamics.get_directional_consistency())
        trending_score = (
            min(drift_vol / 1.5, 1.0) * 0.6 + min(dir_cons / 0.3, 1.0) * 0.4
        )

        # mean_reverting证据
        autocorr = self.dynamics.get_lag1_autocorr()
        mr_strength = self.dynamics.get_mean_reversion_strength()
        mr_score = 0.0
        if autocorr < -0.2:
            mr_score += min(abs(autocorr) / 0.5, 1.0) * 0.5
        if mr_strength > 0.3:
            mr_score += min(mr_strength / 0.8, 1.0) * 0.5

        # 竞争选择（需要明显优势）
        # 竞争选择（需要明显优势）
        # ETF调优：
        # 1. 提高Trending门槛 (0.4 -> 0.6)
        # 2. 降低MR门槛 (0.4 -> 0.25)
        # 3. 增加竞争Buffer (0.15 -> 0.20)

        if trending_score > 0.60 and trending_score > mr_score + 0.20:
            return "trending", trending_score
        elif mr_score > 0.25 and mr_score > trending_score + 0.20:
            return "mean_reverting", mr_score
        else:
            return "neutral", 0.5


# 导出接口保持兼容
TwoTierRegimeDetector = TwoTierRegimeDetector_v11
