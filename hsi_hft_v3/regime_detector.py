"""
HSI HFT V3 - 体制识别系统 (RegimeDetector)
技术实施方案

优先级：🟠 高（Tier 1）
状态：框架文件 - 待用户填充自己的想法
预期收益：体制转换期减少15%回撤，震荡期胜率+5-10%

设计哲学：
基于现有白盒指标的规则驱动检测，避免学习型方法的过拟合。
两份评估报告100%共识，是核心优化点。
"""

import numpy as np
from typing import Dict, Optional, Tuple
from enum import Enum


class MarketRegime(Enum):
    """市场体制枚举"""

    NORMAL = "normal"
    HIGH_VOLATILITY = "high_volatility"
    ILLIQUID = "illiquid"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"


class RegimeDetector:
    """
    市场体制识别器

    核心功能：
    1. 基于白盒指标检测5种市场体制
    2. 为每种体制配置专属的因子权重
    3. 与RiskMonitor联动提供双重风控

    检测指标来源（复用HSI现有白盒）：
    - vpin_z: 成交量不平衡Z-score
    - spread_bps: 价差（基点）
    - depth: 流动性深度
    - [TODO 用户自定义] 动量指标
    - [TODO 用户自定义] 其他微观结构指标
    """

    def __init__(self):
        # ========================================
        # 体制定义规则
        # ========================================
        # TODO: 用户可根据自己的理解调整这些阈值

        self.regime_rules = {
            "normal": {
                "vpin_z_range": (-2, 2),
                "spread_max": 8,
                "depth_min": 5000,
                "description": "正常交易状态，流动性充足",
            },
            "high_volatility": {
                "vpin_z_range": (2, 5),  # VPIN异常高
                "spread_max": 15,
                "description": "高波动期，价格剧烈波动",
            },
            "illiquid": {
                "depth_min": 3000,
                "spread_max": 20,
                "description": "流动性枯竭，大单难成交",
            },
            "trending": {
                # TODO: 用户自定义趋势检测指标
                # 建议：价格动量、autocorrelation、方向性成交量等
                "momentum_z_range": (2, np.inf),  # placeholder
                "description": "单边趋势行情",
            },
            "mean_reverting": {
                "vpin_z_range": (-1, 1),
                "momentum_z_range": (-2, 2),  # placeholder
                "description": "震荡行情，均值回复特征明显",
            },
        }

        # ========================================
        # 体制特定的因子权重配置
        # ========================================
        # TODO: 用户根据自己的策略调整这些权重

        self.alpha_by_regime = {
            MarketRegime.NORMAL: {
                "white_weight": 0.5,
                "black_weight": 0.5,
                "rationale": "正常情况下平衡使用白盒和黑盒",
            },
            MarketRegime.HIGH_VOLATILITY: {
                "white_weight": 0.7,
                "black_weight": 0.3,
                "rationale": "高波动期信任经验因子，降低黑盒权重",
            },
            MarketRegime.ILLIQUID: {
                "white_weight": 0.8,
                "black_weight": 0.2,
                "rationale": "流动性差时保守策略，主要依赖白盒",
            },
            MarketRegime.TRENDING: {
                "white_weight": 0.3,
                "black_weight": 0.7,
                "rationale": "趋势行情下黑盒可能捕捉动量模式",
            },
            MarketRegime.MEAN_REVERTING: {
                "white_weight": 0.6,
                "black_weight": 0.4,
                "rationale": "震荡期偏重白盒的均值回复因子",
            },
        }

        # ========================================
        # 体制特定的入场阈值调整
        # ========================================
        # TODO: 用户调整不同体制下的风控阈值

        self.threshold_multiplier = {
            MarketRegime.NORMAL: 1.0,
            MarketRegime.HIGH_VOLATILITY: 1.2,  # 提高入场门槛
            MarketRegime.ILLIQUID: 1.5,  # 大幅提高门槛
            MarketRegime.TRENDING: 0.9,  # 略降低门槛（捕捉趋势）
            MarketRegime.MEAN_REVERTING: 0.95,  # 略降低门槛
        }

        # 状态管理
        self.current_regime = MarketRegime.NORMAL
        self.regime_history = []
        self.regime_confidence = 1.0

    def detect(self, white_risk: Dict) -> Tuple[MarketRegime, float]:
        """
        检测当前市场体制

        Args:
            white_risk: {
                'vpin_z': VPIN的Z-score,
                'spread_bps': 价差（基点）,
                'depth': 流动性深度,
                'momentum_z': 动量指标Z-score (TODO),
                ... 其他白盒指标
            }

        Returns:
            (regime, confidence): 体制类型和置信度
        """
        # ========================================
        # TODO: 用户实现自己的检测逻辑
        # ========================================

        vpin = white_risk.get("vpin_z", 0)
        spread = white_risk.get("spread_bps", 0)
        depth = white_risk.get("depth", 10000)

        # TODO: 用户添加动量指标的计算
        momentum_z = white_risk.get("momentum_z", 0)

        # 优先级检测（从异常到正常）

        # 1. 流动性枯竭（最高优先级）
        if depth < 3000 or spread > 20:
            regime = MarketRegime.ILLIQUID
            confidence = 0.9

        # 2. 高波动
        elif vpin > 2 or spread > 12:
            regime = MarketRegime.HIGH_VOLATILITY
            confidence = 0.8

        # 3. 趋势行情
        # TODO: 用户完善趋势检测逻辑
        elif abs(momentum_z) > 2:
            regime = MarketRegime.TRENDING
            confidence = 0.7

        # 4. 均值回复
        elif abs(vpin) < 1 and abs(momentum_z) < 1:
            regime = MarketRegime.MEAN_REVERTING
            confidence = 0.8

        # 5. 正常
        else:
            regime = MarketRegime.NORMAL
            confidence = 1.0

        # 平滑切换：如果体制频繁切换，降低置信度
        if len(self.regime_history) > 0 and self.regime_history[-1] != regime:
            if len(self.regime_history) >= 3:
                recent_regimes = self.regime_history[-3:]
                if len(set(recent_regimes)) >= 3:
                    confidence *= 0.7  # 降低置信度

        # 更新状态
        self.current_regime = regime
        self.regime_confidence = confidence
        self.regime_history.append(regime)

        # 限制历史长度
        if len(self.regime_history) > 100:
            self.regime_history.pop(0)

        return regime, confidence

    def get_alpha_weights(self, regime: Optional[MarketRegime] = None) -> Dict:
        """
        获取体制对应的因子权重

        Args:
            regime: 体制类型（None则使用当前体制）

        Returns:
            {'white_weight': float, 'black_weight': float, 'rationale': str}
        """
        if regime is None:
            regime = self.current_regime

        return self.alpha_by_regime.get(
            regime, self.alpha_by_regime[MarketRegime.NORMAL]
        )

    def get_threshold_multiplier(self, regime: Optional[MarketRegime] = None) -> float:
        """
        获取体制对应的入场阈值倍数

        Args:
            regime: 体制类型

        Returns:
            multiplier: 阈值倍数（1.0为基准）
        """
        if regime is None:
            regime = self.current_regime

        return self.threshold_multiplier.get(regime, 1.0)

    def get_regime_summary(self) -> str:
        """生成体制分析报告"""
        report = []
        report.append(f"=== Regime Detector Status ===")
        report.append(f"Current Regime: {self.current_regime.value}")
        report.append(f"Confidence: {self.regime_confidence:.2f}")

        weights = self.get_alpha_weights()
        report.append(f"\nFactor Weights:")
        report.append(f"  White: {weights['white_weight']:.2f}")
        report.append(f"  Black: {weights['black_weight']:.2f}")
        report.append(f"  Rationale: {weights['rationale']}")

        multiplier = self.get_threshold_multiplier()
        report.append(f"\nThreshold Multiplier: {multiplier:.2f}x")

        if len(self.regime_history) >= 10:
            recent = self.regime_history[-10:]
            regime_counts = {}
            for r in recent:
                regime_counts[r] = regime_counts.get(r, 0) + 1

            report.append(f"\nRecent Regime Distribution (last 10 bars):")
            for regime, count in regime_counts.items():
                report.append(f"  {regime.value}: {count}/10")

        return "\n".join(report)


# ========================================
# 高级功能：动量指标计算（TODO用户实现）
# ========================================


class MomentumIndicator:
    """
    动量指标计算器

    TODO: 用户根据自己的策略实现

    建议指标：
    1. 价格动量（短期/长期均线偏离）
    2. 方向性成交量（买卖力量对比）
    3. 自相关性（价格序列的autocorrelation）
    4. RSI/MACD等经典技术指标
    """

    def __init__(self, window_short=20, window_long=100):
        self.window_short = window_short
        self.window_long = window_long
        self.price_history = []
        self.volume_history = []

    def update(self, price: float, volume: int):
        """更新历史数据"""
        self.price_history.append(price)
        self.volume_history.append(volume)

        # 限制长度
        if len(self.price_history) > self.window_long * 2:
            self.price_history.pop(0)
            self.volume_history.pop(0)

    def compute_momentum_z(self) -> float:
        """
        计算动量Z-score

        TODO: 用户实现自己的逻辑

        Returns:
            momentum_z: 标准化的动量指标
        """
        if len(self.price_history) < self.window_long:
            return 0.0

        # 示例：简单的价格变化率
        recent = np.array(self.price_history[-self.window_short :])
        baseline = np.array(self.price_history[-self.window_long : -self.window_short])

        mean_recent = recent.mean()
        mean_baseline = baseline.mean()
        std_baseline = baseline.std()

        if std_baseline > 1e-9:
            momentum_z = (mean_recent - mean_baseline) / std_baseline
        else:
            momentum_z = 0.0

        return momentum_z


# ========================================
# 使用示例
# ========================================

if __name__ == "__main__":
    # 初始化
    detector = RegimeDetector()
    momentum_calc = MomentumIndicator()

    # 模拟数据流
    for i in range(100):
        # 模拟白盒指标
        white_risk = {
            "vpin_z": np.random.randn(),
            "spread_bps": 5 + np.random.rand() * 10,
            "depth": 5000 + np.random.randint(-2000, 2000),
        }

        # 计算动量（用户实现）
        price = 4.5 + np.random.randn() * 0.1
        volume = 10000 + np.random.randint(-3000, 3000)
        momentum_calc.update(price, volume)
        white_risk["momentum_z"] = momentum_calc.compute_momentum_z()

        # 检测体制
        regime, confidence = detector.detect(white_risk)

        if i % 20 == 0:
            print(f"\n--- Bar {i} ---")
            print(f"White Risk: {white_risk}")
            print(detector.get_regime_summary())
