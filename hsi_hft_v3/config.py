#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HSI HFT V3 - 全局配置模块 (Global Configuration)

集中管理整个交易系统的所有配置参数。
包含：
1. 全局身份与范围 (Global Identity)
2. 执行与成本模型 (Execution & Cost)
3. 建模目标与超参 (Modeling & Hyperparams)
4. 策略与阈值 (Strategy & Thresholds)
5. 数据契约 (Data Contract)
6. 训练配置 (Train Config)
7. 预训练配置 (Pretrain Config)
8. IRM配置 (IRM Config)
9. 风控配置 (Risk Config)
"""

from dataclasses import dataclass, field
from typing import List, Dict

# ==========================================
# 1. 全局身份与范围 (Global Identity)
# ==========================================
TARGET_SYMBOL = "sz159920"  # 目标标的 (ETF/期货)
AUX_SYMBOL = "sh513130"  # 辅助标的 (用于跨市场分析)
BAR_SIZE_S = 3  # K线周期 (秒)
TIMEZONE = "Asia/Shanghai"  # 市场时区

# ==========================================
# 2. 执行与成本模型 (Execution & Cost Model)
# ==========================================
COST_RATE = 0.0001  # 交易成本 (双边各1bp, 买入1万成本1元)
TICK_SIZE = 0.001  # 最小价格变动单位
LATENCY_BARS = 1  # 模拟执行延迟 (多少个Bar后成交)
TRADE_QTY = 1000  # 默认模拟交易数量


@dataclass
class ExecutionConfig:
    """回测与实盘执行配置"""

    order_type: str = "TAKER"  # 订单类型: 始终使用Taker (吃单)
    fill_mode: str = "L1_IOC"  # 成交模式: Level 1 即时成交否则取消 (IOC)
    slippage_bps: float = 0.0  # 额外滑点模拟 (基点)


# ==========================================
# 3. 建模目标与超参 (Modeling Objectives)
# ==========================================
PREDICT_HORIZON_S = 120  # 预测视野 (H* = 120秒)
K_BARS = PREDICT_HORIZON_S // BAR_SIZE_S  # 预测步数 (40个Bar)
LOOKBACK_BARS = 100  # 输入特征窗口长度 (Lookback)
BLACKBOX_DIM = 32  # 深度模型潜在因子维度


@dataclass
class ModelConfig:
    """深度模型架构参数"""

    d_model: int = 64  # 模型隐藏层维度
    d_state: int = 16  # Mamba状态维度
    expand: int = 2  # Mamba扩展系数
    projector_dim: int = 64  # 投影层中间维度

    # 残差结合
    default_alpha: float = 0.5  # 默认黑盒权重

    # 损失函数
    vicreg_batch_size: int = 128  # VICReg计算协方差的期望批次大小 (用于维度推断)


@dataclass
class LabelConfig:
    """标签生成配置"""

    use_cost_gate: bool = False  # 是否启用成本过滤 (True则收益>2*Cost才算Hit)
    adverse_bps: float = 20.0  # 逆向波动阈值 (作为止损/风险标签)
    embargo_bars: int = (
        LOOKBACK_BARS + K_BARS + LATENCY_BARS + 10
    )  # 数据分割时的隔离缓冲区


# ==========================================
# 4. 策略与阈值 (Strategy Policy & Thresholds)
# ==========================================
@dataclass
class PolicyConfig:
    """
    交易策略配置 (自适应阈值)
    调整这些参数以平衡 激进程度 vs 精确度
    """

    # 标准阈值 (当有辅助数据Aux时)
    # 针对低成本(1bp)高频场景优化
    th_p_hit: float = 0.55  # Hit概率阈值 (入场门槛)
    th_p_tau_60: float = 0.50  # 60秒内Hit概率阈值 (时间紧迫性)
    th_risk_enter: float = 0.30  # 入场最大允许风险概率

    # 保守阈值 (当无辅助数据时)
    th_p_hit_noaux: float = 0.70
    th_p_tau_60_noaux: float = 0.80
    th_risk_enter_noaux: float = 0.10

    # 离场与风控门槛
    th_risk_exit: float = 0.25  # 触发风险离场的概率阈值
    max_hold_bars: int = K_BARS  # 最大持仓时间 (时间止损)
    cooldown_bars: int = 20  # 交易冷却期 (Bar数, 20*3=1分钟)

    # 微观结构与白盒门槛
    spread_max_bps: float = 5.0  # 最大允许盘口价差 (严格控制)
    depth_min_qty: int = 5000  # 盘口最小挂单量 (流动性过滤)
    vpin_max_z: float = 3.0  # VPIN (毒性流) 入场阈值 (Z-score)
    vpin_exit_z: float = 3.5  # VPIN 强制离场阈值
    lambda_exit_z: float = 3.0  # Kyle Lambda (价格冲击) 离场阈值


@dataclass
class RegimeConfig:
    """体制检测与阈值配置"""

    # 基础窗口
    health_window: int = 300  # 健康度检查窗口
    dynamics_window: int = 40  # 动力学指标窗口
    baseline_bucket_minutes: int = 5  # 基线统计桶大小 (分钟)

    # 驻留时间 (Bar数)
    min_residence_micro: int = 30  # 微观体制最小驻留
    min_residence_action: int = 50  # 动作体制最小驻留

    # 评分阈值
    th_illiquid_enter: float = 0.5  # 进入流动性枯竭的阈值
    th_illiquid_exit: float = 0.3  # 退出流动性枯竭的阈值
    th_highvol_enter: float = 0.5  # 进入高波动的阈值
    th_highvol_exit: float = 0.3  # 退出高波动的阈值

    # 信号阈值
    th_trending_score: float = 0.60  # 趋势得分门槛
    th_trending_diff: float = 0.20  # 趋势 vs 均值回复 优势差
    th_mr_score: float = 0.25  # 均值回复得分门槛

    # 健康度
    health_min_nonzero: float = 0.05  # 特征非零比例最小阈值


# ==========================================
# 5. 数据契约与管道 (Data Contract)
# ==========================================
@dataclass
class DataConfig:
    """数据处理与对齐配置"""

    max_lag_ms: int = 30000  # 双流对齐最大允许延迟 (30秒)


# ==========================================
# 5. 数据契约 (Data Contract)
# ==========================================
# 允许进入数据管道的字段白名单
ALLOWLIST_FIELDS = [
    "tx_local_time",
    "tx_server_time",
    "price",
    "tick_vol",
    "tick_amt",
    "tick_vwap",
    "bp1",
    "bv1",
    "sp1",
    "sv1",
    "bp2",
    "bv2",
    "sp2",
    "sv2",
    "bp3",
    "bv3",
    "sp3",
    "sv3",
    "bp4",
    "bv4",
    "sp4",
    "sv4",
    "bp5",
    "bv5",
    "sp5",
    "sv5",
    "sentiment",
    "premium_rate",
    "iopv",
    "index_price",
    "fx_rate",
    "fut_price",
    "fut_mid",
    "fut_imb",
    "fut_delta_vol",
    "fut_pct",
]

# 需要排除的字段黑名单
BLOCKLIST_FIELDS = ["idx_delay_ms", "fut_delay_ms", "data_flags"]


# ==========================================
# 6. 训练配置 (Train Config)
# ==========================================
@dataclass
class TrainConfig:
    """模型训练超参数"""

    epochs: int = 30  # 总训练轮数
    batch_size: int = 128  # 批次大小
    learning_rate: float = 1e-3  # 学习率
    weight_decay: float = 1e-5  # 权重衰减 (L2正则)
    pos_weight: float = 10.0  # 正样本权重 (解决类别不平衡, 10倍关注盈利机会)

    # 动态数据分割比例
    train_ratio: float = 0.8
    val_ratio: float = 0.1

    # 这里定义需要排除的脏数据日期 (例如缺失指数价格的日子)
    exclude_dates: List[str] = field(
        default_factory=lambda: ["2026-01-27", "2026-01-28", "2026-01-29", "2026-01-30"]
    )

    compute_baseline_stats: bool = True  # 训练结束后是否计算风控基准


# ==========================================
# 7. 预训练配置 (Pretrain Config)
# ==========================================
@dataclass
class PretrainConfig:
    """自监督预训练超参数"""

    epochs: int = 10  # 预训练轮数
    batch_size: int = 128  # 批次大小
    learning_rate: float = 1e-3  # 学习率
    pred_horizon: int = 20  # 预测未来步数 (JEPA目标)
    exclude_dates: List[str] = field(
        default_factory=lambda: ["2026-01-27", "2026-01-28", "2026-01-29", "2026-01-30"]
    )


# ==========================================
# 8. 特征工程配置 (Feature Engineering)
# ==========================================
@dataclass
class FeatureConfig:
    """白盒特征提取参数"""

    windows: List[int] = field(
        default_factory=lambda: [20, 100, 600]
    )  # 滚动窗口 (Tick数)
    depth_levels: int = 5  # LOB深度层数
    iofi_weights: List[float] = field(
        default_factory=lambda: [1.0, 0.8, 0.6, 0.4, 0.2]
    )  # OFI权重衰减
    lead_lag_lags: List[int] = field(
        default_factory=lambda: [1, 2, 3, 5]
    )  # 领滞分析的滞后阶数


# ==========================================
# 9. IRM并配置 (Invariant Risk Minimization)
# ==========================================
@dataclass
class IRMConfig:
    """IRM (不变风险最小化) 配置"""

    use_irm: bool = True  # 是否启用IRM损失
    penalty_weight: float = 1.0  # 梯度惩罚权重 (Lambda)
    anneal_epochs: int = 5  # 惩罚权重退火周期 (前N轮逐渐增加惩罚)


# ==========================================
# 9. 风控监控配置 (Risk Monitor Config)
# ==========================================
@dataclass
class RiskConfig:
    """在线风控监控参数"""

    window_size: int = 60  # 监控滑窗大小 (Bar数)

    # 阈值设置
    drift_zscore_threshold: float = 3.0  # 分布漂移 Z-score 阈值
    ks_pvalue_threshold: float = 0.01  # KS检验 P值阈值 (显著性水平)
    ic_threshold: float = -0.2  # IC (信息系数) 负向阈值
    sharpe_threshold: float = -0.5  # 夏普比率 负向阈值
    jump_sigma_multiplier: float = 5.0  # 突变检测 (Sigma倍数)
    black_loss_threshold: float = -1000.0  # 黑盒最大允许累积亏损

    cooldown_bars: int = 5  # 警报触发后的冷却时间
    circuit_breaker_critical: int = 2  # 触发熔断所需的严重警报数量
