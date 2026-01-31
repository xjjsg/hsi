"""
HSI HFT V3 - 风控监控系统 (RiskMonitor)
技术实施方案

优先级：🔴 最高（Tier 1）
状态：待实施
预期收益：避免黑盒过拟合导致的-10%以上回撤
"""

import numpy as np
import torch
from collections import deque
from scipy.stats import ks_2samp
from typing import Dict, List, Optional
import warnings


class RiskMonitor:
    """
    黑盒模型风控监控器

    功能模块：
    1. 分布漂移检测 (Distribution Drift Detection)
    2. 因子性能监控 (Performance Monitoring)
    3. 异常值检测 (Anomaly Detection)
    4. 自动降权与熔断 (Auto-Downweight & Circuit Breaker)

    设计原则：
    - 低延迟：所有检测<1ms
    - 低误报：多重验证机制
    - 可解释：每个警报都有明确原因
    """

    def __init__(self, baseline_stats: Dict, window_size: int = 60):
        """
        Args:
            baseline_stats: 训练集统计量 {
                'black_mu': 黑盒输出均值,
                'black_sigma': 黑盒输出标准差,
                'black_q99': 99分位数,
                'black_samples': 1000个训练样本(用于KS检验),
                'white_mu': 白盒输出均值,
                'white_sigma': 白盒输出标准差
            }
            window_size: 滑窗大小（60个bar = 3分钟）
        """
        # 基准统计（从训练集离线计算）
        self.baseline = baseline_stats

        # 滑动窗口
        self.window_size = window_size
        self.black_outputs = deque(maxlen=window_size)
        self.white_outputs = deque(maxlen=window_size)
        self.predictions = deque(maxlen=window_size)
        self.realized_returns = deque(maxlen=window_size)

        # 性能跟踪
        self.black_pnl = 0.0
        self.white_pnl = 0.0
        self.total_trades = 0

        # 警报状态
        self.alerts = {}  # {alert_type: alert_info}
        self.alert_history = []
        self.alert_cooldown = {}  # 冷却期管理

        # 降权参数
        self.alpha_adjustment = 0.0  # 叠加到基础α的调整值
        self.circuit_breaker_active = False

        # 配置参数
        # 配置参数 (从Config加载)
        from hsi_hft_v3.config import RiskConfig

        cfg = RiskConfig()

        self.config = {
            "drift_zscore_threshold": cfg.drift_zscore_threshold,
            "ks_pvalue_threshold": cfg.ks_pvalue_threshold,
            "ic_threshold": cfg.ic_threshold,
            "sharpe_threshold": cfg.sharpe_threshold,
            "jump_sigma_multiplier": cfg.jump_sigma_multiplier,
            "black_loss_threshold": cfg.black_loss_threshold,
            "cooldown_bars": cfg.cooldown_bars,
            "circuit_breaker_critical": cfg.circuit_breaker_critical,
        }

    def update(
        self,
        model_output: Dict,
        white_feats: Optional[np.ndarray] = None,
        realized_pnl: Optional[float] = None,
    ):
        """
        每个decision cycle调用一次

        Args:
            model_output: {
                'base_hit': 白盒logit,
                'delta_hit': 黑盒delta,
                'logit_hit': 最终logit,
                'p_hit': sigmoid(logit_hit)
            }
            white_feats: 白盒特征向量（用于后续分析）
            realized_pnl: 如果有完成的交易，提供实际盈亏
        """
        # 1. 存储数据
        base = model_output.get("base_hit", 0)
        delta = model_output.get("delta_hit", 0)
        final = model_output.get("logit_hit", 0)

        self.black_outputs.append(float(delta))
        self.white_outputs.append(float(base))
        self.predictions.append(float(final))

        # 2. PnL归因
        if realized_pnl is not None:
            self.realized_returns.append(realized_pnl)
            self.total_trades += 1

            # 简化归因：根据贡献度分配
            total_signal = abs(base) + abs(delta) + 1e-9
            self.white_pnl += realized_pnl * (abs(base) / total_signal)
            self.black_pnl += realized_pnl * (abs(delta) / total_signal)

        # 3. 运行检测（仅当窗口满）
        if len(self.black_outputs) >= self.window_size:
            self._check_distribution_drift()
            self._check_performance()
            self._check_anomalies()
            self._update_alpha()

    # ========================================
    # 检测模块
    # ========================================

    def _check_distribution_drift(self):
        """检测黑盒输出分布漂移"""
        recent_black = np.array(self.black_outputs)

        # 方法1：3-sigma均值偏移
        mu_recent = recent_black.mean()
        sigma_recent = recent_black.std()

        mu_baseline = self.baseline.get("black_mu", 0)
        sigma_baseline = self.baseline.get("black_sigma", 1)

        # Z-score of mean shift
        z_shift = abs(mu_recent - mu_baseline) / (
            sigma_baseline / np.sqrt(len(recent_black))
        )

        if z_shift > self.config["drift_zscore_threshold"]:
            self._trigger_alert(
                "drift_mean",
                {
                    "severity": "critical",
                    "z_shift": z_shift,
                    "mu_recent": mu_recent,
                    "mu_baseline": mu_baseline,
                    "diff": mu_recent - mu_baseline,
                },
            )

        # 方法2：KS检验（每10个bar检查一次，避免过于频繁）
        if len(self.black_outputs) % 10 == 0:
            baseline_sample = self.baseline.get("black_samples", [])
            if len(baseline_sample) > 30:
                ks_stat, p_value = ks_2samp(recent_black, baseline_sample)
                if p_value < self.config["ks_pvalue_threshold"]:
                    self._trigger_alert(
                        "drift_distribution",
                        {
                            "severity": "critical",
                            "ks_stat": ks_stat,
                            "p_value": p_value,
                        },
                    )

    def _check_performance(self):
        """监控因子性能"""
        if len(self.realized_returns) < 10:
            return

        recent_returns = np.array(list(self.realized_returns)[-10:])
        recent_preds = np.array(list(self.predictions)[-10:])

        # 1. IC (Information Coefficient)
        # DISABLE: Currently broken due to alignment mismatch between realized_returns (trades) and predictions (bars)
        # if len(recent_preds) == len(recent_returns):
        #     ic = np.corrcoef(recent_preds, recent_returns)[0, 1]
        #     if not np.isnan(ic) and ic < self.config["ic_threshold"]:
        #         self._trigger_alert(...)

        # 2. 累积PnL（黑盒）
        if self.black_pnl < self.config["black_loss_threshold"]:
            self._trigger_alert(
                "black_loss",
                {
                    "severity": "critical",
                    "black_pnl": self.black_pnl,
                    "white_pnl": self.white_pnl,
                    "total_trades": self.total_trades,
                },
            )

        # 3. Sharpe（最近10笔）
        if len(recent_returns) >= 5:
            sharpe = recent_returns.mean() / (recent_returns.std() + 1e-9)
            if sharpe < self.config["sharpe_threshold"]:
                self._trigger_alert(
                    "sharpe_negative",
                    {
                        "severity": "warning",
                        "sharpe": sharpe,
                        "mean_return": recent_returns.mean(),
                        "std": recent_returns.std(),
                    },
                )

    def _check_anomalies(self):
        """检测异常值"""
        if len(self.black_outputs) < 2:
            return

        # 1. 单步跳变
        last_two = list(self.black_outputs)[-2:]
        jump = abs(last_two[1] - last_two[0])

        sigma_baseline = self.baseline.get("black_sigma", 1)
        jump_threshold = self.config["jump_sigma_multiplier"] * sigma_baseline

        if jump > jump_threshold:
            self._trigger_alert(
                "anomaly_jump",
                {
                    "severity": "warning",
                    "jump": jump,
                    "threshold": jump_threshold,
                    "from": last_two[0],
                    "to": last_two[1],
                },
            )

        # 2. 极端值
        latest = self.black_outputs[-1]
        q99 = self.baseline.get("black_q99", 10)

        if abs(latest) > q99:
            self._trigger_alert(
                "anomaly_extreme", {"severity": "warning", "value": latest, "q99": q99}
            )

    # ========================================
    # 警报管理
    # ========================================

    def _trigger_alert(self, alert_type: str, meta: Dict):
        """触发警报（带冷却期）"""
        # 检查冷却期
        current_time = len(self.black_outputs)
        if alert_type in self.alert_cooldown:
            if (
                current_time - self.alert_cooldown[alert_type]
                < self.config["cooldown_bars"]
            ):
                return  # 仍在冷却期

        # 记录警报
        alert = {"type": alert_type, "timestamp": current_time, "meta": meta}

        self.alerts[alert_type] = alert
        self.alert_history.append(alert)
        self.alert_cooldown[alert_type] = current_time

        # 打印警报
        severity_emoji = "🔴" if meta.get("severity") == "critical" else "🟡"
        print(f"{severity_emoji} RISK ALERT: {alert_type} | {meta}")

    def _update_alpha(self):
        """根据警报自动调整α"""
        # 严重性映射：不同警报对α的影响
        severity_map = {
            "drift_mean": -0.3,  # 分布均值漂移：降低30%
            "drift_distribution": -0.2,  # KS检验失败：降低20%
            "ic_negative": -0.3,  # IC为负：降低30%
            "black_loss": -0.4,  # 累积亏损：降低40%
            "anomaly_extreme": -0.5,  # 极端异常：降低50%
            "anomaly_jump": -0.2,  # 跳变：降低20%
            "sharpe_negative": -0.1,  # Sharpe差：降低10%
        }

        # 重置调整
        self.alpha_adjustment = 0.0

        # 累加惩罚（多个警报叠加）
        for alert_type, alert in self.alerts.items():
            if alert_type in severity_map:
                self.alpha_adjustment += severity_map[alert_type]

        # 限制范围
        self.alpha_adjustment = max(-1.0, self.alpha_adjustment)

        # 熔断逻辑
        critical_alerts = [
            a for a in self.alerts.values() if a["meta"].get("severity") == "critical"
        ]

        if len(critical_alerts) >= self.config["circuit_breaker_critical"]:
            self.circuit_breaker_active = True
            self.alpha_adjustment = -1.0  # 完全关闭黑盒
            print("🔴 CIRCUIT BREAKER ACTIVATED - Black box disabled!")
        else:
            self.circuit_breaker_active = False

        # 清理过期警报（超过20个bar的警报）
        current_time = len(self.black_outputs)
        self.alerts = {
            k: v for k, v in self.alerts.items() if current_time - v["timestamp"] < 20
        }

    # ========================================
    # 对外接口
    # ========================================

    def get_adjusted_alpha(self, base_alpha: float) -> float:
        """
        获取调整后的α系数

        Args:
            base_alpha: 基础α（来自RegimeDetector）

        Returns:
            adjusted_alpha ∈ [0, 1]
        """
        if self.circuit_breaker_active:
            return 0.0

        adjusted = base_alpha + self.alpha_adjustment
        return max(0.0, min(1.0, adjusted))

    def get_status_report(self) -> str:
        """生成状态报告"""
        report = []
        report.append(f"=== Risk Monitor Status ===")
        report.append(f"Window: {len(self.black_outputs)}/{self.window_size}")
        report.append(f"Black PnL: {self.black_pnl:.2f} ({self.total_trades} trades)")
        report.append(f"White PnL: {self.white_pnl:.2f}")
        report.append(f"Alpha Adjustment: {self.alpha_adjustment:.2f}")
        report.append(
            f"Circuit Breaker: {'🔴 ACTIVE' if self.circuit_breaker_active else '✅ OFF'}"
        )
        report.append(f"\nActive Alerts ({len(self.alerts)}):")
        for alert_type, alert in self.alerts.items():
            severity = alert["meta"].get("severity", "info")
            emoji = "🔴" if severity == "critical" else "🟡"
            report.append(f"  {emoji} {alert_type}: {alert['meta']}")
        return "\n".join(report)

    def reset_circuit_breaker(self):
        """手动重置熔断（需人工确认后调用）"""
        self.circuit_breaker_active = False
        self.alerts.clear()
        print("✅ Circuit breaker manually reset")


# ========================================
# 离线计算baseline_stats的工具函数
# ========================================


def compute_baseline_stats(model, train_dataloader, device="cpu"):
    """
    从训练集计算基准统计量

    Args:
        model: 训练好的模型（ResidualCombine）
        train_dataloader: 训练数据加载器
        device: 设备

    Returns:
        baseline_stats: Dict包含所有统计量
    """
    model.eval()
    black_outputs = []
    white_outputs = []

    with torch.no_grad():
        for batch in train_dataloader:
            # 假设batch包含white_feats和deep_factors
            if isinstance(batch, (list, tuple)):
                white_feats, deep_factors = batch[0].to(device), batch[1].to(device)
            else:
                continue

            # 获取模型输出
            output = model(white_feats, deep_factors)

            black_outputs.extend(output["delta_hit"].cpu().numpy().flatten())
            white_outputs.extend(output["base_hit"].cpu().numpy().flatten())

    black_outputs = np.array(black_outputs)
    white_outputs = np.array(white_outputs)

    # 计算统计量
    baseline_stats = {
        "black_mu": float(black_outputs.mean()),
        "black_sigma": float(black_outputs.std()),
        "black_q99": float(np.percentile(np.abs(black_outputs), 99)),
        "black_samples": black_outputs[:1000].tolist(),  # 保存1000个样本用于KS检验
        "white_mu": float(white_outputs.mean()),
        "white_sigma": float(white_outputs.std()),
    }

    return baseline_stats


# ========================================
# 使用示例
# ========================================

if __name__ == "__main__":
    # 1. 训练后离线计算baseline
    # baseline_stats = compute_baseline_stats(model, train_loader)
    # with open('baseline_stats.pkl', 'wb') as f:
    #     pickle.dump(baseline_stats, f)

    # 2. 推理时使用
    baseline_stats = {
        "black_mu": 0.0,
        "black_sigma": 0.15,
        "black_q99": 0.50,
        "black_samples": np.random.randn(1000).tolist(),
        "white_mu": 0.5,
        "white_sigma": 0.2,
    }

    risk_monitor = RiskMonitor(baseline_stats, window_size=60)

    # 3. 在推理循环中
    for i in range(200):
        model_output = {
            "base_hit": 0.5 + np.random.randn() * 0.1,
            "delta_hit": 0.0 + np.random.randn() * 0.15,
            "logit_hit": 0.5 + np.random.randn() * 0.2,
        }

        # 模拟异常
        if i == 100:
            model_output["delta_hit"] = 2.0  # 触发极端值警报

        realized_pnl = np.random.randn() * 100 if i % 10 == 0 else None

        risk_monitor.update(model_output, realized_pnl=realized_pnl)

        if i % 20 == 0:
            print(f"\n--- Bar {i} ---")
            print(risk_monitor.get_status_report())

            # 获取调整后的alpha
            base_alpha = 0.5
            adjusted_alpha = risk_monitor.get_adjusted_alpha(base_alpha)
            print(f"Alpha: {base_alpha} → {adjusted_alpha}")
