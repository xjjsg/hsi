"""
HSI HFT V3 - ResidualCombine门控优化
技术实施方案

优先级：🟡 中（Tier 1）
状态：待实施
功能：规则驱动的白黑盒动态权重调整

关键纠正：
评估报告的公式错误：y = y_white * (1-α) + y_black * α  ❌
正确公式：y = y_white + α * Δy_black                     ✅
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from enum import Enum


class RegimeAdaptiveResidualCombine(nn.Module):
    """
     体制自适应的残差融合模型

     核心改进：
     1. 规则驱动的α调整（非可训练门控网络）
     2. 来自RegimeDetector的基础α
    3. 来自RiskMonitor的降权调整
     4. 正确的残差公式：y = white + α * Δblack

     设计理念：
     - 规则透明（交易员可理解）
     - 避免门控网络过拟合
     - 多维度风控（体制+风险双重保护）
    """

    def __init__(self, white_dim, black_dim=32, k_bars=40):
        super().__init__()

        # ========================================
        # 白盒代理（可解释基线）
        # ========================================
        self.white_hit = nn.Linear(white_dim, 1)
        self.white_hazard = nn.Linear(white_dim, k_bars)
        self.white_risk = nn.Linear(white_dim, 1)

        # ========================================
        # 黑盒修正（深度因子Delta）
        # ========================================
        self.delta_hit = nn.Linear(black_dim, 1)
        self.delta_hazard = nn.Linear(black_dim, k_bars)
        self.delta_risk = nn.Linear(black_dim, 1)

        # Delta初始化为0（从白盒基线开始训练）
        nn.init.zeros_(self.delta_hit.weight)
        nn.init.zeros_(self.delta_hazard.weight)
        nn.init.zeros_(self.delta_risk.weight)

        # ========================================
        # 默认α配置（无体制检测时的fallback）
        # ========================================
        self.default_alpha = 0.5

    def forward(
        self,
        white_feats: torch.Tensor,
        deep_factors: torch.Tensor,
        regime: Optional[str] = None,
        regime_alpha: Optional[float] = None,
        risk_alpha_adjustment: Optional[float] = None,
    ) -> Dict:
        """
        前向传播（带体制和风控调整）

        Args:
            white_feats: (B, white_dim) 白盒特征
            deep_factors: (B, black_dim) 深度因子
            regime: 当前体制（可选，用于记录）
            regime_alpha: RegimeDetector提供的基础α（可选）
            risk_alpha_adjustment: RiskMonitor提供的α调整（可选）

        Returns:
            {
                'logit_hit': 最终logit,
                'logit_hazard': 最终hazard logit,
                'logit_risk': 最终risk logit,
                'base_hit': 白盒logit（用于RiskMonitor）,
                'delta_hit': 黑盒delta（用于RiskMonitor）,
                'alpha': 实际使用的α系数
            }
        """
        # ========================================
        # 1. 计算白盒基线
        # ========================================
        base_hit = self.white_hit(white_feats)
        base_hazard = self.white_hazard(white_feats)
        base_risk = self.white_risk(white_feats)

        # ========================================
        # 2. 计算黑盒Delta
        # ========================================
        delta_hit = self.delta_hit(deep_factors)
        delta_hazard = self.delta_hazard(deep_factors)
        delta_risk = self.delta_risk(deep_factors)

        # ========================================
        # 3. 计算最终α（多层调整）
        # ========================================

        # 基础α（来自RegimeDetector）
        if regime_alpha is not None:
            alpha_base = regime_alpha
        else:
            alpha_base = self.default_alpha

        # 风控调整（来自RiskMonitor）
        if risk_alpha_adjustment is not None:
            alpha_adjusted = alpha_base + risk_alpha_adjustment
        else:
            alpha_adjusted = alpha_base

        # 限制在[0, 1]
        alpha_final = torch.clamp(
            torch.tensor(alpha_adjusted, device=white_feats.device), min=0.0, max=1.0
        )

        # ========================================
        # 4. 残差组合（关键公式！）
        # ========================================
        # 正确：y = white + α * Δblack
        # 错误：y = white * (1-α) + black * α

        logit_hit = base_hit + alpha_final * delta_hit
        logit_hazard = base_hazard + alpha_final * delta_hazard
        logit_risk = base_risk + alpha_final * delta_risk

        # ========================================
        # 5. 返回完整分解（用于监控和审计）
        # ========================================
        return {
            # 最终输出
            "logit_hit": logit_hit,
            "logit_hazard": logit_hazard,
            "logit_risk": logit_risk,
            # 分解（用于RiskMonitor PnL归因）
            "base_hit": base_hit.detach(),  # 白盒贡献
            "delta_hit": delta_hit.detach(),  # 黑盒贡献
            # α记录（用于分析）
            "alpha": alpha_final.item(),
            "alpha_base": alpha_base,
            "alpha_adjust": (
                risk_alpha_adjustment if risk_alpha_adjustment is not None else 0.0
            ),
            # 元数据
            "regime": regime if regime is not None else "unknown",
        }


# ========================================
# 对比：评估报告中的错误公式
# ========================================


class WrongGatedCombine(nn.Module):
    """
    错误的加权平均公式（评估报告建议）

    问题：当α=0时，y_white也变成0了！
    应该是：完全忽略黑盒，但保留白盒基线
    """

    def forward(self, y_white, y_black, alpha):
        # ❌ 错误公式
        y_wrong = y_white * (1 - alpha) + y_black * alpha

        # 当α=0时：
        # y_wrong = y_white * 1.0 + y_black * 0 = y_white ✓

        # 当α=1时：
        # y_wrong = y_white * 0 + y_black * 1.0 = y_black
        # 问题：这破坏了残差结构！白盒基线被抹掉了

        return y_wrong


class CorrectResidualCombine(nn.Module):
    """
    正确的残差调节公式

    优势：始终保留白盒基线，只调节黑盒Delta的强度
    """

    def forward(self, y_white, delta_black, alpha):
        # ✅ 正确公式
        y_correct = y_white + alpha * delta_black

        # 当α=0时：
        # y_correct = y_white + 0 = y_white ✓ 完全信任白盒

        # 当α=1时：
        # y_correct = y_white + delta_black ✓ 白盒基线+黑盒全修正

        # 当α=0.5时：
        # y_correct = y_white + 0.5 * delta_black ✓ 部分修正

        return y_correct


# ========================================
# 集成示例：在trading层使用
# ========================================


class IntegratedTradingSystem:
    """
    完整的交易系统集成示例

    展示如何联动：
    1. RegimeDetector → 提供基础α
    2. RiskMonitor → 提供α调整
    3. RegimeAdaptiveResidualCombine → 最终预测
    """

    def __init__(self, model, regime_detector, risk_monitor):
        self.model = model
        self.regime_detector = regime_detector
        self.risk_monitor = risk_monitor

    def predict(self, white_feats, deep_factors, white_risk):
        """
        完整的预测流程

        Args:
            white_feats: 白盒特征
            deep_factors: 深度因子
            white_risk: 白盒风控指标（用于体制检测）

        Returns:
            model_output: 包含最终预测和元数据
        """
        # 1. 检测体制
        regime, confidence = self.regime_detector.detect(white_risk)

        # 2. 获取体制对应的基础α
        weights = self.regime_detector.get_alpha_weights(regime)
        alpha_base = weights["black_weight"]  # 黑盒权重即为α

        # 3. 获取风控调整
        alpha_adjustment = self.risk_monitor.alpha_adjustment

        # 4. 模型预测（带α调整）
        output = self.model(
            white_feats,
            deep_factors,
            regime=regime.value,
            regime_alpha=alpha_base,
            risk_alpha_adjustment=alpha_adjustment,
        )

        # 5. 更新RiskMonitor（用于下一次检测）
        self.risk_monitor.update(output)

        # 6. 返回预测结果
        return output


# ========================================
# 使用示例
# ========================================

if __name__ == "__main__":
    # 配置
    white_dim = 114  # HSI的白盒特征维度
    black_dim = 32  # 黑盒潜在因子维度
    batch_size = 16

    # 1. 创建模型
    model = RegimeAdaptiveResidualCombine(white_dim=white_dim, black_dim=black_dim)

    # 2. 模拟输入
    white_feats = torch.randn(batch_size, white_dim)
    deep_factors = torch.randn(batch_size, black_dim)

    # 3. 测试不同场景

    # 场景A：正常市场（α=0.5）
    output_normal = model(
        white_feats,
        deep_factors,
        regime="normal",
        regime_alpha=0.5,
        risk_alpha_adjustment=0.0,
    )
    print("=== Normal Market ===")
    print(f"Alpha: {output_normal['alpha']:.2f}")
    print(f"Base Hit: {output_normal['base_hit'][0].item():.3f}")
    print(f"Delta Hit: {output_normal['delta_hit'][0].item():.3f}")
    print(f"Final Logit: {output_normal['logit_hit'][0].item():.3f}")

    # 场景B：高波动期（α=0.3，降低黑盒）
    output_high_vol = model(
        white_feats,
        deep_factors,
        regime="high_volatility",
        regime_alpha=0.3,
        risk_alpha_adjustment=0.0,
    )
    print("\n=== High Volatility ===")
    print(f"Alpha: {output_high_vol['alpha']:.2f}")

    # 场景C：风控警报（α调整-0.3）
    output_risk_alert = model(
        white_feats,
        deep_factors,
        regime="normal",
        regime_alpha=0.5,
        risk_alpha_adjustment=-0.3,  # RiskMonitor降权
    )
    print("\n=== Risk Alert ===")
    print(f"Alpha Base: {output_risk_alert['alpha_base']:.2f}")
    print(f"Alpha Adjust: {output_risk_alert['alpha_adjust']:.2f}")
    print(f"Alpha Final: {output_risk_alert['alpha']:.2f}")

    # 场景D：熔断（α调整=-1.0）
    output_circuit_breaker = model(
        white_feats,
        deep_factors,
        regime="normal",
        regime_alpha=0.5,
        risk_alpha_adjustment=-1.0,  # 完全关闭黑盒
    )
    print("\n=== Circuit Breaker ===")
    print(f"Alpha: {output_circuit_breaker['alpha']:.2f}")  # 应为0.0
    print("Black box disabled!")

    # 验证残差公式
    print("\n=== Residual Formula Verification ===")
    base = output_normal["base_hit"][0].item()
    delta = output_normal["delta_hit"][0].item()
    alpha = output_normal["alpha"]
    final = output_normal["logit_hit"][0].item()

    expected = base + alpha * delta
    print(f"Base: {base:.3f}")
    print(f"Delta: {delta:.3f}")
    print(f"Alpha: {alpha:.2f}")
    print(f"Expected (base + α*delta): {expected:.3f}")
    print(f"Actual: {final:.3f}")
    print(f"Match: {abs(expected - final) < 1e-6}")
