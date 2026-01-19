#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HSI HFT V3 - Model Layer (Consolidated Module)

整合模块：
- features/blackbox.py - 深度因子挖掘器 (Mamba架构)
- models/heads.py - 预测头 (Hit/Hazard/Risk)

功能：
1. 深度因子挖掘：Dual-Stream Mamba + FiLM融合 + VICReg正则
2. 预测头：Hit (是否命中)、Hazard (时间分布)、Risk (风险评估)
3. 残差结合：白盒基线 + 黑盒修正
4. VICReg损失：方差-不变性-协方差正则化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

# 导入配置常量
from hsi_hft_v3.trading_layer import BLACKBOX_DIM, LOOKBACK_BARS, K_BARS


# ==========================================
# 1. Mamba核心组件 (Selective State Space Model)
# ==========================================


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (Mamba核心)

    纯PyTorch实现，支持Windows/CPU环境
    状态转移: h_t = A_bar * h_{t-1} + B_bar * x_t
    输出映射: y_t = C_t * h_t
    """

    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state

        # In-projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)

        # x_proj takes x to (dt, B, C)
        self.dt_rank = d_model // 16 if d_model // 16 > 0 else 1
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2)

        # Parameters
        # A: (d_inner, d_state)
        self.A_log = nn.Parameter(torch.log(torch.randn(self.d_inner, d_state).abs()))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # dt proj
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.act = nn.SiLU()

    def forward(self, u):
        """
        前向传播

        Args:
            u: (B, T, D) 输入序列

        Returns:
            (B, T, D) 输出序列
        """
        # u: (B, T, D)
        B, T, D = u.shape

        # 1. Expand
        u_inner = self.in_proj(u)
        x, z = u_inner.chunk(2, dim=-1)  # (B, T, d_inner)

        # 2. Dynamic Projections (Selection Mechanism)
        delta_bc = self.x_proj(x)

        # Split
        delta_raw, B_ssm, C_ssm = torch.split(
            delta_bc, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # Softplus delta
        delta = F.softplus(self.dt_proj(delta_raw))  # (B, T, d_inner)

        # 3. Discretization
        A = -torch.exp(self.A_log)  # A must be negative

        # 4. Sequential Scan (Pure PyTorch)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        y_list = []

        for t in range(T):
            # Time step t
            dt_t = delta[:, t, :]  # (B, d_inner)
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt_t, A))  # (B, d_inner, d_state)
            dB = torch.einsum(
                "bd,bn->bdn", dt_t, B_ssm[:, t, :]
            )  # (B, d_inner, d_state)

            xt = x[:, t, :]  # (B, d_inner)

            # State update
            h = h * dA + xt.unsqueeze(-1) * dB

            # Output
            Ct = C_ssm[:, t, :]  # (B, d_state)
            yt = torch.einsum("bdn,bn->bd", h, Ct)
            y_list.append(yt)

        y = torch.stack(y_list, dim=1)  # (B, T, d_inner)

        # 5. Residual + Gate
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        y = y * self.act(z)

        return self.out_proj(y)


class LocalLOBEncoder(nn.Module):
    """局部LOB编码器 (1D卷积)"""

    def __init__(self, input_dim, d_model):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(d_model)

    def forward(self, x):
        """
        Args:
            x: (B, T, F) 输入特征

        Returns:
            (B, T, D) 编码后的特征
        """
        # x: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x.transpose(1, 2)  # (B, T, D)


class FiLMFusion(nn.Module):
    """
    Feature-wise Linear Modulation

    使用辅助流调制目标流：
    output = target * (1 + scale(aux)) + shift(aux)
    """

    def __init__(self, d_model):
        super().__init__()
        self.scale = nn.Linear(d_model, d_model)
        self.shift = nn.Linear(d_model, d_model)

    def forward(self, tgt, aux, aux_mask):
        """
        Args:
            tgt: (B, T, D) 目标流
            aux: (B, T, D) 辅助流
            aux_mask: (B, T, 1) 辅助流可用性掩码

        Returns:
            (B, T, D) 融合后的特征
        """
        # Apply mask to aux
        aux_masked = aux * aux_mask
        gamma = self.scale(aux_masked)
        beta = self.shift(aux_masked)
        return tgt * (1 + gamma) + beta


# ==========================================
# 2. 深度因子挖掘器 (Deep Factor Miner)
# ==========================================


class DeepFactorMinerV5(nn.Module):
    """
    深度因子挖掘器 V5

    架构：
    1. 双流编码器 (Target + Aux)
    2. Mamba序列建模
    3. FiLM跨流融合
    4. 投影器 -> 32维潜在因子

    训练时使用VICReg损失进行自监督预训练
    """

    def __init__(self, input_dim_raw, d_model=64, out_dim=BLACKBOX_DIM):
        super().__init__()
        self.d_model = d_model

        # Encoders
        self.encoder_tgt = LocalLOBEncoder(input_dim_raw, d_model)
        self.encoder_aux = LocalLOBEncoder(input_dim_raw, d_model)

        # SSM Backbone (Mamba-like)
        self.ssm_tgt = SelectiveSSM(d_model)
        self.ssm_aux = SelectiveSSM(d_model)

        # Fusion
        self.film = FiLMFusion(d_model)

        # Mixing
        self.mixer = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model)
        )

        # Projector (Information Bottleneck)
        self.projector = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, out_dim),  # 32-dim latent factors
        )

    def forward(self, x_tgt, x_aux, aux_mask, has_fut_mask=None):
        """
        前向传播

        Args:
            x_tgt: (B, T, F) 目标流原始特征
            x_aux: (B, T, F) 辅助流原始特征
            aux_mask: (B, T, 1) 辅助流可用性掩码
            has_fut_mask: (可选) 期货数据可用性掩码

        Returns:
            (B, 32) 深度因子
        """
        # 1. Local Encode
        e_tgt = self.encoder_tgt(x_tgt)
        e_aux = self.encoder_aux(x_aux)

        # 2. SSM Scan (Sequence Modeling)
        h_tgt = self.ssm_tgt(e_tgt)
        h_aux = self.ssm_aux(e_aux)

        # 3. FiLM Fusion
        h_fused = self.film(h_tgt, h_aux, aux_mask)

        # 4. Mixing
        h_final = self.mixer(h_fused)

        # 5. Pooling (Last state)
        z_pool = h_final[:, -1, :]

        # 6. Projection
        deep_factors = self.projector(z_pool)  # (B, 32)

        return deep_factors


# ==========================================
# 3. 预测头 (Prediction Heads)
# ==========================================


class HitHead(nn.Module):
    """Hit预测头: P(Hit) - 二分类"""

    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        """返回logit"""
        return self.fc(x)


class HazardHead(nn.Module):
    """Hazard预测头: P(T=k | T>=k) - 离散时间生存分析"""

    def __init__(self, input_dim, k_bins=K_BARS):
        super().__init__()
        self.fc = nn.Linear(input_dim, k_bins)

    def forward(self, x):
        """返回每个时间bin的logit"""
        return self.fc(x)


class RiskHead(nn.Module):
    """Risk预测头: P(Adverse | Adverse or Hit) - 竞争风险"""

    def __init__(self, input_dim):
        super().__init__()
        # Simplified Risk Head: P(Adverse Before Hit)
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        """返回logit"""
        return self.fc(x)


class ResidualCombine(nn.Module):
    """
    体制自适应的残差融合模型（优化版）

    改进：
    1. 支持规则驱动的α调整（来自RegimeDetector + RiskMonitor）
    2. 正确的残差公式：y = white + α * Δblack
    3. 返回完整分解用于风控监控

    最终输出 = 白盒基线 + α调整的黑盒修正
    """

    def __init__(self, white_dim, black_dim=BLACKBOX_DIM, k_bars=K_BARS):
        super().__init__()

        # 白盒代理（可解释基线）
        self.white_hit = nn.Linear(white_dim, 1)
        self.white_hazard = nn.Linear(white_dim, k_bars)
        self.white_risk = nn.Linear(white_dim, 1)

        # 黑盒修正（深度因子Delta）
        self.delta_hit = nn.Linear(black_dim, 1)
        self.delta_hazard = nn.Linear(black_dim, k_bars)
        self.delta_risk = nn.Linear(black_dim, 1)

        # Delta初始化为0（从白盒基线开始）
        nn.init.zeros_(self.delta_hit.weight)
        nn.init.zeros_(self.delta_hazard.weight)
        nn.init.zeros_(self.delta_risk.weight)

        # 默认α配置
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
            regime: 当前体制（可选）
            regime_alpha: RegimeDetector提供的基础α
            risk_alpha_adjustment: RiskMonitor提供的α调整

        Returns:
            包含最终预测、分解和α信息的Dict
        """
        # 1. 白盒基线
        base_hit = self.white_hit(white_feats)
        base_hazard = self.white_hazard(white_feats)
        base_risk = self.white_risk(white_feats)

        # 2. 黑盒Delta
        delta_hit = self.delta_hit(deep_factors)
        delta_hazard = self.delta_hazard(deep_factors)
        delta_risk = self.delta_risk(deep_factors)

        # 3. 计算最终α
        alpha_base = regime_alpha if regime_alpha is not None else self.default_alpha

        if risk_alpha_adjustment is not None:
            alpha_adjusted = alpha_base + risk_alpha_adjustment
        else:
            alpha_adjusted = alpha_base

        # 限制在[0, 1]
        alpha_final = torch.clamp(
            torch.tensor(alpha_adjusted, device=white_feats.device), min=0.0, max=1.0
        )

        # 4. 残差组合（正确公式：y = white + α * Δblack）
        logit_hit = base_hit + alpha_final * delta_hit
        logit_hazard = base_hazard + alpha_final * delta_hazard
        logit_risk = base_risk + alpha_final * delta_risk

        # 5. 返回完整信息
        return {
            "logit_hit": logit_hit,
            "logit_hazard": logit_hazard,
            "logit_risk": logit_risk,
            "base_hit": base_hit.detach(),
            "delta_hit": delta_hit.detach(),
            "alpha": (
                alpha_final.item()
                if isinstance(alpha_final, torch.Tensor)
                else alpha_final
            ),
            "alpha_base": alpha_base,
            "alpha_adjust": (
                risk_alpha_adjustment if risk_alpha_adjustment is not None else 0.0
            ),
            "regime": regime if regime is not None else "unknown",
        }


# ==========================================
# 4. 损失函数 (Loss Functions)
# ==========================================


def vicreg_loss(z, batch_size):
    """
    VICReg损失 (Variance-Invariance-Covariance Regularization)

    用于自监督预训练，确保潜在表示：
    1. Variance: 各维度方差足够大
    2. Covariance: 各维度尽可能独立

    Args:
        z: (B, D) 潜在表示
        batch_size: batch大小

    Returns:
        (std_loss, cov_loss)
    """
    # Variance: 惩罚方差小于1的维度
    std_z = torch.sqrt(z.var(dim=0) + 0.0001)
    std_loss = torch.mean(torch.relu(1 - std_z))

    # Covariance: 惩罚非对角元素
    z = z - z.mean(dim=0)
    cov_z = (z.T @ z) / (batch_size - 1)
    # Off-diagonal elements
    off_diag = cov_z.flatten()[:-1].view(31, 33)[:, 1:].flatten()
    cov_loss = off_diag.pow(2).sum() / 32

    return std_loss, cov_loss
