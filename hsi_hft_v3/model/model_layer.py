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
# 导入配置常量
from hsi_hft_v3.config import BLACKBOX_DIM, LOOKBACK_BARS, K_BARS, ModelConfig

# 加载配置
model_cfg = ModelConfig()

# ==========================================
# 1. Mamba核心组件 (Selective State Space Model)
# ==========================================


class SelectiveSSM(nn.Module):
    """Selective State Space Model (Mamba核心)"""

    def __init__(self, d_model, d_state=None, expand=None):
        super().__init__()
        # 使用配置默认值 (如果未提供)
        self.d_model = d_model
        # 使用配置参数
        self.d_state = d_state if d_state else model_cfg.d_state
        self.expand = expand if expand else model_cfg.expand

        self.d_inner = d_model * self.expand

        # 输入投影 (In-projection)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)

        # x_proj 将 x 映射到 (dt, B, C)
        self.dt_rank = d_model // 16 if d_model // 16 > 0 else 1
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2)

        # 参数初始化
        # A: (d_inner, d_state)
        # ... (其余相同)
        self.A_log = nn.Parameter(
            torch.log(torch.randn(self.d_inner, self.d_state).abs())
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # dt 投影
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.act = nn.SiLU()

    def forward(self, u):
        # ... (forward logic same)
        B, T, D = u.shape

        # 1. 扩展 (Expand)
        u_inner = self.in_proj(u)
        x, z = u_inner.chunk(2, dim=-1)  # (B, T, d_inner)

        # 2. 动态投影 (选择机制)
        delta_bc = self.x_proj(x)

        # 分割参数
        delta_raw, B_ssm, C_ssm = torch.split(
            delta_bc, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # delta 激活 (Softplus)
        delta = F.softplus(self.dt_proj(delta_raw))  # (B, T, d_inner)

        # 3. 离散化 (Discretization)
        A = -torch.exp(self.A_log)  # A 必须为负

        # 4. 串行扫描 (纯 PyTorch 实现)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        y_list = []

        for t in range(T):
            # 时间步 t
            dt_t = delta[:, t, :]  # (B, d_inner)
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt_t, A))  # (B, d_inner, d_state)
            dB = torch.einsum(
                "bd,bn->bdn", dt_t, B_ssm[:, t, :]
            )  # (B, d_inner, d_state)

            xt = x[:, t, :]  # (B, d_inner)

            # 状态更新
            h = h * dA + xt.unsqueeze(-1) * dB

            # 输出
            Ct = C_ssm[:, t, :]  # (B, d_state)
            yt = torch.einsum("bdn,bn->bd", h, Ct)
            y_list.append(yt)

        y = torch.stack(y_list, dim=1)  # (B, T, d_inner)

        # 5. 残差 + 门控 (Residual + Gate)
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
    """Feature-wise Linear Modulation"""

    def __init__(self, d_model):
        super().__init__()
        self.scale = nn.Linear(d_model, d_model)
        self.shift = nn.Linear(d_model, d_model)

    def forward(self, tgt, aux, aux_mask):
        # 对 Aux 应用掩码
        aux_masked = aux * aux_mask
        gamma = self.scale(aux_masked)
        beta = self.shift(aux_masked)
        return tgt * (1 + gamma) + beta


# ==========================================
# 2. 深度因子挖掘器 (Deep Factor Miner)
# ==========================================


class DeepFactorMinerV5(nn.Module):
    """深度因子挖掘器 V5 (Configurable)"""

    def __init__(self, input_dim_raw, d_model=None, out_dim=BLACKBOX_DIM):
        super().__init__()

        self.d_model = d_model if d_model else model_cfg.d_model

        # Encoders
        self.encoder_tgt = LocalLOBEncoder(input_dim_raw, self.d_model)
        self.encoder_aux = LocalLOBEncoder(input_dim_raw, self.d_model)

        # SSM Backbone (Mamba-like)
        self.ssm_tgt = SelectiveSSM(self.d_model)
        self.ssm_aux = SelectiveSSM(self.d_model)

        # Fusion
        self.film = FiLMFusion(self.d_model)

        # Mixing
        self.mixer = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.GELU(),
            nn.Linear(self.d_model * 2, self.d_model),
        )

        # Projector (Information Bottleneck)
        proj_dim = model_cfg.projector_dim
        self.projector = nn.Sequential(
            nn.Linear(self.d_model, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, out_dim),  # 32-dim latent factors
        )

    def forward(self, x_tgt, x_aux, aux_mask, has_fut_mask=None):
        # 1. 局部编码 (Local Encode)
        e_tgt = self.encoder_tgt(x_tgt)
        e_aux = self.encoder_aux(x_aux)

        # 2. SSM 扫描 (序列建模)
        h_tgt = self.ssm_tgt(e_tgt)
        h_aux = self.ssm_aux(e_aux)

        # 3. FiLM 融合
        h_fused = self.film(h_tgt, h_aux, aux_mask)

        # 4. 混合 (Mixing)
        h_final = self.mixer(h_fused)

        # 5. 池化 (Pooling - 取最后状态)
        z_pool = h_final[:, -1, :]

        # 6. 投影 (Projection)
        deep_factors = self.projector(z_pool)  # (B, 32)

        return deep_factors


# ==========================================
# 3. 预测头 (Prediction Heads)
# ==========================================


class HitHead(nn.Module):
    """Hit预测头"""

    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x)


class HazardHead(nn.Module):
    """Hazard预测头"""

    def __init__(self, input_dim, k_bins=K_BARS):
        super().__init__()
        self.fc = nn.Linear(input_dim, k_bins)

    def forward(self, x):
        return self.fc(x)


class RiskHead(nn.Module):
    """Risk预测头"""

    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x)


class ResidualCombine(nn.Module):
    """体制自适应的残差融合模型"""

    def __init__(self, white_dim, black_dim=BLACKBOX_DIM, k_bars=K_BARS):
        super().__init__()

        # 白盒归一化
        self.white_norm = nn.BatchNorm1d(white_dim)

        # 白盒代理
        self.white_hit = nn.Linear(white_dim, 1)
        self.white_hazard = nn.Linear(white_dim, k_bars)
        self.white_risk = nn.Linear(white_dim, 1)

        # 黑盒修正
        self.delta_hit = nn.Linear(black_dim, 1)
        self.delta_hazard = nn.Linear(black_dim, k_bars)
        self.delta_risk = nn.Linear(black_dim, 1)

        # Delta初始化为0
        nn.init.zeros_(self.delta_hit.weight)
        nn.init.zeros_(self.delta_hazard.weight)
        nn.init.zeros_(self.delta_risk.weight)

        # 默认α配置
        self.default_alpha = model_cfg.default_alpha

    def forward(
        self,
        white_feats: torch.Tensor,
        deep_factors: torch.Tensor,
        regime: Optional[str] = None,
        regime_alpha: Optional[float] = None,
        risk_alpha_adjustment: Optional[float] = None,
    ) -> Dict:
        # ... (Same logic)
        # 0. 白盒归一化
        white_feats = self.white_norm(white_feats)

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

        # 4. 残差组合
        logit_hit = base_hit + alpha_final * delta_hit
        logit_hazard = base_hazard + alpha_final * delta_hazard
        logit_risk = base_risk + alpha_final * delta_risk

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


def vicreg_loss(z, batch_size=None):
    """VICReg 损失 (动态掩码版)"""
    # 方差 (Variance)
    std_z = torch.sqrt(z.var(dim=0) + 0.0001)
    std_loss = torch.mean(torch.relu(1 - std_z))

    # 协方差 (Covariance)
    bs = batch_size if batch_size else z.shape[0]
    dim = z.shape[1]

    z = z - z.mean(dim=0)
    cov_z = (z.T @ z) / (bs - 1)

    # 非对角元素 (动态掩码)
    # 创建对角掩码
    diag_mask = torch.eye(dim, device=z.device, dtype=torch.bool)
    off_diag = cov_z[~diag_mask]

    cov_loss = off_diag.pow(2).sum() / dim

    return std_loss, cov_loss
