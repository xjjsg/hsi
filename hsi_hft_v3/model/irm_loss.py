"""
HSI HFT V3 - IRM (不变风险最小化) 损失函数
技术实施方案

优先级：🟡 中（Tier 2，需重训练）状态：待实施
来源：第二份评估，优于第一份的T-VICReg

核心思想：
寻找在所有市场环境（低/中/高波动）下梯度方向一致的参数，
自动剔除只在特定体制有效的"伪因子"。

理论基础：
IRM (Invariant Risk Minimization) - 因果ML的SOTA方法
目标：min ∑Error_e + λ·||∇Error_e||²
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict


class IRMLoss(nn.Module):
    """
    不变风险最小化损失

    核心机制：
    1. 将数据按环境划分（低/中/高波动）
    2. 每个环境独立计算损失和梯度
    3. 惩罚跨环境的梯度不一致性
    4. 迫使模型学习环境不变的因果特征
    """

    def __init__(
        self, penalty_weight=1.0, penalty_anneal_epochs=10, compute_grads_every_k=1
    ):
        super().__init__()

        self.penalty_weight = penalty_weight
        self.penalty_anneal_epochs = penalty_anneal_epochs
        self.compute_grads_every_k = compute_grads_every_k

        self.epoch = 0
        self.grad_computation_count = 0

    def forward(
        self,
        model: nn.Module,
        data_by_env: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        base_loss_fn: nn.Module = nn.MSELoss(),
    ) -> Dict:
        """
        计算IRM损失

        Args:
            model: 要训练的模型
            data_by_env: {
                'low_vol': (X_low, y_low),
                'mid_vol': (X_mid, y_mid),
                'high_vol': (X_high, y_high)
            }
            base_loss_fn: 基础损失函数（MSE/BCE等）

        Returns:
            {
                'total_loss': 总损失,
                'env_losses': 各环境损失,
                'grad_penalty': 梯度惩罚,
                'current_penalty_weight': 当前惩罚权重
            }
        """
        env_losses = []
        env_names = []

        # ========================================
        # 1. 计算每个环境的损失
        # ========================================
        for env_name, (X, y) in data_by_env.items():
            pred = model(X)
            loss_e = base_loss_fn(pred, y)

            env_losses.append(loss_e)
            env_names.append(env_name)

        # 环境平均损失
        mean_env_loss = sum(env_losses) / len(env_losses)

        # ========================================
        # 2. 计算梯度惩罚（计算密集，不是每步都算）
        # ========================================

        if self.grad_computation_count % self.compute_grads_every_k == 0:
            grad_penalty = self._compute_grad_penalty(model, env_losses)
        else:
            # 复用上次的梯度惩罚（近似）
            grad_penalty = torch.tensor(0.0, device=env_losses[0].device)

        self.grad_computation_count += 1

        # ========================================
        # 3. Annealing：逐渐增加惩罚权重
        # ========================================
        # 前几个epoch让模型先学习基础模式，再强制不变性

        if self.epoch < self.penalty_anneal_epochs:
            current_penalty_weight = self.penalty_weight * (
                self.epoch / self.penalty_anneal_epochs
            )
        else:
            current_penalty_weight = self.penalty_weight

        # ========================================
        # 4. 总损失
        # ========================================
        total_loss = mean_env_loss + current_penalty_weight * grad_penalty

        return {
            "total_loss": total_loss,
            "mean_env_loss": mean_env_loss,
            "env_losses": {
                name: loss.item() for name, loss in zip(env_names, env_losses)
            },
            "grad_penalty": (
                grad_penalty.item()
                if isinstance(grad_penalty, torch.Tensor)
                else grad_penalty
            ),
            "current_penalty_weight": current_penalty_weight,
        }

    def _compute_grad_penalty(
        self, model: nn.Module, env_losses: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        计算跨环境的梯度惩罚

        核心思想：
        如果某个参数在环境A中梯度为正，在环境B中梯度为负，
        说明该参数依赖环境特定的模式（伪因子），应该惩罚。
        """
        env_grads = []

        # 获取模型参数
        params = [p for p in model.parameters() if p.requires_grad]

        # 计算每个环境的梯度
        for loss_e in env_losses:
            grads_e = torch.autograd.grad(
                loss_e, params, create_graph=True, retain_graph=True  # 允许二阶导数
            )
            env_grads.append(grads_e)

        # 计算梯度不一致性惩罚
        penalty = 0.0
        num_envs = len(env_grads)

        for i in range(num_envs):
            for j in range(i + 1, num_envs):
                # 对每个参数，计算环境i和环境j的梯度差
                for grad_i, grad_j in zip(env_grads[i], env_grads[j]):
                    penalty += torch.norm(grad_i - grad_j)

        # 归一化
        num_pairs = num_envs * (num_envs - 1) / 2
        penalty = penalty / num_pairs

        return penalty

    def step_epoch(self):
        """每个epoch结束时调用"""
        self.epoch += 1


# ========================================
# 环境划分策略
# ========================================


class EnvironmentSplitter:
    """
    将数据按市场环境划分

    策略：
    1. 按波动率划分（推荐）
    2. 按成交量划分
    3. 按时间段划分
    4. 自定义规则
    """

    @staticmethod
    def split_by_volatility(
        data: pd.DataFrame, returns_col="returns", thresholds=[0.01, 0.03], window=20
    ) -> Dict[str, pd.DataFrame]:
        """
        按波动率划分环境

        Args:
            data: DataFrame包含returns列
            returns_col: 收益率列名
            thresholds: [低波界限, 高波界限]
            window: 滚动窗口

        Returns:
            {
                'low_vol': 低波动数据,
                'mid_vol': 中等波动数据,
                'high_vol': 高波动数据
            }
        """
        # 计算滚动波动率
        vol = data[returns_col].rolling(window).std()
        data = data.copy()
        data["volatility"] = vol

        # 划分
        low_vol = data[vol < thresholds[0]].copy()
        mid_vol = data[(vol >= thresholds[0]) & (vol < thresholds[1])].copy()
        high_vol = data[vol >= thresholds[1]].copy()

        return {"low_vol": low_vol, "mid_vol": mid_vol, "high_vol": high_vol}

    @staticmethod
    def split_by_time_period(
        data: pd.DataFrame, time_col="timestamp"
    ) -> Dict[str, pd.DataFrame]:
        """
        按时间段划分环境

        适用场景：避免时序泄露，确保环境独立

        Args:
            data: DataFrame包含时间列
            time_col: 时间戳列名

        Returns:
            {
                'env_1': 第一时间段,
                'env_2': 第二时间段,
                'env_3': 第三时间段
            }
        """
        data = data.sort_values(time_col)
        n = len(data)

        env_1 = data.iloc[: n // 3]
        env_2 = data.iloc[n // 3 : 2 * n // 3]
        env_3 = data.iloc[2 * n // 3 :]

        return {"env_1": env_1, "env_2": env_2, "env_3": env_3}


# ========================================
# 训练流程集成
# ========================================


def train_with_irm(model, train_data, val_data, epochs=50, device="cpu"):
    """
    使用IRM损失训练模型

    Steps:
    1. 划分环境
    2. 为每个环境创建DataLoader（或使用批次采样）
    3. IRM损失计算
    4. 反向传播
    """

    # 1. 划分训练数据为环境
    train_envs = EnvironmentSplitter.split_by_volatility(train_data)

    print("Environment Statistics:")
    for env_name, env_data in train_envs.items():
        print(f"  {env_name}: {len(env_data)} samples")

    # 2. 准备优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    irm_loss = IRMLoss(penalty_weight=1.0, penalty_anneal_epochs=10)
    base_loss_fn = nn.MSELoss()

    # 3. 训练循环
    for epoch in range(epochs):
        model.train()

        # 从每个环境采样batch
        data_by_env = {}
        for env_name, env_data in train_envs.items():
            # 简化：每个epoch随机采样一个batch
            batch_size = min(128, len(env_data))
            indices = np.random.choice(len(env_data), batch_size, replace=False)

            # 假设env_data有X和y
            X = torch.tensor(env_data.iloc[indices]["features"].values, device=device)
            y = torch.tensor(env_data.iloc[indices]["target"].values, device=device)

            data_by_env[env_name] = (X, y)

        # IRM损失计算
        optimizer.zero_grad()

        loss_dict = irm_loss(
            model=model, data_by_env=data_by_env, base_loss_fn=base_loss_fn
        )

        total_loss = loss_dict["total_loss"]
        total_loss.backward()
        optimizer.step()

        # 打印进度
        if epoch % 5 == 0:
            print(f"\nEpoch {epoch}/{epochs}")
            print(f"Total Loss: {total_loss.item():.4f}")
            print(f"Env Losses: {loss_dict['env_losses']}")
            print(f"Grad Penalty: {loss_dict['grad_penalty']:.4f}")
            print(f"Penalty Weight: {loss_dict['current_penalty_weight']:.2f}")

        irm_loss.step_epoch()

    return model


# ========================================
# 与T-VICReg对比
# ========================================


def compare_irm_vs_tvicreg():
    """
    IRM vs T-VICReg 对比分析
    """
    comparison = {
        "IRM": {
            "目标": "跨环境梯度一致性",
            "优化对象": "模型参数的因果性",
            "理论基础": "因果ML，寻找不变预测",
            "计算复杂度": "高（需要二阶导数）",
            "适用场景": "环境明确可划分",
            "优势": "自动剔除伪因子",
            "劣势": "训练慢，超参敏感",
        },
        "T-VICReg [Research]": {
            "目标": "跨环境协方差一致性",
            "优化对象": "表示的统计结构",
            "理论基础": "自监督学习，表示解耦",
            "计算复杂度": "中（只需一阶导数）",
            "适用场景": "预训练阶段",
            "优势": "训练快，稳定",
            "劣势": "理论不如IRM强",
        },
    }

    print("=== IRM vs T-VICReg ===")
    for method, props in comparison.items():
        print(f"\n{method}:")
        for key, val in props.items():
            print(f"  {key}: {val}")


# ========================================
# 使用建议
# ========================================

"""
何时使用IRM：
1. 数据量充足（>10k样本）
2. 环境划分明确（波动率、时间段等）
3. 追求模型鲁棒性，可以接受训练慢

何时使用T-VICReg：
1. 预训练阶段
2. 数据量有限
3. 追求训练效率

推荐策略：
- 预训练：T-VICReg
- 微调：IRM（在标注数据上）
"""


if __name__ == "__main__":
    # 演示IRM损失计算

    # 1. 模拟模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 1)

        def forward(self, x):
            return self.fc(x)

    model = SimpleModel()

    # 2. 模拟多环境数据
    data_by_env = {
        "low_vol": (torch.randn(32, 10), torch.randn(32, 1)),
        "mid_vol": (torch.randn(32, 10), torch.randn(32, 1)),
        "high_vol": (torch.randn(32, 10), torch.randn(32, 1)),
    }

    # 3. 计算IRM损失
    irm = IRMLoss(penalty_weight=1.0)

    loss_dict = irm(model, data_by_env)

    print("=== IRM Loss Computation ===")
    print(f"Total Loss: {loss_dict['total_loss'].item():.4f}")
    print(f"Env Losses: {loss_dict['env_losses']}")
    print(f"Grad Penalty: {loss_dict['grad_penalty']:.4f}")

    # 4. 对比分析
    print("\n")
    compare_irm_vs_tvicreg()
