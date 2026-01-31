"""
HSI HFT V3 - 可微分特征工程
技术实施方案

优先级：🟠 高（Tier 1）
状态：待实施
来源：第二份评估的新亮点

核心思想：
让whitebox.py中的硬编码权重变成nn.Parameter，
通过反向传播自动学习最优权重，同时保留可解释性。

简单有效：改动量小，收益大
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
from collections import deque


class LearnableDepthWeights(nn.Module):
    """
    可学习的档位权重

    原代码（whitebox.py）：
    weights = [1.0, 0.8, 0.6, 0.4, 0.2]  # 硬编码

    改进：
    self.weights = nn.Parameter(torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2]))
    """

    def __init__(self, num_levels=5):
        super().__init__()

        # 初始化为经验权重
        init_weights = torch.linspace(1.0, 0.2, num_levels)
        self.weights = nn.Parameter(init_weights)

    def forward(self):
        """返回归一化后的权重"""
        # Softmax归一化（确保和为1）
        return torch.softmax(self.weights, dim=0)

    def get_raw_weights(self):
        """返回原始权重（用于分析）"""
        return self.weights.detach().cpu().numpy()


class LearnableDecayRate(nn.Module):
    """
    可学习的时间衰减率

    原代码：
    decay_rate = 0.5  # 固定

    改进：
    让模型学习最优的衰减速度
    """

    def __init__(self, init_rate=0.5):
        super().__init__()

        # 用sigmoid确保在(0, 1)
        self.logit_rate = nn.Parameter(
            torch.tensor(np.log(init_rate / (1 - init_rate)))
        )

    def forward(self):
        """返回(0, 1)范围的衰减率"""
        return torch.sigmoid(self.logit_rate)


class LearnableWindowScales(nn.Module):
    """
    可学习的滚动窗口大小

    原代码（whitebox.py）：
    W_set = [20, 100, 600]  # 固定

    改进：
    让模型微调窗口大小（在合理范围内）
    """

    def __init__(self, base_windows=[20, 100, 600]):
        super().__init__()

        # 学习偏移量（±20%）
        self.base_windows = torch.tensor(base_windows, dtype=torch.float32)
        self.scale_factors = nn.Parameter(torch.ones(len(base_windows)))

    def forward(self):
        """返回调整后的窗口大小"""
        # 限制scale在[0.8, 1.2]
        scales = torch.clamp(self.scale_factors, 0.8, 1.2)
        windows = self.base_windows * scales
        return windows.int()


class LearnableWhiteBoxFactory(nn.Module):
    """
    可微分的白盒特征工厂（部分可学习）

    策略：
    阶段1：只让关键权重可学习（推荐）
    - 档位权重
    - 时间衰减率
    - （可选）窗口大小微调

    阶段2：全可微（需谨慎）
    - 所有参数可学习
    - 风险：可能破坏可解释性
    """

    def __init__(self, full_differentiable=False):
        super().__init__()

        self.full_differentiable = full_differentiable

        # ========================================
        # 可学习的参数
        # ========================================

        # 1. 档位权重
        self.depth_weights = LearnableDepthWeights(num_levels=5)

        # 2. 时间衰减率
        self.decay_rate = LearnableDecayRate(init_rate=0.5)

        # 3. 窗口大小（可选）
        if full_differentiable:
            self.window_scales = LearnableWindowScales([20, 100, 600])
        else:
            # 固定窗口
            self.window_set = [20, 100, 600]

    def compute_ofi(self, bids, asks, use_learnable=True):
        """
        计算订单流不平衡（OFI）

        Args:
            bids: List[(price, volume)] 买盘
            asks: List[(price, volume)] 卖盘
            use_learnable: 是否使用可学习权重

        Returns:
            ofi: 加权订单流不平衡
        """
        if use_learnable:
            weights = self.depth_weights()
        else:
            weights = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2])

        # 计算OFI
        bid_weighted = sum(w * vol for w, (_, vol) in zip(weights, bids[:5]))
        ask_weighted = sum(w * vol for w, (_, vol) in zip(weights, asks[:5]))

        ofi = (bid_weighted - ask_weighted) / (bid_weighted + ask_weighted + 1e-9)
        return ofi

    def get_learnable_params_summary(self) -> Dict:
        """返回可学习参数的当前值"""
        summary = {
            "depth_weights": self.depth_weights.get_raw_weights(),
            "decay_rate": self.decay_rate().item(),
        }

        if self.full_differentiable:
            summary["window_scales"] = self.window_scales().cpu().numpy()

        return summary


# ========================================
# 训练时的正则化
# ========================================


class LearnableParamsRegularization:
    """
    防止可学习参数diverge的正则化

    策略：
    1. L2正则（权重不要偏离初始值太远）
    2. 单调性约束（档位权重应该递减）
    3. 范围约束（已通过nn.Parameter的限制实现）
    """

    @staticmethod
    def l2_regularization(model: LearnableWhiteBoxFactory, weight=0.01):
        """L2正则：惩罚权重偏离初始值"""
        reg_loss = 0.0

        # 档位权重正则
        init_depth_weights = torch.linspace(1.0, 0.2, 5)
        reg_loss += weight * torch.norm(
            model.depth_weights.weights - init_depth_weights
        )

        # 衰减率正则
        init_decay = 0.5
        reg_loss += weight * (model.decay_rate() - init_decay) ** 2

        return reg_loss

    @staticmethod
    def monotonicity_constraint(model: LearnableWhiteBoxFactory, weight=0.1):
        """
        单调性约束：档位权重应该递减

        惩罚违反w[i] > w[i+1]的情况
        """
        weights = model.depth_weights.weights

        # 计算相邻权重差
        diffs = weights[:-1] - weights[1:]

        # 惩罚负差值（违反递减）
        violations = torch.relu(-diffs)  # 负差值变正，正差值为0

        return weight * violations.sum()


# ========================================
# 集成到训练流程
# ========================================


def train_with_learnable_features(model, dataloader, optimizer, epochs=10):
    """
    训练时同时优化模型和白盒参数

    Args:
        model: 包含LearnableWhiteBoxFactory的模型
        dataloader: 训练数据
        optimizer: 优化器（包含白盒参数）
        epochs: 训练轮数
    """

    for epoch in range(epochs):
        total_loss = 0

        for batch in dataloader:
            optimizer.zero_grad()

            # 前向传播
            output = model(batch)

            # 任务损失（如MSE）
            task_loss = compute_task_loss(output, batch["target"])

            # 正则化损失
            reg_loss = LearnableParamsRegularization.l2_regularization(
                model.white_box_factory, weight=0.01
            )
            reg_loss += LearnableParamsRegularization.monotonicity_constraint(
                model.white_box_factory, weight=0.1
            )

            # 总损失
            loss = task_loss + reg_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 每个epoch后打印参数
        if epoch % 2 == 0:
            print(f"\nEpoch {epoch}")
            print(f"Loss: {total_loss:.4f}")
            print("Learnable Params:")
            summary = model.white_box_factory.get_learnable_params_summary()
            for key, val in summary.items():
                print(f"  {key}: {val}")


# ========================================
# 可解释性分析
# ========================================


def analyze_learned_weights(model: LearnableWhiteBoxFactory):
    """
    分析学习到的权重是否合理

    检查：
    1. 档位权重是否单调递减
    2. 衰减率是否在合理范围
    3. 窗口大小是否合理
    """
    summary = model.get_learnable_params_summary()

    print("=== Learned Parameters Analysis ===")

    # 1. 档位权重
    depth_weights = summary["depth_weights"]
    print(f"\nDepth Weights: {depth_weights}")

    is_monotonic = all(
        depth_weights[i] >= depth_weights[i + 1] for i in range(len(depth_weights) - 1)
    )
    print(f"Monotonic: {is_monotonic} {'✓' if is_monotonic else '✗'}")

    # 2. 衰减率
    decay = summary["decay_rate"]
    print(f"\nDecay Rate: {decay:.3f}")
    print(
        f"Reasonable (0.3-0.7): {0.3 <= decay <= 0.7} {'✓' if 0.3 <= decay <= 0.7 else '✗'}"
    )

    # 3. 窗口大小（如果有）
    if "window_scales" in summary:
        windows = summary["window_scales"]
        print(f"\nWindow Scales: {windows}")


# ========================================
# 使用示例
# ========================================

if __name__ == "__main__":
    # 1. 创建可微分白盒工厂
    white_box = LearnableWhiteBoxFactory(full_differentiable=False)

    # 2. 查看初始参数
    print("=== Initial Parameters ===")
    print(white_box.get_learnable_params_summary())

    # 3. 模拟训练（参数会更新）
    optimizer = torch.optim.Adam(white_box.parameters(), lr=0.01)

    for step in range(10):
        # 模拟损失
        loss = torch.randn(1, requires_grad=True).sum()

        # 加正则
        loss = loss + LearnableParamsRegularization.l2_regularization(white_box)
        loss = loss + LearnableParamsRegularization.monotonicity_constraint(white_box)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 4. 查看学习后的参数
    print("\n=== After Training ===")
    analyze_learned_weights(white_box)

    # 5. 使用学习到的权重计算OFI
    bids = [(4.50, 10000), (4.49, 8000), (4.48, 6000), (4.47, 5000), (4.46, 4000)]
    asks = [(4.51, 9000), (4.52, 7000), (4.53, 5500), (4.54, 4500), (4.55, 3500)]

    ofi = white_box.compute_ofi(bids, asks, use_learnable=True)
    print(f"\nComputed OFI: {ofi:.4f}")
