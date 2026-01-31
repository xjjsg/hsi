#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HSI HFT V3 - Trading Layer (Consolidated Module)

整合模块：
- core/config.py - 全局配置和策略参数
- policy/state_machine.py - 状态机交易策略
- backtest/engine.py - 回测引擎
- backtest/metrics.py - 性能指标计算

功能：
1. 配置管理：全局参数、执行配置、策略阈值
2. 状态机策略：FLAT/LONG/COOLDOWN 三态决策
3. 回测引擎：事件驱动回测，支持延迟模拟
4. 性能指标：PnL、夏普、回撤等评估指标
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Tuple, Deque
from collections import deque

# 导入数据层的依赖（需要 AlignedSample）
from .data_layer import AlignedSample

# ==========================================
# 1. 全局配置 (Global Configuration)
# ==========================================
from hsi_hft_v3.config import (
    TARGET_SYMBOL,
    AUX_SYMBOL,
    BAR_SIZE_S,
    TIMEZONE,
    COST_RATE,
    TICK_SIZE,
    LATENCY_BARS,
    TRADE_QTY,
    ExecutionConfig,
    LabelConfig,
    PolicyConfig,
    ALLOWLIST_FIELDS,
    BLOCKLIST_FIELDS,
    PREDICT_HORIZON_S,
    K_BARS,
    LOOKBACK_BARS,
    BLACKBOX_DIM,
)


# ==========================================
# 2. 状态机策略 (State Machine Policy)
# ==========================================


class State(Enum):
    """Trading states"""

    FLAT = 0
    LONG = 1
    COOLDOWN = 2


class Action(Enum):
    """Trading actions"""

    HOLD = 0
    ENTER_LONG = 1
    EXIT_LONG = 2


@dataclass
class PolicyOutput:
    """Output from policy decision"""

    action: Action
    reason: str
    meta: Dict


class StateMachine:
    """
    状态机交易策略

    状态 (States):
    - FLAT: 空仓，寻找入场信号
    - LONG: 持有多头仓位，监控出场条件
    - COOLDOWN: 交易后冷却期

    转换 (Transitions):
    FLAT -> LONG: 入场信号 (p_hit, p_tau, 风控通过)
    LONG -> COOLDOWN: 出场信号 (最大持有时间, 风控触发, 或逆向条件)
    COOLDOWN -> FLAT: 冷却期结束
    """

    def __init__(self, config: PolicyConfig):
        self.cfg = config
        self.state = State.FLAT
        self.pending_action = None  # Track if we are waiting for execution
        self.hold_bars = 0
        self.cooldown_rem = 0
        self.entry_price = 0.0

    def decide(
        self,
        model_probs: Dict[str, float],
        white_risk: Dict[str, float],
        aux_available: bool,
    ) -> PolicyOutput:
        """
        主要决策函数

        Args:
            model_probs: 模型预测 (p_hit, P_tau_le_60s, risk)
            white_risk: 白盒风控指标 (spread_bps, vpin_z, etc.)
            aux_available: 是否有辅助数据

        Returns:
            PolicyOutput 包含动作和原因
        """

        # 0. 检查挂单
        if self.pending_action is not None:
            return PolicyOutput(Action.HOLD, "PENDING_EXEC", {"state": self.state.name})

        # 1. 更新冷却状态
        if self.state == State.COOLDOWN:
            self.cooldown_rem -= 1
            if self.cooldown_rem <= 0:
                self.state = State.FLAT
                return PolicyOutput(Action.HOLD, "COOLDOWN_END", {})
            return PolicyOutput(Action.HOLD, "IN_COOLDOWN", {})

        # 2. 选择阈值 (根据辅助数据可用性自适应)
        if aux_available:
            th_hit = self.cfg.th_p_hit
            th_tau = self.cfg.th_p_tau_60
            th_risk = self.cfg.th_risk_enter
        else:
            th_hit = self.cfg.th_p_hit_noaux
            th_tau = self.cfg.th_p_tau_60_noaux
            th_risk = self.cfg.th_risk_enter_noaux

        # 3. 提取概率
        p_hit = model_probs.get("p_hit", 0.0)
        p_tau = model_probs.get("P_tau_le_60s", 0.0)
        p_risk = model_probs.get("risk", 1.0)  # 默认高风险

        meta = {
            "p_hit": p_hit,
            "p_tau": p_tau,
            "p_risk": p_risk,
            "state": self.state.name,
        }

        # 4. 出场逻辑 (EXIT LOGIC)
        if self.state == State.LONG:
            self.hold_bars += 1

            # 最大持有时间
            if self.hold_bars >= self.cfg.max_hold_bars:
                return PolicyOutput(Action.EXIT_LONG, "MAX_HOLD", meta)

            # 风控门槛
            if p_risk > self.cfg.th_risk_exit:
                return PolicyOutput(Action.EXIT_LONG, "RISK_EXIT", meta)

            # 白盒反转 / 微观风控 (简化版)
            if white_risk.get("vpin_z", 0) > self.cfg.vpin_exit_z:
                return PolicyOutput(Action.EXIT_LONG, "VPIN_EXIT", meta)

            if white_risk.get("kyle_z", 0) > self.cfg.lambda_exit_z:
                return PolicyOutput(Action.EXIT_LONG, "KYLE_EXIT", meta)

            return PolicyOutput(Action.HOLD, "HOLD_LONG", meta)

        # 5. 入场逻辑 (ENTRY LOGIC - FLAT)
        if self.state == State.FLAT:
            # 白盒门槛 (Entry)
            if white_risk.get("spread_bps", 0) > self.cfg.spread_max_bps:
                return PolicyOutput(Action.HOLD, "SPREAD_GATE", meta)

            if white_risk.get("depth_qty", 999999) < self.cfg.depth_min_qty:
                return PolicyOutput(Action.HOLD, "DEPTH_GATE", meta)

            if white_risk.get("vpin_z", 0) > self.cfg.vpin_max_z:
                return PolicyOutput(Action.HOLD, "VPIN_GATE", meta)

            # 信号判断
            if p_hit >= th_hit and p_tau >= th_tau and p_risk <= th_risk:
                # 此时不立即转换状态，等待执行回调
                return PolicyOutput(Action.ENTER_LONG, "SIGNAL_ENTRY", meta)

            return PolicyOutput(Action.HOLD, "NO_SIGNAL", meta)

        return PolicyOutput(Action.HOLD, "UNKNOWN_STATE", meta)

    # Callbacks for execution events
    def on_order_sent(self, action: Action):
        """Called when order is sent"""
        self.pending_action = action

    def on_fill(self, action: Action):
        """Called when order is filled"""
        self.pending_action = None
        if action == Action.ENTER_LONG:
            self.state = State.LONG
            self.hold_bars = 0

        elif action == Action.EXIT_LONG:
            self._transition_to_cooldown()

    def on_reject(self, action: Action):
        """Called when order is rejected"""
        self.pending_action = None
        # Stay in current state (Retry next bar)
        # Note: If EXIT rejected (liquidity dryup), we are stuck in LONG.
        # Strategy will keep trying to Exit.
        pass

    def _transition_to_cooldown(self):
        """Transition to cooldown state"""
        self.state = State.COOLDOWN
        self.cooldown_rem = self.cfg.cooldown_bars


# ==========================================
# 3. 回测引擎 (Backtest Engine)
# ==========================================


class Order:
    """Order representation"""

    def __init__(self, action: str, create_time: int, trade_time: int, qty: int):
        self.action = action
        self.create_time = create_time
        self.trade_time = trade_time
        self.qty = qty
        self.is_filled = False


class BacktestEngine:
    """
    事件驱动的回测引擎，支持延迟模拟

    特性:
    - 延迟队列模拟真实的订单执行
    - L1 Taker 执行 (IOC @ 最佳买卖价)
    - 流动性检查 (全部成交或不成交)
    - 逐日盯市 (Mark-to-market) 权益跟踪
    - RiskMonitor 集成 (NEW)
    """

    def __init__(self, policy: StateMachine, baseline_stats=None):
        self.policy = policy
        self.latency_queue: Deque[Order] = deque()
        self.trades = []
        self.cash = 1_000_000.0  # Initial Capital
        self.position = 0
        self.equity_curve = []

        # NEW: RiskMonitor集成
        if baseline_stats:
            from hsi_hft_v3.risk_monitor import RiskMonitor

            self.risk_monitor = RiskMonitor(baseline_stats, window_size=60)
            print("✅ RiskMonitor initialized with baseline stats")
        else:
            self.risk_monitor = None
            print("⚠️ RiskMonitor disabled (no baseline_stats provided)")

    def run(self, samples: List[AlignedSample], model_outputs: List[Dict]):
        """
        运行回测

        Args:
            samples: 对齐后的样本列表 (数据)
            model_outputs: 模型预测列表 (与样本对齐)
        """
        n = len(samples)
        for i in range(n):
            sample = samples[i]
            model_out = model_outputs[i] if i < len(model_outputs) else {}

            # NEW: RiskMonitor更新
            if self.risk_monitor:
                # 计算realized PnL（从equity变化）
                realized_pnl = None
                if i > 0 and len(self.equity_curve) > 0:
                    realized_pnl = self.equity_curve[-1]["equity"] - (
                        self.equity_curve[-2]["equity"]
                        if len(self.equity_curve) > 1
                        else self.cash
                    )

                # 更新监控
                self.risk_monitor.update(
                    model_output=model_out, realized_pnl=realized_pnl
                )

                # NEW: 应用α调整（如果模型输出包含regime_alpha）
                if "alpha_base" in model_out or "regime_alpha" in model_out:
                    base_alpha = model_out.get(
                        "alpha_base", model_out.get("regime_alpha", 0.5)
                    )
                    adjusted_alpha = self.risk_monitor.get_adjusted_alpha(base_alpha)

                    # 重新计算logit（应用调整后的α）
                    if "base_hit" in model_out and "delta_hit" in model_out:
                        model_out["logit_hit"] = (
                            model_out["base_hit"]
                            + adjusted_alpha * model_out["delta_hit"]
                        )
                        model_out["alpha_final"] = adjusted_alpha

                # 每50个bar打印风控状态
                if i % 50 == 0 and len(self.risk_monitor.alerts) > 0:
                    print(f"\n[Bar {i}] {self.risk_monitor.get_status_report()}")

            # 1. 撮合订单 (时间 >= trade_time)
            self._match_orders(sample)

            # 2. 更新策略 (获取动作)
            # 读取从 pipeline 传递的 white_risk (由特征计算得出)
            white_risk = model_out.get("white_risk", {})
            if not white_risk:
                # 如果未提供则回退 (例如旧版 pipeline)
                white_risk = {"spread_bps": 0.0, "vpin_z": 0.0}

            pol_out = self.policy.decide(
                model_probs=model_out,
                white_risk=white_risk,
                aux_available=sample.aux_available,
            )

            # 3. 生成订单 (延迟)
            if pol_out.action == Action.ENTER_LONG:
                trade_ts = sample.ts_ms + (LATENCY_BARS * 3000)
                order = Order("BUY", sample.ts_ms, trade_ts, TRADE_QTY)
                self.latency_queue.append(order)
                self.policy.on_order_sent(Action.ENTER_LONG)

            elif pol_out.action == Action.EXIT_LONG:
                trade_ts = sample.ts_ms + (LATENCY_BARS * 3000)
                order = Order("SELL", sample.ts_ms, trade_ts, TRADE_QTY)
                self.latency_queue.append(order)
                self.policy.on_order_sent(Action.EXIT_LONG)

            # 4. 逐日盯市 (模拟)
            mid = sample.target.mid
            equity = self.cash + (self.position * mid)
            self.equity_curve.append(
                {"ts": sample.ts_ms, "equity": equity, "pos": self.position}
            )

    def _match_orders(self, sample: AlignedSample):
        """Process latency queue and execute ready orders"""
        while self.latency_queue:
            # Check head
            if self.latency_queue[0].trade_time > sample.ts_ms:
                break  # Not ready yet

            order = self.latency_queue.popleft()
            self._execute(order, sample)

    def _execute(self, order: Order, sample: AlignedSample):
        """Execute order at L1 with liquidity check"""

        # 0. Safety Check
        assert (
            sample.target.symbol == TARGET_SYMBOL
        ), f"CRITICAL: Attempting to trade wrong symbol {sample.target.symbol}"

        # Taker Execution @ L1
        if order.action == "BUY":
            # Buy @ Ask1
            if sample.target.asks:
                ask_px = sample.target.asks[0][0]
                ask_vol = sample.target.asks[0][1]

                # Liquidity Check (All or None)
                if ask_vol >= order.qty:
                    cost = order.qty * ask_px * (1 + COST_RATE)
                    self.cash -= cost
                    self.position += order.qty
                    self._log_trade(sample.ts_ms, "BUY", ask_px, order.qty, cost)
                    self.policy.on_fill(Action.ENTER_LONG)
                else:
                    self._log_trade(sample.ts_ms, "SKIP_BUY", ask_px, 0, 0)
                    self.policy.on_reject(Action.ENTER_LONG)

        elif order.action == "SELL":
            # Sell @ Bid1
            if sample.target.bids:
                bid_px = sample.target.bids[0][0]
                bid_vol = sample.target.bids[0][1]

                if bid_vol >= order.qty:
                    revenue = order.qty * bid_px * (1 - COST_RATE)
                    self.cash += revenue
                    self.position -= order.qty
                    self._log_trade(sample.ts_ms, "SELL", bid_px, order.qty, revenue)
                    self.policy.on_fill(Action.EXIT_LONG)
                else:
                    self._log_trade(sample.ts_ms, "SKIP_SELL", bid_px, 0, 0)
                    self.policy.on_reject(Action.EXIT_LONG)

    def _log_trade(self, ts, side, px, qty, amt):
        """Log trade execution"""
        self.trades.append({"ts": ts, "side": side, "px": px, "qty": qty, "amt": amt})


# ==========================================
# 4. 性能指标 (Performance Metrics)
# ==========================================


def calculate_metrics(
    trades: List[Dict], equity_curve: List[Dict], initial_cap: float = 1e6
) -> Dict:
    """
    计算综合回测指标

    Args:
        trades: 交易记录列表
        equity_curve: 权益曲线列表
        initial_cap: 初始资金

    Returns:
        Dict 包含性能指标
    """
    if not trades:
        return {
            "n_trades": 0,
            "total_pnl": 0.0,
            "sharpe": 0.0,
            "max_drawdown_pct": 0.0,
            "fill_rate_pct": 0.0,
            "est_cost": 0.0,
        }

    # Trade Analysis
    df_tx = pd.DataFrame(trades)

    # Equity Analysis
    df_eq = pd.DataFrame(equity_curve)
    df_eq["pnl"] = df_eq["equity"] - initial_cap
    df_eq["ret"] = df_eq["equity"].pct_change().fillna(0)

    # Risk Metrics
    total_ret = (df_eq["equity"].iloc[-1] / initial_cap) - 1.0

    # Sharpe (Annualized assuming 4 hours trading ~ 4800 bars)
    ret_std = df_eq["ret"].std()
    sharpe = (
        (df_eq["ret"].mean() / ret_std) * np.sqrt(250 * 4800) if ret_std > 1e-9 else 0.0
    )

    # Drawdown
    cum_max = df_eq["equity"].cummax()
    dd = (df_eq["equity"] - cum_max) / cum_max
    max_dd = dd.min()

    # Trade Stats
    buys = df_tx[df_tx["side"] == "BUY"]
    sells = df_tx[df_tx["side"] == "SELL"]
    skips = df_tx[df_tx["side"].str.startswith("SKIP")]

    # Cost Analysis
    total_vol_traded = buys["qty"].sum() + sells["qty"].sum()
    est_cost = total_vol_traded * buys["px"].mean() * COST_RATE if len(buys) > 0 else 0

    return {
        "n_trades": len(buys),
        "n_skips": len(skips),
        "total_pnl": df_eq["equity"].iloc[-1] - initial_cap,
        "total_ret_pct": total_ret * 100,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd * 100,
        "fill_rate_pct": (
            len(buys) / (len(buys) + len(skips)) * 100
            if (len(buys) + len(skips)) > 0
            else 0
        ),
        "est_cost": est_cost,
    }
