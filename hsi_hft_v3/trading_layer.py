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
from hsi_hft_v3.data_layer import AlignedSample


# ==========================================
# 1. 全局配置 (Global Configuration)
# ==========================================

# 1.1 Global Identity & Scope
TARGET_SYMBOL = "sz159920"
AUX_SYMBOL = "sh513130"
BAR_SIZE_S = 3  # 3-second bars
TIMEZONE = "Asia/Shanghai"

# 1.2 Execution & Cost Model
COST_RATE = 0.0001        # 1bp, single-side
TICK_SIZE = 0.001          # Minimum price move
LATENCY_BARS = 1           # Signal at t -> Trade at t + 1
TRADE_QTY = 1000           # Simulation quantity


@dataclass
class ExecutionConfig:
    """Execution configuration"""
    order_type: str = "TAKER"  # Always taker
    fill_mode: str = "L1_IOC"  # Only fill at Level 1, IOC
    slippage_bps: float = 0.0  # Optional slippage simulation


# 1.3 Modeling Objectives
PREDICT_HORIZON_S = 120    # H* = 120s
K_BARS = PREDICT_HORIZON_S // BAR_SIZE_S  # 40 bars
LOOKBACK_BARS = 100        # Feature window
BLACKBOX_DIM = 32          # Latent dimension (Spec Requirement)


@dataclass
class LabelConfig:
    """Label generation configuration for training"""
    use_cost_gate: bool = False  # Strict hit definition
    adverse_bps: float = 20.0    # Risk threshold in bps
    embargo_bars: int = LOOKBACK_BARS + K_BARS + LATENCY_BARS + 10  # Data split buffer


# 1.4 Policy Thresholds
@dataclass
class PolicyConfig:
    """Trading policy configuration with adaptive thresholds"""
    
    # Standard Thresholds (with Aux)
    # Relaxed for initial testing
    th_p_hit: float = 0.51
    th_p_tau_60: float = 0.50
    th_risk_enter: float = 0.40
    
    # Conservative Thresholds (No Aux)
    th_p_hit_noaux: float = 0.70
    th_p_tau_60_noaux: float = 0.80
    th_risk_enter_noaux: float = 0.10
    
    # Exit & Gates
    th_risk_exit: float = 0.25
    max_hold_bars: int = K_BARS
    cooldown_bars: int = 20
    
    # Microstructure Gates
    spread_max_bps: float = 10.0
    depth_min_qty: int = 5000 
    vpin_max_z: float = 3.0
    vpin_exit_z: float = 3.5
    lambda_exit_z: float = 3.0


# 1.5 Data Contract (Field Lists)
ALLOWLIST_FIELDS = [
    "tx_local_time", "tx_server_time", 
    "price", "tick_vol", "tick_amt", "tick_vwap",
    "bp1", "bv1", "sp1", "sv1",
    "bp2", "bv2", "sp2", "sv2",
    "bp3", "bv3", "sp3", "sv3",
    "bp4", "bv4", "sp4", "sv4",
    "bp5", "bv5", "sp5", "sv5",
    "sentiment", "premium_rate", "iopv",
    "index_price", "fx_rate",
    # Only for 159920
    "fut_price", "fut_mid", "fut_imb", "fut_delta_vol", "fut_pct"
]

BLOCKLIST_FIELDS = [
    "idx_delay_ms", "fut_delay_ms", "data_flags"
]


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
    State machine trading policy
    
    States:
    - FLAT: No position, looking for entry signals
    - LONG: Holding long position, monitoring exit conditions
    - COOLDOWN: Post-trade cooldown period
    
    Transitions:
    FLAT -> LONG: Entry signal (p_hit, p_tau, risk gates passed)
    LONG -> COOLDOWN: Exit signal (max hold, risk, or adverse conditions)
    COOLDOWN -> FLAT: Cooldown period expired
    """
    
    def __init__(self, config: PolicyConfig):
        self.cfg = config
        self.state = State.FLAT
        self.pending_action = None  # Track if we are waiting for execution
        self.hold_bars = 0
        self.cooldown_rem = 0
        self.entry_price = 0.0
        
    def decide(self, 
               model_probs: Dict[str, float], 
               white_risk: Dict[str, float], 
               aux_available: bool) -> PolicyOutput:
        """
        Main decision function
        
        Args:
            model_probs: Model predictions (p_hit, P_tau_le_60s, risk)
            white_risk: White-box risk indicators (spread_bps, vpin_z, etc.)
            aux_available: Whether auxiliary data is available
            
        Returns:
            PolicyOutput with action and reason
        """
        
        # 0. Check Pending
        if self.pending_action is not None:
             return PolicyOutput(Action.HOLD, "PENDING_EXEC", {"state": self.state.name})

        # 1. Update Cooldown
        if self.state == State.COOLDOWN:
            self.cooldown_rem -= 1
            if self.cooldown_rem <= 0:
                self.state = State.FLAT
                return PolicyOutput(Action.HOLD, "COOLDOWN_END", {})
            return PolicyOutput(Action.HOLD, "IN_COOLDOWN", {})

        # 2. Select Thresholds (Adaptive based on aux availability)
        if aux_available:
            th_hit = self.cfg.th_p_hit
            th_tau = self.cfg.th_p_tau_60
            th_risk = self.cfg.th_risk_enter
        else:
            th_hit = self.cfg.th_p_hit_noaux
            th_tau = self.cfg.th_p_tau_60_noaux
            th_risk = self.cfg.th_risk_enter_noaux

        # 3. Extract Probabilities
        p_hit = model_probs.get("p_hit", 0.0)
        p_tau = model_probs.get("P_tau_le_60s", 0.0)
        p_risk = model_probs.get("risk", 1.0)  # default high risk
        
        meta = {
            "p_hit": p_hit, "p_tau": p_tau, "p_risk": p_risk,
            "state": self.state.name
        }

        # 4. EXIT LOGIC
        if self.state == State.LONG:
            self.hold_bars += 1
            
            # Max Hold
            if self.hold_bars >= self.cfg.max_hold_bars:
                return PolicyOutput(Action.EXIT_LONG, "MAX_HOLD", meta)
            
            # Risk Gates
            if p_risk > self.cfg.th_risk_exit:
                return PolicyOutput(Action.EXIT_LONG, "RISK_EXIT", meta)
                
            # Whitebox Reversal / Micro Risk (Simplified)
            if white_risk.get("vpin_z", 0) > self.cfg.vpin_exit_z:
                 return PolicyOutput(Action.EXIT_LONG, "VPIN_EXIT", meta)
            
            if white_risk.get("kyle_z", 0) > self.cfg.lambda_exit_z:
                 return PolicyOutput(Action.EXIT_LONG, "KYLE_EXIT", meta)

            return PolicyOutput(Action.HOLD, "HOLD_LONG", meta)

        # 5. ENTRY LOGIC (FLAT)
        if self.state == State.FLAT:
            # Whitebox Gates (Entry)
            if white_risk.get("spread_bps", 0) > self.cfg.spread_max_bps:
                return PolicyOutput(Action.HOLD, "SPREAD_GATE", meta)
            
            if white_risk.get("depth_qty", 999999) < self.cfg.depth_min_qty:
                return PolicyOutput(Action.HOLD, "DEPTH_GATE", meta)
                
            if white_risk.get("vpin_z", 0) > self.cfg.vpin_max_z:
                return PolicyOutput(Action.HOLD, "VPIN_GATE", meta)
            
            # Signals
            if p_hit >= th_hit and p_tau >= th_tau and p_risk <= th_risk:
                # Do NOT transition here. Wait for execution callback.
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
    Event-driven backtest engine with latency simulation
    
    Features:
    - Latency queue for realistic order execution
    - L1 taker execution (IOC at best bid/ask)
    - Liquidity checks (all-or-none fills)
    - Mark-to-market equity tracking
    """
    
    def __init__(self, policy: StateMachine):
        self.policy = policy
        self.latency_queue: Deque[Order] = deque()
        self.trades = []
        self.cash = 1_000_000.0  # Initial Capital
        self.position = 0
        self.equity_curve = []
        
    def run(self, samples: List[AlignedSample], model_outputs: List[Dict]):
        """
        Run backtest
        
        Args:
            samples: List of aligned samples (data)
            model_outputs: List of model predictions (aligned with samples)
        """
        n = len(samples)
        for i in range(n):
            sample = samples[i]
            model_out = model_outputs[i] if i < len(model_outputs) else {}
            
            # 1. Match Orders (Time >= trade_time)
            self._match_orders(sample)
            
            # 2. Update Policy (Get Action)
            # Read white_risk passed from pipeline (calculated from features)
            white_risk = model_out.get("white_risk", {})
            if not white_risk:
                # Fallback if not provided (e.g. earlier pipeline version)
                white_risk = {"spread_bps": 0.0, "vpin_z": 0.0} 
            
            pol_out = self.policy.decide(
                model_probs=model_out,
                white_risk=white_risk,
                aux_available=sample.aux_available
            )
            
            # 3. Generate Order (Latency)
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
                
            # 4. Mark to Market (Simulated)
            mid = sample.target.mid
            equity = self.cash + (self.position * mid)
            self.equity_curve.append({
                "ts": sample.ts_ms,
                "equity": equity,
                "pos": self.position
            })

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
        assert sample.target.symbol == TARGET_SYMBOL, \
            f"CRITICAL: Attempting to trade wrong symbol {sample.target.symbol}"

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
        self.trades.append({
            "ts": ts, "side": side, "px": px, "qty": qty, "amt": amt
        })


# ==========================================
# 4. 性能指标 (Performance Metrics)
# ==========================================

def calculate_metrics(trades: List[Dict], equity_curve: List[Dict], initial_cap: float = 1e6) -> Dict:
    """
    Calculate comprehensive backtest metrics
    
    Args:
        trades: List of trade records
        equity_curve: List of equity snapshots over time
        initial_cap: Initial capital
        
    Returns:
        Dict of performance metrics
    """
    if not trades:
        return {
            "total_trades": 0,
            "pnl": 0.0,
            "sharpe": 0.0,
            "drawdown": 0.0
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
    sharpe = (df_eq["ret"].mean() / ret_std) * np.sqrt(250 * 4800) if ret_std > 1e-9 else 0.0
    
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
        "fill_rate_pct": len(buys) / (len(buys) + len(skips)) * 100 if (len(buys)+len(skips))>0 else 0,
        "est_cost": est_cost
    }
