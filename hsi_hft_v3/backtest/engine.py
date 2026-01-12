import pandas as pd
from typing import List, Dict, Deque
from collections import deque
from hsi_hft_v3.core.config import LATENCY_BARS, COST_RATE, TRADE_QTY
from hsi_hft_v3.policy.state_machine import Action, PolicyOutput, StateMachine
from hsi_hft_v3.core.data_contract import AlignedSample

class Order:
    def __init__(self, action: str, create_time: int, trade_time: int, qty: int):
        self.action = action
        self.create_time = create_time
        self.trade_time = trade_time
        self.qty = qty
        self.is_filled = False

class BacktestEngine:
    def __init__(self, policy: StateMachine):
        self.policy = policy
        self.latency_queue: Deque[Order] = deque()
        self.trades = []
        self.cash = 0.0
        self.position = 0
        self.equity_curve = []
        
    def run(self, samples: List[AlignedSample], model_outputs: List[Dict]):
        """Event-Driven Loop"""
        n = len(samples)
        for i in range(n):
            sample = samples[i]
            model_out = model_outputs[i] if i < len(model_outputs) else {}
            
            # 1. Match Orders (Time >= trade_time)
            self._match_orders(sample)
            
            # 2. Update Policy (Get Action)
            # 2. Update Policy (Get Action)
            # Read white_risk passed from pipeline (calculated from features)
            white_risk = model_out.get("white_risk", {})
            if not white_risk:
                # Fallback if not provided (e.g. earlier pipeline version)
                # But typically pipeline sends it now.
                white_risk = {"spread_bps": 0.0, "vpin_z": 0.0} 
            
            pol_out = self.policy.decide(
                model_probs=model_out,
                white_risk=white_risk,
                aux_available=sample.aux_available
            )
            
            # 3. Generate Order (Latency)
            if pol_out.action == Action.ENTER_LONG:
                # Ask: How to get next expected valid time?
                # Simple: next bar ts = current ts + 3000 * latency
                trade_ts = sample.ts_ms + (LATENCY_BARS * 3000)
                order = Order("BUY", sample.ts_ms, trade_ts, TRADE_QTY)
                self.latency_queue.append(order)
                
            elif pol_out.action == Action.EXIT_LONG:
                trade_ts = sample.ts_ms + (LATENCY_BARS * 3000)
                order = Order("SELL", sample.ts_ms, trade_ts, TRADE_QTY)
                self.latency_queue.append(order)
                
            # 4. Mark to Market (Simulated)
            mid = sample.target.mid
            equity = self.cash + (self.position * mid)
            self.equity_curve.append({
                "ts": sample.ts_ms,
                "equity": equity,
                "pos": self.position
            })

    def _match_orders(self, sample: AlignedSample):
        # Process queue
        while self.latency_queue:
            # Check head
            if self.latency_queue[0].trade_time > sample.ts_ms:
                break # Not ready yet
                
            order = self.latency_queue.popleft()
            self._execute(order, sample)
            
    def _execute(self, order: Order, sample: AlignedSample):
        # 0. Safety Check
        from hsi_hft_v3.core.config import TARGET_SYMBOL
        assert sample.target.symbol == TARGET_SYMBOL, f"CRITICAL: Attempting to trade wrong symbol {sample.target.symbol}"

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
                else:
                    self._log_trade(sample.ts_ms, "SKIP_BUY", ask_px, 0, 0)
                    
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
                else:
                    self._log_trade(sample.ts_ms, "SKIP_SELL", bid_px, 0, 0)

    def _log_trade(self, ts, side, px, qty, amt):
        self.trades.append({
            "ts": ts, "side": side, "px": px, "qty": qty, "amt": amt
        })
