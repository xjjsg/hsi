from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
from hsi_hft_v3.core.config import PolicyConfig, TARGET_SYMBOL

class State(Enum):
    FLAT = 0
    LONG = 1
    COOLDOWN = 2

class Action(Enum):
    HOLD = 0
    ENTER_LONG = 1
    EXIT_LONG = 2

@dataclass
class PolicyOutput:
    action: Action
    reason: str
    meta: Dict

class StateMachine:
    def __init__(self, config: PolicyConfig):
        self.cfg = config
        self.state = State.FLAT
        self.hold_bars = 0
        self.cooldown_rem = 0
        self.entry_price = 0.0
        
    def decide(self, 
               model_probs: Dict[str, float], 
               white_risk: Dict[str, float], 
               aux_available: bool) -> PolicyOutput:
        
        # 1. Update Cooldown
        if self.state == State.COOLDOWN:
            self.cooldown_rem -= 1
            if self.cooldown_rem <= 0:
                self.state = State.FLAT
                return PolicyOutput(Action.HOLD, "COOLDOWN_END", {})
            return PolicyOutput(Action.HOLD, "IN_COOLDOWN", {})

        # 2. Select Thresholds
        if aux_available:
            th_hit = self.cfg.th_p_hit
            th_tau = self.cfg.th_p_tau_60
            th_risk = self.cfg.th_risk_enter
        else:
            th_hit = self.cfg.th_p_hit_noaux
            th_tau = self.cfg.th_p_tau_60_noaux
            th_risk = self.cfg.th_risk_enter_noaux

        # 3. Decision Logic
        p_hit = model_probs.get("p_hit", 0.0)
        p_tau = model_probs.get("P_tau_le_60s", 0.0)
        p_risk = model_probs.get("risk", 1.0) # default high risk
        
        meta = {
            "p_hit": p_hit, "p_tau": p_tau, "p_risk": p_risk,
            "state": self.state.name
        }

        # EXIT LOGIC
        if self.state == State.LONG:
            self.hold_bars += 1
            
            # Max Hold
            if self.hold_bars >= self.cfg.max_hold_bars:
                self._transition_to_cooldown()
                return PolicyOutput(Action.EXIT_LONG, "MAX_HOLD", meta)
            
            # Risk Gates
            if p_risk > self.cfg.th_risk_exit:
                self._transition_to_cooldown()
                return PolicyOutput(Action.EXIT_LONG, "RISK_EXIT", meta)
                
            # Whitebox Reversal / Micro Risk (Simplified)
            if white_risk.get("vpin_z", 0) > self.cfg.vpin_exit_z:
                 self._transition_to_cooldown()
                 return PolicyOutput(Action.EXIT_LONG, "VPIN_EXIT", meta)
            
            if white_risk.get("kyle_z", 0) > self.cfg.lambda_exit_z:
                 self._transition_to_cooldown()
                 return PolicyOutput(Action.EXIT_LONG, "KYLE_EXIT", meta)

            return PolicyOutput(Action.HOLD, "HOLD_LONG", meta)

        # ENTRY LOGIC (FLAT)
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
                self.state = State.LONG
                self.hold_bars = 0
                return PolicyOutput(Action.ENTER_LONG, "SIGNAL_ENTRY", meta)
                
            return PolicyOutput(Action.HOLD, "NO_SIGNAL", meta)
            
        return PolicyOutput(Action.HOLD, "UNKNOWN_STATE", meta)

    def _transition_to_cooldown(self):
        self.state = State.COOLDOWN
        self.cooldown_rem = self.cfg.cooldown_bars
