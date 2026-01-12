from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# ==========================================
# 1. Global Identity & Scope
# ==========================================
TARGET_SYMBOL = "sz159920"
AUX_SYMBOL = "sh513130"
BAR_SIZE_S = 3  # 3-second bars
TIMEZONE = "Asia/Shanghai"

# ==========================================
# 2. Execution & Cost Model
# ==========================================
COST_RATE = 0.0001        # 1bp, single-side
TICK_SIZE = 0.001          # Minimum price move
LATENCY_BARS = 1           # Signal at t -> Trade at t + 1
TRADE_QTY = 1000           # Simulation quantity

@dataclass
class ExecutionConfig:
    order_type: str = "TAKER"  # Always taker
    fill_mode: str = "L1_IOC"  # Only fill at Level 1, IOC
    slippage_bps: float = 0.0  # Optional slippage simulation

# ==========================================
# 3. Modeling Objectives
# ==========================================
PREDICT_HORIZON_S = 120    # H* = 120s
K_BARS = PREDICT_HORIZON_S // BAR_SIZE_S  # 40 bars
LOOKBACK_BARS = 100        # Feature window
BLACKBOX_DIM = 32          # Latent dimension (Spec Requirement)

@dataclass
class LabelConfig:
    use_cost_gate: bool = False  # Strict hit definition
    adverse_bps: float = 20.0    # Risk threshold in bps
    embargo_bars: int = LOOKBACK_BARS + K_BARS + LATENCY_BARS + 10 # Data split buffer

# ==========================================
# 4. Policy Thresholds
# ==========================================
@dataclass
class PolicyConfig:
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

# ==========================================
# 5. Data Contract
# ==========================================
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
