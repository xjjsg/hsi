import pandas as pd
import numpy as np
from typing import Dict, List

def calculate_metrics(trades: List[Dict], equity_curve: List[Dict], initial_cap: float = 1e6) -> Dict:
    """Detailed V4-style metrics"""
    if not trades:
        return {
            "total_trades": 0,
            "pnl": 0.0,
            "sharpe": 0.0,
            "drawdown": 0.0
        }
        
    # Trade Analysis
    df_tx = pd.DataFrame(trades)
    
    # Calculate PnL per round-trip (Simplified: Buy Amt - Sell Amt)
    # Note: Precise FIFO/LIFO matching is complex, here we use Cash Flow PnL approximation for quick stats
    # Real PnL comes from Equity Curve
    
    # Equity Analysis
    df_eq = pd.DataFrame(equity_curve)
    df_eq["pnl"] = df_eq["equity"] - initial_cap
    df_eq["ret"] = df_eq["equity"].pct_change().fillna(0)
    
    # Risk Metrics
    total_ret = (df_eq["equity"].iloc[-1] / initial_cap) - 1.0
    
    # Sharpe (Annualized assuming 4 hours trading ~ 4800 bars)
    # Using simple std of bar returns
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
    # Estimate total cost paid
    total_vol_traded = buys["qty"].sum() + sells["qty"].sum()
    est_cost = total_vol_traded * buys["px"].mean() * 0.0001 # approx
    
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
