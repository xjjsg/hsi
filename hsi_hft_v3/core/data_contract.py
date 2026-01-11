from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

@dataclass
class Bar:
    """Standard 3s Bar Structure"""
    ts_ms: int  # Bucket end time
    symbol: str
    
    # Market Data
    mid: float
    vwap: float
    volume: int
    amount: float
    
    # LOB (Snapshot at bucket end)
    bids: List[tuple] = field(default_factory=list)  # [(price, vol), ...] for 5 levels
    asks: List[tuple] = field(default_factory=list)
    
    # External / Derived
    sentiment: float = 0.0
    premium_rate: float = 0.0
    
    # Futures (Target Only, Optional)
    fut_price: Optional[float] = None
    fut_imb: Optional[float] = None
    
    def sanity_check(self) -> bool:
        """Basic data integrity checks"""
        # 1. Price Logic
        if self.mid <= 0 or not np.isfinite(self.mid):
            return False
            
        # 2. LOB Logic
        if len(self.bids) > 0 and len(self.asks) > 0:
            best_bid = self.bids[0][0]
            best_ask = self.asks[0][0]
            if best_bid > best_ask: # Crossed book
                return False
                
        # 3. Volume Logic
        if self.volume < 0 or self.amount < 0:
            return False
            
        return True

@dataclass
class AlignedSample:
    """Dual-Stream Aligned Input for Features"""
    ts_ms: int
    target: Bar
    aux: Optional[Bar]
    
    # Masks
    aux_available: bool  # 1 if aux exists and lag <= max_lag
    aux_lag_ms: Optional[int]
    has_fut: bool
    
    def to_whitebox_input(self) -> Dict:
        """Convert to dict structure for WhiteBoxFactory"""
        return {
            "target": self.target,
            "aux": self.aux,
            "masks": {
                "aux_available": 1.0 if self.aux_available else 0.0,
                "has_fut": 1.0 if self.has_fut else 0.0
            }
        }
