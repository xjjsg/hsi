from typing import List, Optional
from hsi_hft_v3.core.data_contract import Bar, AlignedSample

class DualStreamAligner:
    """Strict Causal Alignment using Asof Logic"""
    def __init__(self, max_lag_ms: int = 30000):
        self.max_lag_ms = max_lag_ms

    def align(self, target_bars: List[Bar], aux_bars: List[Bar]) -> List[AlignedSample]:
        # Ensure strict sorted order
        target_bars.sort(key=lambda x: x.ts_ms)
        aux_bars.sort(key=lambda x: x.ts_ms)
        
        aligned_samples = []
        
        # Dual Pointer Logic
        aux_idx = 0
        n_aux = len(aux_bars)
        last_valid_aux = None
        
        for t_bar in target_bars:
            # Advance aux pointer to find the snapshot right before or at t_bar.ts_ms
            # We want last aux where aux.ts_ms <= t_bar.ts_ms
            
            while aux_idx < n_aux and aux_bars[aux_idx].ts_ms <= t_bar.ts_ms:
                last_valid_aux = aux_bars[aux_idx]
                aux_idx += 1
                
            # Determine logic
            aux_val = None
            aux_available = False
            aux_lag = None
            
            if last_valid_aux is not None:
                lag = t_bar.ts_ms - last_valid_aux.ts_ms
                if lag <= self.max_lag_ms:
                    aux_val = last_valid_aux
                    aux_available = True
                    aux_lag = lag
                else:
                    # Stale aux data
                    aux_available = False
                    aux_lag = lag # Keep lag for debugging, but not available
            
            # Futures check
            has_fut = (t_bar.fut_price is not None and t_bar.fut_imb is not None)
            
            sample = AlignedSample(
                ts_ms=t_bar.ts_ms,
                target=t_bar,
                aux=aux_val,
                aux_available=aux_available,
                aux_lag_ms=aux_lag,
                has_fut=has_fut
            )
            aligned_samples.append(sample)
            
        return aligned_samples
