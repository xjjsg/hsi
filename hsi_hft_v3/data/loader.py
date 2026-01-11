import os
import glob
import re
import pandas as pd
from typing import List, Tuple, Dict
from hsi_hft_v3.core.config import TARGET_SYMBOL, AUX_SYMBOL
from hsi_hft_v3.data.bar_builder import BarBuilder
from hsi_hft_v3.data.aligner import DualStreamAligner
from hsi_hft_v3.core.data_contract import AlignedSample

class V5DataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.target_symbol = TARGET_SYMBOL
        self.aux_symbol = AUX_SYMBOL
        
    def load_date_range(self, start_date: str = None, end_date: str = None) -> Dict[str, List[AlignedSample]]:
        """
        Load data for a range of dates.
        Returns: Dict {date_str: List[AlignedSample]}
        """
        pairs = self._match_files()
        results = {}
        
        # Filter by date
        filtered_pairs = []
        for date, tgt_path, aux_path in pairs:
            if start_date and date < start_date: continue
            if end_date and date > end_date: continue
            filtered_pairs.append((date, tgt_path, aux_path))
            
        print(f"[Loader] Found {len(filtered_pairs)} valid days in {self.data_dir}")
        
        # Process each day
        bb_tgt = BarBuilder(self.target_symbol)
        bb_aux = BarBuilder(self.aux_symbol)
        aligner = DualStreamAligner()
        
        for date, tgt_path, aux_path in filtered_pairs:
            print(f"Loading {date}...")
            try:
                # Read CSVs
                df_tgt = pd.read_csv(tgt_path)
                df_aux = pd.read_csv(aux_path) if aux_path else pd.DataFrame()
                
                # Build Bars
                bars_tgt = bb_tgt.process_dataframe(df_tgt)
                bars_aux = bb_aux.process_dataframe(df_aux) if not df_aux.empty else []
                
                # Align
                samples = aligner.align(bars_tgt, bars_aux)
                
                if samples:
                    results[date] = samples
                    print(f"  -> {len(samples)} samples")
                else:
                    print("  -> No samples produced")
                    
            except Exception as e:
                print(f"  -> Error loading {date}: {e}")
                
        return results

    def _match_files(self) -> List[Tuple[str, str, str]]:
        """Find paired files by date"""
        # Assume structure: data_dir/sz159920/*.csv
        tgt_pattern = os.path.join(self.data_dir, self.target_symbol, "*.csv")
        # Assume structure: data_dir/sh513130/*.csv
        aux_pattern = os.path.join(self.data_dir, self.aux_symbol, "*.csv")
        
        tgt_files = glob.glob(tgt_pattern)
        aux_files = glob.glob(aux_pattern)
        
        print(f"[Loader] Scanning: {len(tgt_files)} target files, {len(aux_files)} aux files")
        
        def get_date(path):
            # Expecting *-YYYY-MM-DD.csv
            m = re.search(r"(\d{4}-\d{2}-\d{2})", path)
            return m.group(1) if m else None
            
        tgt_map = {get_date(f): f for f in tgt_files if get_date(f)}
        aux_map = {get_date(f): f for f in aux_files if get_date(f)}
        
        common_dates = sorted(tgt_map.keys())
        
        pairs = []
        for d in common_dates:
            pairs.append((d, tgt_map[d], aux_map.get(d))) # Aux is optional but preferred
            
        return pairs
