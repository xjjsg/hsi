import sys
import os
sys.path.append(os.getcwd())
import unittest
import numpy as np
from hsi_hft_v5.features.whitebox import WhiteBoxFeatureFactory, RollingStats, RollingCov
from hsi_hft_v5.core.data_contract import AlignedSample, Bar

class TestWhiteBox(unittest.TestCase):
    def setUp(self):
        self.wb = WhiteBoxFeatureFactory()
        
    def _create_sample(self, mid=100, aux_avail=True):
        t = Bar(ts_ms=1000, symbol="tgt", mid=mid, asks=[(mid+0.5, 100)]*5, bids=[(mid-0.5, 100)]*5, volume=100, vwap=mid, amount=mid*100)
        a = Bar(ts_ms=1000, symbol="aux", mid=10, asks=[(10.1, 1000)], bids=[(9.9, 1000)], volume=500, vwap=10, amount=5000, sentiment=0.5) if aux_avail else None
        return AlignedSample(
            ts_ms=1000, target=t, aux=a, aux_available=aux_avail, 
            has_fut=True, aux_lag_ms=0
        )

    def test_rolling_stats(self):
        rs = RollingStats(5)
        for i in range(10):
            rs.update(float(i))
            
        # Window should be [5, 6, 7, 8, 9]
        self.assertAlmostEqual(rs.mean(), 7.0)
        self.assertAlmostEqual(rs.slope(), 4.0) # 9 - 5
        
    def test_rolling_cov(self):
        rc = RollingCov(5)
        # Perfectly correlated: y = 2x
        for i in range(10):
            rc.update(float(i), float(i)*2)
            
        self.assertAlmostEqual(rc.corr(), 1.0)
        
    def test_aux_masking(self):
        # 1. With Aux
        s1 = self._create_sample(aux_avail=True)
        o1 = self.wb.compute(s1)
        self.assertNotEqual(o1["white_cross_raw"]["flow_ratio"], 0.0)
        
        # 2. Without Aux
        s2 = self._create_sample(aux_avail=False)
        o2 = self.wb.compute(s2)
        # Cross features should be 0
        self.assertEqual(o2["white_cross_raw"]["flow_ratio"], 0.0)
        self.assertEqual(o2["white_cross_raw"]["LLT_rs"], 0.0)
        
        # Aux Raw should be zeroed
        self.assertEqual(o2["white_aux_raw"]["aux_iOFI"], 0.0)

    def test_unified_regime_z(self):
        # Feed some noise
        for i in range(50):
            s = self._create_sample(mid=100 + i%5, aux_avail=True)
            out = self.wb.compute(s)
            
        derived = out["white_derived"]
        # Check if we have z_20 keys for A1 features
        self.assertIn("tgt_iOFI_z_20", derived)
        self.assertIn("tgt_VPIN_20_z_20", derived) 
        # Note: VPIN_20 is a raw feature name, so derived is VPIN_20_z_20 (meta!)

if __name__ == "__main__":
    unittest.main()
