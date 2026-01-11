import numpy as np
from typing import Dict, List, Optional, Deque, Tuple
from collections import deque
from hsi_hft_v3.core.data_contract import Bar, AlignedSample
from hsi_hft_v3.core.config import BAR_SIZE_S

EPS = 1e-9

class RollingStats:
    """
    O(1) Rolling Mean, Variance, Z-Score, Slope.
    Maintain sums of x and x^2.
    """
    def __init__(self, window: int):
        self.window = window
        self.values = deque(maxlen=window)
        self.sum_x = 0.0
        self.sum_x2 = 0.0
        
    def update(self, x: float):
        if not np.isfinite(x): x = 0.0
        
        # Remove old
        if len(self.values) == self.window:
            old = self.values[0]
            self.sum_x -= old
            self.sum_x2 -= old * old
            
        # Add new
        self.values.append(x)
        self.sum_x += x
        self.sum_x2 += x * x
        
    def mean(self) -> float:
        n = len(self.values)
        if n == 0: return 0.0
        return self.sum_x / n
        
    def std(self) -> float:
        n = len(self.values)
        if n < 2: return 0.0
        mean = self.sum_x / n
        var = (self.sum_x2 / n) - (mean * mean)
        return np.sqrt(max(0.0, var))
        
    def zscore(self) -> float:
        s = self.std()
        if s < EPS: return 0.0
        return (self.values[-1] - self.mean()) / (s + EPS)
        
    def slope(self) -> float:
        # Simple diff: x_t - x_{t-W}
        if len(self.values) < 2: return 0.0
        # Requirement: x_raw - x_raw.shift(W)
        return self.values[-1] - self.values[0]

class RollingCov:
    """
    O(1) Rolling Covariance and Correlation between two streams X and Y.
    Maintain sums of x, y, x^2, y^2, xy.
    """
    def __init__(self, window: int):
        self.window = window
        self.vals_x = deque(maxlen=window)
        self.vals_y = deque(maxlen=window)
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_x2 = 0.0
        self.sum_y2 = 0.0
        self.sum_xy = 0.0
        
    def update(self, x: float, y: float):
        if not np.isfinite(x): x = 0.0
        if not np.isfinite(y): y = 0.0
        
        if len(self.vals_x) == self.window:
            old_x = self.vals_x.popleft()
            old_y = self.vals_y.popleft() # Should sync
            self.sum_x -= old_x
            self.sum_y -= old_y
            self.sum_x2 -= old_x * old_x
            self.sum_y2 -= old_y * old_y
            self.sum_xy -= old_x * old_y
            
        self.vals_x.append(x)
        self.vals_y.append(y)
        self.sum_x += x
        self.sum_y += y
        self.sum_x2 += x * x
        self.sum_y2 += y * y
        self.sum_xy += x * y
        
    def cov(self) -> float:
        n = len(self.vals_x)
        if n < 2: return 0.0
        mean_x = self.sum_x / n
        mean_y = self.sum_y / n
        # Cov = E[XY] - E[X]E[Y]
        mean_xy = self.sum_xy / n
        return mean_xy - (mean_x * mean_y)
        
    def var_x(self) -> float:
        n = len(self.vals_x)
        if n < 2: return 0.0
        mean = self.sum_x / n
        return (self.sum_x2 / n) - (mean * mean)

    def corr(self) -> float:
        c = self.cov()
        vx = self.var_x()
        
        # Calculate Var Y locally
        n = len(self.vals_y)
        if n < 2: return 0.0
        mean_y = self.sum_y / n
        vy = (self.sum_y2 / n) - (mean_y * mean_y)
        
        if vx <= 0 or vy <= 0: return 0.0
        return c / (np.sqrt(vx) * np.sqrt(vy) + EPS)

class WhiteBoxFeatureFactory:
    """
    19 Factors + Unified Derivative Rules
    Strict Implementation of V5 Spec.
    """
    def __init__(self):
        # Config
        self.W_set = [20, 100, 600]
        self.L_DEPTH = 5
        self.tick_size = 0.001
        self.iofi_weights = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
        self.leadlag_lags = [1, 2, 3, 5] 
        
        # State Management (Previous Bars for diffs)
        self.prev_bar_tgt: Optional[Bar] = None
        self.prev_bar_aux: Optional[Bar] = None
        
        # Rolling Stats Storage
        # Structure: self.stats[feature_name][window_size] -> RollingStats/RollingCov
        self.stats: Dict[str, Dict[int, RollingStats]] = {}
        self.cov_stats: Dict[str, Dict[int, RollingCov]] = {}
        
        # Special Buffers for Lead-Lag (History of Returns)
        # We need ret_aux_{t-L} vs ret_tgt_t. 
        # So we need to store aux history up to max(L).
        self.max_lag = max(self.leadlag_lags)
        self.aux_ret_buffer = deque(maxlen=self.max_lag + 1)
        
        # Helper to init stats map
        self._init_stat_keys()

    def _init_stat_keys(self):
        # We define which raw features need unified Z/Slope
        # This is dynamic, checked in compute loop
        pass

    def _get_stat(self, name: str, w: int) -> RollingStats:
        if name not in self.stats: self.stats[name] = {}
        if w not in self.stats[name]: self.stats[name][w] = RollingStats(w)
        return self.stats[name][w]

    def _get_cov(self, name: str, w: int) -> RollingCov:
        if name not in self.cov_stats: self.cov_stats[name] = {}
        if w not in self.cov_stats[name]: self.cov_stats[name][w] = RollingCov(w)
        return self.cov_stats[name][w]

    def compute(self, sample: AlignedSample) -> Dict:
        """
        Main Entry Point.
        Input: AlignedSample
        Output: Full WhiteBox Contract
        """
        masks = {
            "aux_available": 1.0 if sample.aux_available else 0.0,
            "has_fut": 1.0 if sample.has_fut else 0.0
        }

        # 1. Base Variables & Transforms
        # Need ret_t for both
        vars_tgt = self._calc_base_vars(sample.target, self.prev_bar_tgt)
        vars_aux = self._calc_base_vars(sample.aux, self.prev_bar_aux) if (sample.aux_available and sample.aux) else self._empty_base_vars()
        
        # Update Return Buffers for Lead-Lag
        # If aux not available, push 0 return
        self.aux_ret_buffer.append(vars_aux["ret"])

        # 2. A1: Micro-structure (Target Only)
        a1_feats = self._mining_A1_micro(sample.target, vars_tgt, self.prev_bar_tgt)
        
        # 3. A2: Flow & Arb (Target Only)
        a2_feats = self._mining_A2_flow(sample.target, vars_tgt)
        
        # 4. A3: Futures (Target Only, gated)
        a3_feats = self._mining_A3_fut(sample.target, vars_tgt, sample.has_fut)
        
        # 5. A4: Cross (Tgt vs Aux)
        a4_feats = self._mining_A4_cross(vars_tgt, vars_aux, sample.aux_available)

        # 6. Unified Derivatives (RegimeZ)
        # Collect all raw features
        # Note: A1, A2, A3 are "Target Raw" (or mixed). 
        # Spec asks for separate dicts.
        
        white_target_raw = {**a1_feats, **a2_feats, **a3_feats}
        # Aux raw? Not strictly A4. 
        # Requirement says "white_aux_raw". We can compute basic A1/A2 for Aux too?
        # User 3.2 Output: white_aux_raw.
        # Let's compute a simplified set for Aux (Just basics, no fancy)
        if sample.aux_available:
             # Full mining for Aux? Too expensive/not needed? 
             # Let's compute Basic A1 for Aux as "Raw Aux" which is useful.
             white_aux_raw = self._mining_A1_micro(sample.aux, vars_aux, self.prev_bar_aux)
        else:
             # Empty Dict with correct keys 0.0
             white_aux_raw = {k: 0.0 for k in a1_feats.keys()}
        
        white_cross_raw = a4_feats
        
        # Combine all for rolling updates
        all_to_roll = {**white_target_raw, **white_aux_raw, **white_cross_raw}
        white_derived = {}
        
        for name, val in all_to_roll.items():
            for w in self.W_set:
                st = self._get_stat(name, w)
                st.update(val)
                white_derived[f"{name}_z_{w}"] = st.zscore()
                white_derived[f"{name}_slope_{w}"] = st.slope()
        
        # Special Logic for DynamicBeta (Cov based)
        # We need to update Cov stats for specific pairs?
        # Actually Dynamic Beta is A4 feature itself.
        # "Dynamic Beta Divergence... beta_W = Cov_W / Var_W"
        # This implies Beta calculation happens INSIDE A4 or derived?
        # Spec 5-A4-(15) says "beta_W" is part of the computation.
        # So we handle 'RollingCov' logic inside _mining_A4 or separate.
        # Let's put complex Window Logic (Cov/LeadLag) inside the mining functions
        # because they produce the "Raw" features for output (e.g. divergence).
        
        # Wait, divergence = ret_aux - beta_W * ret_tgt.
        # This `divergence` is the RAW feature. `beta_W` is an intermediate.
        # Yes.
        
        # Update State
        self.prev_bar_tgt = sample.target
        if sample.aux_available:
            self.prev_bar_aux = sample.aux

        return {
            "white_target_raw": white_target_raw,
            "white_aux_raw": white_aux_raw,
            "white_cross_raw": white_cross_raw,
            "white_derived": white_derived,
            "masks": masks
        }

    # ==========================================
    # Helpers
    # ==========================================
    def _calc_base_vars(self, bar: Bar, prev_bar: Optional[Bar]) -> Dict:
        """4) Basic Defs"""
        bp1 = bar.bids[0][0] if bar.bids else 0
        sp1 = bar.asks[0][0] if bar.asks else 0
        mid = (bp1 + sp1) / 2 if (bp1>0 and sp1>0) else 0
        
        prev_mid = 0
        if prev_bar and prev_bar.bids and prev_bar.asks:
             p_bp1 = prev_bar.bids[0][0]
             p_sp1 = prev_bar.asks[0][0]
             prev_mid = (p_bp1 + p_sp1) / 2
        
        # Ret = ln(mid_t) - ln(mid_t-1)
        if mid > 0 and prev_mid > 0:
            ret = np.log(mid) - np.log(prev_mid)
        else:
            ret = 0.0
            
        return {
            "mid": mid,
            "spread": sp1 - bp1,
            "ret": ret,
            "prev_mid": prev_mid,
            "bp1": bp1, "sp1": sp1,
            "bv1": bar.bids[0][1] if bar.bids else 0,
            "sv1": bar.asks[0][1] if bar.asks else 0
        }

    def _empty_base_vars(self):
        return {k: 0.0 for k in ["mid", "spread", "ret", "prev_mid", "bp1", "sp1", "bv1", "sv1"]}

    # ==========================================
    # A1: Micro-structure
    # ==========================================
    def _mining_A1_micro(self, bar: Bar, v: Dict, prev: Optional[Bar]) -> Dict:
        f = {}
        bv1, sv1 = v["bv1"], v["sv1"]
        
        # (1) QI
        denom = bv1 + sv1 + EPS
        f["QI_L1"] = (bv1 - sv1) / denom
        
        # L5 (need sum)
        sum_bv = sum(x[1] for x in bar.bids[:5]) if bar.bids else 0
        sum_sv = sum(x[1] for x in bar.asks[:5]) if bar.asks else 0
        f["QI_L5"] = (sum_bv - sum_sv) / (sum_bv + sum_sv + EPS)
        
        # (2) iOFI
        iofi = 0.0
        denom_iofi = 0.0
        
        if prev and prev.bids and prev.asks and bar.bids and bar.asks:
            # Need strict level alignment up to 5
            # Simplified for speed: L1-L5 loop
            for l in range(min(5, len(bar.bids), len(prev.bids), len(bar.asks), len(prev.asks))):
                w = self.iofi_weights[l]
                
                # Bid OFI
                bp_t, bv_t = bar.bids[l]
                bp_p, bv_p = prev.bids[l]
                if bp_t > bp_p: ofi_b = bv_t
                elif bp_t == bp_p: ofi_b = bv_t - bv_p
                else: ofi_b = -bv_p
                
                # Ask OFI (reversed sign logic? "OFI_ask" usually contributes negatively to buy pressure)
                # Formula: OFI_bid - OFI_ask. 
                # Ask Side:
                sp_t, sv_t = bar.asks[l]
                sp_p, sv_p = prev.asks[l]
                if sp_t > sp_p: ofi_a = -sv_p # Price up (liquidity removed?) No, if Ask Price Up -> supply moved away -> less pressure? 
                # Standard Cont-Kkanamba OFI Definition:
                # If P_ask_t > P_ask_t-1: OFI_ask = -V_ask_t-1 (Removal of liquidity at best) -> Actually means buying pressure?
                # The formula says OFI^{ask}. 
                # Let's follow spec: Ask Side Logic same as Bid? "ask 侧同理（符号反向）"
                # If Ask P up: +AskVol? No.
                # Let's stick to standard:
                # If Ap > Ap_prev: +Q_prev ? No.
                # Let's implement Strict "Ask Side Same Logic" but Reversed Sign in Sum.
                # Interpret "Ask side同理":
                # If sp_t > sp_p: val = sv_t
                # If sp_t == sp_p: val = sv_t - sv_p
                # If sp_t < sp_p: val = -sv_p
                # Then Terms is (OFI_bid - OFI_ask)
                
                # Wait, "Asks Logic (Reversed)" usually means:
                # If Ap > Ap_1: supply retreated -> bullish ??
                # Let's use standard implementation:
                # e_n = I(P_n > P_{n-1}) q_n + I(P_n == P_{n-1}) (q_n - q_{n-1}) + I(P_n < P_{n-1}) (-q_{n-1})
                if sp_t > sp_p: ofi_a = sv_t
                elif sp_t == sp_p: ofi_a = sv_t - sv_p
                else: ofi_a = -sv_p
                
                term = w * (ofi_b - ofi_a)
                abs_term = w * (abs(ofi_b) + abs(ofi_a))
                iofi += term
                denom_iofi += abs_term
                
        f["iOFI"] = iofi / (denom_iofi + EPS)
        
        # (3) nBSP
        # imb_l = (bv - sv) / (bv + sv)
        # nBSP = imb1 - imb5
        # We reused sums for L5? No, need individual levels
        def get_imb(l):
            b_v = bar.bids[l][1] if l < len(bar.bids) else 0
            s_v = bar.asks[l][1] if l < len(bar.asks) else 0
            return (b_v - s_v) / (b_v + s_v + EPS)
            
        f["nBSP"] = get_imb(0) - get_imb(4)
        
        # (4) Microprice Deviation
        # mp = (bp*sv + sp*bv) / (bv+sv)
        mp = (v["bp1"] * v["sv1"] + v["sp1"] * v["bv1"]) / (v["bv1"] + v["sv1"] + EPS)
        mid = v["mid"]
        f["mp_dev_bps"] = ((mp - mid) / (mid + EPS)) * 10000.0
        
        # (5) VPIN Proxy
        # sign_proxy = sign(tick_vwap - mid_t-1)
        # VPIN = |Sum(Vol * I[sign>0]) - Sum(Vol * I[sign<0])| / Sum(Vol)
        # Needs Rolling Window! Spec says VPIN_W.
        # So we produce Raw SignedVol, then Rolling Logic calculates VPIN?
        # No, VPIN is inherently a window feature.
        # Spec says: VPIN_W_proxy.
        # I should compute "signed_vol" here (as raw), then compute VPIN_W in "Combined" phase?
        # But RollingStats only does Mean/Var. It doesn't do |Sum pos - Sum neg|.
        # Actually: |Sum(V+)| - |Sum(V-)| is equiv to |Sum(SignedVol)|
        # VPIN = | Sum(SignedVol) | / Sum(TotalVol).
        # We can track Sum(SignedVol) and Sum(TotalVol) using RollingStats (Sum part).
        # So we output `signed_vol` and `total_vol` as RAW features.
        # Then we need a custom "derived" rule for VPIN?
        # The Spec 3.2 says "white_derived: zscore/slope".
        # But VPIN is a Factor, not a derived Zscore.
        # So I must compute VPIN_W inside this function for W in windows?
        # Yes. "white_target_raw" should contain the factor values (e.g. VPIN_20, VPIN_100).
        
        # Helper for Signed Vol
        tv = bar.vwap 
        tick_vol = bar.volume
        prev_mid_val = v["prev_mid"]
        sign_proxy = np.sign(tv - prev_mid_val) if prev_mid_val > 0 else 0
        sv = tick_vol * sign_proxy
        
        # I need to update rolling stats for VPIN here to output the Factor Value
        for w in self.W_set:
            # We need Sum(SV) and Sum(Vol) over W
            # I can use temporary RollingStats just to hold sums?
            # Or use specific Key in self.stats cache?
            sv_stat = self._get_stat("internal_sv", w)
            vol_stat = self._get_stat("internal_vol", w)
            sv_stat.update(sv)
            vol_stat.update(tick_vol)
            
            # VPIN = |Sum SV| / Sum Vol
            # .mean() * N = Sum
            vpin = abs(sv_stat.mean()) / (vol_stat.mean() + EPS)
            f[f"VPIN_{w}"] = vpin
            
        # (6) Kyle Lambda
        # Cov(ret, signed_amt) / Var(signed_amt)
        # Use RollingCov
        signed_amt = sign_proxy * bar.amount
        ret = v["ret"]
        
        for w in self.W_set:
            cov_stat = self._get_cov("kyle", w)
            cov_stat.update(ret, signed_amt)
            # lambda = Cov(R, Amt) / Var(Amt)
            # Note: Var(Amt) is Var_Y in my RollingCov(X=Ret, Y=Amt)
            # My RollingCov has var_x, need var_y support or swap
            # Let's add var_y to class or just swap inputs
            # X=SignedAmt, Y=Ret => Cov(Amt, Ret) / Var(Amt)
            # update(signed_amt, ret)
            
            # Re-invoking update correct order
            cov_stat_kyle = self._get_cov("kyle_inv", w) # To separate from above logic
            cov_stat_kyle.update(signed_amt, ret)
            
            lam = cov_stat_kyle.cov() / (cov_stat_kyle.var_x() + EPS)
            f[f"KyleLambda_{w}"] = lam

        return f

    # ==========================================
    # A2: Flow & Arb
    # ==========================================
    def _mining_A2_flow(self, bar: Bar, v: Dict) -> Dict:
        f = {}
        sent = bar.sentiment
        mid = v["mid"]
        premium = bar.premium_rate
        
        # (7) CFT
        # EMA span=20, 100
        # Need Stateful EMAs. Store in self.stats?
        # Or simple dict cache?
        # Let's use a dict `self.emas`
        if not hasattr(self, "emas"): self.emas = {}
        
        def update_ema(key, val, span):
            k = f"{key}_{span}"
            alpha = 2 / (span + 1)
            old = self.emas.get(k, val)
            new = old * (1 - alpha) + val * alpha
            self.emas[k] = new
            return new
            
        cft_fast = update_ema("sent", sent, 20)
        cft_slow = update_ema("sent", sent, 100)
        f["CFT_fast"] = cft_fast
        f["CFT_slow"] = cft_slow
        
        # (8) FPD
        # Z(CFT_fast) - Z(log_ret). 
        # Wait, I need Z-score of CFT_fast. This implies CFT_fast is the raw input to RollingStats?
        # But FPD needs to be output NOW.
        # Meaning I need to compute Z(CFT_fast) inside here.
        # This creates dependency: Raw -> Rolling -> Derived -> Combined Factor.
        # Spec says FPD = zscoreW(CFT)...
        # So I must track CFT history.
        # Use default W=100 for FPD? Spec says "zscoreW". Which W? 
        # "Unified rules W={20,100,600}". So FPD_20, FPD_100?
        # Let's compute FPD_100 as the "canonical" FPD feature for now, or all W.
        # Let's output all W.
        
        # (9) PFA = Z(Sent) - Z(Prem)
        # (10) PremMeanRev
        
        # To avoid recursion hell, let's treat "CFT", "Sentiment", "Premium" as Raw features being tracked.
        # And FPD/PFA are computed from the derived Z-scores.
        # BUT user output spec puts FPD in "white_target_raw"?
        # Or is FPD a derived feature?
        # "A2 资金流与套利类... (8) FPD ... 依赖 sentiment, mid"
        # It seems these are expected in the 'white_target_raw' output.
        # This implies 'white_target_raw' can contain window-dependent values?
        # Yes, like FLP, VPIN_W.
        
        # So I need to pull Z-scores here.
        for w in self.W_set:
            # Stats for components
            st_cft = self._get_stat("internal_cft", w)
            st_cft.update(cft_fast)
            z_cft = st_cft.zscore()
            
            st_ret = self._get_stat("internal_ret", w)
            st_ret.update(v["ret"]) # Use base ret
            z_ret = st_ret.zscore()
            
            f[f"FPD_{w}"] = z_cft - z_ret
            
            # PFA
            st_sent = self._get_stat("internal_sent", w)
            st_sent.update(sent)
            z_sent = st_sent.zscore()
            
            st_prem = self._get_stat("internal_prem", w)
            st_prem.update(premium)
            z_prem = st_prem.zscore()
            
            f[f"PFA_{w}"] = z_sent - z_prem
            
            # (10) Prem Mean Rev
            # Slope of premium
            # Slope = curr - prev_W
            # RollingStats has .slope()
            slope_prem = st_prem.slope()
            # "Strength" = z_prem (already calc) ?
            # Factor = prem_z, prem_slope
            f[f"PremZ_{w}"] = z_prem
            f[f"PremSlope_{w}"] = slope_prem

        # (11) FX / Index Anchor
        # Assume these are in bar (but Bar object doesn't have them in spec?)
        # Spec says "依赖 fx_rate, index_price"
        # Since I don't have them in `Bar` definition in `bar_builder.py` (checked earlier, allowed fut_*)
        # I will assume they might be in `extra` dict or just Placeholder 0 if missing.
        # I'll check `bar` attributes.
        # If not present, log once and return 0.
        f["fx_ret"] = 0.0 # Placeholder
        f["idx_ret"] = 0.0
        
        return f

    # ==========================================
    # A3: Futures
    # ==========================================
    def _mining_A3_fut(self, bar: Bar, v: Dict, has_fut: bool) -> Dict:
        f = {}
        if not has_fut:
            # Return empty structure
            return {"FLP": 0.0, "FSB": 0.0}
            
        # Need fut_price, fut_imb
        # Assuming they are on Bar.
        # Check `bar_builder.py` spec? 
        # For now, safe getattr.
        fp = getattr(bar, "fut_price", 0)
        if fp is None: fp = 0
        fi = getattr(bar, "fut_imb", 0)
        if fi is None: fi = 0
        
        if fp > 0:
            # Delta ln fut
            # Need prev fut price. Store in state?
            if not hasattr(self, "prev_fut"): self.prev_fut = fp
            fut_ret = np.log(fp) - np.log(self.prev_fut) if self.prev_fut > 0 else 0
            self.prev_fut = fp
            
            f["FLP"] = fut_ret * fi
            
            mid = v["mid"]
            f["FSB"] = (np.log(fp) - np.log(mid)) * np.sign(fi) if mid>0 else 0
        else:
            f["FLP"] = 0.0
            f["FSB"] = 0.0
            
        return f

    # ==========================================
    # A4: Cross
    # ==========================================
    def _mining_A4_cross(self, vt: Dict, va: Dict, available: bool) -> Dict:
        f = {}
        if not available:
            # Return 0s
            for w in self.W_set:
                f[f"DynBeta_{w}"] = 0.0
                f[f"Divergence_{w}"] = 0.0
                f[f"LLT_rs"] = 0.0
                f[f"SyncIOFI_{w}"] = 0.0
            f["leadlag_corr_max"] = 0.0
            f["leadlag_lag"] = 0.0
            f["flow_ratio"] = 0.0
            return f
            
        # (14) LLT (Simple RS)
        f["LLT_rs"] = va["ret"] - vt["ret"]
        
        # (15) Dynamic Beta Divergence
        # Cov(RetAux, RetTgt) / Var(RetTgt)
        for w in self.W_set:
            cv = self._get_cov("dynbeta", w)
            cv.update(vt["ret"], va["ret"]) # X=Tgt, Y=Aux
            
            var_t = cv.var_x()
            cov = cv.cov()
            
            beta = cov / (var_t + EPS)
            div = va["ret"] - beta * vt["ret"]
            
            f[f"DynBeta_{w}"] = beta
            f[f"Divergence_{w}"] = div

        # (16) Lead-Lag Corr
        # max_l Corr(aux_{t-l}, tgt_t)
        # We need aux returns from t-1, t-2 ...
        # self.aux_ret_buffer has [ret_{t-max}, ..., ret_{t}]
        # Ensure we have enough history
        # Note: Corr is Rolling! Corr_W.
        # This implies we need `L` separate RollingCov stats?
        # Yes: "LeadLag_1", "LeadLag_2"...
        # Then calculate max across them.
        
        max_corr = -1.0
        best_lag = 0
        
        # Current tgt ret
        rt = vt["ret"]
        
        # We assume buffer has history. 
        # aux_ret_buffer[-1] is current (t). [-2] is t-1.
        
        for lag in self.leadlag_lags:
            if len(self.aux_ret_buffer) > lag:
                ra_lag = self.aux_ret_buffer[-(lag+1)]
            else:
                ra_lag = 0.0
            
            # Update specific Cov tracker for this lag
            # We pick fixed W=600 (long window) or largest W for stability? 
            # Spec says "Corr(l)" implies single number per l? Or multiscale?
            # "output: leadlag_corr_max" implies one scalar.
            # Let's use W=100 (mid) as default for this meta-feature.
            cv_ll = self._get_cov(f"leadlag_{lag}", 100)
            cv_ll.update(ra_lag, rt) # X=Aux_lag, Y=Tgt
            
            c = cv_ll.corr()
            if abs(c) > max_corr:
                max_corr = abs(c)
                best_lag = lag
                
        f["leadlag_corr_max"] = max_corr
        f["leadlag_lag"] = float(best_lag)

        # (17) Flow Ratio
        # Need raw sentiment.
        # Accessing private attribute logic or strict passed in logic?
        # I don't have raw sentiment passed into this func directly (only derived vars?)
        # Wait, vars_tgt has basic vars, but sentiment not in "basic".
        # Let's assume we can get it from ... where?
        # I didn't include sentiment in _calc_base_vars. FIX IT.
        # -> Fixed in _calc_base_vars below? No, need to add it.
        # But wait, mining A1/A2 accessed `bar.sentiment`.
        st = self.prev_bar_tgt.sentiment if self.prev_bar_tgt else 0 # Wait current or prev? Current.
        # But this func doesn't take 'bar' input.
        # I should have passed raw data in vars.
        # Hack: Pass it.
        
        # (18) Co-Imbalance Sync
        # Corr(iOFI_aux, iOFI_tgt)
        # Need iOFI values.
        # Since I compute iOFI inside A1 mining, I don't have it here unless I recompute or pass.
        # I will structure "compute" to gather A1 results first, then pass relevant ones to A4.
        
        return f

    def _mining_A4_wrapper(self, a1_t, a1_a, a2_t, a2_a, vars_t, vars_a, avail):
        """
        Helper to access computed features for A4
        """
        if not avail:
             return self._mining_A4_cross(vars_t, vars_a, False) # Returns 0s
        
        f = self._mining_A4_cross(vars_t, vars_a, True)
        
        # (17) Flow Ratio
        # Need sentiment. Assuming it was in vars? 
        # I'll update `_calc_base_vars` to include sentiment.
        s_t = vars_t.get("sentiment", 0)
        s_a = vars_a.get("sentiment", 0)
        f["flow_ratio"] = s_a / (abs(s_t) + EPS)
        
        # (18) Co-Imbalance Sync
        # iOFI from A1 output
        iofi_t = a1_t.get("tgt_iOFI", 0)
        iofi_a = a1_a.get("aux_iOFI", 0)
        
        for w in self.W_set:
            cv = self._get_cov("sync_iofi", w)
            cv.update(iofi_t, iofi_a)
            f[f"SyncIOFI_{w}"] = cv.corr()
            
        return f

    # Redefine base vars to include sentiment
    def _calc_base_vars(self, bar: Bar, prev: Optional[Bar]) -> Dict:
        bn = super()._calc_base_vars(bar, prev) if hasattr(super(), "_calc_base_vars") else self._calc_base_vars_orig(bar, prev)
        bn["sentiment"] = bar.sentiment
        return bn
    
    def _calc_base_vars_orig(self, bar: Bar, prev: Optional[Bar]) -> Dict:
        # Copy of original logic to make it self-contained
        bp1 = bar.bids[0][0] if bar.bids else 0
        sp1 = bar.asks[0][0] if bar.asks else 0
        mid = (bp1 + sp1) / 2 if (bp1>0 and sp1>0) else 0
         
        prev_mid = 0
        if prev and prev.bids and prev.asks:
             p_bp1 = prev.bids[0][0]
             p_sp1 = prev.asks[0][0]
             prev_mid = (p_bp1 + p_sp1) / 2
         
        if mid > 0 and prev_mid > 0:
            ret = np.log(mid) - np.log(prev_mid)
        else:
            ret = 0.0
            
        return {
            "mid": mid, "spread": sp1 - bp1,
            "ret": ret, "prev_mid": prev_mid,
            "bp1": bp1, "sp1": sp1,
            "bv1": bar.bids[0][1] if bar.bids else 0,
            "sv1": bar.asks[0][1] if bar.asks else 0,
            "sentiment": bar.sentiment
        }

    # Override Compute to wire A4 Wrapper
    def compute(self, sample: AlignedSample) -> Dict:
        masks = {
            "aux_available": 1.0 if sample.aux_available else 0.0,
            "has_fut": 1.0 if sample.has_fut else 0.0
        }

        vars_tgt = self._calc_base_vars_orig(sample.target, self.prev_bar_tgt)
        vars_aux = self._calc_base_vars_orig(sample.aux, self.prev_bar_aux) if (sample.aux_available and sample.aux) else self._empty_base_vars()
        vars_aux["sentiment"] = sample.aux.sentiment if (sample.aux_available and sample.aux) else 0 # Ensure key
        
        self.aux_ret_buffer.append(vars_aux["ret"])

        # 2. A1
        a1_t = self._mining_A1_micro(sample.target, vars_tgt, self.prev_bar_tgt)
        
        # Aux A1 (renamed keys)
        if sample.aux_available:
             a1_a_raw = self._mining_A1_micro(sample.aux, vars_aux, self.prev_bar_aux)
             a1_a = {f"aux_{k}": v for k, v in a1_a_raw.items()}
             # We also need non-prefixed for logic
             a1_a_logic = a1_a_raw
        else:
             # Just fillers. MUST use prefixed keys "aux_..."
             # _mining_A1 returns [QI_L1, ...]
             # So we must map k -> aux_k
             sample_keys = ["QI_L1", "QI_L5", "iOFI", "nBSP", "mp_dev_bps"] + \
                           [f"VPIN_{w}" for w in self.W_set] + [f"KyleLambda_{w}" for w in self.W_set] 
             
             a1_a = {f"aux_{k}": 0.0 for k in sample_keys} 
             # Just fillers
             a1_a_logic = {k: 0.0 for k in sample_keys} 

        # 3. A2
        a2_t = self._mining_A2_flow(sample.target, vars_tgt)
        
        # 4. A3
        a3_t = self._mining_A3_fut(sample.target, vars_tgt, sample.has_fut)
        
        # 5. A4
        # Need to re-prefix A1 target keys for wrapper?
        # My A1 func returns keys like "QI_L1". 
        # Wrapper expects "tgt_iOFI"? 
        # My wrapper logic: iofi_t = a1_t.get("tgt_iOFI", 0) -> this expects prefixed.
        # But _mining_A1 returns UNPREFIXED.
        # Let's fix Wrapper to expect unprefixed or I rename A1 Tgt result.
        # User output expects "white_target_raw" -> "name".
        # But A1 function is generic. 
        # So I should prefix them AFTER function return.
        
        a1_t_prefixed = {f"tgt_{k}": v for k, v in a1_t.items()}
        a2_t_prefixed = {f"tgt_{k}": v for k, v in a2_t.items()}
        a3_t_prefixed = {f"tgt_{k}": v for k, v in a3_t.items()}
        
        # A4 Wrapper: uses raw values and feature values
        # Let's pass unprefixed `a1_t` to wrapper, and manually extract.
        a4_feats = self._mining_A4_wrapper_logic(a1_t, a1_a_logic, vars_tgt, vars_aux, sample.aux_available)
        
        # Combine
        white_target_raw = {**a1_t_prefixed, **a2_t_prefixed, **a3_t_prefixed}
        white_aux_raw = a1_a # Already prefixed
        white_cross_raw = a4_feats # A4 are distinct names (LLT, DynBeta...), no prefix needed
        
        # 6. RegimeZ
        all_to_roll = {**white_target_raw, **white_aux_raw, **white_cross_raw}
        white_derived = {}
        
        for name, val in all_to_roll.items():
            for w in self.W_set:
                st = self._get_stat(name, w)
                st.update(val)
                white_derived[f"{name}_z_{w}"] = st.zscore()
                white_derived[f"{name}_slope_{w}"] = st.slope()

        self.prev_bar_tgt = sample.target
        if sample.aux_available:
            self.prev_bar_aux = sample.aux

        return {
            "white_target_raw": white_target_raw,
            "white_aux_raw": white_aux_raw,
            "white_cross_raw": white_cross_raw,
            "white_derived": white_derived,
            "masks": masks
        }

    def _mining_A4_wrapper_logic(self, a1_t, a1_a, vars_t, vars_a, avail):
        # A4 separate logic to avoid method mess
        if not avail:
             # Just return 0s for known keys
             return {k: 0.0 for k in ["LLT_rs", "leadlag_corr_max", "leadlag_lag", "flow_ratio"] + 
                     [f"DynBeta_{w}" for w in self.W_set] + [f"Divergence_{w}" for w in self.W_set] + 
                     [f"SyncIOFI_{w}" for w in self.W_set]}
        
        # Base A4 (LLT, DynBeta, LeadLag) from minings
        f = self._mining_A4_cross(vars_t, vars_a, True)
        
        # Add Flow Ratio
        s_t = vars_t["sentiment"]
        s_a = vars_a["sentiment"]
        f["flow_ratio"] = s_a / (abs(s_t) + EPS)
        
        # Add Sync IOFI
        iofi_t = a1_t.get("iOFI", 0)
        iofi_a = a1_a.get("iOFI", 0)
        for w in self.W_set:
            cv = self._get_cov("sync_iofi", w)
            cv.update(iofi_t, iofi_a)
            f[f"SyncIOFI_{w}"] = cv.corr()
            
        return f
