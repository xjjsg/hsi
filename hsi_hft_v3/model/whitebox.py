import numpy as np
from typing import Dict, List, Optional, Deque, Tuple
from collections import deque
from hsi_hft_v3.model.data_layer import Bar, AlignedSample, BAR_SIZE_S

EPS = 1e-9


class RollingStats:
    """
    O(1) 滚动均值、方差、Z-Score、斜率。
    维护 x 和 x^2 的累计和。
    """

    def __init__(self, window: int):
        self.window = window
        self.values = deque(maxlen=window)
        self.sum_x = 0.0
        self.sum_x2 = 0.0

    def update(self, x: float):
        if not np.isfinite(x):
            x = 0.0

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
        if n == 0:
            return 0.0
        return self.sum_x / n

    def std(self) -> float:
        n = len(self.values)
        if n < 2:
            return 0.0
        mean = self.sum_x / n
        var = (self.sum_x2 / n) - (mean * mean)
        return np.sqrt(max(0.0, var))

    def zscore(self) -> float:
        s = self.std()
        if s < EPS:
            return 0.0
        return (self.values[-1] - self.mean()) / (s + EPS)

    def slope(self) -> float:
        # Simple diff: x_t - x_{t-W}
        if len(self.values) < 2:
            return 0.0
        # Requirement: x_raw - x_raw.shift(W)
        return self.values[-1] - self.values[0]


class RollingCov:
    """
    O(1) 滚动协方差和相关系数 (两个流 X 和 Y)。
    维护 x, y, x^2, y^2, xy 的累计和。
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
        if not np.isfinite(x):
            x = 0.0
        if not np.isfinite(y):
            y = 0.0

        if len(self.vals_x) == self.window:
            old_x = self.vals_x.popleft()
            old_y = self.vals_y.popleft()  # Should sync
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
        if n < 2:
            return 0.0
        mean_x = self.sum_x / n
        mean_y = self.sum_y / n
        # Cov = E[XY] - E[X]E[Y]
        mean_xy = self.sum_xy / n
        return mean_xy - (mean_x * mean_y)

    def var_x(self) -> float:
        n = len(self.vals_x)
        if n < 2:
            return 0.0
        mean = self.sum_x / n
        return (self.sum_x2 / n) - (mean * mean)

    def var_y(self) -> float:
        """🔧 新增：计算Y的方差（Kyle Lambda需要）"""
        n = len(self.vals_y)
        if n < 2:
            return 0.0
        mean = self.sum_y / n
        return (self.sum_y2 / n) - (mean * mean)

    def corr(self) -> float:
        c = self.cov()
        vx = self.var_x()

        # Calculate Var Y locally
        n = len(self.vals_y)
        if n < 2:
            return 0.0
        mean_y = self.sum_y / n
        vy = (self.sum_y2 / n) - (mean_y * mean_y)

        if vx <= 0 or vy <= 0:
            return 0.0
        return c / (np.sqrt(vx) * np.sqrt(vy) + EPS)


class WhiteBoxFeatureFactory:
    """
    19 个因子 + 统一衍生规则
    V5 规范的严格实现。
    """

    def __init__(self):
        # Config (Load from Central Config)
        from hsi_hft_v3.config import FeatureConfig, TICK_SIZE

        cfg = FeatureConfig()

        self.W_set = cfg.windows
        self.L_DEPTH = cfg.depth_levels
        self.tick_size = TICK_SIZE
        self.iofi_weights = np.array(cfg.iofi_weights)
        self.leadlag_lags = cfg.lead_lag_lags

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

    def get_derived_keys(self) -> List[str]:
        """
        返回确定性的特征键列表 (审计要求)。
        顺序: 对每个原始特征 (排序)，对每个窗口 (排序)，先 z 后 slope。
        """
        # 基于挖掘逻辑硬编码的原始特征名称
        # 此列表必须与 compute() 中生成的匹配
        raw_names = [
            "QI_L1",
            "QI_L5",
            "iOFI",
            "nBSP",
            "mp_dev_bps",
            "CFT_fast",
            "CFT_slow",
            "LLT_rs",
            "leadlag_corr_max",
            "leadlag_lag",
            "flow_ratio",
            "idx_ret",
            "fx_ret",
        ]
        # Weighted raw features (VPIN_{w}, etc)
        for w in self.W_set:
            raw_names.append(f"VPIN_{w}")
            raw_names.append(f"KyleLambda_{w}")
            raw_names.append(f"FPD_{w}")
            raw_names.append(f"PFA_{w}")
            raw_names.append(f"PremZ_{w}")
            raw_names.append(f"PremSlope_{w}")
            raw_names.append(f"DynBeta_{w}")
            raw_names.append(f"Divergence_{w}")
            raw_names.append(f"SyncIOFI_{w}")
            # Aux keys
            raw_names.append(f"aux_QI_L1")
            raw_names.append(f"aux_QI_L5")
            raw_names.append(f"aux_iOFI")
            raw_names.append(f"aux_nBSP")
            raw_names.append(f"aux_mp_dev_bps")
            raw_names.append(f"aux_VPIN_{w}")
            raw_names.append(f"aux_KyleLambda_{w}")

        # Futures
        raw_names.append("FLP")
        raw_names.append("FSB")

        raw_names.sort()

        final_keys = []
        for name in raw_names:
            for w in [20, 100, 600]:  # Explicit W_set for stability
                final_keys.append(f"{name}_z_{w}")
                final_keys.append(f"{name}_slope_{w}")

        return final_keys

    def _init_stat_keys(self):
        # We define which raw features need unified Z/Slope
        # This is dynamic, checked in compute loop
        pass

    def _get_stat(self, name: str, w: int) -> RollingStats:
        if name not in self.stats:
            self.stats[name] = {}
        if w not in self.stats[name]:
            self.stats[name][w] = RollingStats(w)
        return self.stats[name][w]

    def _get_cov(self, name: str, w: int) -> RollingCov:
        if name not in self.cov_stats:
            self.cov_stats[name] = {}
        if w not in self.cov_stats[name]:
            self.cov_stats[name][w] = RollingCov(w)
        return self.cov_stats[name][w]

    def compute(self, sample: AlignedSample) -> Dict:
        """
        主入口点。
        输入: AlignedSample
        输出: 完整的 WhiteBox 契约
        """
        masks = {
            "aux_available": 1.0 if sample.aux_available else 0.0,
            "has_fut": 1.0 if sample.has_fut else 0.0,
        }

        # 1. 基础变量与变换
        # 两者都需要 ret_t
        vars_tgt = self._calc_base_vars(sample.target, self.prev_bar_tgt)
        vars_aux = (
            self._calc_base_vars(sample.aux, self.prev_bar_aux)
            if (sample.aux_available and sample.aux)
            else self._empty_base_vars()
        )

        # 更新 Lead-Lag 的收益率缓冲区
        # 如果 aux 不可用，推入 0 收益
        self.aux_ret_buffer.append(vars_aux["ret"])

        # 2. A1: 微观结构 (Micro-structure) - 仅 Target
        a1_feats = self._mining_A1_micro(sample.target, vars_tgt, self.prev_bar_tgt)

        # 3. A2: 资金流与套利 (Flow & Arb) - 仅 Target
        a2_feats = self._mining_A2_flow(sample.target, vars_tgt)

        # 4. A3: 期货 (Futures) - 仅 Target (门控)
        a3_feats = self._mining_A3_fut(sample.target, vars_tgt, sample.has_fut)

        # 5. A4: 交叉流 (Cross Tgt vs Aux)
        a4_feats = self._mining_A4_cross(vars_tgt, vars_aux, sample.aux_available)

        # 6. 统一衍生 (Unified Derivatives - RegimeZ)
        # 收集所有原始特征 (Raw Features)
        # 注意: A1, A2, A3 是 "Target Raw" (或混合)。
        # 规范要求分开的字典。

        white_target_raw = {**a1_feats, **a2_feats, **a3_feats}
        # Aux raw? 并非严格的 A4。
        # 需求说 "white_aux_raw"。我们可以计算 Aux 的基本 A1/A2 吗？
        # 用户需求 3.2 输出: white_aux_raw。
        # 让我们为 Aux 计算一个简化集 (仅基础，无花哨功能)
        if sample.aux_available:
            # [Research] Aux 全量挖掘？当前仅计算基础 A1 作为 "Raw Aux" (效率考量)。
            # 让我们计算 Aux 的基础 A1 作为 "Raw Aux"，这很有用。
            white_aux_raw = self._mining_A1_micro(
                sample.aux, vars_aux, self.prev_bar_aux
            )
        else:
            # 空字典，带有正确的键 (值为 0.0)
            white_aux_raw = {k: 0.0 for k in a1_feats.keys()}

        white_cross_raw = a4_feats

        # 合并所有特征以进行滚动更新
        all_to_roll = {**white_target_raw, **white_aux_raw, **white_cross_raw}
        white_derived = {}

        for name, val in all_to_roll.items():
            for w in self.W_set:
                st = self._get_stat(name, w)
                st.update(val)
                white_derived[f"{name}_z_{w}"] = st.zscore()
                white_derived[f"{name}_slope_{w}"] = st.slope()

        # DynamicBeta 的特殊逻辑 (基于 Cov)
        # 我们是否需要更新特定对的 Cov 统计？
        # 实际上 Dynamic Beta 是 A4 特征本身。
        # "Dynamic Beta Divergence... beta_W = Cov_W / Var_W"
        # 这意味着 Beta 计算发生在 A4 内部或衍生中？
        # 规范 5-A4-(15) 说 "beta_W" 是计算的一部分。
        # 所以我们在 _mining_A4 或单独处理 'RollingCov' 逻辑。
        # 让我们把复杂的窗口逻辑 (Cov/LeadLag) 放在挖掘函数内部
        # 因为它们生成用于输出的 "Raw" 特征 (例如 divergence)。

        # Wait, divergence = ret_aux - beta_W * ret_tgt.
        # 这个 `divergence` 是原始特征。`beta_W` 是中间变量。
        # 是的。

        # 更新状态
        self.prev_bar_tgt = sample.target
        if sample.aux_available:
            self.prev_bar_aux = sample.aux

        return {
            "white_target_raw": white_target_raw,
            "white_aux_raw": white_aux_raw,
            "white_cross_raw": white_cross_raw,
            "white_derived": white_derived,
            "masks": masks,
        }

    # ==========================================
    # Helpers
    # ==========================================
    def _calc_base_vars(self, bar: Bar, prev_bar: Optional[Bar]) -> Dict:
        """4) 基础定义"""
        bp1 = bar.bids[0][0] if bar.bids else 0
        sp1 = bar.asks[0][0] if bar.asks else 0
        mid = (bp1 + sp1) / 2 if (bp1 > 0 and sp1 > 0) else 0

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
            "bp1": bp1,
            "sp1": sp1,
            "bv1": bar.bids[0][1] if bar.bids else 0,
            "sv1": bar.asks[0][1] if bar.asks else 0,
        }

    def _empty_base_vars(self):
        return {
            k: 0.0
            for k in ["mid", "spread", "ret", "prev_mid", "bp1", "sp1", "bv1", "sv1"]
        }

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
            for l in range(
                min(5, len(bar.bids), len(prev.bids), len(bar.asks), len(prev.asks))
            ):
                w = self.iofi_weights[l]

                # Bid OFI
                bp_t, bv_t = bar.bids[l]
                bp_p, bv_p = prev.bids[l]
                if bp_t > bp_p:
                    ofi_b = bv_t
                elif bp_t == bp_p:
                    ofi_b = bv_t - bv_p
                else:
                    ofi_b = -bv_p

                # Ask OFI (reversed sign logic? "OFI_ask" usually contributes negatively to buy pressure)
                # Formula: OFI_bid - OFI_ask.
                # Ask Side:
                sp_t, sv_t = bar.asks[l]
                sp_p, sv_p = prev.asks[l]
                if sp_t > sp_p:
                    ofi_a = (
                        -sv_p
                    )  # Price up (liquidity removed?) No, if Ask Price Up -> supply moved away -> less pressure?
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
                if sp_t > sp_p:
                    ofi_a = sv_t
                elif sp_t == sp_p:
                    ofi_a = sv_t - sv_p
                else:
                    ofi_a = -sv_p

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

        # (3b) Spread BPS (Explicit for Policy)
        # spread_bps = (ask - bid) / mid * 10000
        mid = v["mid"]
        spread = v["spread"]
        if mid > 0:
            f["spread_bps"] = (spread / mid) * 10000.0
        else:
            f["spread_bps"] = 0.0

        # (4) Microprice Deviation
        # mp = (bp*sv + sp*bv) / (bv+sv)
        mp = (v["bp1"] * v["sv1"] + v["sp1"] * v["bv1"]) / (v["bv1"] + v["sv1"] + EPS)
        mid = v["mid"]
        f["mp_dev_bps"] = ((mp - mid) / (mid + EPS)) * 10000.0

        # (5) VPIN 代理
        # sign_proxy = sign(tick_vwap - mid_t-1)
        # VPIN = |Sum(Vol * I[sign>0]) - Sum(Vol * I[sign<0])| / Sum(Vol)
        # 需要滚动窗口！规范说 VPIN_W。
        # 所以我们产生原始 SignedVol，然后滚动逻辑计算 VPIN？
        # 不，VPIN 本质上是一个窗口特征。
        # 规范说: VPIN_W_proxy。
        # 我应该在这里计算 "signed_vol" (作为 raw)，然后在 "Combined" 阶段计算 VPIN_W？
        # 但 RollingStats 只做 Mean/Var。它不做 |Sum pos - Sum neg|。
        # 实际上: |Sum(V+)| - |Sum(V-)| 等价于 |Sum(SignedVol)|
        # VPIN = | Sum(SignedVol) | / Sum(TotalVol).
        # We can track Sum(SignedVol) and Sum(TotalVol) using RollingStats (Sum part).
        # So we output `signed_vol` and `total_vol` as RAW features.
        # Then we need a custom "derived" rule for VPIN?
        # The Spec 3.2 says "white_derived: zscore/slope".
        # But VPIN is a Factor, not a derived Zscore.
        # So I must compute VPIN_W inside this function for W in windows?
        # Yes. "white_target_raw" should contain the factor values (e.g. VPIN_20, VPIN_100).

        # 🔧 修复：Helper for Signed Vol
        # 使用mid价格变化方向而非vwap
        curr_mid = bar.mid
        tick_vol = bar.volume
        prev_mid_val = v["prev_mid"]

        # 🔧 修复：使用mid-to-mid的变化判断方向
        # 如果当前mid > prev_mid，说明价格上涨，买盘主导，signed volume为正
        # 如果curr mid < prev_mid，说明价格下跌，卖盘主导，signed volume为负
        if prev_mid_val > 0 and curr_mid > 0:
            price_change = curr_mid - prev_mid_val
            sign_proxy = np.sign(price_change)
        else:
            sign_proxy = 0

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

            # 🔧 修复：VPIN = |Sum SV| / Sum Vol
            # sum = mean * n, 但这里我们直接用mean也可以因为分子分母的n会约掉
            # 但为了数值稳定性，用sum更好
            sum_sv = sv_stat.sum_x  # 直接访问累计和
            sum_vol = vol_stat.sum_x

            if sum_vol > EPS:
                vpin = abs(sum_sv) / sum_vol
            else:
                vpin = 0.0

            f[f"VPIN_{w}"] = vpin

        # (6) Kyle Lambda
        # Cov(ret, signed_vol) / Var(signed_vol)
        # 🔧 修复：使用signed_vol而非signed_amt
        # sv已在L518定义
        ret = v["ret"]

        for w in self.W_set:
            # 🔧 修复：正确的Kyle Lambda公式
            # λ = Cov(ret, signed_vol) / Var(signed_vol)
            cov_stat = self._get_cov(f"kyle_{w}", w)
            cov_stat.update(ret, sv)  # X=ret, Y=signed_vol

            # 🔧 修复：使用var_y()而非var_x()
            cov_val = cov_stat.cov()
            var_sv = cov_stat.var_y()  # Var of signed volume

            if abs(var_sv) > EPS:
                lam = cov_val / var_sv
            else:
                lam = 0.0

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
        # 需要有状态的 EMA。存储在 self.stats 中？
        # 或者简单的字典缓存？
        # 让我们使用字典 `self.emas`
        if not hasattr(self, "emas"):
            self.emas = {}

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
        # 等等，我需要 CFT_fast 的 Z-score。这意味着 CFT_fast 是 RollingStats 的原始输入？
        # 但 FPD 需要立即输出。
        # 意味着我必须在这里计算 Z(CFT_fast)。
        # 这创建了依赖关系：Raw -> Rolling -> Derived -> Combined Factor。
        # 规范说 FPD = zscoreW(CFT)...
        # 所以我必须跟踪 CFT 历史。
        # 对 FPD 使用默认 W=100？规范说 "zscoreW"。哪个 W？
        # "统一规则 W={20,100,600}"。所以 FPD_20, FPD_100?
        # 让我们暂时计算 FPD_100 作为 "canonical" FPD 特征，或全部 W。
        # 让我们输出所有 W。

        # (9) PFA = Z(Sent) - Z(Prem)
        # (10) PremMeanRev

        # 为避免递归地狱，让我们将 "CFT", "Sentiment", "Premium" 视为正在跟踪的原始特征。
        # FPD/PFA 根据衍生的 Z-scores 计算。
        # 但用户输出规范将 FPD 放在 "white_target_raw" 中？
        # 还是 FPD 是衍生特征？
        # "A2 资金流与套利类... (8) FPD ... 依赖 sentiment, mid"
        # 看来这些应包含在 'white_target_raw' 输出中。
        # 这意味着 'white_target_raw' 可以包含窗口相关的值？
        # 是的，比如 FLP, VPIN_W。

        # 所以我需要在这里提取 Z-scores。
        for w in self.W_set:
            # Stats for components
            st_cft = self._get_stat("internal_cft", w)
            st_cft.update(cft_fast)
            z_cft = st_cft.zscore()

            st_ret = self._get_stat("internal_ret", w)
            st_ret.update(v["ret"])  # Use base ret
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
        curr_idx = getattr(bar, "index_price", 0)
        curr_fx = getattr(bar, "fx_rate", 0)

        # State Init (if first bar)
        if not hasattr(self, "prev_idx_price"):
            self.prev_idx_price = curr_idx
        if not hasattr(self, "prev_fx_rate"):
            self.prev_fx_rate = curr_fx

        # Log Ret
        if curr_idx > 0 and self.prev_idx_price > 0:
            f["idx_ret"] = np.log(curr_idx / self.prev_idx_price)
        else:
            f["idx_ret"] = 0.0

        if curr_fx > 0 and self.prev_fx_rate > 0:
            f["fx_ret"] = np.log(curr_fx / self.prev_fx_rate)
        else:
            f["fx_ret"] = 0.0

        # Update State
        self.prev_idx_price = curr_idx
        self.prev_fx_rate = curr_fx

        return f

    # ==========================================
    # A3: Futures
    # ==========================================
    def _mining_A3_fut(self, bar: Bar, v: Dict, has_fut: bool) -> Dict:
        f = {}
        if not has_fut:
            # Return empty structure
            return {"FLP": 0.0, "FSB": 0.0}

        # 需要 fut_price, fut_imb
        # 假设它们在 Bar 上。
        # 检查 `bar_builder.py` 规范？
        # 目前使用安全的 getattr。
        fp = getattr(bar, "fut_price", 0)
        if fp is None:
            fp = 0
        fi = getattr(bar, "fut_imb", 0)
        if fi is None:
            fi = 0

        if fp > 0:
            # Delta ln fut
            # 需要 prev fut price。存储在状态中？
            if not hasattr(self, "prev_fut"):
                self.prev_fut = fp
            fut_ret = np.log(fp) - np.log(self.prev_fut) if self.prev_fut > 0 else 0
            self.prev_fut = fp

            f["FLP"] = fut_ret * fi

            mid = v["mid"]
            f["FSB"] = (np.log(fp) - np.log(mid)) * np.sign(fi) if mid > 0 else 0
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

        # (14) LLT (简单 RS)
        f["LLT_rs"] = va["ret"] - vt["ret"]

        # (15) Dynamic Beta Divergence
        # Cov(RetAux, RetTgt) / Var(RetTgt)
        for w in self.W_set:
            cv = self._get_cov("dynbeta", w)
            cv.update(vt["ret"], va["ret"])  # X=Tgt, Y=Aux

            var_t = cv.var_x()
            cov = cv.cov()

            beta = cov / (var_t + EPS)
            div = va["ret"] - beta * vt["ret"]

            f[f"DynBeta_{w}"] = beta
            f[f"Divergence_{w}"] = div

        # (16) Lead-Lag Corr
        # max_l Corr(aux_{t-l}, tgt_t)
        # 我们需要 aux t-1, t-2... 的收益率
        # self.aux_ret_buffer 包含 [ret_{t-max}, ..., ret_{t}]
        # 确保有足够的历史
        # 注意: Corr 是滚动的！Corr_W。
        # 这意味着我们需要 `L` 个单独的 RollingCov 统计？
        # 是的: "LeadLag_1", "LeadLag_2"...
        # 然后取最大值。

        max_corr = -1.0
        best_lag = 0

        # Current tgt ret
        rt = vt["ret"]

        # We assume buffer has history.
        # aux_ret_buffer[-1] is current (t). [-2] is t-1.

        for lag in self.leadlag_lags:
            if len(self.aux_ret_buffer) > lag:
                ra_lag = self.aux_ret_buffer[-(lag + 1)]
            else:
                ra_lag = 0.0

            # 更新此时延的特定 Cov 追踪器
            # 为了稳定性，我们选择固定 W=600 (长窗口) 或最大 W？
            # 规范说 "Corr(l)" 暗示每个 l 一个值？还是多尺度？
            # "output: leadlag_corr_max" 暗示一个标量。
            # 让我们使用 W=100 (中等) 作为此元特征的默认值。
            cv_ll = self._get_cov(f"leadlag_{lag}", 100)
            cv_ll.update(ra_lag, rt)  # X=Aux_lag, Y=Tgt

            c = cv_ll.corr()
            if abs(c) > max_corr:
                max_corr = abs(c)
                best_lag = lag

        f["leadlag_corr_max"] = max_corr
        f["leadlag_lag"] = float(best_lag)

        # (17) 资金流比率 (Flow Ratio)
        # 需要原始情绪值。在此函数中通过 hack 获取 (此处实际上应由 wrapper 传入)
        st = self.prev_bar_tgt.sentiment if self.prev_bar_tgt else 0

        # (18) Co-Imbalance Sync
        # Corr(iOFI_aux, iOFI_tgt)
        # Need iOFI values.
        # Since I compute iOFI inside A1 mining, I don't have it here unless I recompute or pass.
        # I will structure "compute" to gather A1 results first, then pass relevant ones to A4.

        return f

    def _mining_A4_wrapper(self, a1_t, a1_a, a2_t, a2_a, vars_t, vars_a, avail):
        """
        A4 特征计算辅助函数 (访问已计算的特征)
        """
        if not avail:
            return self._mining_A4_cross(vars_t, vars_a, False)  # Returns 0s

        f = self._mining_A4_cross(vars_t, vars_a, True)

        # (17) 资金流比率 (Flow Ratio)
        # 使用来自 vars 的 sentiment
        s_t = vars_t.get("sentiment", 0)
        s_a = vars_a.get("sentiment", 0)
        f["flow_ratio"] = s_a / (abs(s_t) + EPS)

        # (18) Co-Imbalance Sync
        # 使用 A1 输出的 iOFI
        iofi_t = a1_t.get("iOFI", 0)
        iofi_a = a1_a.get("iOFI", 0)

        for w in self.W_set:
            cv = self._get_cov("sync_iofi", w)
            cv.update(iofi_t, iofi_a)
            f[f"SyncIOFI_{w}"] = cv.corr()

        return f

    # 重定义基础变量以包含 sentiment
    def _calc_base_vars(self, bar: Bar, prev: Optional[Bar]) -> Dict:
        bn = (
            super()._calc_base_vars(bar, prev)
            if hasattr(super(), "_calc_base_vars")
            else self._calc_base_vars_orig(bar, prev)
        )
        bn["sentiment"] = bar.sentiment
        return bn

    def _calc_base_vars_orig(self, bar: Bar, prev: Optional[Bar]) -> Dict:
        # 原始逻辑副本 (使其自包含)
        bp1 = bar.bids[0][0] if bar.bids else 0
        sp1 = bar.asks[0][0] if bar.asks else 0
        mid = (bp1 + sp1) / 2 if (bp1 > 0 and sp1 > 0) else 0

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
            "mid": mid,
            "spread": sp1 - bp1,
            "ret": ret,
            "prev_mid": prev_mid,
            "bp1": bp1,
            "sp1": sp1,
            "bv1": bar.bids[0][1] if bar.bids else 0,
            "sv1": bar.asks[0][1] if bar.asks else 0,
            "sentiment": bar.sentiment,
        }

    # 重写 Compute 以连接 A4 Wrapper
    def compute(self, sample: AlignedSample) -> Dict:
        masks = {
            "aux_available": 1.0 if sample.aux_available else 0.0,
            "has_fut": 1.0 if sample.has_fut else 0.0,
        }

        vars_tgt = self._calc_base_vars_orig(sample.target, self.prev_bar_tgt)
        vars_aux = (
            self._calc_base_vars_orig(sample.aux, self.prev_bar_aux)
            if (sample.aux_available and sample.aux)
            else self._empty_base_vars()
        )
        vars_aux["sentiment"] = (
            sample.aux.sentiment if (sample.aux_available and sample.aux) else 0
        )  # Ensure key

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
            sample_keys = (
                ["QI_L1", "QI_L5", "iOFI", "nBSP", "mp_dev_bps"]
                + [f"VPIN_{w}" for w in self.W_set]
                + [f"KyleLambda_{w}" for w in self.W_set]
            )

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
        a4_feats = self._mining_A4_wrapper_logic(
            a1_t, a1_a_logic, vars_tgt, vars_aux, sample.aux_available
        )

        # Combine
        white_target_raw = {**a1_t_prefixed, **a2_t_prefixed, **a3_t_prefixed}
        white_aux_raw = a1_a  # Already prefixed
        white_cross_raw = (
            a4_feats  # A4 are distinct names (LLT, DynBeta...), no prefix needed
        )

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
            "masks": masks,
        }

    def _mining_A4_wrapper_logic(self, a1_t, a1_a, vars_t, vars_a, avail):
        # A4 separate logic to avoid method mess
        if not avail:
            # Just return 0s for known keys
            return {
                k: 0.0
                for k in ["LLT_rs", "leadlag_corr_max", "leadlag_lag", "flow_ratio"]
                + [f"DynBeta_{w}" for w in self.W_set]
                + [f"Divergence_{w}" for w in self.W_set]
                + [f"SyncIOFI_{w}" for w in self.W_set]
            }

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
